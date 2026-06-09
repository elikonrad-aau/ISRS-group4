import numpy as np
import pickle
from pathlib import Path
from django.conf import settings
from scipy.sparse import load_npz
from apps.data.models import Movie

CACHE_DIR = settings.BASE_DIR / "apps" / "recommender" / "embeddings" / "item_knn"


def load_db_model():
    knn_path = CACHE_DIR / "knn_model.pkl"
    ids_path = CACHE_DIR / "movie_ids.pkl"
    matrix_path = CACHE_DIR / "rating_matrix_sparse.npz"

    if not all([knn_path.exists(), ids_path.exists(), matrix_path.exists()]):
        raise FileNotFoundError(f"Model files missing in {CACHE_DIR}")

    with open(knn_path, 'rb') as f:
        knn_model = pickle.load(f)

    with open(ids_path, 'rb') as f:
        movie_ids = pickle.load(f)

    if movie_ids and not isinstance(movie_ids[0], str):
        print(f"WARNING: movie_ids contains {type(movie_ids[0])}s. Converting to strings...")
        movie_ids = [str(m) for m in movie_ids]

    sparse_matrix = load_npz(matrix_path)
    return knn_model, movie_ids, sparse_matrix


def recommend_item_knn(reference_movie_id, limit=20, top_n_neighbors=50):
    try:
        knn_model, movie_ids, sparse_matrix = load_db_model()
    except FileNotFoundError as e:
        print(f"CRITICAL: {e}")
        return []

    # Convert reference ID to STRING to match the loaded list
    ref_id_str = str(reference_movie_id)

    if ref_id_str not in movie_ids:
        # Fallback: Try integer check just in case
        ref_id_int = int(reference_movie_id)
        if ref_id_int in movie_ids:  # This only works if list has ints
            print("DEBUG: Found via integer match. Converting list to int? No, let's fix the build script.")

        try:
            movie = Movie.objects.get(movie_id=ref_id_str)
            movie.similarity_score = 0.0
            return []
        except Movie.DoesNotExist:
            return []

    # Find Index
    ref_index = movie_ids.index(ref_id_str)
    # Get the column vector (Users x 1)
    col_vector = sparse_matrix[:, ref_index]

    # The model was fitted on .T (Movies x Users)
    ref_sample = col_vector.T.tocsr()
    distances, indices = knn_model.kneighbors(ref_sample, n_neighbors=top_n_neighbors + 1)
    similarities = 1 - distances.flatten()

    # Skip the first one (it's the movie itself)
    rec_indices = indices.flatten()[1:limit + 1]
    rec_similarities = similarities[1:limit + 1]

    rec_movie_ids = [movie_ids[i] for i in rec_indices]
    movies_qs = Movie.objects.filter(movie_id__in=rec_movie_ids)
    movies_dict = {str(m.movie_id): m for m in movies_qs}

    recommendations = []
    for idx, sim in zip(rec_indices, rec_similarities):
        mid = movie_ids[idx]
        mid_str = str(mid)

        if mid_str in movies_dict:
            movie = movies_dict[mid_str]
            movie.similarity_score = float(sim)
            recommendations.append(movie)
        else:
            print(f"Warning: Movie {mid} found in model but not in DB.")

    return recommendations