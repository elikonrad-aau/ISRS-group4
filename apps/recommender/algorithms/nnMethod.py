import numpy as np
import pickle
from pathlib import Path
from django.conf import settings
from scipy.sparse import load_npz
from apps.data.models import Movie

CACHE_DIR = settings.BASE_DIR / "apps" / "recommender" / "embeddings" / "item_knn"
KNN_PATH = CACHE_DIR / "knn_model.pkl"
IDS_PATH = CACHE_DIR / "movie_ids.pkl"
MATRIX_PATH = CACHE_DIR / "rating_matrix_sparse.npz"

_knn_model = None
_movie_ids_map = {}  # { 'movie_id_str': index } -> O(1) lookup
_normalized_ids_list = []  # [ 'id1', 'id2', ... ] -> Index -> ID mapping
_sparse_matrix = None
_is_loaded = False


def _ensure_model_loaded():
    global _knn_model, _movie_ids_map, _normalized_ids_list, _sparse_matrix, _is_loaded
    if _is_loaded:
        return
    try:
        if not all([KNN_PATH.exists(), IDS_PATH.exists(), MATRIX_PATH.exists()]):
            raise FileNotFoundError(f"Model files missing in {CACHE_DIR}")
        _sparse_matrix = load_npz(MATRIX_PATH)
        with open(KNN_PATH, 'rb') as f:
            _knn_model = pickle.load(f)
        with open(IDS_PATH, 'rb') as f:
            raw_ids = pickle.load(f)

        _normalized_ids_list = [str(x) for x in raw_ids]
        _movie_ids_map = {mid: idx for idx, mid in enumerate(_normalized_ids_list)}

        _is_loaded = True
    except Exception as e:
        raise e


def recommend_item_knn(reference_movie_id, limit=20, top_n_neighbors=50):
    global _is_loaded
    try:
        _ensure_model_loaded()
    except Exception as e:
        print(f"Model load failed: {e}")
        return []  # CRITICAL FIX: Return empty list on crash

    ref_id_str = str(reference_movie_id)

    if ref_id_str not in _movie_ids_map:
        #  prevents crashes if the ID is valid in DB but not in the model yet
        try:
            movie = Movie.objects.get(movie_id=ref_id_str)
            return []
        except Movie.DoesNotExist:
            return []  # Movie doesn't exist at all

    ref_index = _movie_ids_map[ref_id_str]

    col_vector = _sparse_matrix[:, ref_index]
    ref_sample = col_vector.T.tocsr()

    distances, indices = _knn_model.kneighbors(ref_sample, n_neighbors=top_n_neighbors + 1)
    similarities = 1 - distances.flatten()

    actual_count = len(indices[0]) - 1
    if actual_count == 0:
        return []

    count_to_take = min(limit, actual_count)

    rec_indices = indices[0, 1: count_to_take + 1]
    rec_similarities = similarities[1: count_to_take + 1]

    rec_movie_ids = [_normalized_ids_list[i] for i in rec_indices]

    movies_qs = Movie.objects.filter(movie_id__in=rec_movie_ids)
    movies_dict = {str(m.movie_id): m for m in movies_qs}

    recommendations = []
    for idx, sim in zip(rec_indices, rec_similarities):
        mid = _normalized_ids_list[idx]
        mid_str = str(mid)

        if mid_str in movies_dict:
            movie = movies_dict[mid_str]
            movie.similarity_score = float(sim)
            recommendations.append(movie)
        else:
            continue

    return recommendations