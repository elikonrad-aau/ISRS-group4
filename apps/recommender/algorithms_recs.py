import os
import ast
import requests
import json
import numpy as np
from django.http import JsonResponse
from django.conf import settings

from apps.data.models import Movie, MovieLink, MovieMetadata
from apps.recommender.cast_overlap import CastOverlapRecommender


def get_recommendation_rows(reference_movie_id, limit=20):
    '''
    title -> name of the algorithm for the frontend
    algorithm -> slug for the algorithm
    movies -> list of movies returned from the algorithm
    '''

    # all recommendations
    rows = []

    # baseline algorithm
    rows.append({
        "title": "TMDB API Recommendations",
        "algorithm": "tmdb",
        "movies": recommend_by_tmdb(reference_movie_id, limit),
    })

    # TODO - Add Algorithms
    # function 1 algorithm – Elisabeth

    # function 2 algorithm
    rows.append({
        "title": "Shared Cast Overlap",
        "algorithm": "tmdb",
        "movies": recommend_cast_overlap(reference_movie_id, limit),
    })

    # function 3 algorithm – Julian

    # function 4 algorithm
    rows.append({
        "title": "Visual Image Similarity",
        "algorithm": "image_similarity",
        "movies": recommend_image_similarity(reference_movie_id, limit),
    })

    # function 5 algorithm – Elisabeth


    # function 6 algorithm –  ???

    # collection – only append if the reference movie is part of a collection
    collection_movies = recommend_by_collection(reference_movie_id, limit)

    if collection_movies:
        rows.append({
            "title": "Have you seen these?",
            "algorithm": "collection",
            "movies": collection_movies,
        })


    return rows


# baseline algorithm – implementation
def recommend_by_tmdb(reference_movie_id, limit=20):
    token = os.environ.get("TMDB_API_TOKEN")

    if not token:
        return []

    reference_link = MovieLink.objects.filter(
        movie__movie_id=reference_movie_id,
        tmdb_id__isnull=False,
    ).first()

    if not reference_link:
        return []

    matched_movies = []
    seen_movie_ids = set()
    page = 1
    max_pages = 5

    while len(matched_movies) < limit and page <= max_pages:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{reference_link.tmdb_id}/recommendations",
            headers={
                "Authorization": f"Bearer {token}",
                "accept": "application/json",
            },
            params={
                "page": page,
            },
            timeout=10,
        )

        if response.status_code != 200:
            break

        data = response.json()

        tmdb_ids = [
            item["id"]
            for item in data.get("results", [])
        ]

        if not tmdb_ids:
            break

        links = (
            MovieLink.objects
            .filter(tmdb_id__in=tmdb_ids)
            .select_related("movie")
        )

        movie_by_tmdb_id = {
            link.tmdb_id: link.movie
            for link in links
        }

        for tmdb_id in tmdb_ids:
            movie = movie_by_tmdb_id.get(tmdb_id)

            if not movie:
                continue

            if movie.movie_id == reference_movie_id:
                continue

            if movie.movie_id in seen_movie_ids:
                continue

            matched_movies.append(movie)
            seen_movie_ids.add(movie.movie_id)

            if len(matched_movies) >= limit:
                break

        if page >= data.get("total_pages", 1):
            break

        page += 1

    return matched_movies


def recommend_cast_overlap(reference_movie_id, limit=20):
    # TODO: crew, collections
    recommender = CastOverlapRecommender()
    result = recommender.recommend_cast_overlap(reference_movie_id, limit=limit)

    movie_objects = []
    for rec in result:
        try:
            movie = Movie.objects.get(movie_id=rec['MovieID'])
            movie.overlap_count = rec['OverlapCount']
            movie.average_rating = rec['AverageRating']
            movie_objects.append(movie)
        except Movie.DoesNotExist:
            continue

    return movie_objects


# visual image style recommender
def recommend_image_similarity(reference_movie_id, limit=20):
    # ResNet
    # model_name = "resnet50"

    # CLIP
    # model_name = "clip-vit-base-patch32"
    model_name = "clip-vit-large-patch14"

    # DINOv3
    # model_name = "dinov3-vits16"
    # model_name = "dinov3-vitl16"

    # get path to saved embedding files
    project_root = settings.BASE_DIR
    embeddings_dir = (project_root / "apps" / "recommender" / "embeddings" / model_name)
    embeddings_path = embeddings_dir / "movie_embeddings.npy"
    movie_ids_path = embeddings_dir / "movie_ids.json"

    # return no recommendations if one of the embedding files are missing
    if not embeddings_path.exists() or not movie_ids_path.exists():
        return []

    # load all movie embeddings
    embeddings = np.load(embeddings_path)

    # load movie ids (same order than embeddings)
    with open(movie_ids_path, "r", encoding="utf-8") as file:
        movie_ids = [str(mid).strip() for mid in json.load(file)]

    reference_movie_id = str(reference_movie_id).strip()

    # return no recommendations if there are no image embeddings
    if reference_movie_id not in movie_ids:
        return []

    # find the correct row in the embeddings for the reference movie
    query_index = movie_ids.index(reference_movie_id)
    query_embedding = embeddings[query_index]

    # calculate similarities (cosine similarity with dot product)
    similarities = embeddings @ query_embedding

    # exclude reference movie
    similarities[query_index] = -1

    # get indices of the most similar movies sorted by score
    top_indices = np.argsort(similarities)[::-1][:limit]

    # convert the top embedding indices back into MovieLens movie IDs
    recommended_movie_ids = [int(movie_ids[index]) for index in top_indices]

    # fetch movies from the database and restore their ranking
    movies = (Movie.objects.filter(movie_id__in=recommended_movie_ids))
    movie_lookup = {movie.movie_id: movie for movie in movies}

    # build movie recommendation list
    result = []

    for index in top_indices:
        movie_id = int(movie_ids[index])

        movie = movie_lookup.get(movie_id)

        if movie:
            result.append(movie)

    return result


# show all movies from the same collection
def recommend_by_collection(reference_movie_id, limit=20):
    # get the model
    reference_metadata = (
        MovieMetadata.objects
        .filter(movie__movie_id=reference_movie_id)
        .first()
    )

    if not reference_metadata:
        return []

    # check if movie belongs to collection
    collection = reference_metadata.belongs_to_collection

    if not collection:
        return []

    if isinstance(collection, str):
        try:
            collection = ast.literal_eval(collection)
        except (ValueError, SyntaxError):
            return []

    # get name of the collection
    collection_name = collection.get("name")

    if not collection_name:
        return []

    # get all movies from the same collection
    collection_movies = (
        MovieMetadata.objects
        .filter(belongs_to_collection__icontains=collection_name)
        .exclude(movie__movie_id=reference_movie_id)        # exclude the reference movie from the collection
        .select_related("movie")
        .order_by("release_date")
    )

    return [
        metadata.movie
        for metadata in collection_movies[:limit]
    ]