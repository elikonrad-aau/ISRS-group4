import os
import ast
import requests
from django.http import JsonResponse

from apps.data.models import Movie, MovieLink, MovieMetadata
from apps.recommender.cast_overlap import CastOverlapRecommender

# TODO FINAL - Delete
MOCK_ROWS = [
    "Collaborative Filtering",
    "Visual Mood",
    "Text Semantic",
    "Genome Score",
    "Hybrid Recommendations",
]


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
    # function 1 algorithm
    # function 2 algorithm
    rows.append({
        "title": "Shared Cast Overlap",
        "algorithm": "tmdb",
        "movies": recommend_cast_overlap(reference_movie_id, limit),
    })

    # function 3 algorithm
    # function 4 algorithm
    # function 5 algorithm
    # function 6 algorithm

    # TODO FINAL - Delete
    for row_title in MOCK_ROWS:
        rows.append({
            "title": row_title,
            "algorithm": row_title.lower().replace(" ", "_"),
            "movies": get_random_movies(reference_movie_id, limit),
        })

    # collection – only append if the reference movie is part of a collection
    collection_movies = recommend_by_collection(reference_movie_id, limit)

    if collection_movies:
        rows.append({
            "title": "Have you seen these?",
            "algorithm": "collection",
            "movies": collection_movies,
        })


    return rows

# TODO FINAL - Delete
def get_random_movies(reference_movie_id, limit=20):
    return list(
        Movie.objects
        .exclude(movie_id=reference_movie_id)
        .order_by("?")[:limit]
    )


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