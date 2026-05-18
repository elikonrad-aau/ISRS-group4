import os
import requests

from apps.data.models import Movie, MovieLink


MOCK_ROWS = [
    "Collaborative Filtering",
    "Shared Cast Overlap",
    "Visual Mood",
    "Text Semantic",
    "Genome Score",
    "Hybrid Recommendations",
]


def get_recommendation_rows(reference_movie_id, limit=20):
    rows = []

    # baseline algorithm
    rows.append({
        "title": "TMDB Recommendations",
        "algorithm": "tmdb",
        "movies": recommend_by_tmdb(reference_movie_id, limit),
    })

    # dummy data
    for row_title in MOCK_ROWS:
        rows.append({
            "title": row_title,
            "algorithm": row_title.lower().replace(" ", "_"),
            "movies": get_random_movies(reference_movie_id, limit),
        })

    return rows


def get_random_movies(reference_movie_id, limit=20):
    return list(
        Movie.objects
        .exclude(movie_id=reference_movie_id)
        .order_by("?")[:limit]
    )


# get baseline recommendations from tmdb
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