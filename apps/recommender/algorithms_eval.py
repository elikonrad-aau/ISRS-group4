import json

from apps.data.models import Movie
from apps.recommender.algorithms_recs import (
    get_recommendation_rows,
    recommend_by_tmdb,
)

SKIP_EVAL = False

def evaluate_algorithm(our_movies, tmdb_movies, algorithm_name):
    if SKIP_EVAL:
        return {}

    # check which of those movies are in the tmdb_recommendations
    our_movie_ids = set()
    tmdb_movie_ids = set()

    for movie in our_movies:
        if hasattr(movie, 'movie_id'):
            our_movie_ids.add(movie.movie_id)
        elif isinstance(movie, dict) and "movie_id" in movie:
            our_movie_ids.add(movie["movie_id"])

    for movie in tmdb_movies:
        if hasattr(movie, 'movie_id'):
            tmdb_movie_ids.add(movie.movie_id)
        elif isinstance(movie, dict) and "movie_id" in movie:
            tmdb_movie_ids.add(movie["movie_id"])

    # Calculate overlap between our recommendations and TMDB baseline
    overlap_ids = our_movie_ids.intersection(tmdb_movie_ids)
    overlap_count = len(overlap_ids)

    # Identify unique recommendations
    unique_to_our_algo = our_movie_ids - tmdb_movie_ids

    evaluation_result = {
        "our_algorithm_count": len(our_movie_ids),
        "tmdb_baseline_count": len(tmdb_movie_ids),
        "overlap_count": overlap_count,
        "overlap_rate": round(
            overlap_count / min(len(our_movie_ids), len(tmdb_movie_ids)),
            4,
        ) if our_movie_ids and tmdb_movie_ids else 0,
        "unique_to_algorithm": len(unique_to_our_algo),
        "overlap_movie_ids": list(overlap_ids)[:10],
        "algorithm_only_movie_ids": list(unique_to_our_algo)[:10],
    }

    # Print summary for debugging/evaluation
    print(f"\n{'=' * 50}")
    print(f"ALGORITHM EVALUATION {algorithm_name}")
    print(f"{'=' * 50}")
    print(f"Our Algorithm Recommendations: {len(our_movie_ids)}")
    print(f"TMDB Baseline Recommendations: {len(tmdb_movie_ids)}")
    print(f"Overlap Count: {overlap_count} ({evaluation_result['overlap_rate']:.1%})")
    print(f"Unique to Our Algorithm: {len(unique_to_our_algo)}")
    print(f"{'=' * 50}\n")

    return evaluation_result


def get_distinct_movieids_for_eval(rows):
    distinct_ids = set()

    for row in rows:
        if row.get("algorithm") == "tmdb":
            continue

        movies_list = row.get("movies", [])

        for movie in movies_list:
            if hasattr(movie, 'movie_id'):
                distinct_ids.add(movie.movie_id)
            elif isinstance(movie, dict) and "movie_id" in movie:
                distinct_ids.add(movie["movie_id"])

    print(f"\n--- Distinct Movie IDs Found ({len(distinct_ids)} total) ---")
    print(", ".join(map(str, sorted(distinct_ids))))

    return distinct_ids


def evaluate_recommendation_rows(
    reference_movie_id,
    user_selection,
    limit=10,
):
    # get recommendation rows from recommender module
    rows = get_recommendation_rows(
        reference_movie_id=reference_movie_id,
        user_selection=user_selection,
        limit=limit,
    )

    # TMDB baseline used for comparison
    tmdb_movies = recommend_by_tmdb(
        reference_movie_id,
        50,
    )

    evaluated_rows = []

    for row in rows:
        algorithm = row["algorithm"]

        # skip baseline row
        if algorithm == "tmdb":
            continue

        evaluation = evaluate_algorithm(
            our_movies=row["movies"],
            tmdb_movies=tmdb_movies,
            algorithm_name=algorithm,
        )

        evaluated_rows.append({
            **row,
            "evaluation": evaluation,
        })

    return evaluated_rows