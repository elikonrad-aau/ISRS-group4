import os
from apps.data.models import Movie, MovieLink, MovieMetadata, MovieGenomeProfile, GenomeRecommendation
import requests

def evaluate_using_tmdb(reference_movie_id, distinct_ids, url_ending="recommendations", limit=20):
    token = os.environ.get("TMDB_API_TOKEN")
    reference_link = MovieLink.objects.filter(
        movie__movie_id=reference_movie_id,
        tmdb_id__isnull=False,
    ).first()

    tmdb_ref_id = reference_link.tmdb_id
    fetched_similar_ids = set()
    seen_tmdb_ids = set()
    page = 1
    max_pages = 50

    while len(fetched_similar_ids) < limit and page <= max_pages:
        try:
            response = requests.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_ref_id}/{url_ending}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "accept": "application/json",
                },
                params={"page": page},
                timeout=10,
            )

            if response.status_code != 200:
                print(f"API Error: Status {response.status_code}")
                break

            data = response.json()
            results = data.get("results", [])
            if not results:
                break

            # Map TMDB IDs to your local Movie objects efficiently
            # Extract all TMDB IDs from current page
            current_page_tmdb_ids = [item["id"] for item in results]
            for item in results:
                print(item["title"])

            # Fetch corresponding local movies in one query
            links = MovieLink.objects.filter(tmdb_id__in=current_page_tmdb_ids).select_related("movie")
            local_movies_by_tmdb = {link.tmdb_id: link.movie for link in links}

            for tmdb_id in current_page_tmdb_ids:
                movie = local_movies_by_tmdb.get(tmdb_id)

                if not movie:
                    continue  # Movie exists on TMDB but not in our DB

                # Prevent duplicates and self-reference
                if movie.movie_id == reference_movie_id or movie.movie_id in seen_tmdb_ids:
                    continue

                if len(fetched_similar_ids) < limit: fetched_similar_ids.add(movie.movie_id)
                seen_tmdb_ids.add(movie.movie_id)

            if len(fetched_similar_ids) >= limit:
                break

            # Check if we are at the last page
            if page >= data.get("total_pages", 1):
                break

            page += 1

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break

    total_fetched = len(fetched_similar_ids)
    overlap_set = distinct_ids.intersection(fetched_similar_ids)
    overlap_count = len(overlap_set)

    print(f"\n--- Evaluation Results ---")
    print(f"Total similar movies fetched from TMDB: {total_fetched}")
    print(f"Overlap with recommendations: {overlap_count} movies")
    print(f"How many of the recommended movies were in the tmdb movies: {overlap_count}/{len(distinct_ids)}")

    if overlap_count > 0:
        sorted_overlap = sorted(list(overlap_set))
        print(f"Overlapping IDs: {sorted_overlap[:10]}{'...' if len(sorted_overlap) > 10 else ''}")

    return overlap_count, total_fetched