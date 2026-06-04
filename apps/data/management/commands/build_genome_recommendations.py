from django.core.management.base import BaseCommand
import numpy as np

from apps.data.models import Movie, GenomeScore, GenomeRecommendation


#
# command: docker compose exec web python manage.py build_genome_recommendations
#
class Command(BaseCommand):
    help = "Build genome recommendations from all genome scores"

    def handle(self, *args, **options):
        # get all movies with genome scores
        movie_ids = list(
            GenomeScore.objects
            .values_list("movie__movie_id", flat=True)
            .distinct()
        )

        # get all genome tags
        tag_ids = list(
            GenomeScore.objects
            .values_list("genome_tag_id", flat=True)
            .distinct()
        )

        # create lookup dictionaries for matrix positions
        movie_index = {
            movie_id: index
            for index, movie_id in enumerate(movie_ids)
        }

        tag_index = {
            tag_id: index
            for index, tag_id in enumerate(tag_ids)
        }

        # create empty movie x genome-tag matrix
        genome_matrix = np.zeros(
            (len(movie_ids), len(tag_ids)),
            dtype=np.float32,
        )

        # fill matrix with relevance scores
        for movie_id, tag_id, relevance in (
                GenomeScore.objects
                        .values_list("movie__movie_id", "genome_tag_id", "relevance")
        ):
            genome_matrix[
                movie_index[movie_id],
                tag_index[tag_id],
            ] = relevance

        # load movie objects once
        movies = {
            movie.movie_id: movie
            for movie in Movie.objects.filter(movie_id__in=movie_ids)
        }

        # remove old genome recommendations
        GenomeRecommendation.objects.all().delete()

        # progress tracking
        movie_count = 0
        recommendation_count = 0

        # normalize genome matrix once for cosine similarity
        norms = np.linalg.norm(
            genome_matrix,
            axis=1,
            keepdims=True,
        )

        # avoid division by zero
        norms[norms == 0] = 1
        normalized_matrix = genome_matrix / norms

        # compare every movie against every other movie
        for reference_movie_id in movie_ids:
            # get normalized reference movie vector
            reference_vector = normalized_matrix[
                movie_index[reference_movie_id]
            ]

            # calculate cosine similarity against all other movies
            similarities = normalized_matrix @ reference_vector

            # exclude the reference movie itself
            similarities[
                movie_index[reference_movie_id]
            ] = -1

            # get top 100 most similar movies
            top_indices = np.argsort(similarities)[::-1][:100]

            # save recommendations in one JSON list
            recommended_movies = []

            for rank, index in enumerate(top_indices, start=1):
                recommended_movie_id = movie_ids[index]
                score = float(similarities[index])

                if score <= 0:
                    continue

                recommended_movies.append({
                    "movie_id": recommended_movie_id,
                    "score": score,
                    "rank": rank,
                })

            # save one genome recommendation object per movie
            GenomeRecommendation.objects.create(
                movie=movies[reference_movie_id],
                recommended_movies=recommended_movies,
            )

            recommendation_count += len(recommended_movies)

            # progress tracking
            movie_count += 1
            if movie_count % 1000 == 0:
                print(f"{movie_count} movies processed...")

        print(f"{movie_count} movies were processed in total.")
        print(f"{recommendation_count} recommendations were created in total.")
