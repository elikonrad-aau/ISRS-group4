from collections import defaultdict
from django.db.models import Avg
from apps.data.models import Movie, MovieCredits, Rating
import ast
import json

class CastOverlapRecommender:
    def __init__(self):
        self.actor_index = None

    def _safe_parse_cast(self, cast_value):
        if not cast_value:
            return []

        if isinstance(cast_value, list):
            return [actor for actor in cast_value if isinstance(actor, dict)]

        s = str(cast_value).strip()

        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1]

        try:
            cast_list = ast.literal_eval(s)
            if isinstance(cast_list, list):
                return [actor for actor in cast_list if isinstance(actor, dict)]
        except (ValueError, SyntaxError):
            pass

        try:
            s_json = s.replace("None", "null")
            cast_list = json.loads(s_json)
            if isinstance(cast_list, list):
                return [actor for actor in cast_list if isinstance(actor, dict)]
        except (ValueError, json.JSONDecodeError):
            pass

        return []

    def build_actor_index_from_db(self):
        actor_index = defaultdict(set)

        # Use iterator to avoid loading all records into memory
        for credits in MovieCredits.objects.select_related('movie').iterator(chunk_size=5000):
            movie_id = credits.movie.movie_id
            cast_list = self._safe_parse_cast(credits.cast)

            for actor in cast_list:
                if 'name' in actor:
                    actor_index[actor['name']].add(movie_id)

        self.actor_index = dict(actor_index)
        return self.actor_index

    def get_movie_by_id(self, movie_id):
        try:
            return Movie.objects.get(movie_id=movie_id)
        except Movie.DoesNotExist:
            return None

    def get_average_rating_for_movie(self, movie_id):
        try:
            result = Rating.objects.filter(
                movie__movie_id=movie_id
            ).aggregate(avg_rating=Avg('rating'))
            return result['avg_rating']
        except:
            return None

    def recommend_cast_overlap(self, reference_movie_id, limit=20, max_actors=10):
        ref_movie = self.get_movie_by_id(reference_movie_id)
        if not ref_movie:
            return []
        if self.actor_index is None:
            self.build_actor_index_from_db()

        try:
            # get reference movie actors
            credits = MovieCredits.objects.get(movie=ref_movie)
            cast_list = self._safe_parse_cast(credits.cast)

            if 'order' in cast_list[0]:
                cast_list.sort(key=lambda x: x.get('order', float('inf')))

            reference_actors = cast_list[:max_actors]
        except MovieCredits.DoesNotExist:
            return []

        # Count overlaps
        overlap_counts = defaultdict(int)

        for actor in reference_actors:
            actor_name = actor['name']
            if actor_name in self.actor_index:
                for movie_id in self.actor_index[actor_name]:
                    if movie_id != reference_movie_id:
                        overlap_counts[movie_id] += 1

        sorted_movies = sorted(overlap_counts.items(), key=lambda x: x[1], reverse=True) # get top n movies
        top_movie_ids = [mid for mid, _ in sorted_movies[:limit]]

        movies = Movie.objects.filter(movie_id__in=top_movie_ids)
        movie_dict = {m.movie_id: m for m in movies}

        results = []
        for movie_id, score in sorted_movies[:limit]:
            if movie_id in movie_dict:
                movie = movie_dict[movie_id]
                avg_rating = self.get_average_rating_for_movie(movie_id)

                results.append({
                    'MovieID': movie_id,
                    'Title': movie.title,
                    'OverlapCount': score,
                    'AverageRating': avg_rating
                })

        # sorting by overlap max and avg rating
        results.sort(key=lambda x: (x['OverlapCount'], x['AverageRating'] or 0), reverse=True)
        return results
