from django.conf import settings
PKL_PATH = settings.BASE_DIR / "apps" / "recommender" / "embeddings" / "movie_overlaps_lookup.pkl"
import pickle
rec_inst = None

def get_recommender():
    global rec_inst
    if rec_inst is None:
        rec_inst = CastOverlapRecommender()
    return rec_inst

class CastOverlapRecommender:
    def __init__(self, filepath=PKL_PATH):
        self.overlaps = {}
        self.metadata = {}
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.overlaps = data['overlaps']
        self.metadata = data['metadata']

    def recommend_cast_overlap(self, movie_id, limit=20):
        if movie_id not in self.overlaps:
            return []

        results = []
        for target_id, score in self.overlaps[movie_id][:limit]:
            meta = self.metadata.get(target_id, {})
            results.append({
                'MovieID': target_id,
                'Title': meta.get('title', 'Unknown'),
                'OverlapCount': score,
                'AverageRating': meta.get('rating')
            })
        return results