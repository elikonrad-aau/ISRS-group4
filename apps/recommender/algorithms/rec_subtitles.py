import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings

class SubtitleRecommender:
    _instance = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_engine()
        return cls._instance

    def _init_engine(self):
        base_dir = getattr(settings, 'EMBEDDINGS_PATH', None)
        if not base_dir:
            base_dir = Path(__file__).resolve().parent.parent / "static" / "embeddings"

        self.model_path = base_dir / "model"
        self.vectors_path = base_dir / "vectors.npy"
        self.ids_path = base_dir / "movie_ids.json"
        self.index_path = base_dir / "id_to_index.json"

        self.model = None
        self.vectors = None
        self.movie_ids = None
        self.id_to_index = None

    def load(self):
        if self._loaded:
            return
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            self.model = SentenceTransformer(str(self.model_path))

            if not self.vectors_path.exists():
                raise FileNotFoundError(f"Vectors not found at {self.vectors_path}")
            self.vectors = np.load(self.vectors_path)

            with open(self.ids_path, 'r') as f:
                self.movie_ids = json.load(f)

            with open(self.index_path, 'r') as f:
                self.id_to_index = json.load(f)

            self._loaded = True
        except Exception as e:
            raise

    def get_recommendations(self, movie_id, n=5):
        self.load()

        if str(movie_id) not in self.id_to_index:
            return None, f"Movie ID {movie_id} not found."

        query_idx = self.id_to_index[str(movie_id)]
        query_vec = self.vectors[query_idx:query_idx + 1]

        # Cosine Similarity with all movies ---- shape: (1, num_movies)
        similarities = cosine_similarity(query_vec, self.vectors)[0]

        #makinge list of (index, score)
        candidates = [
            (idx, score) for idx, score in enumerate(similarities)
            if idx != query_idx
        ]

        candidates.sort(key=lambda x: x[1], reverse=True) #Sort by score desc
        top_n = candidates[:n] #top n

        results = []
        for idx, score in top_n:
            results.append({
                "MovieID": self.movie_ids[idx],
                "similarity_score": round(float(score), 4)
            })

        return results, None


