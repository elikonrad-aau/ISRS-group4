import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, save_npz
import sys
import gc

# GENERATES item_knn folder (pickle files and npz) knn Algorithm
# folder content must be saved in apps/recommender/embeddings/item_knn

def build_and_cache_model(ratings_csv_path, output_dir="model_cache"):
    cache_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(ratings_csv_path)

    # Get valid movie IDs
    counts = ratings.groupby('movieId')['rating'].count()
    valid_movies = counts[counts >= 1].index.tolist()

    df_filtered = ratings[ratings['movieId'].isin(valid_movies)]
    del ratings
    gc.collect()

    # Create mapping dictionaries
    movie_to_col = {str(mid): i for i, mid in enumerate(valid_movies)}

    unique_users = df_filtered['userId'].unique()
    user_to_row = {uid: i for i, uid in enumerate(unique_users)}

    num_users = len(unique_users)
    num_movies = len(valid_movies)

    rows = []
    cols = []
    data = []

    batch_size = 500000
    total_rows = len(df_filtered)

    # Process in chunks to avoid memory spike
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        chunk = df_filtered.iloc[start:end]

        rows.extend([user_to_row[uid] for uid in chunk['userId']])
        cols.extend([movie_to_col[str(mid)] for mid in chunk['movieId']])
        data.extend(chunk['rating'].values)

    del df_filtered
    gc.collect()

    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_movies))

    del rows, cols, data
    gc.collect()

    movie_ids = list(movie_to_col.keys())


    knn = NearestNeighbors(
        n_neighbors=50,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1            # use all avail. cores
    )

    knn.fit(sparse_matrix.T)

    with open(cache_dir / "knn_model.pkl", "wb") as f:
        pickle.dump(knn, f)

    with open(cache_dir / "movie_ids.pkl", "wb") as f:
        pickle.dump(movie_ids, f)
    save_npz(cache_dir / "rating_matrix_sparse.npz", sparse_matrix)

    print(f"Saved to: {cache_dir.absolute()}")

if __name__ == "__main__":
    csv_path = "C:/Users/elisa/PycharmProjects/ISRS-group4/dataset/ratings.csv"
    build_and_cache_model(csv_path)