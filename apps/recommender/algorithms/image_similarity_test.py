import json
from pathlib import Path

import numpy as np
import pandas as pd


#
# helper functions
#
def load_titles(project_root):
    # read moviedata
    movies_path = project_root / "dataset" / "movies.csv"
    df = pd.read_csv(movies_path)
    df = df[["movieId", "title"]].dropna()
    df["movieId"] = df["movieId"].astype("Int64").astype(str)

    # return dictionary
    return dict(zip(df["movieId"], df["title"]))


def get_recommendations(movie_id, model_name, embeddings_dir, titles, top_k=20):
    # load the created embeddings
    model_dir = embeddings_dir / model_name
    embeddings_path = model_dir / "movie_embeddings.npy"
    movie_ids_path = model_dir / "movie_ids.json"

    if not embeddings_path.exists():
        print(f"\n{model_name}: embeddings not found")
        return

    embeddings = np.load(embeddings_path)

    with open(movie_ids_path, "r", encoding="utf-8") as file:
        movie_ids = json.load(file)

    movie_ids = [str(movie_id) for movie_id in movie_ids]

    if movie_id not in movie_ids:
        print(f"\n{model_name}: movie id {movie_id} not found")
        return

    query_index = movie_ids.index(movie_id)
    query_embedding = embeddings[query_index]

    # calculate similarities
    similarities = embeddings @ query_embedding

    # exclude movie from recommendations
    similarities[query_index] = -1

    # sort
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"{model_name.upper()} Recommendations:")
    print(f"{movie_id} - {titles.get(movie_id, 'Unknown title')}")
    print()

    # print recommendations to console
    for rank, index in enumerate(top_indices, start=1):
        recommended_id = movie_ids[index]
        recommended_title = titles.get(
            recommended_id,
            "Unknown title",
        )

        score = similarities[index]

        print(
            f"{rank:2d}. "
            f"{recommended_id:<8} "
            f"{score:.4f} "
            f"{recommended_title}"
        )


#
# main logic
#
def main():
    # get project root
    project_root = Path(__file__).resolve().parents[3]

    # define folder for embeddings
    embeddings_dir = (project_root / "dataset" / "tmdb" / "image_similarities_embeddings")

    # load titles
    titles = load_titles(project_root)

    # models and movies for testing
    models_to_evaluate = ["resnet50", "clipsmall", "dinov3small", "clip", "dinov3"]
    movie_id = "527"

    # evaluate all models for defined movie
    for model_name in models_to_evaluate:
        get_recommendations(movie_id=movie_id, model_name=model_name, embeddings_dir=embeddings_dir, titles=titles, top_k=20)


if __name__ == "__main__":
    main()