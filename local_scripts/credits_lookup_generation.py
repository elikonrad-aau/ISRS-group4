import pandas as pd
import ast
import json
from collections import defaultdict
import pickle
import os
import sys

# GENERATES PICKLE FILE for Credits Overlap Algorithmx
# file must be saved in apps/recommender/embeddings

BASE_DIR = r"C:/Users/elisa/PycharmProjects/ISRS-group4/dataset"
MOVIES_FILE = os.path.join(BASE_DIR, "movies.csv")
CREDITS_FILE = os.path.join(BASE_DIR, "credits.csv")
RATINGS_FILE = os.path.join(BASE_DIR, "ratings.csv")
OUTPUT_PICKLE = "simple_movie_lookups.pkl"

TOP_OVERLAPS_PER_MOVIE = 20
MAX_CAST_TO_CHECK = 10 # limit for cast members
MAX_CREW_TO_CHECK = 15 # limit for crew members

def safe_parse_list(cast_string):
    if pd.isna(cast_string) or cast_string == '' or cast_string is None: return []
    s = str(cast_string).strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')): s = s[1:-1]
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list): return [item for item in parsed if isinstance(item, dict)]
    except:
        pass
    try:
        s_json = s.replace("None", "null")
        parsed = json.loads(s_json)
        if isinstance(parsed, list): return [item for item in parsed if isinstance(item, dict)]
    except:
        pass
    return []


def load_data():
    print("Loading datasets...")
    movies_df = pd.read_csv(MOVIES_FILE)
    credits_df = pd.read_csv(CREDITS_FILE)

    # Normalize IDs
    if 'id' in movies_df.columns and 'movieId' not in movies_df.columns: movies_df.rename(columns={'id': 'movieId'},
                                                                                          inplace=True)
    if 'id' in credits_df.columns and 'movieId' not in credits_df.columns: credits_df.rename(columns={'id': 'movieId'},
                                                                                             inplace=True)

    ratings_df = pd.read_csv(RATINGS_FILE)
    return movies_df, credits_df, ratings_df


def build_indices(credits_df, valid_movie_ids):
    print("Building Indices...")
    actor_index = defaultdict(set)
    crew_index = defaultdict(set)

    valid_credits = credits_df[credits_df['movieId'].isin(valid_movie_ids)]

    for _, row in valid_credits.iterrows():
        mid = row['movieId']

        # Cast
        cast_list = safe_parse_list(row.get('cast'))
        for p in cast_list:
            if 'name' in p: actor_index[p['name']].add(mid)

        # Crew
        crew_list = safe_parse_list(row.get('crew'))
        for p in crew_list:
            if 'name' in p: crew_index[p['name']].add(mid)

    return actor_index, crew_index


def calculate_overlaps(actor_index, crew_index, valid_movie_ids, movies_df):
    movie_cast_map = {}
    movie_crew_map = {}

    valid_credits = pd.read_csv(CREDITS_FILE)
    if 'id' in valid_credits.columns and 'movieId' not in valid_credits.columns: valid_credits.rename(
        columns={'id': 'movieId'}, inplace=True)
    valid_credits = valid_credits[valid_credits['movieId'].isin(valid_movie_ids)]

    for _, row in valid_credits.iterrows():
        mid = row['movieId']
        c_list = safe_parse_list(row.get('cast'))
        w_list = safe_parse_list(row.get('crew'))

        if c_list and any('order' in x for x in c_list):
            c_list.sort(key=lambda x: x.get('order', float('inf')))
            movie_cast_map[mid] = set(p['name'] for p in c_list[:MAX_CAST_TO_CHECK])
        else:
            movie_cast_map[mid] = set(p['name'] for p in c_list[:MAX_CAST_TO_CHECK])

        movie_crew_map[mid] = set(p['name'] for p in w_list[:MAX_CREW_TO_CHECK])

    results = {}
    total = len(valid_movie_ids)

    for i, ref_id in enumerate(sorted(list(valid_movie_ids))):
        ref_people = movie_cast_map.get(ref_id, set()).union(movie_crew_map.get(ref_id, set()))
        strict_counts = defaultdict(set)

        for person in ref_people:
            sources = []
            if person in actor_index: sources.extend(actor_index[person])
            if person in crew_index: sources.extend(crew_index[person])

            for m_id in sources:
                if m_id != ref_id:
                    strict_counts[m_id].add(person)

        # Convert to (count, id) for sorting, then flip back to (id, count)
        scored = [(m_id, len(people)) for m_id, people in strict_counts.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        results[ref_id] = scored[:TOP_OVERLAPS_PER_MOVIE]

    return results


def main():
    movies_df, credits_df, ratings_df = load_data()
    valid_ids = set(movies_df['movieId'].unique())

    # Build indices
    actor_idx, crew_idx = build_indices(credits_df, valid_ids)

    # Calculate overlaps
    overlap_data = calculate_overlaps(actor_idx, crew_idx, valid_ids, movies_df)
    # Metadata Map -> Simple Dict of Dicts)
    title_map = movies_df.set_index('movieId')['title'].to_dict()

    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.rename(columns={'rating': 'AverageRating'}, inplace=True)
    rating_map = avg_ratings.set_index('movieId')['AverageRating'].to_dict()

    metadata = {}
    for mid in valid_ids:
        metadata[mid] = {
            "title": title_map.get(mid, f"Unknown {mid}"),
            "rating": rating_map.get(mid)
        }

    # Save both structures in one file
    final_data = {
        "overlaps": overlap_data,  # Key: movie_id, Value: [(target_id, score), ...]
        "metadata": metadata  # Key: movie_id, Value: {title, rating}
    }

    print(f"\nSaving to {OUTPUT_PICKLE}...")
    with open(OUTPUT_PICKLE, 'wb') as f:
        pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()