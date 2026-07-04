# ISRS-group4
Movie Recommendation

## Setup
1. Create `.env` file:
```bash
cp .env.example .env
```

2. Build Docker:
```bash
docker compose up --build -d
```

3. Migrate Database:
```bash
docker compose exec web python manage.py migrate
```

4. Restore Database from dump in `root` folder 
```bash
docker compose exec -T db psql -U postgres -d app < db_dump.sql
```

5. Download embeddings folder and place it in apps/recommender/embeddings

5. Open:
```text
http://127.0.0.1:8000
```


# Algorithms
## Function 0: Baseline (TMDB API)
using the recommendations from the TMDB API as a baseline for the other recommendations – requested via the API and filtered to only show movies existing in the dataset.
For this, a tmdb account is necessary (for API token)

## Function 1: Collaborative Filtering
finding the highest rated movies among all users that gave the reference movie the highest rating
1. use the data from the movies.csv and rating.csv to (pre-)calculate the nearest neighbor
   Creates a sparse matrix where rows represent users, columns represent movies, and values are the ratings. Each movie becomes a vector of how all users rated it.
2. trained a Nearest Neighbors model on the transposed matrix (movies × users). Using cosine similarity, find the top 50 most similar movies for every movie in the dataset. Results are saved as cached files (pickle + npz) for fast retrieval.
3. When given a reference movie ID, load the pre-computed model and return up to n similar movies, ranked by similarity score

## Function 2: Content-Based Filtering (Credits Overlap) 
identifies movies that share actors/crew with the reference movie - assumes that if a user enjoyed a film because of its cast/crew, they might enjoy other films with the same actors.
1.	Parse Credits.csv: Extract the list of actor names and crew member names from the entry for the reference movie. Cast is sorted by billing order and limited to the top 10 members. Crew is limited to the top 15 members.
2.	Indexing:  Create lookup tables mapping every person name to the list of movie IDs they have worked on (one for actors, one for crew).
3.	 Retrieve the sets of movies associated with each of the reference movie's cast and crew members. Count how many shared people each candidate movie has with the reference. The reference movie itself is excluded.
4.	Ranking: Return the top movies with the highest overlap count..

## Optional – Function 3: Content-Based Filtering (Genome Score Based)
uses the genome scores of the dataset to recommend movies with similar semantic profiles

## Function 4: Visual Mood Recommender (Image Based Deep Learning)
finds the movies that have a visual similarity to the reference movie.
1. download the posters and backdrops from TMDB (working)
2. use a pretrained CNN such as ResNet to extract image embeddings.
3. create embedding for each downloaded image and combine them into one visual embedding per movie
4. compare the movie embedding of the reference movie with all other embeddings 
5. return the most similar movies (visually)

## Function 5: Semantic Subtitle Recommender
Uses natural language processing to find movies with similar dialogue styles, themes, and storytelling contexts.
1. Cleans raw subtitle files (.srt) by removing timestamps, sequence numbers, and formatting tags to isolate pure dialogue.
2. Splits long transcripts into 200-word segments to stay within model token limits while preserving narrative context.
3. Semantic Embedding: Pass chunks through all-MiniLM-L6-v2 (Sentence Transformer) to generate vector representations, then average them to create a single, unique "semantic signature" for each movie.
4. Stores vectors in a matrix and retrieve recommendations by calculating Cosine Similarity between the target movie's signature and all others in the database.

## Function 6: Hybrid Recommender (Top Recommendations)
Takes the ranks from each recommender and sorts them by their ranks across recommender algorithms with reciprocal rank fusion (RRF) and using the users preferences as weighting for the scores

## "Function 7": Collection
If the reference movie is in a collection, these movies will be excluded from our custom algorithms.
However, there will be a section at the bottom of the page featuring all movies from the same collection so that the user can make sure that he/she has seen all of them already.
