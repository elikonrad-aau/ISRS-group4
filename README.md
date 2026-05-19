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

4. Restore Database from dump in `root` folder – Download from: https://drive.google.com/drive/folders/1d023Ph-vgjgNFjDr7sQCIrq2LfBkPS6x?usp=sharing
```bash
docker compose exec -T db pg_restore -U postgres -d app --clean --if-exists < db_dump.dump
```

5. Open:
```text
http://127.0.0.1:8000
```


# Algorithm Ideas (not covered in the course)
## Function 0: Baseline (TMDB API)
using the recommendations from the TMDB API as a baseline for the other recommendations – requested via the API and filtered to only show movies existing in the dataset.

## Function 1: Collaborative Filtering (Item-Item kNN)
finding the highest rated movies among all users that gave the reference movie the highest rating
1. use the data from the movies.csv and rating.csv to (pre-)calculate the nearest neighbor (item-item) instead of (user-user)
2. using scikit-learn (sklearn) and comparing different neighborhood sizes and distance measures

## Function 2: Content-Based Filtering (Actor Overlap) // IDEA: Director and Crew Overlap based on Preferences ???
identifies movies that share actors with the reference movie - assumes that if a user enjoyed a film because of its cast, they might enjoy other films with the same actors.
1.	Parse Credits.csv: Extract the list of actor names from the entry for the reference movie
2.	Indexing: Create a lookup table (or inverted index) mapping every actor name to the list of movie IDs they have appeared in.
3.	Intersection -> retrieve the sets of movies associated with each of its top 5–10 actors.
4.	Ranking > Returning the top 5 movies with the highest overlap count, excluding the reference movie itself.

## Optional – Function 3: Content-Based Filtering (Genome Score Based)
uses the genome scores of the dataset to recommend movies with similar semantic profiles

## Function 4: Visual Mood Recommender (Image Based Deep Learning)
finds the movies that have a visual similarity to the reference movie.
1. download the posters and backdrops from TMDB (working)
2. use a pretrained CNN such as ResNet to extract image embeddings.
3. create embedding for each downloaded image and combine them into one visual embedding per movie
4. compare the movie embedding of the reference movie with all other embeddings 
5. return the most similar movies (visually)
- Large Vision Language Models?

## Function 5: Semantic Recommender based on subtitles
uses textual movie information to recommend movies from subtitles
- word count
- compare length of the parts in a conversation
- in combination with genre?
- spaCy for vectors similarity
- https://dl.acm.org/doi/epdf/10.1145/3503252.3531291
  
## Function 6: Hybrid Recommender (Top Recommendations)
Takes the ranks from each recommender and sorts them by their ranks across recommender algorithms with reciprocal rank fusion (RRF) and using the users preferences as weighting for the scores

## "Function 7": Collection
If the reference movie is in a collection, these movies will be excluded from our custom algorithms.
However, there will be a section at the bottom of the page featuring all movies from the same collection so that the user can make sure that he/she has seen all of them already.

## Optional Function: Letterboxd (API Access uncertain)
Collaborative Filtering (User Behavior and Taste Profile) -> Instead of looking at the movie metadata – looking for who liked the movie -> identify users who have LIKED or rated the reference movie highly and then recommend other movies those users liked.
Steps:
1.	Fetch the reference movie's ID (?)
2.	Retrieve a sample of users who have LIKED this movie 
3.	For each of these users, fetch their LIKED list or top-rated films
4.	Maybe also ranking the movies by frequency: How many of the other  movie fans also liked this movie -> order by frequency


### Evaluation:
- Choice of Movies (popular, not popular, ...)
- Friends and Family?
- LLM as a judge for the recommendations
