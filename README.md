# ISRS-group4
Movie Recommendation

## Function 0: Baseline

## Function 1: recommend_by_shared_cast_overlap
Content-Based Filtering (Actor occurrence): identifies movies that share actors with the reference movie- assumes that if a user enjoyed a film because of its cast, they might enjoy other films with the same actors.
1.	Parse Credits.csv: Extract the list of actor names from the entry for the reference movie
2.	Indexing: Create a lookup table (or inverted index) mapping every actor name to the list of movie IDs they have appeared in.
3.	Intersection -> retrieve the sets of movies associated with each of its top 5–10 actors.
4.	Ranking > Returning the top 5 movies with the highest overlap count, excluding the reference movie itself.

## Function 2: Deep Learning

## Function 3: Image Base Deep Learning

## Function 4: Tag Based


### Optional Letterboxd
Collaborative Filtering (User Behavior  and Taste Profile) -> Instead of looking at the movie metadata – looking for who liked the movie -> identify users who have LIKED or rated the reference movie highly and then recommend other movies those users liked.
Steps:
1.	Fetch the reference movie's ID (?)
2.	Retrieve a sample of users who have LIKED this movie 
3.	For each of these users, fetch their LIKED list or top-rated films
4.	Maybe also ranking the movies by frequency: How many of the other  movie fans also liked this movie -> order by frequency
   
Letterboxd API -> uncertain access
