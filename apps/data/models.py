import re

from django.db import models

#
# define models for the data/db
#
# movies.csv
class Movie(models.Model):
    movie_id = models.IntegerField(unique=True)
    title = models.CharField(max_length=255)
    genres = models.JSONField(default=list)
    year = models.IntegerField(null=True, blank=True)

    # fix titles
    @property
    def display_title(self):
        title = self.title

        # remove the year from the title
        title = re.sub(r"\s*\(\d{4}\)$", "", title)

        # fix the formatting of the articles "the", "a", "an"
        patterns = [
            # titles with year
            r"^(.*), The (\(.*\))$",
            r"^(.*), A (\(.*\))$",
            r"^(.*), An (\(.*\))$",

            # titles without year
            r"^(.*), The$",
            r"^(.*), A$",
            r"^(.*), An$",
        ]

        replacements = [
            r"The \1 \2",
            r"A \1 \2",
            r"An \1 \2",

            r"The \1",
            r"A \1",
            r"An \1",
        ]

        for pattern, replacement in zip(patterns, replacements):
            if re.match(pattern, title):
                return re.sub(pattern, replacement, title)

        return title

    @property
    def extracted_year(self):
        match = re.search(r"\((\d{4})\)$", self.title)
        return int(match.group(1)) if match else None

    def save(self, *args, **kwargs):
        self.year = self.extracted_year
        super().save(*args, **kwargs)

    def __str__(self):
        return self.display_title


# ratings.csv
class Rating(models.Model):
    user_id = models.IntegerField()

    movie = models.ForeignKey(
        Movie,
        on_delete=models.CASCADE,
        related_name="ratings",
    )

    rating = models.FloatField()

    timestamp = models.BigIntegerField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["user_id", "movie", "timestamp"],
                name="unique_user_movie_rating_timestamp",
            )
        ]
        indexes = [
            models.Index(fields=["user_id"]),
            models.Index(fields=["movie"]),
        ]

    def __str__(self):
        return f"User {self.user_id} rated {self.movie.title}: {self.rating}"


# tags.csv
class Tag(models.Model):
    user_id = models.IntegerField()

    movie = models.ForeignKey(
        Movie,
        on_delete=models.CASCADE,
        related_name="tags",
    )

    tag = models.CharField(max_length=255)

    timestamp = models.BigIntegerField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["user_id", "movie", "tag", "timestamp"],
                name="unique_user_movie_tag_timestamp",
            )
        ]
        indexes = [
            models.Index(fields=["user_id"]),
            models.Index(fields=["movie"]),
            models.Index(fields=["tag"]),
        ]

    def __str__(self):
        return self.tag


# links.csv
class MovieLink(models.Model):
    movie = models.OneToOneField(
        Movie,
        on_delete=models.CASCADE,
        related_name="links",
    )

    imdb_id = models.CharField(max_length=20, blank=True, null=True)
    tmdb_id = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"Links for {self.movie.title}"


# genome-tags.csv
class GenomeTag(models.Model):
    tag_id = models.IntegerField(unique=True)

    tag = models.CharField(
        max_length=255,
        unique=True,
    )

    def __str__(self):
        return self.tag


# genome-scores.csv
class GenomeScore(models.Model):
    movie = models.ForeignKey(
        Movie,
        on_delete=models.CASCADE,
        related_name="genome_scores",
    )

    genome_tag = models.ForeignKey(
        GenomeTag,
        on_delete=models.CASCADE,
        related_name="movie_scores",
    )

    relevance = models.FloatField()

    class Meta:
        unique_together = ("movie", "genome_tag")

        indexes = [
            models.Index(fields=["movie"]),
            models.Index(fields=["genome_tag"]),
        ]

    def __str__(self):
        return f"{self.movie.title} - {self.genome_tag.tag}: {self.relevance}"


# movies_metadata.csv
class MovieMetadata(models.Model):
    movie = models.OneToOneField(
        Movie,
        on_delete=models.CASCADE,
        related_name="metadata",
    )

    tmdb_id = models.IntegerField(unique=True)
    imdb_id = models.CharField(max_length=20, blank=True, null=True)

    adult = models.BooleanField(default=False)
    budget = models.BigIntegerField(null=True, blank=True)
    homepage = models.URLField(max_length=500, blank=True, null=True)
    original_language = models.CharField(max_length=20, blank=True, null=True)
    original_title = models.CharField(max_length=255, blank=True, null=True)
    overview = models.TextField(blank=True, null=True)
    popularity = models.FloatField(null=True, blank=True)
    poster_path = models.CharField(max_length=500, blank=True, null=True)
    release_date = models.DateField(null=True, blank=True)
    revenue = models.BigIntegerField(null=True, blank=True)
    runtime = models.FloatField(null=True, blank=True)
    status = models.CharField(max_length=100, blank=True, null=True)
    tagline = models.TextField(blank=True, null=True)
    video = models.BooleanField(default=False)
    vote_average = models.FloatField(null=True, blank=True)
    vote_count = models.IntegerField(null=True, blank=True)
    belongs_to_collection = models.JSONField(null=True, blank=True)

    genres = models.JSONField(default=list, blank=True)
    production_companies = models.JSONField(default=list, blank=True)
    production_countries = models.JSONField(default=list, blank=True)
    spoken_languages = models.JSONField(default=list, blank=True)

    raw_data = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Metadata for {self.movie.title}"


# credits.csv
class MovieCredits(models.Model):
    movie = models.OneToOneField(
        Movie,
        on_delete=models.CASCADE,
        related_name="credits",
    )

    source_id = models.IntegerField(null=True, blank=True)

    cast = models.JSONField(default=list, blank=True)
    crew = models.JSONField(default=list, blank=True)

    def __str__(self):
        return f"Credits for {self.movie.title}"