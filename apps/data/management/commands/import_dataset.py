import ast
import csv
from datetime import datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.data.models import (Movie, Rating, Tag, MovieLink, GenomeTag, GenomeScore, MovieMetadata, MovieCredits,)

#
# command: docker compose exec web python manage.py import_dataset
#
class Command(BaseCommand):
    help = "Import dataset CSV files into the database"

    # import handler
    def handle(self, *args, **options):
        dataset_dir = Path("dataset")

        if not dataset_dir.exists():
            raise CommandError("Dataset folder not found. Expected: dataset/")

        self.import_movies(dataset_dir / "movies.csv")
        self.import_links(dataset_dir / "links.csv")
        self.import_ratings(dataset_dir / "ratings.csv")
        self.import_tags(dataset_dir / "tags.csv")
        self.import_genome_tags(dataset_dir / "genome-tags.csv")
        self.import_genome_scores(dataset_dir / "genome-scores.csv")
        self.import_movies_metadata(dataset_dir / "movies_metadata.csv")
        self.import_credits(dataset_dir / "credits.csv")

        self.stdout.write(self.style.SUCCESS("Dataset import completed."))

    # helper/parser methods
    def read_csv(self, path):
        if not path.exists():
            raise CommandError(f"Missing required file: {path}")

        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                yield row

    def bulk_create_in_batches(self, model, objects, batch_size=10000):
        model.objects.bulk_create(
            objects,
            batch_size=batch_size,
            ignore_conflicts=True,
        )

    def parse_int(self, value):
        if value in ("", None):
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def parse_float(self, value):
        if value in ("", None):
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def parse_bool(self, value):
        return str(value).lower() == "true"

    def parse_date(self, value):
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None

    def parse_json_like(self, value, default):
        if not value:
            return default
        try:
            return ast.literal_eval(value)
        except Exception:
            return default


    # import movies.csv
    @transaction.atomic
    def import_movies(self, path):
        rows = self.read_csv(path)
        count = 0

        for row in rows:
            Movie.objects.update_or_create(
                movie_id=int(row["movieId"]),
                defaults={
                    "title": row["title"],
                    "genres": row["genres"].split("|") if row["genres"] else [],
                },
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f"Imported movies: {count}"))


    # import links.csv
    @transaction.atomic
    def import_links(self, path):
        rows = self.read_csv(path)
        count = 0
        skipped = 0

        for row in rows:
            movie = Movie.objects.filter(movie_id=int(row["movieId"])).first()

            if not movie:
                skipped += 1
                continue

            MovieLink.objects.update_or_create(
                movie=movie,
                defaults={
                    "imdb_id": row["imdbId"] or None,
                    "tmdb_id": self.parse_int(row["tmdbId"]),
                },
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f"Imported links: {count}, skipped: {skipped}"))


    # import ratings.csv
    @transaction.atomic
    def import_ratings(self, path):
        rows = self.read_csv(path)

        movie_map = dict(Movie.objects.values_list("movie_id", "id"))

        batch = []
        count = 0
        skipped = 0
        batch_size = 10000

        for row in rows:
            movie_db_id = movie_map.get(int(row["movieId"]))

            if not movie_db_id:
                skipped += 1
                continue

            batch.append(
                Rating(
                    user_id=int(row["userId"]),
                    movie_id=movie_db_id,
                    rating=float(row["rating"]),
                    timestamp=int(row["timestamp"]),
                )
            )

            count += 1

            if len(batch) >= batch_size:
                self.bulk_create_in_batches(Rating, batch, batch_size)
                batch.clear()
                self.stdout.write(f"Imported ratings so far: {count}")

        if batch:
            self.bulk_create_in_batches(Rating, batch, batch_size)

        self.stdout.write(self.style.SUCCESS(f"Imported ratings: {count}, skipped: {skipped}"))


    # import tags.csv
    @transaction.atomic
    def import_tags(self, path):
        rows = self.read_csv(path)

        movie_map = dict(Movie.objects.values_list("movie_id", "id"))

        batch = []
        count = 0
        skipped = 0
        batch_size = 10000

        for row in rows:
            movie_db_id = movie_map.get(int(row["movieId"]))

            if not movie_db_id:
                skipped += 1
                continue

            batch.append(
                Tag(
                    user_id=int(row["userId"]),
                    movie_id=movie_db_id,
                    tag=row["tag"],
                    timestamp=int(row["timestamp"]),
                )
            )

            count += 1

            if len(batch) >= batch_size:
                self.bulk_create_in_batches(Tag, batch, batch_size)
                batch.clear()
                self.stdout.write(f"Imported tags so far: {count}")

        if batch:
            self.bulk_create_in_batches(Tag, batch, batch_size)

        self.stdout.write(self.style.SUCCESS(f"Imported tags: {count}, skipped: {skipped}"))


    # import genome_tags.csv
    @transaction.atomic
    def import_genome_tags(self, path):
        rows = self.read_csv(path)
        count = 0

        for row in rows:
            GenomeTag.objects.update_or_create(
                tag_id=int(row["tagId"]),
                defaults={
                    "tag": row["tag"],
                },
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f"Imported genome tags: {count}"))


    # import genome_scores.csv
    @transaction.atomic
    def import_genome_scores(self, path):
        rows = self.read_csv(path)

        movie_map = dict(Movie.objects.values_list("movie_id", "id"))
        genome_tag_map = dict(GenomeTag.objects.values_list("tag_id", "id"))

        batch = []
        count = 0
        skipped = 0
        batch_size = 10000

        for row in rows:
            movie_db_id = movie_map.get(int(row["movieId"]))
            genome_tag_db_id = genome_tag_map.get(int(row["tagId"]))

            if not movie_db_id or not genome_tag_db_id:
                skipped += 1
                continue

            batch.append(
                GenomeScore(
                    movie_id=movie_db_id,
                    genome_tag_id=genome_tag_db_id,
                    relevance=float(row["relevance"]),
                )
            )

            count += 1

            if len(batch) >= batch_size:
                self.bulk_create_in_batches(GenomeScore, batch, batch_size)
                batch.clear()
                self.stdout.write(f"Imported genome scores so far: {count}")

        if batch:
            self.bulk_create_in_batches(GenomeScore, batch, batch_size)

        self.stdout.write(self.style.SUCCESS(f"Imported genome scores: {count}, skipped: {skipped}"))


    # import movies_metadata.csv
    @transaction.atomic
    def import_movies_metadata(self, path):
        rows = self.read_csv(path)
        count = 0
        skipped = 0

        for row in rows:
            tmdb_id = self.parse_int(row["id"])

            if not tmdb_id:
                skipped += 1
                continue

            link = MovieLink.objects.filter(tmdb_id=tmdb_id).first()

            if not link:
                skipped += 1
                continue

            MovieMetadata.objects.update_or_create(
                movie=link.movie,
                defaults={
                    "tmdb_id": tmdb_id,
                    "imdb_id": row["imdb_id"] or None,
                    "adult": self.parse_bool(row["adult"]),
                    "budget": self.parse_int(row["budget"]),
                    "homepage": row["homepage"] or None,
                    "original_language": row["original_language"] or None,
                    "original_title": row["original_title"] or None,
                    "overview": row["overview"] or None,
                    "popularity": self.parse_float(row["popularity"]),
                    "poster_path": row["poster_path"] or None,
                    "release_date": self.parse_date(row["release_date"]),
                    "revenue": self.parse_int(row["revenue"]),
                    "runtime": self.parse_float(row["runtime"]),
                    "status": row["status"] or None,
                    "tagline": row["tagline"] or None,
                    "video": self.parse_bool(row["video"]),
                    "vote_average": self.parse_float(row["vote_average"]),
                    "vote_count": self.parse_int(row["vote_count"]),
                    "belongs_to_collection": self.parse_json_like(row["belongs_to_collection"], None),
                    "genres": self.parse_json_like(row["genres"], []),
                    "production_companies": self.parse_json_like(row["production_companies"], []),
                    "production_countries": self.parse_json_like(row["production_countries"], []),
                    "spoken_languages": self.parse_json_like(row["spoken_languages"], []),
                    "raw_data": dict(row),
                },
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f"Imported movie metadata: {count}, skipped: {skipped}"))


    # import credits.csv
    @transaction.atomic
    def import_credits(self, path):
        rows = self.read_csv(path)
        count = 0
        skipped = 0

        for row in rows:
            movie_id = self.parse_int(row["id"])

            if not movie_id:
                skipped += 1
                continue

            movie = Movie.objects.filter(movie_id=movie_id).first()

            if not movie:
                skipped += 1
                continue

            MovieCredits.objects.update_or_create(
                movie=movie,
                defaults={
                    "source_id": movie_id,
                    "cast": self.parse_json_like(row["cast"], []),
                    "crew": self.parse_json_like(row["crew"], []),
                },
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f"Imported credits: {count}, skipped: {skipped}"))