import os
import time
import requests

from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from apps.data.models import MovieLink

# API variables
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"

# download settings
BATCH_SIZE = 1000
BACKDROPS_PER_MOVIE = 5
POSTER_SIZE = "w500"
MAIN_BACKDROP_SIZE = "w1280"
RESNET_BACKDROP_SIZE = "w300"
SLEEP_BETWEEN_MOVIES = 0.25
SLEEP_BETWEEN_BATCHES = 60
RETRY_COUNT = 5

# bookkeeping file (progress for request fails)
PROGRESS_FILE = Path("dataset/tmdb/progress.txt")

#
# command: docker compose exec web python manage.py tmdb_download
#
class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--restart-batch", action="store_true")

    def handle(self, *args, **options):
        token = os.environ.get("TMDB_API_TOKEN")

        if not token:
            raise CommandError("TMDB_API_TOKEN missing in .env")

        overwrite = options["overwrite"]

        poster_dir = Path("dataset/tmdb/posters")
        main_backdrop_dir = Path("dataset/tmdb/main_backdrops")
        resnet_backdrop_base_dir = Path("dataset/tmdb/resnet_backdrops")

        poster_dir.mkdir(parents=True, exist_ok=True)
        main_backdrop_dir.mkdir(parents=True, exist_ok=True)
        resnet_backdrop_base_dir.mkdir(parents=True, exist_ok=True)
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

        start = self.read_progress()

        links_count = (
            MovieLink.objects
            .exclude(tmdb_id__isnull=True)
            .count()
        )

        while start < links_count:
            end = start + BATCH_SIZE

            self.stdout.write(
                self.style.WARNING(
                    f"Starting batch {start}–{min(end, links_count)} of {links_count}"
                )
            )

            try:
                stats = self.process_batch(
                    token=token,
                    overwrite=overwrite,
                    start=start,
                    end=end,
                    poster_dir=poster_dir,
                    main_backdrop_dir=main_backdrop_dir,
                    resnet_backdrop_base_dir=resnet_backdrop_base_dir,
                )

            except Exception as error:
                self.stderr.write(
                    self.style.ERROR(
                        f"Batch failed at start={start}. Progress was NOT advanced."
                    )
                )
                self.stderr.write(str(error))
                return

            start = end
            self.write_progress(start)

            self.stdout.write(
                self.style.SUCCESS(
                    f"Finished batch. "
                    f"Posters: {stats['posters']}, "
                    f"Main backdrops: {stats['main_backdrops']}, "
                    f"ResNet backdrops: {stats['resnet_backdrops']}, "
                    f"Skipped: {stats['skipped']}"
                )
            )

            if start < links_count:
                self.stdout.write(
                    f"Waiting {SLEEP_BETWEEN_BATCHES} seconds before next batch..."
                )
                time.sleep(SLEEP_BETWEEN_BATCHES)

        self.stdout.write(self.style.SUCCESS("All batches completed."))

    def process_batch(
        self,
        token,
        overwrite,
        start,
        end,
        poster_dir,
        main_backdrop_dir,
        resnet_backdrop_base_dir,
    ):
        links = (
            MovieLink.objects
            .exclude(tmdb_id__isnull=True)
            .select_related("movie")
            .order_by("movie__movie_id")
        )[start:end]

        downloaded_posters = 0
        downloaded_main_backdrops = 0
        downloaded_resnet_backdrops = 0
        skipped = 0

        for link in links:
            movie = link.movie
            tmdb_id = link.tmdb_id

            images = self.get_tmdb_images(token, tmdb_id)

            if not images:
                skipped += 1
                time.sleep(SLEEP_BETWEEN_MOVIES)
                continue

            posters = self.pick_best_images(images.get("posters", []), limit=1)
            backdrops = self.pick_best_images(
                images.get("backdrops", []),
                limit=BACKDROPS_PER_MOVIE,
            )

            if posters:
                poster_path = posters[0]["file_path"]
                output_path = poster_dir / f"{movie.movie_id}.jpg"

                if overwrite or not output_path.exists():
                    if self.download_image(poster_path, POSTER_SIZE, output_path):
                        downloaded_posters += 1

            if backdrops:
                best_backdrop = backdrops[0]
                output_path = main_backdrop_dir / f"{movie.movie_id}.jpg"

                if overwrite or not output_path.exists():
                    if self.download_image(
                        best_backdrop["file_path"],
                        MAIN_BACKDROP_SIZE,
                        output_path,
                    ):
                        downloaded_main_backdrops += 1

                movie_resnet_dir = resnet_backdrop_base_dir / str(movie.movie_id)
                movie_resnet_dir.mkdir(parents=True, exist_ok=True)

                for index, backdrop in enumerate(backdrops, start=1):
                    output_path = movie_resnet_dir / f"{index:03}.jpg"

                    if overwrite or not output_path.exists():
                        if self.download_image(
                            backdrop["file_path"],
                            RESNET_BACKDROP_SIZE,
                            output_path,
                        ):
                            downloaded_resnet_backdrops += 1

            self.stdout.write(
                f"{movie.display_title}: "
                f"posters={downloaded_posters}, "
                f"main_backdrops={downloaded_main_backdrops}, "
                f"resnet_backdrops={downloaded_resnet_backdrops}, "
                f"skipped={skipped}"
            )

            time.sleep(SLEEP_BETWEEN_MOVIES)

        return {
            "posters": downloaded_posters,
            "main_backdrops": downloaded_main_backdrops,
            "resnet_backdrops": downloaded_resnet_backdrops,
            "skipped": skipped,
        }

    def get_tmdb_images(self, token, tmdb_id):
        for attempt in range(RETRY_COUNT):
            try:
                response = requests.get(
                    f"{TMDB_API_BASE}/movie/{tmdb_id}/images",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "accept": "application/json",
                    },
                    timeout=30,
                )

                if response.status_code == 429:
                    self.wait_for_rate_limit(response)
                    continue

                if response.status_code != 200:
                    return None

                return response.json()

            except requests.exceptions.RequestException:
                time.sleep(5)

        return None

    def download_image(self, file_path, size, output_path):
        for attempt in range(RETRY_COUNT):
            try:
                response = requests.get(
                    f"{TMDB_IMAGE_BASE}/{size}{file_path}",
                    timeout=60,
                )

                if response.status_code == 200:
                    output_path.write_bytes(response.content)
                    return True

                if response.status_code == 429:
                    self.wait_for_rate_limit(response)
                    continue

                return False

            except requests.exceptions.RequestException:
                time.sleep(5)

        return False

    def wait_for_rate_limit(self, response):
        retry_after = response.headers.get("Retry-After")

        if retry_after:
            wait_seconds = int(retry_after)
        else:
            wait_seconds = 30

        self.stdout.write(
            self.style.WARNING(
                f"Rate limited. Waiting {wait_seconds} seconds..."
            )
        )

        time.sleep(wait_seconds)

    def pick_best_images(self, images, limit=3):
        if not images:
            return []

        return sorted(
            images,
            key=lambda image: (
                image.get("vote_average") or 0,
                image.get("vote_count") or 0,
                image.get("width") or 0,
            ),
            reverse=True,
        )[:limit]

    def read_progress(self):
        if not PROGRESS_FILE.exists():
            return 0

        try:
            return int(PROGRESS_FILE.read_text().strip())
        except ValueError:
            return 0

    def write_progress(self, start):
        PROGRESS_FILE.write_text(str(start))