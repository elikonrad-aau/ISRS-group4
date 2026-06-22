from django.core.management.base import BaseCommand
from apps.data.models import Movie, GenomeScore, MovieGenomeProfile

#
# command: docker compose exec web python manage.py build_genomes
#
class Command(BaseCommand):
    help = "Build genome profiles from genome scores"

    def handle(self, *args, **options):
        # get all movies in the database
        movies = Movie.objects.all()

        # progress tracking
        updated_count = 0

        # iterate over all movies
        for movie in movies:
            # get the 50 most relevant genome tags for every movie
            top_scores = (
                GenomeScore.objects
                .filter(movie=movie)
                .order_by("-relevance")[:50]    # reduce to 50 from ~1k
            )

            # build the genome profile
            top_genome_tags = []

            for score in top_scores:
                top_genome_tags.append({
                    "tag_id": score.genome_tag_id,
                    "relevance": score.relevance,
                })

            # save the genome profile
            MovieGenomeProfile.objects.update_or_create(
                movie=movie,
                defaults={
                    "top_genome_tags": top_genome_tags,
                }
            )

            # progress tracking
            updated_count += 1
            if updated_count % 2000 == 0:
                print(f"{updated_count} movies processed...")

        print(f"{updated_count} were processed in total.")