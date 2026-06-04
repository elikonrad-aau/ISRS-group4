from django.contrib import admin
from .models import Movie, Rating, Tag, MovieLink, GenomeTag, GenomeScore, MovieGenomeProfile, GenomeRecommendation, MovieMetadata, MovieCredits

# exposes models/data to the admin interface
admin.site.register(Movie)
admin.site.register(Rating)
admin.site.register(Tag)
admin.site.register(MovieLink)
admin.site.register(GenomeTag)
admin.site.register(GenomeScore)
admin.site.register(MovieGenomeProfile)
admin.site.register(GenomeRecommendation)
admin.site.register(MovieMetadata)
admin.site.register(MovieCredits)