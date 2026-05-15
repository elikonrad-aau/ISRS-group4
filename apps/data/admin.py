from django.contrib import admin

from .models import (
    Movie,
    Rating,
    Tag,
    MovieLink,
    GenomeTag,
    GenomeScore,
    MovieMetadata,
    MovieCredits,
)

admin.site.register(Movie)
admin.site.register(Rating)
admin.site.register(Tag)
admin.site.register(MovieLink)
admin.site.register(GenomeTag)
admin.site.register(GenomeScore)
admin.site.register(MovieMetadata)
admin.site.register(MovieCredits)