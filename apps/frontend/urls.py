from django.urls import path
from .views import movie_selection, movie_search, movie_detail, recommendations

app_name = "frontend"

urlpatterns = [
    path("", movie_selection, name="movie_selection"),
    path("api/movies/search/", movie_search, name="movie_search"),
    path("api/movies/<int:movie_id>/", movie_detail, name="movie_detail"),
    path("recommendations/", recommendations, name="recommendations"),
]