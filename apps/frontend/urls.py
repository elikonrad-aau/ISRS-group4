from django.urls import path
from .views import movie_selection, movie_search, movie_detail, recommendations, recommendation_algorithm_row, movie_evaluation, set_mode

app_name = "frontend"

urlpatterns = [
    path("", movie_selection, name="movie_selection"),
    path("api/movies/search/", movie_search, name="movie_search"),
    path("api/movies/<int:movie_id>/", movie_detail, name="movie_detail"),
    path("recommendations/", recommendations, name="recommendations"),
    path(
        "recommendations/<int:movie_id>/<str:algorithm>/",
        recommendation_algorithm_row,
        name="recommendation-algorithm-row",
    ),
    path("evaluation/", movie_evaluation, name="movie_evaluation"),
    path("mode/<str:mode>/", set_mode, name="set_mode"),

]
