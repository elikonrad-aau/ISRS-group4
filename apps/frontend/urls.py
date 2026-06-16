from django.urls import path
from .views import movie_selection, movie_search, movie_selection_modal, movie_detail_modal, recommendations, recommendation_algorithm_row, movie_evaluation, set_mode, evaluation_step, evaluation_overall, finish_evaluation

app_name = "frontend"

urlpatterns = [
    path("", movie_selection, name="movie_selection"),
    path("api/movies/search/", movie_search, name="movie_search"),
    path(
        "api/movies/<int:movie_id>/selection/",
        movie_selection_modal,
        name="movie_selection_modal",
    ),
    path(
        "api/movies/<int:movie_id>/detail/",
        movie_detail_modal,
        name="movie_detail_modal",
    ),
    path("recommendations/", recommendations, name="recommendations"),
    path(
        "recommendations/<int:movie_id>/<str:algorithm>/",
        recommendation_algorithm_row,
        name="recommendation-algorithm-row",
    ),
    path("evaluation/", movie_evaluation, name="movie_evaluation"),
    path("mode/<str:mode>/", set_mode, name="set_mode"),
    path("evaluation/step/<int:step>/", evaluation_step, name="evaluation_step"),
    path("evaluation/overall/", evaluation_overall, name="evaluation_overall"),
    path("evaluation/finish/", finish_evaluation, name="finish_evaluation"),

]
