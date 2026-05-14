from django.urls import path
from .views import movie_selection

app_name = "frontend"

urlpatterns = [
    path("", movie_selection, name="movie_selection"),
]