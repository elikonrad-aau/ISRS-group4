from django.db.models import Q, Avg, Count
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404

from apps.data.models import Movie
from apps.recommender.algorithms_recs import get_recommendation_rows


def movie_search(request):
    query = request.GET.get("q", "").strip()

    if len(query) < 2:
        return render(request, "partials/movie_search_results.html", {"movies": []})

    lowered = query.lower()

    search = Q(title__icontains=query)

    if lowered.startswith("the "):
        without_article = query[4:]
        search |= Q(title__icontains=without_article)
        search |= Q(title__icontains=f"{without_article}, The")

    elif lowered.startswith("a "):
        without_article = query[2:]
        search |= Q(title__icontains=without_article)
        search |= Q(title__icontains=f"{without_article}, A")

    elif lowered.startswith("an "):
        without_article = query[3:]
        search |= Q(title__icontains=without_article)
        search |= Q(title__icontains=f"{without_article}, An")

    movies = (
        Movie.objects
        .filter(search)
        .annotate(
            avg_rating=Avg("ratings__rating"),
            rating_count=Count("ratings"),
        )
        .order_by("-rating_count", "-avg_rating")[:10]
    )

    return render(request, "partials/movie_search_results.html", {"movies": movies})



def movie_selection(request):
    return render(request, "movie_selection.html")


def movie_detail(request, movie_id):
    movie = get_object_or_404(
        Movie.objects
        .annotate(
            avg_rating=Avg("ratings__rating"),
            rating_count=Count("ratings"),
        ),
        movie_id=movie_id,
    )

    return render(request, "partials/movie_selection_modal.html", {
        "movie": movie,
    })

def recommendations(request):
    reference_movie_id = request.GET.get("movie_id")

    reference_movie = get_object_or_404(Movie, movie_id=reference_movie_id)

    recommendation_rows = get_recommendation_rows(
        reference_movie_id=reference_movie.movie_id,
        limit=10,
    )

    return render(request, "recommendations.html", {
        "reference_movie": reference_movie,
        "recommendation_rows": recommendation_rows,
    })