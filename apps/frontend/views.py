from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404

from apps.data.models import Movie


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
        .order_by(
            "-metadata__popularity",
            "-metadata__vote_count",
        )[:5]
    )

    return render(request, "partials/movie_search_results.html", {"movies": movies})



def movie_selection(request):
    return render(request, "movie_selection.html")


def movie_detail(request, movie_id):
    movie = get_object_or_404(
        Movie.objects.select_related("metadata"),
        movie_id=movie_id,
    )

    return render(request, "partials/movie_detail_modal.html", {
        "movie": movie,
    })

def recommendations(request):
    movie_id = request.GET.get("movie_id")
    return render(request, "recommendations.html", {"movie_id": movie_id})