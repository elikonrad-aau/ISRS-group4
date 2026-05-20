from django.db.models import Q, Avg, Count, Case, When, Value, IntegerField
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404

from apps.data.models import Movie
from apps.recommender.algorithms_recs import get_recommendation_rows


# live search views
def movie_search(request):
    query = request.GET.get("q", "").strip()

    if len(query) < 2:
        return render(request, "partials/movie_search_results.html", {"movies": []})

    lowered = query.lower()

    search = Q(title__icontains=query)

    article_title = None

    if lowered.startswith("the "):
        without_article = query[4:]
        article_title = f"{without_article}, The"
        search |= Q(title__icontains=without_article)
        search |= Q(title__icontains=article_title)

    elif lowered.startswith("a "):
        without_article = query[2:]
        article_title = f"{without_article}, A"
        search |= Q(title__icontains=without_article)
        search |= Q(title__icontains=article_title)

    elif lowered.startswith("an "):
        without_article = query[3:]
        article_title = f"{without_article}, An"
        search |= Q(title__icontains=without_article)
        search |= Q(title__icontains=article_title)

    movies = (
        Movie.objects
        .filter(search)
        .annotate(
            avg_rating=Avg("ratings__rating"),
            rating_count=Count("ratings"),
            match_rank=Case(
                # best: exact title match
                When(title__iexact=query, then=Value(1)),

                # best for MovieLens article format
                When(title__iexact=article_title, then=Value(1)),

                # good: title starts with query
                When(title__istartswith=query, then=Value(2)),

                # okay: title contains query
                When(title__icontains=query, then=Value(3)),

                # fallback
                default=Value(4),
                output_field=IntegerField(),
            ),
        )
        .order_by("match_rank", "-rating_count", "-avg_rating")[:10]
    )

    return render(request, "partials/movie_search_results.html", {"movies": movies})


# movie selection page
def movie_selection(request):
    return render(request, "movie_selection.html")


# movie selection modal
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


# recommendations page
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