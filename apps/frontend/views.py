from django.db.models import Q, Avg, Count, Case, When, Value, IntegerField
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect

from apps.data.models import Movie
from apps.recommender.algorithms_recs import get_recommendation_rows, get_recommendation_row
from django.conf import settings
import os
import random
import traceback


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
    try:
        collages_dir = os.path.join(settings.BASE_DIR, 'static', 'collages')

        if not os.path.exists(collages_dir):
            print(f"ERROR: Directory not found: {collages_dir}")
            selected_bg = None
        else:
            collages = [f for f in os.listdir(collages_dir) if f.endswith('.jpg')]

            if not collages:
                print(f"WARNING: No .jpg files found in {collages_dir}")
                selected_bg = None
            else:
                selected_filename = random.choice(collages)
                selected_bg = f"/static/collages/{selected_filename}"
            print(selected_bg)
        return render(request, 'movie_selection.html', {'background_image': selected_bg})

    except Exception as e:
        print(f"CRITICAL ERROR in movie_selection: {e}")
        traceback.print_exc()
        return render(request, '500.html', status=500)

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

#
# # recommendations page
# def recommendations(request):
#     reference_movie_id = request.GET.get("movie_id")
#
#     reference_movie = get_object_or_404(Movie, movie_id=reference_movie_id)
#
#     user_selection = {}
#
#     # Check which keys are present in the GET request
#     possible_prefs = ['story', 'cast', 'mood']
#     for pref in possible_prefs:
#         user_selection[pref] = request.GET.get(pref)
#
#     print(user_selection)
#     recommendation_rows = get_recommendation_rows(
#         reference_movie_id=reference_movie.movie_id,
#         user_selection = user_selection,
#         limit=10,
#     )
#
#     return render(request, "recommendations.html", {
#         "reference_movie": reference_movie,
#         "recommendation_rows": recommendation_rows
#     })

# --------------------------------------------------
# Recommendations Page
# --------------------------------------------------

def recommendations(request):
    movie_id = request.GET.get("movie_id")

    if not movie_id:
        return JsonResponse(
            {"error": "movie_id is required"},
            status=400,
        )

    reference_movie = get_object_or_404(
        Movie,
        movie_id=movie_id,
    )

    user_selection = {
        "story": request.GET.get("story"),
        "cast": request.GET.get("cast"),
        "mood": request.GET.get("mood"),
    }

    # Load TMDB recommendations for hero section
    tmdb_row = get_recommendation_row(
        algorithm="tmdb",
        reference_movie_id=movie_id,
        user_selection=user_selection,
        limit=10,
    )

    tmdb_recs = []

    if tmdb_row:
        tmdb_recs = tmdb_row.get("movies", [])

    return render(
        request,
        "recommendations.html",
        {
            "reference_movie": reference_movie,
            "user_selection": user_selection,
            "tmdb_recs": tmdb_recs,
            "algorithms": [
                "tmdb",
                "knn",
                "castoverlap",
                "genome_overlap",
                "image_similarity",
                "image_text_similarity",
                "subtitles",
                "weighted_hybrid",
            ],
        },
    )



# --------------------------------------------------
# HTMX Recommendation Row By Algorithm
# --------------------------------------------------

def recommendation_algorithm_row(request, movie_id, algorithm):
    prefs = request.GET.get("prefs", "")
    user_selection = {
        "story": "story" in prefs,
        "cast": "cast" in prefs,
        "mood": "mood" in prefs,
    }

    row = get_recommendation_row(
        algorithm=algorithm,
        reference_movie_id=movie_id,
        limit=10,
        user_selection = user_selection
    )

    # TODO ADD EVAL
    if algorithm != "tmdb":
        print("EVAL for " + algorithm)

    if not row:
        row = {
            "algorithm": algorithm,
            "title": "No results",
            "movies": [],
        }

    return render(
        request,
        "partials/recommendation_row.html",
        {
            "row": row,
        },
    )




### evaluation
def set_mode(request, mode):
    if mode in ["recommendation", "evaluation"]:
        request.session["mode"] = mode

    return redirect(request.META.get("HTTP_REFERER", "/"))


def movie_evaluation(request):
    movie_id = request.GET.get("movie_id")

    if not movie_id:
        return JsonResponse({"error": "movie_id is required"}, status=400)

    reference_movie = get_object_or_404(Movie, movie_id=movie_id)

    user_selection = {
        "story": request.GET.get("story"),
        "cast": request.GET.get("cast"),
        "mood": request.GET.get("mood"),
    }

    recommendation_rows = get_recommendation_rows(
        reference_movie_id=movie_id,
        user_selection=user_selection,
        limit=10,
    )

    unique_movies = {}

    for row in recommendation_rows:
        for movie in row.get("movies", []):
            unique_movies[movie.movie_id] = movie

    movies = list(unique_movies.values())
    random.shuffle(movies)

    evaluation_row = {
        "algorithm": "evaluation",
        "title": "All Unique Recommendations",
        "movies": movies,
    }

    return render(
        request,
        "evaluation.html",
        {
            "reference_movie": reference_movie,
            "row": evaluation_row,
        },
    )