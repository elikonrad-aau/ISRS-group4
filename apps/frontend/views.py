import os
import random
import traceback
import json
import uuid

from datetime import datetime
from django.db.models import Q, Avg, Count, Case, When, Value, IntegerField
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect

from apps.data.models import Movie
from apps.recommender.algorithms_recs import get_recommendation_rows, get_recommendation_row
from django.conf import settings

from apps.recommender.tmdb_similarity_eval import evaluate_single_algorithm

SKIP_SINGLE_EVAL = True

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
def movie_selection_modal(request, movie_id):
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

    # movie selection modal
def movie_detail_modal(request, movie_id):
    movie = get_object_or_404(
        Movie.objects
        .annotate(
            avg_rating=Avg("ratings__rating"),
            rating_count=Count("ratings"),
        ),
        movie_id=movie_id,
    )

    return render(request, "partials/movie_detail.html", {
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
                "collection",
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

    # Single TMDB eval
    if not SKIP_SINGLE_EVAL and algorithm != "tmdb":
        print("EVAL for " + algorithm)
        evaluate_single_algorithm(movie_id, algorithm, row, 50)


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

    try:
        movie_id = int(movie_id)
    except ValueError:
        return JsonResponse({"error": "movie_id must be an integer"}, status=400)

    reference_movie = get_object_or_404(Movie, movie_id=movie_id)

    user_selection = {
        "story": request.GET.get("story"),
        "cast": request.GET.get("cast"),
        "mood": request.GET.get("mood"),
    }

    recommendation_rows = [
        row
        for row in get_recommendation_rows(
            reference_movie_id=movie_id,
            user_selection=user_selection,
            limit=10,
        )
        if row["algorithm"] != "collection"
    ]

    print("\n=== RECOMMENDATION ROWS ===", flush=True)

    for row in recommendation_rows:
        algorithm = row.get("algorithm", "unknown")
        movies = row.get("movies", [])

        print(f"\n{algorithm}", flush=True)

        for movie in movies:
            print(
                f"  {movie.movie_id} - {movie.title}",
                flush=True,
            )

    PER_ALGO_LIMIT = 5
    unique_movies = {}

    for row in recommendation_rows:
        algorithm = row.get("algorithm", "unknown")
        selected_movies = row.get("movies", [])[:PER_ALGO_LIMIT]

        for rank, movie in enumerate(selected_movies, start=1):

            if movie.movie_id not in unique_movies:
                unique_movies[movie.movie_id] = {
                    "movie": movie,
                    "algorithms": {},
                }

            unique_movies[movie.movie_id]["algorithms"][algorithm] = rank

    movies = list(unique_movies.values())
    random.shuffle(movies)

    evaluation_movies = []

    for item in movies:
        evaluation_movies.append({
            "movie_id": item["movie"].movie_id,
            "algorithms": item["algorithms"],
        })

    request.session["evaluation_movies"] = evaluation_movies
    request.session["evaluation_reference_movie_id"] = reference_movie.movie_id
    request.session["evaluation_responses"] = {}

    return redirect("frontend:evaluation_step", step=0)


def evaluation_step(request, step):
    evaluation_movies = request.session.get("evaluation_movies", [])
    reference_movie_id = request.session.get("evaluation_reference_movie_id")

    if not evaluation_movies or not reference_movie_id:
        return redirect("frontend:movie_selection")

    if step >= len(evaluation_movies):
        return redirect("frontend:evaluation_overall")

    movie_data = evaluation_movies[step]
    movie = get_object_or_404(Movie, movie_id=movie_data["movie_id"])
    reference_movie = get_object_or_404(Movie, movie_id=reference_movie_id)

    responses = request.session.get("evaluation_responses", {})
    movie_id = str(movie.movie_id)

    if request.method == "POST":
        action = request.POST.get("action", "next")

        familiarity = request.POST.get("familiarity")

        responses[movie_id] = {
            "movie_id": movie_id,
            "algorithms": movie_data["algorithms"],
            "familiarity": familiarity,
            "rating": (
                request.POST.get("rating")
                if familiarity == "watched"
                else None
            ),
            "relevance": request.POST.get("relevance"),
            "watch_likelihood": (
                request.POST.get("watch_likelihood")
                if familiarity in ["heard", "unknown"]
                else None
            ),
        }

        request.session["evaluation_responses"] = responses
        request.session.modified = True

        if action == "previous":
            return redirect("frontend:evaluation_step", step=max(0, step - 1))

        return redirect("frontend:evaluation_step", step=step + 1)

    existing_response = responses.get(movie_id, {})

    progress = int(((step + 1) / len(evaluation_movies)) * 100)

    return render(request, "partials/evaluation_step.html", {
        "reference_movie": reference_movie,
        "movie": movie,
        "algorithms": movie_data["algorithms"],
        "step": step,
        "total_steps": len(evaluation_movies) + 1,
        "progress": progress,
        "response": existing_response,
    })


def evaluation_overall(request):
    evaluation_movies = request.session.get("evaluation_movies", [])
    reference_movie_id = request.session.get("evaluation_reference_movie_id")

    if not evaluation_movies or not reference_movie_id:
        return redirect("frontend:movie_selection")

    reference_movie = get_object_or_404(Movie, movie_id=reference_movie_id)

    movies = []

    for item in evaluation_movies:
        movie = get_object_or_404(Movie, movie_id=item["movie_id"])
        movies.append(movie)

    return render(
        request,
        "partials/evaluation_overall.html",
        {
            "reference_movie": reference_movie,
            "movies": movies,
        },
    )


def finish_evaluation(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    reference_movie_id = request.session.get("evaluation_reference_movie_id")
    responses_dict = request.session.get("evaluation_responses", {})

    responses = []

    for movie_id, response in responses_dict.items():
        familiarity = response.get("familiarity")

        novelty_score = {
            "watched": 0,
            "heard": 1,
            "unknown": 2,
        }.get(familiarity)

        response["novelty_score"] = novelty_score
        responses.append(response)

    payload = {
        "created_at": datetime.utcnow().isoformat(),
        "reference_movie_id": reference_movie_id,
        "responses": responses,
        "overall": {
            "relevance": request.POST.get("overall_relevance"),
            "diversity": request.POST.get("overall_diversity"),
            "satisfaction": request.POST.get("overall_satisfaction"),
            "trust": request.POST.get("overall_trust"),
            "usefulness": request.POST.get("overall_usefulness"),
            "top_5": request.POST.getlist("top_5"),
        },
    }

    folder = os.path.join(settings.BASE_DIR, "evaluation_results")
    os.makedirs(folder, exist_ok=True)

    filename = f"evaluation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.json"
    path = os.path.join(folder, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    request.session.pop("evaluation_movies", None)
    request.session.pop("evaluation_reference_movie_id", None)
    request.session.pop("evaluation_responses", None)

    return render(request, "partials/evaluation_thanks.html")