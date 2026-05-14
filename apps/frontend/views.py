from django.shortcuts import render
# from apps.recommender.algorithms import get_all_recommendation_rows

# Create your views here.
def movie_selection(request):
    return render(request, "movie_selection.html")

    # recommendation_rows = get_all_recommendation_rows(limit=20)
    #
    # return render(
    #     request,
    #     "movie_selection.html",
    #     {
    #         "recommendation_rows": recommendation_rows,
    #     },
    # )