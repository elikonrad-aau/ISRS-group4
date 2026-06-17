import json
import os
import csv
from collections import defaultdict
from statistics import mean


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


folder = os.path.dirname(os.path.abspath(__file__))

per_algorithm = defaultdict(lambda: {
    "n": 0,
    "ratings": [],
    "relevance": [],
    "watch_likelihood": [],
    "novelty": [],
    "top_5_hits": 0,
    "ranks": [],
})

overall_scores = []

for filename in os.listdir(folder):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(folder, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_5 = set(data.get("overall", {}).get("top_5", []))

    overall_scores.append({
        "file": filename,
        "overall_relevance": to_int(data.get("overall", {}).get("relevance")),
        "overall_diversity": to_int(data.get("overall", {}).get("diversity")),
        "overall_satisfaction": to_int(data.get("overall", {}).get("satisfaction")),
        "overall_trust": to_int(data.get("overall", {}).get("trust")),
        "overall_usefulness": to_int(data.get("overall", {}).get("usefulness")),
    })

    for response in data.get("responses", []):
        movie_id = str(response.get("movie_id"))

        rating = to_int(response.get("rating"))
        relevance = to_int(response.get("relevance"))
        watch_likelihood = to_int(response.get("watch_likelihood"))
        novelty_score = to_int(response.get("novelty_score"))

        for algorithm, rank in response.get("algorithms", {}).items():
            stats = per_algorithm[algorithm]

            stats["n"] += 1
            stats["ranks"].append(to_int(rank))

            if rating is not None:
                stats["ratings"].append(rating)

            if relevance is not None:
                stats["relevance"].append(relevance)

            if watch_likelihood is not None:
                stats["watch_likelihood"].append(watch_likelihood)

            if novelty_score is not None:
                stats["novelty"].append(novelty_score)

            if movie_id in top_5:
                stats["top_5_hits"] += 1


def avg(values):
    return round(mean(values), 3) if values else None


algorithm_rows = []

for algorithm, stats in sorted(per_algorithm.items()):
    n = stats["n"]

    algorithm_rows.append({
        "algorithm": algorithm,
        "n_recommendations": n,
        "avg_rating_watched": avg(stats["ratings"]),
        "avg_relevance": avg(stats["relevance"]),
        "avg_watch_likelihood_unwatched": avg(stats["watch_likelihood"]),
        "avg_novelty": avg(stats["novelty"]),
        "top_5_hits": stats["top_5_hits"],
        "top_5_hit_rate": round(stats["top_5_hits"] / n, 3) if n else None,
    })


overall_summary = {
    "n_files": len(overall_scores),
    "avg_overall_relevance": avg([x["overall_relevance"] for x in overall_scores if x["overall_relevance"] is not None]),
    "avg_overall_diversity": avg([x["overall_diversity"] for x in overall_scores if x["overall_diversity"] is not None]),
    "avg_overall_satisfaction": avg([x["overall_satisfaction"] for x in overall_scores if x["overall_satisfaction"] is not None]),
    "avg_overall_trust": avg([x["overall_trust"] for x in overall_scores if x["overall_trust"] is not None]),
    "avg_overall_usefulness": avg([x["overall_usefulness"] for x in overall_scores if x["overall_usefulness"] is not None]),
}


print("\n=== Overall Summary ===")
for key, value in overall_summary.items():
    print(f"{key}: {value}")

print("\n=== Per Algorithm Summary ===")
for row in algorithm_rows:
    print(row)


with open(os.path.join(folder, "algorithm_summary.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=algorithm_rows[0].keys())
    writer.writeheader()
    writer.writerows(algorithm_rows)

with open(os.path.join(folder, "overall_summary.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=overall_summary.keys())
    writer.writeheader()
    writer.writerow(overall_summary)

print("\nSaved:")
print("algorithm_summary.csv")
print("overall_summary.csv")