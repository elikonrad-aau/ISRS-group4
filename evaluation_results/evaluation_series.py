import json
import os
from collections import defaultdict
from datetime import datetime
from statistics import mean

import matplotlib.pyplot as plt


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def avg(values):
    return mean(values) if values else None


def parse_timestamp(data, filename):
    """
    Tries to find a timestamp in common places.
    Falls back to the file's modified time if none exists.
    """

    candidates = [
        data.get("timestamp"),
        data.get("created_at"),
        data.get("evaluated_at"),
        data.get("overall", {}).get("timestamp"),
        data.get("overall", {}).get("created_at"),
        data.get("overall", {}).get("evaluated_at"),
    ]

    for value in candidates:
        if not value:
            continue

        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)

        if isinstance(value, str):
            cleaned = value.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(cleaned)
            except ValueError:
                pass

            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y"):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    pass

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    return datetime.fromtimestamp(os.path.getmtime(path))


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(folder, "algorithm_rating_graphs")
    os.makedirs(output_folder, exist_ok=True)

    files = []

    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(folder, filename)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        timestamp = parse_timestamp(data, filename)

        files.append({
            "filename": filename,
            "timestamp": timestamp,
            "data": data,
        })

    files.sort(key=lambda x: x["timestamp"])

    per_algorithm_over_time = defaultdict(list)

    for file_entry in files:
        filename = file_entry["filename"]
        timestamp = file_entry["timestamp"]
        data = file_entry["data"]

        ratings_by_algorithm = defaultdict(list)

        for response in data.get("responses", []):
            rating = to_int(response.get("rating"))

            if rating is None:
                continue

            for algorithm in response.get("algorithms", {}).keys():
                ratings_by_algorithm[algorithm].append(rating)

        for algorithm, ratings in ratings_by_algorithm.items():
            per_algorithm_over_time[algorithm].append({
                "filename": filename,
                "timestamp": timestamp,
                "avg_rating": avg(ratings),
                "n_ratings": len(ratings),
            })

    if not per_algorithm_over_time:
        print("No algorithm rating data found.")
        return

    for algorithm, points in sorted(per_algorithm_over_time.items()):
        points.sort(key=lambda x: x["timestamp"])

        x_labels = [
            f"{p['timestamp'].strftime('%Y-%m-%d %H:%M')}\n{p['filename']}"
            for p in points
        ]
        y_values = [p["avg_rating"] for p in points]

        plt.figure(figsize=(max(10, len(points) * 1.4), 6))
        plt.plot(x_labels, y_values, marker="o")
        plt.title(f"Average Rating Over Time: {algorithm}")
        plt.xlabel("Evaluation file ordered by timestamp")
        plt.ylabel("Average rating")
        plt.ylim(0, 10)
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        safe_algorithm_name = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in algorithm
        )

        output_path = os.path.join(
            output_folder,
            f"{safe_algorithm_name}_ratings_over_time.png"
        )

        plt.savefig(output_path, dpi=200)
        plt.close()

        print(f"Saved graph: {output_path}")

    print(f"\nDone. Graphs saved in: {output_folder}")


if __name__ == "__main__":
    main()