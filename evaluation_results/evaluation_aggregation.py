import csv
import json

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

RESPONSE_METRICS = {
    # Internal storage name: (metric key, output-column suffix)
    "ratings": (
        "rating",
        "rating_watched",
    ),
    "relevance": (
        "relevance",
        "relevance",
    ),
    "watch_likelihood": (
        "watch_likelihood",
        "watch_likelihood_unwatched",
    ),
    "preference_scores": (
        "preference_score",
        "preference_score",
    ),
    "novelty": (
        "novelty_score",
        "novelty",
    ),
}


# Metrics used in the evaluator-level wide and long tables.
#
# Rating and watch likelihood are excluded because preference_score
# already combines them into one comparable measure.
ANALYSIS_METRICS = (
    "relevance",
    "preference_score",
    "novelty_score",
)


OVERALL_METRICS = (
    "relevance",
    "diversity",
    "satisfaction",
    "trust",
    "usefulness",
)


# The algorithm ranking uses four conceptually distinct criteria.
RANKING_METRICS = (
    "mean_relevance",
    "top_5_hit_rate",
    "mean_preference_score",
    "mean_novelty",
)


# Reciprocal rank fusion gives every metric equal importance.
RRF_METRICS = RANKING_METRICS


# Every weight set must:
#
# 1. contain the same metrics as RANKING_METRICS;
# 2. add up to 1.0.
WEIGHT_SETS = {
    "equal": {
        "mean_relevance": 0.25,
        "top_5_hit_rate": 0.25,
        "mean_preference_score": 0.25,
        "mean_novelty": 0.25,
    },

    "quality_focused": {
        "mean_relevance": 0.30,
        "top_5_hit_rate": 0.30,
        "mean_preference_score": 0.20,
        "mean_novelty": 0.20,
    },

    "discovery_focused": {
        "mean_relevance": 0.30,
        "top_5_hit_rate": 0.20,
        "mean_preference_score": 0.20,
        "mean_novelty": 0.30,
    },
}


# Determines how evaluation_results.csv is ordered.
PRIMARY_WEIGHT_SET = "equal"


# -------------------------------------------------------------------
# General helpers
# -------------------------------------------------------------------

def to_int(value):
    """Convert a value to an integer, or return None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def descriptive_stats(values, name, include_count=True):
    """
    Calculate mean, median, and sample standard deviation.

    The count is included unless include_count is False.
    """
    stats = {
        f"mean_{name}": (
            round(mean(values), 3)
            if values
            else None
        ),
        f"median_{name}": (
            round(median(values), 3)
            if values
            else None
        ),
        f"std_{name}": (
            round(stdev(values), 3)
            if len(values) >= 2
            else None
        ),
    }

    if include_count:
        stats[f"n_{name}"] = len(values)

    return stats


def write_csv(path, rows):
    """Write a list of dictionaries to a CSV file."""
    if not rows:
        print(f"No data available for {path.name}.")
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {path.name}")


def validate_weight_sets():
    """Check that all weighting configurations are valid."""
    required_metrics = set(RANKING_METRICS)

    if PRIMARY_WEIGHT_SET not in WEIGHT_SETS:
        raise ValueError(
            f"PRIMARY_WEIGHT_SET '{PRIMARY_WEIGHT_SET}' "
            "does not exist in WEIGHT_SETS."
        )

    for weight_set_name, weights in WEIGHT_SETS.items():
        supplied_metrics = set(weights)

        if supplied_metrics != required_metrics:
            missing = required_metrics - supplied_metrics
            extra = supplied_metrics - required_metrics

            raise ValueError(
                f"Invalid metrics in weight set "
                f"'{weight_set_name}'. "
                f"Missing: {sorted(missing)}. "
                f"Extra: {sorted(extra)}."
            )

        weight_sum = sum(weights.values())

        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(
                f"Weights in '{weight_set_name}' must add up "
                f"to 1.0, but they add up to {weight_sum}."
            )


# -------------------------------------------------------------------
# Loading JSON evaluations
# -------------------------------------------------------------------

def parse_created_at(value):
    """Parse an ISO timestamp and normalize it to UTC."""
    minimum_date = datetime.min.replace(
        tzinfo=timezone.utc
    )

    if not isinstance(value, str):
        return minimum_date

    try:
        timestamp = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return minimum_date

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(
            tzinfo=timezone.utc
        )

    return timestamp.astimezone(timezone.utc)


def load_evaluations(folder):
    """
    Load all JSON files in the script folder.

    Evaluator IDs are assigned according to created_at.
    Evaluator 1 is the oldest evaluation.
    """
    evaluations = []

    for path in folder.glob("*.json"):
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        evaluations.append({
            "path": path,
            "data": data,
            "created_at": parse_created_at(
                data.get("created_at")
            ),
        })

    evaluations.sort(
        key=lambda evaluation: (
            evaluation["created_at"],
            evaluation["path"].name,
        )
    )

    for evaluator_id, evaluation in enumerate(
        evaluations,
        start=1,
    ):
        evaluation["evaluator_id"] = evaluator_id

    return evaluations


# -------------------------------------------------------------------
# Collecting evaluation values
# -------------------------------------------------------------------

def create_empty_algorithm_stats():
    """Create the statistics container for one algorithm."""
    return {
        "n": 0,
        "ratings": [],
        "relevance": [],
        "watch_likelihood": [],
        "preference_scores": [],
        "novelty": [],
        "top_5_flags": [],
    }


def collect_evaluation_data(evaluations):
    """
    Collect the values required for the aggregate, wide, and long tables.
    """
    per_algorithm = defaultdict(
        create_empty_algorithm_stats
    )

    # Structure:
    #
    # evaluator_totals[evaluator_id][algorithm][metric]
    #     -> [sum, count]
    evaluator_totals = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: [0, 0]
            )
        )
    )

    overall_scores = []
    overall_by_evaluator = {}

    algorithms = set()
    evaluated_movies = set()

    for evaluation in evaluations:
        evaluator_id = evaluation["evaluator_id"]
        filename = evaluation["path"].name
        data = evaluation["data"]

        overall = data.get("overall", {})

        overall_values = {
            f"overall_{metric}": to_int(
                overall.get(metric)
            )
            for metric in OVERALL_METRICS
        }

        overall_scores.append({
            "file": filename,
            **overall_values,
        })

        overall_by_evaluator[
            evaluator_id
        ] = overall_values

        top_5 = {
            str(movie_id)
            for movie_id in overall.get(
                "top_5",
                [],
            )
        }

        for response in data.get("responses", []):
            movie_id = str(
                response.get("movie_id")
            )

            evaluated_movies.add(
                (evaluator_id, movie_id)
            )

            rating = to_int(
                response.get("rating")
            )

            watch_likelihood = to_int(
                response.get("watch_likelihood")
            )

            # Combined preference measure:
            #
            # - Use watch likelihood when available.
            # - Otherwise, use the rating.
            preference_score = (
                watch_likelihood
                if watch_likelihood is not None
                else rating
            )

            metric_values = {
                "rating": rating,
                "relevance": to_int(
                    response.get("relevance")
                ),
                "watch_likelihood": watch_likelihood,
                "preference_score": preference_score,
                "novelty_score": to_int(
                    response.get("novelty_score")
                ),
            }

            in_top_5 = int(
                movie_id in top_5
            )

            algorithms_for_response = response.get(
                "algorithms",
                {},
            )

            for algorithm in algorithms_for_response:
                algorithms.add(algorithm)

                # ---------------------------------------------------
                # Algorithm-level aggregate statistics
                # ---------------------------------------------------

                stats = per_algorithm[algorithm]
                stats["n"] += 1

                for (
                    storage_name,
                    metric_config,
                ) in RESPONSE_METRICS.items():

                    metric_key = metric_config[0]
                    value = metric_values[metric_key]

                    if value is not None:
                        stats[storage_name].append(
                            value
                        )

                stats["top_5_flags"].append(
                    in_top_5
                )

                # ---------------------------------------------------
                # Evaluator-level values for statistical tests
                # ---------------------------------------------------

                for metric in ANALYSIS_METRICS:
                    value = metric_values[metric]

                    if value is None:
                        continue

                    total_and_count = evaluator_totals[
                        evaluator_id
                    ][algorithm][metric]

                    total_and_count[0] += value
                    total_and_count[1] += 1

    return {
        "per_algorithm": per_algorithm,
        "overall_scores": overall_scores,
        "overall_by_evaluator": overall_by_evaluator,
        "evaluator_totals": evaluator_totals,
        "algorithms": algorithms,
        "evaluated_movies": evaluated_movies,
    }


# -------------------------------------------------------------------
# Building aggregate and summary tables
# -------------------------------------------------------------------

def build_algorithm_rows(per_algorithm):
    """Create one aggregate row for every algorithm."""
    rows = []

    for algorithm, stats in sorted(
        per_algorithm.items()
    ):
        row = {
            "algorithm": algorithm,
            "n_recommendations": stats["n"],
        }

        for (
            storage_name,
            metric_config,
        ) in RESPONSE_METRICS.items():

            output_name = metric_config[1]

            row.update(
                descriptive_stats(
                    stats[storage_name],
                    output_name,
                )
            )

        top_5_flags = stats["top_5_flags"]

        row["top_5_hits"] = sum(
            top_5_flags
        )

        row.update(
            descriptive_stats(
                top_5_flags,
                "top_5_hit",
            )
        )

        # The mean of binary zero/one values equals the hit rate.
        row["top_5_hit_rate"] = row[
            "mean_top_5_hit"
        ]

        rows.append(row)

    return rows


def build_overall_summary(overall_scores):
    """
    Create descriptive statistics across all evaluations.

    Only one overall count, n_files, is included.
    """
    summary = {
        "n_files": len(overall_scores),
    }

    for metric in OVERALL_METRICS:
        column_name = f"overall_{metric}"

        values = [
            row[column_name]
            for row in overall_scores
            if row[column_name] is not None
        ]

        summary.update(
            descriptive_stats(
                values,
                column_name,
                include_count=False,
            )
        )

    return summary


# -------------------------------------------------------------------
# Building evaluator-level analysis tables
# -------------------------------------------------------------------

def get_evaluator_metric(
    evaluator_totals,
    evaluator_id,
    algorithm,
    metric,
):
    """
    Return an evaluator-level mean and its observation count.

    The mean represents all movies assigned to the evaluator by the
    selected algorithm.
    """
    total, count = evaluator_totals[
        evaluator_id
    ][algorithm][metric]

    value = (
        round(total / count, 3)
        if count
        else ""
    )

    return value, count


def build_wide_rows(
    evaluations,
    algorithms,
    evaluator_totals,
    overall_by_evaluator,
):
    """
    Create one row per evaluator.

    Example columns:
        relevance_knn
        preference_score_knn
        novelty_score_knn
    """
    rows = []
    sorted_algorithms = sorted(algorithms)

    for evaluation in evaluations:
        evaluator_id = evaluation[
            "evaluator_id"
        ]

        row = {
            "evaluator_id": evaluator_id,
        }

        for algorithm in sorted_algorithms:
            for metric in ANALYSIS_METRICS:
                value, _ = get_evaluator_metric(
                    evaluator_totals=evaluator_totals,
                    evaluator_id=evaluator_id,
                    algorithm=algorithm,
                    metric=metric,
                )

                column_name = (
                    f"{metric}_{algorithm}"
                )

                row[column_name] = value

        row.update(
            overall_by_evaluator[evaluator_id]
        )

        rows.append(row)

    return rows


def build_long_rows(
    evaluations,
    algorithms,
    evaluator_totals,
):
    """
    Create evaluator-level long-format data.

    Each row represents one:

        evaluator × algorithm × metric

    The value is the evaluator's mean across all movies assigned
    to that algorithm.

    This is the appropriate granularity for Friedman and paired
    Wilcoxon tests when the evaluator is the repeated-measures unit.
    """
    rows = []
    sorted_algorithms = sorted(algorithms)

    for evaluation in evaluations:
        evaluator_id = evaluation[
            "evaluator_id"
        ]

        for algorithm in sorted_algorithms:
            for metric in ANALYSIS_METRICS:
                value, count = get_evaluator_metric(
                    evaluator_totals=evaluator_totals,
                    evaluator_id=evaluator_id,
                    algorithm=algorithm,
                    metric=metric,
                )

                # Omit evaluator-algorithm combinations where no
                # observations exist for this metric.
                if count == 0:
                    continue

                rows.append({
                    "evaluator_id": evaluator_id,
                    "algorithm": algorithm,
                    "metric": metric,
                    "value": value,
                    "n_observations": count,
                })

    return rows


# -------------------------------------------------------------------
# Building per-movie data
# -------------------------------------------------------------------

def build_per_movie_rows(evaluations):
    """
    Create one row per evaluator, movie, and algorithm.

    This preserves the original recommendation-level observations.

    When one movie was produced by multiple algorithms, one row is
    created for each associated algorithm.
    """
    rows = []

    for evaluation in evaluations:
        evaluator_id = evaluation[
            "evaluator_id"
        ]
        filename = evaluation["path"].name
        data = evaluation["data"]

        reference_movie_id = data.get(
            "reference_movie_id"
        )

        overall = data.get("overall", {})

        top_5 = {
            str(movie_id)
            for movie_id in overall.get(
                "top_5",
                [],
            )
        }

        for response in data.get("responses", []):
            movie_id = str(
                response.get("movie_id")
            )

            rating = to_int(
                response.get("rating")
            )

            watch_likelihood = to_int(
                response.get("watch_likelihood")
            )

            if watch_likelihood is not None:
                preference_score = watch_likelihood
                preference_source = "watch_likelihood"

            elif rating is not None:
                preference_score = rating
                preference_source = "rating"

            else:
                preference_score = None
                preference_source = ""

            relevance = to_int(
                response.get("relevance")
            )

            novelty_score = to_int(
                response.get("novelty_score")
            )

            in_top_5 = int(
                movie_id in top_5
            )

            algorithms_for_response = response.get(
                "algorithms",
                {},
            )

            for (
                algorithm,
                algorithm_rank,
            ) in algorithms_for_response.items():

                rows.append({
                    "evaluator_id": evaluator_id,
                    "source_file": filename,
                    "reference_movie_id": reference_movie_id,
                    "movie_id": movie_id,
                    "algorithm": algorithm,
                    "algorithm_rank": algorithm_rank,
                    "familiarity": response.get(
                        "familiarity"
                    ),
                    "rating": rating,
                    "watch_likelihood": watch_likelihood,
                    "preference_score": preference_score,
                    "preference_source": preference_source,
                    "relevance": relevance,
                    "novelty_score": novelty_score,
                    "in_top5": in_top_5,
                })

    return rows


# -------------------------------------------------------------------
# Algorithm ranking
# -------------------------------------------------------------------

def get_metric_ranges(rows, metrics):
    """Get the minimum and maximum value for each ranking metric."""
    metric_ranges = {}

    for metric in metrics:
        values = [
            row[metric]
            for row in rows
            if row.get(metric) is not None
        ]

        metric_ranges[metric] = (
            (min(values), max(values))
            if values
            else None
        )

    return metric_ranges


def normalize_value(value, minimum, maximum):
    """Normalize a value to the range zero to one."""
    if minimum == maximum:
        return 1.0

    return (
        (value - minimum)
        / (maximum - minimum)
    )


def add_weighted_ranking(
    rows,
    weight_set_name,
    weights,
):
    """
    Add score and rank columns for one weighting scenario.

    Example output columns:
        weighted_score_equal
        weighted_rank_equal
    """
    score_column = (
        f"weighted_score_{weight_set_name}"
    )
    rank_column = (
        f"weighted_rank_{weight_set_name}"
    )

    metric_ranges = get_metric_ranges(
        rows,
        weights.keys(),
    )

    for row in rows:
        score = 0.0

        for metric, weight in weights.items():
            value = row.get(metric)
            value_range = metric_ranges[metric]

            if value is None or value_range is None:
                continue

            minimum, maximum = value_range

            normalized_value = normalize_value(
                value,
                minimum,
                maximum,
            )

            score += normalized_value * weight

        row[score_column] = round(
            score,
            6,
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: (
            -row[score_column],
            row["algorithm"],
        ),
    )

    for rank, row in enumerate(
        ranked_rows,
        start=1,
    ):
        row[rank_column] = rank


def add_rrf_ranking(rows, metrics, k=10):
    """Add reciprocal-rank-fusion scores and ranks."""
    scores = {
        row["algorithm"]: 0.0
        for row in rows
    }

    for metric in metrics:
        valid_rows = [
            row
            for row in rows
            if row.get(metric) is not None
        ]

        ranked_rows = sorted(
            valid_rows,
            key=lambda row: (
                -row[metric],
                row["algorithm"],
            ),
        )

        for position, row in enumerate(
            ranked_rows,
            start=1,
        ):
            scores[row["algorithm"]] += (
                1 / (k + position)
            )

    for row in rows:
        row["rrf_score"] = round(
            scores[row["algorithm"]],
            6,
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: (
            -row["rrf_score"],
            row["algorithm"],
        ),
    )

    for rank, row in enumerate(
        ranked_rows,
        start=1,
    ):
        row["rrf_rank"] = rank


def add_algorithm_rankings(rows):
    """
    Add rankings for every weighting scenario and one RRF ranking.
    """
    for weight_set_name, weights in WEIGHT_SETS.items():
        add_weighted_ranking(
            rows=rows,
            weight_set_name=weight_set_name,
            weights=weights,
        )

    add_rrf_ranking(
        rows=rows,
        metrics=RRF_METRICS,
        k=10,
    )

    primary_rank_column = (
        f"weighted_rank_{PRIMARY_WEIGHT_SET}"
    )

    return sorted(
        rows,
        key=lambda row: row[primary_rank_column],
    )


# -------------------------------------------------------------------
# Main program
# -------------------------------------------------------------------

def main():
    folder = Path(__file__).resolve().parent

    validate_weight_sets()

    evaluations = load_evaluations(folder)

    if not evaluations:
        print("No JSON evaluation files found.")
        return

    print("Evaluator mapping (oldest = 1):")

    for evaluation in evaluations:
        evaluator_id = evaluation[
            "evaluator_id"
        ]
        filename = evaluation["path"].name

        print(
            f"  {evaluator_id}  {filename}"
        )

    print()

    collected = collect_evaluation_data(
        evaluations
    )

    algorithm_rows = build_algorithm_rows(
        collected["per_algorithm"]
    )

    overall_summary = build_overall_summary(
        collected["overall_scores"]
    )

    wide_rows = build_wide_rows(
        evaluations=evaluations,
        algorithms=collected["algorithms"],
        evaluator_totals=collected[
            "evaluator_totals"
        ],
        overall_by_evaluator=collected[
            "overall_by_evaluator"
        ],
    )

    long_rows = build_long_rows(
        evaluations=evaluations,
        algorithms=collected["algorithms"],
        evaluator_totals=collected[
            "evaluator_totals"
        ],
    )

    per_movie_rows = build_per_movie_rows(
        evaluations
    )

    if algorithm_rows:
        algorithm_rows = add_algorithm_rankings(
            algorithm_rows
        )

    # ---------------------------------------------------------------
    # CSV outputs
    # ---------------------------------------------------------------

    write_csv(
        folder / "evaluation_results.csv",
        algorithm_rows,
    )

    write_csv(
        folder / "evaluation_results_summary.csv",
        [overall_summary],
    )

    write_csv(
        folder / "evaluation_results_wide.csv",
        wide_rows,
    )

    write_csv(
        folder / "evaluation_results_long.csv",
        long_rows,
    )

    write_csv(
        folder / "evaluation_results_per_movie.csv",
        per_movie_rows,
    )


if __name__ == "__main__":
    main()