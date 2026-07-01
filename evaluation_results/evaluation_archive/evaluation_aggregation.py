#!/usr/bin/env python3
"""
Complete evaluation pipeline for recommendation-algorithm experiments.

The script performs the full workflow in one readable file:

1. Load and validate JSON evaluation files.
2. Export one row per evaluator × movie × algorithm.
3. Aggregate to one row per evaluator × algorithm.
4. Export long- and wide-format analysis datasets.
5. Produce algorithm-level descriptive statistics.
6. Run repeated-measures inferential tests.
7. Run exploratory serendipity and any-hit sensitivity analyses.
8. Calculate a conservative repeated-measures power approximation.
9. Export documentation-ready CSV and LaTeX tables.
10. Write a text report and an output manifest.

The pipeline can also resume from an existing recommendation-level CSV. This
is useful when the JSON-to-CSV stage has already been completed.

-------------------------------------------------------------------------------
Installation
-------------------------------------------------------------------------------

    pip install pandas numpy scipy jinja2

-------------------------------------------------------------------------------
Typical use
-------------------------------------------------------------------------------

Place this file in the folder containing the JSON evaluation files and run:

    python evaluation_pipeline_complete.py

Outputs are written to:

    evaluation_outputs/

To resume from an existing recommendation-level CSV:

    python evaluation_pipeline_complete.py \
        --resume-per-movie evaluation_results_per_movie.csv \
        --overall-csv evaluation_results_wide.csv

-------------------------------------------------------------------------------
Statistical design
-------------------------------------------------------------------------------

The evaluator is the repeated-measures subject. Each evaluator contributes one
aggregated value for every algorithm.

Primary family of four outcomes:
    - mean relevance
    - mean preference score
    - mean novelty score
    - precision@5

The four omnibus p-values are Holm-corrected. Pairwise Wilcoxon tests are only
run for outcomes whose omnibus test remains significant after this correction,
and their 28 pairwise p-values are corrected separately within each outcome.

Exploratory analyses:
    - multiplicative serendipity score
    - any-top-5-hit sensitivity analysis using Cochran's Q

The exploratory tests are deliberately kept outside the primary four-outcome
family and must be described as exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import (
    binomtest,
    chi2,
    f as f_distribution,
    friedmanchisquare,
    ncf,
    rankdata,
    wilcoxon,
)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Expected number of displayed recommendations for each evaluator and algorithm.
EXPECTED_RECOMMENDATIONS_PER_CELL = 5
STRICT_CELL_SIZE = True

# Questionnaire scale bounds. These are used for validation and fixed-scale
# normalization. Fixed bounds are preferable to sample min-max normalization,
# because scores do not change merely because another algorithm is added.
RELEVANCE_MIN = 1
RELEVANCE_MAX = 5
PREFERENCE_MIN = 1
PREFERENCE_MAX = 5
NOVELTY_MIN = 0
NOVELTY_MAX = 2
OVERALL_MIN = 1
OVERALL_MAX = 5

# Meaning assigned to novelty_score. Update only if the questionnaire used a
# different coding scheme.
NOVELTY_LABELS = {
    0: "watched",
    1: "heard_of",
    2: "unknown",
}

OVERALL_METRICS = (
    "relevance",
    "diversity",
    "satisfaction",
    "trust",
    "usefulness",
)

# Primary confirmatory family. precision@5 retains the full 0--5 hit count
# rather than collapsing it to whether at least one hit occurred.
PRIMARY_METRICS = {
    "relevance": {
        "column": "mean_relevance",
        "label": "Relevance",
    },
    "preference_score": {
        "column": "mean_preference_score",
        "label": "Preference score",
    },
    "novelty_score": {
        "column": "mean_novelty_score",
        "label": "Novelty",
    },
    "precision_at_5": {
        "column": "precision_at_5",
        "label": "Precision@5",
    },
}

ALPHA = 0.05
P_ADJUST_METHOD = "holm"

# The quality index combines three related quality signals before the broader
# quality/discovery trade-off is calculated. This avoids giving three separate
# quality metrics three times the conceptual influence of novelty.
QUALITY_COMPONENT_WEIGHTS = {
    "mean_relevance_normalized": 1 / 3,
    "mean_preference_normalized": 1 / 3,
    "precision_at_5": 1 / 3,
}

DECISION_SCENARIOS = {
    "balanced": {
        "quality_index": 0.50,
        "discovery_index": 0.50,
    },
    "quality_focused": {
        "quality_index": 0.75,
        "discovery_index": 0.25,
    },
    "discovery_focused": {
        "quality_index": 0.25,
        "discovery_index": 0.75,
    },
}

PRIMARY_DECISION_SCENARIO = "balanced"
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 20260630

# Conservative planning assumptions for Pingouin's repeated-measures ANOVA
# power approximation. This is not an exact Friedman-test power calculation.
POWER_COHENS_F = 0.25
POWER_TARGET = 0.90
POWER_AVERAGE_CORRELATION = 0.30
POWER_EPSILON = 0.50
POWER_DROPOUT_RATE = 0.15

# Friendly display names used only in documentation tables.
ALGORITHM_LABELS = {
    "castoverlap": "Cast overlap",
    "genome_overlap": "Genome overlap",
    "image_similarity": "Image similarity",
    "image_text_similarity": "Image--text similarity",
    "knn": "KNN",
    "subtitles": "Subtitles",
    "tmdb": "TMDb",
    "weighted_hybrid": "Weighted hybrid",
}


# =============================================================================
# 2. SMALL GENERAL-PURPOSE HELPERS
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse optional command-line paths while keeping sensible defaults."""
    script_folder = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run the complete recommendation evaluation pipeline."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_folder,
        help="Folder containing JSON evaluation files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output folder. Defaults to <input-dir>/evaluation_outputs."
        ),
    )
    parser.add_argument(
        "--resume-per-movie",
        type=Path,
        default=None,
        help=(
            "Optional existing recommendation-level CSV. When supplied, "
            "the JSON loading stage is skipped."
        ),
    )
    parser.add_argument(
        "--overall-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing evaluator_id and overall_* columns. "
            "Useful with --resume-per-movie."
        ),
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip ranking-uncertainty bootstrap for faster test runs.",
    )
    parser.add_argument(
        "--skip-power",
        action="store_true",
        help="Skip the repeated-measures power approximation.",
    )

    return parser.parse_args()


def validate_configuration() -> None:
    """Check all weight dictionaries before any data are processed."""
    if PRIMARY_DECISION_SCENARIO not in DECISION_SCENARIOS:
        raise ValueError(
            f"Unknown primary scenario: {PRIMARY_DECISION_SCENARIO!r}."
        )

    quality_sum = sum(QUALITY_COMPONENT_WEIGHTS.values())
    if not math.isclose(quality_sum, 1.0, abs_tol=1e-12):
        raise ValueError(
            "QUALITY_COMPONENT_WEIGHTS must add up to 1.0; "
            f"received {quality_sum}."
        )

    expected_dimensions = {"quality_index", "discovery_index"}

    for scenario_name, weights in DECISION_SCENARIOS.items():
        if set(weights) != expected_dimensions:
            raise ValueError(
                f"Scenario {scenario_name!r} must contain exactly "
                f"{sorted(expected_dimensions)}."
            )

        weight_sum = sum(weights.values())
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-12):
            raise ValueError(
                f"Scenario {scenario_name!r} must add up to 1.0; "
                f"received {weight_sum}."
            )


def display_algorithm(name: str) -> str:
    """Return a readable algorithm name for reports and LaTeX tables."""
    return ALGORITHM_LABELS.get(name, name.replace("_", " ").title())


def normalize_known_scale(
    value: pd.Series | float,
    minimum: float,
    maximum: float,
) -> pd.Series | float:
    """Normalize a score using its known questionnaire bounds."""
    if maximum <= minimum:
        raise ValueError("Scale maximum must be greater than scale minimum.")
    return (value - minimum) / (maximum - minimum)


def weighted_sum(row: pd.Series, weights: dict[str, float]) -> float:
    """Calculate a weighted score and fail if a component is missing."""
    values = row[list(weights)]
    if values.isna().any():
        missing = values.index[values.isna()].tolist()
        raise ValueError(f"Missing weighted-score components: {missing}")
    return float(sum(row[column] * weight for column, weight in weights.items()))


def package_version(package: str) -> str:
    """Return an installed package version without failing the analysis."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV with full analytical precision and a consistent format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {len(df):>5} rows -> {path.name}")


def holm_correction(
    p_values: Iterable[float],
    alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the Holm--Bonferroni step-down correction.

    The implementation returns arrays in the original p-value order:
        reject[i]     -> whether hypothesis i is rejected
        adjusted[i]   -> Holm-adjusted p-value
    """
    p_array = np.asarray(list(p_values), dtype=float)
    if p_array.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if len(p_array) == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    if np.any(~np.isfinite(p_array)):
        raise ValueError("Holm correction requires finite p-values.")

    order = np.argsort(p_array)
    ordered = p_array[order]
    m = len(ordered)

    # Adjusted p-values must be monotonic as the ordered raw p-values grow.
    adjusted_ordered = np.maximum.accumulate(
        [(m - index) * p for index, p in enumerate(ordered)]
    )
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)

    # Step-down rejection: once one hypothesis fails, all larger p-values fail.
    reject_ordered = np.zeros(m, dtype=bool)
    continue_rejecting = True
    for index, p_value in enumerate(ordered):
        threshold = alpha / (m - index)
        if continue_rejecting and p_value <= threshold:
            reject_ordered[index] = True
        else:
            continue_rejecting = False

    adjusted = np.empty(m, dtype=float)
    reject = np.empty(m, dtype=bool)
    adjusted[order] = adjusted_ordered
    reject[order] = reject_ordered
    return reject, adjusted


def format_p(value: Any) -> str:
    """Format p-values for documentation tables without reporting p = 0."""
    if pd.isna(value):
        return "--"
    value = float(value)
    return "< .001" if value < 0.001 else f"{value:.3f}".lstrip("0")


def format_number(value: Any, digits: int = 2) -> str:
    """Format a number for human-readable tables."""
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


# =============================================================================
# 3. JSON LOADING AND VALIDATION
# =============================================================================


def parse_created_at(
    value: Any,
    *,
    filename: str,
    warnings: list[str],
) -> datetime:
    """Parse an ISO timestamp; invalid timestamps are sorted last."""
    maximum = datetime.max.replace(tzinfo=timezone.utc)

    if not isinstance(value, str):
        warnings.append(
            f"{filename}: missing created_at; file ordered last."
        )
        return maximum

    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        warnings.append(
            f"{filename}: invalid created_at={value!r}; file ordered last."
        )
        return maximum

    if timestamp.tzinfo is None:
        warnings.append(
            f"{filename}: created_at had no timezone; UTC assumed."
        )
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    return timestamp.astimezone(timezone.utc)


def content_signature(data: dict[str, Any]) -> str:
    """Hash response content to identify apparent duplicate submissions."""
    payload = {
        "reference_movie_id": data.get("reference_movie_id"),
        "overall": data.get("overall"),
        "responses": data.get("responses"),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def as_optional_integer(value: Any, field: str) -> int | None:
    """Parse an optional integer while rejecting booleans and decimals."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, not a boolean.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer; received {value!r}.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer; received {value!r}.")
    return number


def bounded_integer(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
    allow_none: bool,
) -> int | None:
    """Parse and range-check a questionnaire value."""
    number = as_optional_integer(value, field)
    if number is None:
        if allow_none:
            return None
        raise ValueError(f"{field} is required.")
    if not minimum <= number <= maximum:
        raise ValueError(
            f"{field} must be between {minimum} and {maximum}; "
            f"received {number}."
        )
    return number


def load_json_evaluations(
    input_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Load all JSON files and assign evaluator IDs chronologically."""
    evaluations: list[dict[str, Any]] = []

    for path in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"{path.name} must contain a top-level object.")

        evaluations.append(
            {
                "path": path,
                "data": data,
                "created_at": parse_created_at(
                    data.get("created_at"),
                    filename=path.name,
                    warnings=warnings,
                ),
                "signature": content_signature(data),
            }
        )

    evaluations.sort(
        key=lambda item: (item["created_at"], item["path"].name)
    )

    for evaluator_id, evaluation in enumerate(evaluations, start=1):
        evaluation["evaluator_id"] = evaluator_id

    signature_groups: dict[str, list[str]] = {}
    for evaluation in evaluations:
        signature_groups.setdefault(evaluation["signature"], []).append(
            evaluation["path"].name
        )

    for filenames in signature_groups.values():
        if len(filenames) > 1:
            warnings.append(
                "Possible duplicate submission content: "
                + ", ".join(sorted(filenames))
            )

    return evaluations


def json_to_dataframes(
    evaluations: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert nested JSON into recommendation, overall, and mapping tables.

    One recommendation can be attributed to multiple algorithms. In that case,
    one row is emitted for every associated algorithm, while the shared origin
    is retained through n_algorithms_for_movie and is_shared_recommendation.
    """
    recommendation_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    for evaluation in evaluations:
        evaluator_id = evaluation["evaluator_id"]
        path: Path = evaluation["path"]
        data: dict[str, Any] = evaluation["data"]
        created_at = evaluation["created_at"]
        created_at_text = (
            ""
            if created_at == datetime.max.replace(tzinfo=timezone.utc)
            else created_at.isoformat()
        )

        mapping_rows.append(
            {
                "evaluator_id": evaluator_id,
                "source_file": path.name,
                "created_at_utc": created_at_text,
            }
        )

        overall = data.get("overall", {})
        if not isinstance(overall, dict):
            raise ValueError(f"{path.name}: overall must be an object.")

        overall_row: dict[str, Any] = {
            "evaluator_id": evaluator_id,
            "source_file": path.name,
        }
        for metric in OVERALL_METRICS:
            overall_row[f"overall_{metric}"] = bounded_integer(
                overall.get(metric),
                field=f"{path.name}.overall.{metric}",
                minimum=OVERALL_MIN,
                maximum=OVERALL_MAX,
                allow_none=True,
            )
        overall_rows.append(overall_row)

        top_5_raw = overall.get("top_5", []) or []
        if not isinstance(top_5_raw, list):
            raise ValueError(f"{path.name}: overall.top_5 must be a list.")
        top_5 = {str(movie_id) for movie_id in top_5_raw}

        responses = data.get("responses", [])
        if not isinstance(responses, list):
            raise ValueError(f"{path.name}: responses must be a list.")

        seen_movie_ids: set[str] = set()

        for response_number, response in enumerate(responses, start=1):
            context = f"{path.name}.responses[{response_number}]"
            if not isinstance(response, dict):
                raise ValueError(f"{context} must be an object.")

            movie_id_raw = response.get("movie_id")
            if movie_id_raw is None or movie_id_raw == "":
                raise ValueError(f"{context}.movie_id is required.")
            movie_id = str(movie_id_raw)

            if movie_id in seen_movie_ids:
                raise ValueError(
                    f"{path.name}: movie {movie_id!r} appears twice in "
                    "responses. Shared recommendations should be represented "
                    "once with several algorithms in the algorithms mapping."
                )
            seen_movie_ids.add(movie_id)

            rating = bounded_integer(
                response.get("rating"),
                field=f"{context}.rating",
                minimum=PREFERENCE_MIN,
                maximum=PREFERENCE_MAX,
                allow_none=True,
            )
            watch_likelihood = bounded_integer(
                response.get("watch_likelihood"),
                field=f"{context}.watch_likelihood",
                minimum=PREFERENCE_MIN,
                maximum=PREFERENCE_MAX,
                allow_none=True,
            )

            if (rating is None) == (watch_likelihood is None):
                raise ValueError(
                    f"{context}: exactly one of rating and watch_likelihood "
                    "must be populated."
                )

            if watch_likelihood is not None:
                preference_score = watch_likelihood
                preference_source = "watch_likelihood"
            else:
                preference_score = rating
                preference_source = "rating"

            relevance = bounded_integer(
                response.get("relevance"),
                field=f"{context}.relevance",
                minimum=RELEVANCE_MIN,
                maximum=RELEVANCE_MAX,
                allow_none=False,
            )
            novelty_score = bounded_integer(
                response.get("novelty_score"),
                field=f"{context}.novelty_score",
                minimum=NOVELTY_MIN,
                maximum=NOVELTY_MAX,
                allow_none=False,
            )

            algorithms = response.get("algorithms", {})
            if not isinstance(algorithms, dict) or not algorithms:
                raise ValueError(
                    f"{context}.algorithms must be a non-empty object."
                )

            for algorithm, rank_value in algorithms.items():
                if not isinstance(algorithm, str) or not algorithm.strip():
                    raise ValueError(f"{context}: invalid algorithm name.")
                algorithm_rank = as_optional_integer(
                    rank_value,
                    f"{context}.algorithms[{algorithm!r}]",
                )
                if algorithm_rank is None or algorithm_rank < 1:
                    raise ValueError(
                        f"{context}: algorithm rank must be at least 1."
                    )

                recommendation_rows.append(
                    {
                        "evaluator_id": evaluator_id,
                        "source_file": path.name,
                        "created_at_utc": created_at_text,
                        "reference_movie_id": data.get("reference_movie_id"),
                        "movie_id": movie_id,
                        "algorithm": algorithm,
                        "algorithm_rank": algorithm_rank,
                        "familiarity": response.get("familiarity", ""),
                        "rating": rating,
                        "watch_likelihood": watch_likelihood,
                        "preference_score": preference_score,
                        "preference_source": preference_source,
                        "relevance": relevance,
                        "novelty_score": novelty_score,
                        "in_top5": int(movie_id in top_5),
                    }
                )

        missing_top_5 = top_5 - seen_movie_ids
        if missing_top_5:
            warnings.append(
                f"{path.name}: top_5 contains unevaluated movie IDs: "
                + ", ".join(sorted(missing_top_5))
            )

    return (
        pd.DataFrame(recommendation_rows),
        pd.DataFrame(overall_rows),
        pd.DataFrame(mapping_rows),
    )


# =============================================================================
# 4. RESUME MODE AND COMMON RECOMMENDATION-LEVEL VALIDATION
# =============================================================================


def load_optional_overall_csv(path: Path | None) -> pd.DataFrame:
    """Extract evaluator-level overall ratings from a supplied CSV."""
    if path is None:
        return pd.DataFrame()

    df = pd.read_csv(path)
    required = {"evaluator_id"}
    overall_columns = [
        column for column in df.columns if column.startswith("overall_")
    ]

    if not required.issubset(df.columns) or not overall_columns:
        raise ValueError(
            "--overall-csv must contain evaluator_id and overall_* columns."
        )

    return df[["evaluator_id", *overall_columns]].drop_duplicates(
        subset=["evaluator_id"]
    )


def load_resume_csv(
    per_movie_path: Path,
    overall_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load an existing recommendation-level CSV and optional overall data."""
    recommendation_df = pd.read_csv(per_movie_path)
    overall_df = load_optional_overall_csv(overall_path)

    mapping_columns = [
        column
        for column in (
            "evaluator_id",
            "source_file",
            "created_at_utc",
        )
        if column in recommendation_df.columns
    ]

    mapping_df = (
        recommendation_df[mapping_columns]
        .drop_duplicates(subset=["evaluator_id"])
        .sort_values("evaluator_id")
        if mapping_columns
        else pd.DataFrame(
            {"evaluator_id": sorted(recommendation_df["evaluator_id"].unique())}
        )
    )

    return recommendation_df, overall_df, mapping_df


def validate_and_enrich_recommendations(
    df: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    """
    Validate recommendation-level columns and add all derived metrics.

    No analytical values are rounded here. Rounding is reserved for display
    copies used in LaTeX tables and the text report.
    """
    required_columns = {
        "evaluator_id",
        "movie_id",
        "algorithm",
        "algorithm_rank",
        "rating",
        "watch_likelihood",
        "preference_score",
        "preference_source",
        "relevance",
        "novelty_score",
        "in_top5",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Recommendation-level data are missing columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    numeric_columns = (
        "evaluator_id",
        "algorithm_rank",
        "rating",
        "watch_likelihood",
        "preference_score",
        "relevance",
        "novelty_score",
        "in_top5",
    )
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["evaluator_id", "algorithm_rank", "algorithm"]].isna().any().any():
        raise ValueError(
            "evaluator_id, algorithm, and algorithm_rank may not be missing."
        )

    range_checks = {
        "relevance": (RELEVANCE_MIN, RELEVANCE_MAX),
        "preference_score": (PREFERENCE_MIN, PREFERENCE_MAX),
        "novelty_score": (NOVELTY_MIN, NOVELTY_MAX),
        "in_top5": (0, 1),
    }
    for column, (minimum, maximum) in range_checks.items():
        invalid = df[column].isna() | ~df[column].between(minimum, maximum)
        if invalid.any():
            sample = df.loc[invalid, ["evaluator_id", "movie_id", column]].head()
            raise ValueError(
                f"Invalid {column} values. Example rows:\n{sample.to_string(index=False)}"
            )

    rating_present = df["rating"].notna()
    watch_present = df["watch_likelihood"].notna()
    invalid_source = rating_present == watch_present
    if invalid_source.any():
        raise ValueError(
            "Every row must contain exactly one of rating and "
            "watch_likelihood."
        )

    expected_preference = df["watch_likelihood"].where(
        watch_present,
        df["rating"],
    )
    if not np.allclose(
        df["preference_score"].to_numpy(float),
        expected_preference.to_numpy(float),
        equal_nan=False,
    ):
        raise ValueError(
            "preference_score does not consistently equal watch_likelihood "
            "when available and rating otherwise."
        )

    expected_source = np.where(watch_present, "watch_likelihood", "rating")
    if not np.array_equal(
        df["preference_source"].astype(str).to_numpy(),
        expected_source,
    ):
        warnings.append(
            "preference_source contained inconsistent labels and was rebuilt."
        )
        df["preference_source"] = expected_source

    duplicate_rows = df.duplicated(
        subset=["evaluator_id", "movie_id", "algorithm"],
        keep=False,
    )
    if duplicate_rows.any():
        sample = df.loc[
            duplicate_rows,
            ["evaluator_id", "movie_id", "algorithm"],
        ].head()
        raise ValueError(
            "Duplicate evaluator × movie × algorithm rows found:\n"
            + sample.to_string(index=False)
        )

    # Identify movies shared by several algorithms for the same evaluator.
    overlap_count = df.groupby(
        ["evaluator_id", "movie_id"],
        observed=True,
    )["algorithm"].transform("nunique")
    df["n_algorithms_for_movie"] = overlap_count.astype(int)
    df["is_shared_recommendation"] = (overlap_count > 1).astype(int)

    # Fixed-scale normalized components.
    df["relevance_normalized"] = normalize_known_scale(
        df["relevance"], RELEVANCE_MIN, RELEVANCE_MAX
    )
    df["preference_normalized"] = normalize_known_scale(
        df["preference_score"], PREFERENCE_MIN, PREFERENCE_MAX
    )
    df["novelty_normalized"] = normalize_known_scale(
        df["novelty_score"], NOVELTY_MIN, NOVELTY_MAX
    )

    # Exploratory multiplicative serendipity: a recommendation must be novel
    # and positively evaluated to receive a high score.
    df["serendipity_score"] = df["novelty_normalized"] * (
        (df["relevance_normalized"] + df["preference_normalized"]) / 2
    )

    # Validate evaluator × algorithm cell size and rank structure.
    cell_counts = df.groupby(
        ["evaluator_id", "algorithm"],
        observed=True,
    ).size()

    bad_cell_counts = cell_counts[
        cell_counts != EXPECTED_RECOMMENDATIONS_PER_CELL
    ]
    if not bad_cell_counts.empty:
        message = (
            "Unexpected recommendations per evaluator × algorithm cell:\n"
            + bad_cell_counts.to_string()
        )
        if STRICT_CELL_SIZE:
            raise ValueError(message)
        warnings.append(message)

    for (evaluator_id, algorithm), group in df.groupby(
        ["evaluator_id", "algorithm"], observed=True
    ):
        ranks = group["algorithm_rank"].astype(int)
        if ranks.duplicated().any():
            raise ValueError(
                f"Evaluator {evaluator_id}, algorithm {algorithm}: "
                "duplicate ranks."
            )

        expected_ranks = set(range(1, len(group) + 1))
        actual_ranks = set(ranks.tolist())
        if actual_ranks != expected_ranks:
            warnings.append(
                f"Evaluator {evaluator_id}, algorithm {algorithm}: "
                f"ranks {sorted(actual_ranks)} instead of "
                f"{sorted(expected_ranks)}."
            )

    # Check that every evaluator has every algorithm.
    evaluators = sorted(df["evaluator_id"].unique())
    algorithms = sorted(df["algorithm"].unique())
    observed_cells = set(
        map(
            tuple,
            df[["evaluator_id", "algorithm"]].drop_duplicates().to_numpy(),
        )
    )
    expected_cells = {
        (evaluator_id, algorithm)
        for evaluator_id in evaluators
        for algorithm in algorithms
    }
    missing_cells = expected_cells - observed_cells
    if missing_cells:
        raise ValueError(
            "Incomplete repeated-measures matrix. Missing cells: "
            + ", ".join(map(str, sorted(missing_cells)))
        )

    return df.sort_values(
        ["evaluator_id", "algorithm", "algorithm_rank", "movie_id"]
    ).reset_index(drop=True)


# =============================================================================
# 5. EVALUATOR × ALGORITHM AGGREGATION
# =============================================================================


def dcg(gains: np.ndarray) -> float:
    """Discounted cumulative gain for an already rank-ordered gain vector."""
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum((np.power(2.0, gains) - 1.0) / discounts))


def ndcg(group: pd.DataFrame, gain_column: str) -> float:
    """
    Calculate NDCG for the displayed list relative to its ideal reordering.

    This evaluates ordering among the five displayed recommendations. It does
    not evaluate unshown candidate movies.
    """
    ordered = group.sort_values("algorithm_rank")
    gains = ordered[gain_column].to_numpy(dtype=float)
    ideal = np.sort(gains)[::-1]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0


def reciprocal_rank_first_hit(group: pd.DataFrame) -> float:
    """Return 1/rank of the first top-5 hit, or zero if no hit occurred."""
    hit_ranks = group.loc[group["in_top5"] == 1, "algorithm_rank"]
    return 0.0 if hit_ranks.empty else 1.0 / float(hit_ranks.min())


def aggregate_evaluator_algorithm(df: pd.DataFrame) -> pd.DataFrame:
    """Create one full-precision row for every evaluator and algorithm."""
    rows: list[dict[str, Any]] = []

    for (evaluator_id, algorithm), group in df.groupby(
        ["evaluator_id", "algorithm"],
        observed=True,
        sort=True,
    ):
        group = group.sort_values("algorithm_rank")
        n_recommendations = len(group)
        top5_hit_count = int(group["in_top5"].sum())
        precision_at_5 = top5_hit_count / n_recommendations

        row: dict[str, Any] = {
            "evaluator_id": evaluator_id,
            "algorithm": algorithm,
            "n_recommendations": n_recommendations,
            "mean_relevance": group["relevance"].mean(),
            "median_relevance": group["relevance"].median(),
            "sd_relevance_within_cell": group["relevance"].std(ddof=1),
            "mean_preference_score": group["preference_score"].mean(),
            "median_preference_score": group["preference_score"].median(),
            "sd_preference_within_cell": group["preference_score"].std(ddof=1),
            "mean_novelty_score": group["novelty_score"].mean(),
            "median_novelty_score": group["novelty_score"].median(),
            "sd_novelty_within_cell": group["novelty_score"].std(ddof=1),
            "top5_hit_count": top5_hit_count,
            "precision_at_5": precision_at_5,
            "any_top5_hit": int(top5_hit_count > 0),
            "reciprocal_rank_first_top5_hit": reciprocal_rank_first_hit(group),
            "ndcg_relevance_at_5": ndcg(group, "relevance"),
            "ndcg_preference_at_5": ndcg(group, "preference_score"),
            "rating_count": int((group["preference_source"] == "rating").sum()),
            "watch_likelihood_count": int(
                (group["preference_source"] == "watch_likelihood").sum()
            ),
            "rating_share": float(
                (group["preference_source"] == "rating").mean()
            ),
            "watch_likelihood_share": float(
                (group["preference_source"] == "watch_likelihood").mean()
            ),
            "mean_relevance_normalized": group[
                "relevance_normalized"
            ].mean(),
            "mean_preference_normalized": group[
                "preference_normalized"
            ].mean(),
            "mean_novelty_normalized": group["novelty_normalized"].mean(),
            "mean_serendipity_score": group["serendipity_score"].mean(),
            "shared_recommendation_count": int(
                group["is_shared_recommendation"].sum()
            ),
            "shared_recommendation_share": group[
                "is_shared_recommendation"
            ].mean(),
        }

        for score, label in NOVELTY_LABELS.items():
            count = int((group["novelty_score"] == score).sum())
            row[f"novelty_{label}_count"] = count
            row[f"novelty_{label}_share"] = count / n_recommendations

        row["quality_index"] = weighted_sum(
            pd.Series(row), QUALITY_COMPONENT_WEIGHTS
        )
        row["discovery_index"] = row["mean_novelty_normalized"]

        for scenario_name, weights in DECISION_SCENARIOS.items():
            row[f"decision_score_{scenario_name}"] = weighted_sum(
                pd.Series(row), weights
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["evaluator_id", "algorithm"]
    ).reset_index(drop=True)


def build_analysis_long(evaluator_algorithm: pd.DataFrame) -> pd.DataFrame:
    """Create one evaluator × algorithm × metric row for analysis software."""
    metric_columns = {
        **{name: spec["column"] for name, spec in PRIMARY_METRICS.items()},
        "any_top5_hit": "any_top5_hit",
        "top5_hit_count": "top5_hit_count",
        "ndcg_relevance_at_5": "ndcg_relevance_at_5",
        "ndcg_preference_at_5": "ndcg_preference_at_5",
        "reciprocal_rank_first_top5_hit": "reciprocal_rank_first_top5_hit",
        "serendipity_score": "mean_serendipity_score",
        "quality_index": "quality_index",
        "discovery_index": "discovery_index",
    }

    frames = []
    for metric_name, column_name in metric_columns.items():
        frame = evaluator_algorithm[
            ["evaluator_id", "algorithm", "n_recommendations", column_name]
        ].copy()
        frame = frame.rename(columns={column_name: "value"})
        frame["metric"] = metric_name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)[
        [
            "evaluator_id",
            "algorithm",
            "metric",
            "value",
            "n_recommendations",
        ]
    ]


def build_analysis_wide(
    evaluator_algorithm: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create one evaluator row with metric_algorithm columns."""
    metric_columns = {
        **{name: spec["column"] for name, spec in PRIMARY_METRICS.items()},
        "any_top5_hit": "any_top5_hit",
        "top5_hit_count": "top5_hit_count",
        "serendipity_score": "mean_serendipity_score",
        "ndcg_relevance_at_5": "ndcg_relevance_at_5",
        "ndcg_preference_at_5": "ndcg_preference_at_5",
    }

    wide_parts = []
    for metric_name, column_name in metric_columns.items():
        pivot = evaluator_algorithm.pivot(
            index="evaluator_id",
            columns="algorithm",
            values=column_name,
        )
        pivot.columns = [f"{metric_name}_{algorithm}" for algorithm in pivot.columns]
        wide_parts.append(pivot)

    wide = pd.concat(wide_parts, axis=1).reset_index()

    if not overall_df.empty:
        overall_columns = [
            column for column in overall_df.columns if column.startswith("overall_")
        ]
        overall_clean = overall_df[["evaluator_id", *overall_columns]].copy()
        wide = wide.merge(overall_clean, on="evaluator_id", how="left")

    return wide


# =============================================================================
# 6. DESCRIPTIVE SUMMARIES AND DECISION RANKING
# =============================================================================


SUMMARY_METRICS = {
    "relevance": "mean_relevance",
    "preference_score": "mean_preference_score",
    "novelty_score": "mean_novelty_score",
    "precision_at_5": "precision_at_5",
    "top5_hit_count": "top5_hit_count",
    "any_top5_hit": "any_top5_hit",
    "ndcg_relevance_at_5": "ndcg_relevance_at_5",
    "ndcg_preference_at_5": "ndcg_preference_at_5",
    "reciprocal_rank_first_top5_hit": "reciprocal_rank_first_top5_hit",
    "serendipity_score": "mean_serendipity_score",
    "quality_index": "quality_index",
    "discovery_index": "discovery_index",
}


def summarize_algorithm_metrics(
    evaluator_algorithm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize across evaluator-level values.

    The SD therefore represents variation between evaluators, which matches the
    repeated-measures inferential unit. Recommendations are not incorrectly
    treated as 45 independent observations per algorithm.
    """
    rows: list[dict[str, Any]] = []

    for algorithm, group in evaluator_algorithm.groupby(
        "algorithm", observed=True, sort=True
    ):
        row: dict[str, Any] = {
            "algorithm": algorithm,
            "n_evaluators": group["evaluator_id"].nunique(),
        }

        for metric_name, column_name in SUMMARY_METRICS.items():
            values = group[column_name]
            row[f"mean_{metric_name}"] = values.mean()
            row[f"median_{metric_name}"] = values.median()
            row[f"sd_{metric_name}"] = values.std(ddof=1)
            row[f"min_{metric_name}"] = values.min()
            row[f"max_{metric_name}"] = values.max()

        row["evaluators_with_any_top5_hit"] = int(group["any_top5_hit"].sum())
        row["proportion_evaluators_with_any_top5_hit"] = group[
            "any_top5_hit"
        ].mean()
        rows.append(row)

    return pd.DataFrame(rows)


def preference_source_summary(per_movie: pd.DataFrame) -> pd.DataFrame:
    """Report how much of each algorithm's preference score came from each source."""
    table = pd.crosstab(
        per_movie["algorithm"],
        per_movie["preference_source"],
    ).reindex(columns=["rating", "watch_likelihood"], fill_value=0)
    table["total"] = table.sum(axis=1)
    table["rating_share"] = table["rating"] / table["total"]
    table["watch_likelihood_share"] = table["watch_likelihood"] / table["total"]
    return table.reset_index()


def novelty_distribution_summary(per_movie: pd.DataFrame) -> pd.DataFrame:
    """Report interpretable novelty-category counts and shares by algorithm."""
    count_table = pd.crosstab(
        per_movie["algorithm"],
        per_movie["novelty_score"],
    ).reindex(columns=sorted(NOVELTY_LABELS), fill_value=0)

    output = pd.DataFrame({"algorithm": count_table.index})
    total = count_table.sum(axis=1)

    for score, label in NOVELTY_LABELS.items():
        output[f"{label}_count"] = count_table[score].to_numpy()
        output[f"{label}_share"] = (count_table[score] / total).to_numpy()

    output["total"] = total.to_numpy()
    return output.reset_index(drop=True)


def overlap_summaries(
    per_movie: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-algorithm overlap statistics and an algorithm-pair matrix."""
    by_algorithm = (
        per_movie.groupby("algorithm", observed=True)
        .agg(
            attributed_recommendations=("movie_id", "size"),
            shared_attributed_recommendations=(
                "is_shared_recommendation",
                "sum",
            ),
            shared_recommendation_share=(
                "is_shared_recommendation",
                "mean",
            ),
        )
        .reset_index()
    )

    algorithms = sorted(per_movie["algorithm"].unique())
    matrix = pd.DataFrame(0, index=algorithms, columns=algorithms, dtype=int)

    for _, group in per_movie.groupby(
        ["evaluator_id", "movie_id"], observed=True
    ):
        present = sorted(group["algorithm"].unique())
        for algorithm in present:
            matrix.loc[algorithm, algorithm] += 1
        for algorithm_a, algorithm_b in combinations(present, 2):
            matrix.loc[algorithm_a, algorithm_b] += 1
            matrix.loc[algorithm_b, algorithm_a] += 1

    matrix.index.name = "algorithm"
    return by_algorithm, matrix.reset_index()


def overall_summary(overall_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize session-level overall ratings with a valid n per construct."""
    if overall_df.empty:
        return pd.DataFrame(
            columns=["construct", "n", "mean", "median", "sd", "minimum", "maximum"]
        )

    rows = []
    for metric in OVERALL_METRICS:
        column = f"overall_{metric}"
        if column not in overall_df.columns:
            continue
        values = pd.to_numeric(overall_df[column], errors="coerce").dropna()
        rows.append(
            {
                "construct": metric,
                "n": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "sd": values.std(ddof=1),
                "minimum": values.min(),
                "maximum": values.max(),
            }
        )
    return pd.DataFrame(rows)


def decision_ranking(
    evaluator_algorithm: pd.DataFrame,
    skip_bootstrap: bool,
) -> pd.DataFrame:
    """
    Rank algorithms under each pre-defined decision scenario.

    Scores are first calculated per evaluator and then averaged by algorithm.
    Equal scores receive equal average ranks. Bootstrap resampling treats each
    evaluator as a complete within-subject block.
    """
    algorithms = sorted(evaluator_algorithm["algorithm"].unique())
    result = pd.DataFrame({"algorithm": algorithms})

    for scenario_name in DECISION_SCENARIOS:
        score_column = f"decision_score_{scenario_name}"
        means = evaluator_algorithm.groupby("algorithm", observed=True)[
            score_column
        ].mean()
        result[f"score_{scenario_name}"] = result["algorithm"].map(means)
        result[f"rank_{scenario_name}"] = result[
            f"score_{scenario_name}"
        ].rank(method="average", ascending=False)

    if skip_bootstrap:
        return result.sort_values(
            f"rank_{PRIMARY_DECISION_SCENARIO}"
        ).reset_index(drop=True)

    evaluators = sorted(evaluator_algorithm["evaluator_id"].unique())
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for scenario_name in DECISION_SCENARIOS:
        score_column = f"decision_score_{scenario_name}"
        pivot = evaluator_algorithm.pivot(
            index="evaluator_id",
            columns="algorithm",
            values=score_column,
        ).reindex(index=evaluators, columns=algorithms)

        if pivot.isna().any().any():
            raise ValueError(
                f"Cannot bootstrap {scenario_name}: incomplete matrix."
            )

        matrix = pivot.to_numpy(dtype=float)
        bootstrap_scores = np.empty((BOOTSTRAP_ITERATIONS, len(algorithms)))
        bootstrap_ranks = np.empty_like(bootstrap_scores)
        probability_best_credit = np.zeros(len(algorithms), dtype=float)

        for iteration in range(BOOTSTRAP_ITERATIONS):
            sampled_rows = rng.integers(0, len(evaluators), size=len(evaluators))
            means = matrix[sampled_rows, :].mean(axis=0)
            bootstrap_scores[iteration, :] = means

            ranks = pd.Series(means).rank(
                method="average", ascending=False
            ).to_numpy()
            bootstrap_ranks[iteration, :] = ranks

            best = np.max(means)
            tied_best = np.flatnonzero(np.isclose(means, best, atol=1e-12, rtol=0))
            probability_best_credit[tied_best] += 1 / len(tied_best)

        result[f"score_ci95_low_{scenario_name}"] = np.quantile(
            bootstrap_scores, 0.025, axis=0
        )
        result[f"score_ci95_high_{scenario_name}"] = np.quantile(
            bootstrap_scores, 0.975, axis=0
        )
        result[f"bootstrap_mean_rank_{scenario_name}"] = bootstrap_ranks.mean(axis=0)
        result[f"bootstrap_median_rank_{scenario_name}"] = np.median(
            bootstrap_ranks, axis=0
        )
        result[f"probability_best_{scenario_name}"] = (
            probability_best_credit / BOOTSTRAP_ITERATIONS
        )

    return result.sort_values(
        f"rank_{PRIMARY_DECISION_SCENARIO}"
    ).reset_index(drop=True)


# =============================================================================
# 7. PRIMARY INFERENTIAL ANALYSIS
# =============================================================================


def metric_wide(
    evaluator_algorithm: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Create a consistently ordered evaluator × algorithm matrix."""
    return evaluator_algorithm.pivot(
        index="evaluator_id",
        columns="algorithm",
        values=value_column,
    ).sort_index().sort_index(axis=1)


def run_friedman(
    wide: pd.DataFrame,
    metric_name: str,
    metric_label: str,
    family: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run a Friedman test and return the result plus complete-case matrix."""
    complete = wide.dropna(axis=0, how="any")
    n_subjects = len(complete)
    n_conditions = complete.shape[1]

    if n_subjects < 2:
        return (
            {
                "family": family,
                "metric": metric_name,
                "metric_label": metric_label,
                "test": "Friedman",
                "n_subjects": n_subjects,
                "n_conditions": n_conditions,
                "statistic": np.nan,
                "df": n_conditions - 1,
                "kendalls_w": np.nan,
                "p_unc": np.nan,
            },
            complete,
        )

    test = friedmanchisquare(
        *[complete[column].to_numpy(dtype=float) for column in complete.columns]
    )
    statistic = float(test.statistic)
    p_value = float(test.pvalue)
    kendalls_w = statistic / (n_subjects * (n_conditions - 1))

    return (
        {
            "family": family,
            "metric": metric_name,
            "metric_label": metric_label,
            "test": "Friedman",
            "n_subjects": n_subjects,
            "n_conditions": n_conditions,
            "statistic": statistic,
            "df": n_conditions - 1,
            "kendalls_w": kendalls_w,
            "p_unc": p_value,
        },
        complete,
    )


def run_wilcoxon_posthoc(
    wide_complete: pd.DataFrame,
    metric_name: str,
    family: str,
) -> pd.DataFrame:
    """Run every paired Wilcoxon comparison and Holm-correct within metric."""
    rows = []

    for algorithm_a, algorithm_b in combinations(wide_complete.columns, 2):
        paired = wide_complete[[algorithm_a, algorithm_b]].dropna()
        values_a = np.round(paired[algorithm_a].to_numpy(float), 10)
        values_b = np.round(paired[algorithm_b].to_numpy(float), 10)
        differences = np.round(values_a - values_b, 10)
        n_nonzero = int(np.count_nonzero(differences))

        if n_nonzero == 0:
            w_value = 0.0
            p_value = 1.0
            rbc = 0.0
            cles = 0.5
        else:
            test = wilcoxon(
                values_a,
                values_b,
                alternative="two-sided",
                correction=False,
                zero_method="wilcox",
                method="auto",
            )
            w_value = float(test.statistic)
            p_value = float(test.pvalue)

            nonzero_differences = differences[differences != 0]
            signed_ranks = rankdata(
                np.abs(nonzero_differences),
                method="average",
            )
            positive_rank_sum = float(
                signed_ranks[nonzero_differences > 0].sum()
            )
            negative_rank_sum = float(
                signed_ranks[nonzero_differences < 0].sum()
            )
            total_rank_sum = positive_rank_sum + negative_rank_sum
            rbc = (
                (positive_rank_sum - negative_rank_sum) / total_rank_sum
                if total_rank_sum > 0
                else 0.0
            )

            # Paired common-language effect size: the proportion of paired
            # differences favouring A, with ties contributing one half.
            cles = float(
                (np.sum(differences > 0) + 0.5 * np.sum(differences == 0))
                / len(differences)
            )

        rows.append(
            {
                "family": family,
                "metric": metric_name,
                "algorithm_a": algorithm_a,
                "algorithm_b": algorithm_b,
                "w_statistic": w_value,
                "p_unc": p_value,
                "rank_biserial_correlation": rbc,
                "cles": cles,
                "mean_difference_a_minus_b": float(np.mean(differences)),
                "median_difference_a_minus_b": float(np.median(differences)),
                "n_pairs": len(paired),
                "n_nonzero": n_nonzero,
            }
        )

    posthoc = pd.DataFrame(rows)
    reject, corrected = holm_correction(
        posthoc["p_unc"].to_numpy(),
        alpha=ALPHA,
    )
    posthoc["p_holm_within_metric"] = corrected
    posthoc["significant"] = reject

    return posthoc.sort_values(
        ["p_holm_within_metric", "p_unc", "algorithm_a", "algorithm_b"]
    ).reset_index(drop=True)


def run_primary_analysis(
    evaluator_algorithm: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the four primary omnibus tests and gated pairwise tests."""
    omnibus_rows = []
    complete_matrices: dict[str, pd.DataFrame] = {}

    for metric_name, specification in PRIMARY_METRICS.items():
        wide = metric_wide(evaluator_algorithm, specification["column"])
        result, complete = run_friedman(
            wide,
            metric_name,
            specification["label"],
            family="primary",
        )
        omnibus_rows.append(result)
        complete_matrices[metric_name] = complete

    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["p_holm_across_primary_metrics"] = np.nan
    omnibus["significant"] = False
    valid = omnibus["p_unc"].notna()
    reject, corrected = holm_correction(
        omnibus.loc[valid, "p_unc"].to_numpy(),
        alpha=ALPHA,
    )
    omnibus.loc[valid, "p_holm_across_primary_metrics"] = corrected
    omnibus.loc[valid, "significant"] = reject
    omnibus["significant"] = omnibus["significant"].astype(bool)

    posthoc_frames = []
    for row in omnibus.itertuples(index=False):
        if row.significant:
            posthoc_frames.append(
                run_wilcoxon_posthoc(
                    complete_matrices[row.metric],
                    metric_name=row.metric,
                    family="primary",
                )
            )

    posthoc = (
        pd.concat(posthoc_frames, ignore_index=True)
        if posthoc_frames
        else pd.DataFrame(
            columns=[
                "family",
                "metric",
                "algorithm_a",
                "algorithm_b",
                "w_statistic",
                "p_unc",
                "p_holm_within_metric",
                "rank_biserial_correlation",
                "cles",
                "mean_difference_a_minus_b",
                "median_difference_a_minus_b",
                "n_pairs",
                "n_nonzero",
                "significant",
            ]
        )
    )

    return omnibus, posthoc


# =============================================================================
# 8. EXPLORATORY SERENDIPITY AND ANY-HIT SENSITIVITY
# =============================================================================


def run_serendipity_analysis(
    evaluator_algorithm: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one exploratory Friedman test for the serendipity composite."""
    wide = metric_wide(evaluator_algorithm, "mean_serendipity_score")
    result, complete = run_friedman(
        wide,
        metric_name="serendipity_score",
        metric_label="Exploratory serendipity",
        family="exploratory",
    )
    omnibus = pd.DataFrame([result])
    omnibus["significant"] = omnibus["p_unc"] < ALPHA

    posthoc = (
        run_wilcoxon_posthoc(
            complete,
            metric_name="serendipity_score",
            family="exploratory",
        )
        if bool(omnibus["significant"].iloc[0])
        else pd.DataFrame()
    )
    return omnibus, posthoc


def run_any_hit_sensitivity(
    evaluator_algorithm: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run exploratory Cochran's Q on the binary any-hit outcome.

    This is retained as a sensitivity analysis because it discards the
    difference between one hit and several hits. precision@5 is the primary
    top-5 outcome. Pairwise McNemar p-values are calculated with the exact
    binomial test when the omnibus result is significant.
    """
    wide = metric_wide(evaluator_algorithm, "any_top5_hit").dropna()
    matrix = wide.to_numpy(dtype=int)
    n_subjects, n_conditions = matrix.shape

    column_totals = matrix.sum(axis=0)
    row_totals = matrix.sum(axis=1)
    grand_total = matrix.sum()
    denominator = n_conditions * grand_total - np.sum(row_totals**2)

    if denominator == 0:
        q_statistic = 0.0
        p_value = 1.0
    else:
        q_statistic = float(
            (n_conditions - 1)
            * (
                n_conditions * np.sum(column_totals**2)
                - grand_total**2
            )
            / denominator
        )
        p_value = float(chi2.sf(q_statistic, n_conditions - 1))

    omnibus = pd.DataFrame(
        [
            {
                "family": "exploratory_sensitivity",
                "metric": "any_top5_hit",
                "metric_label": "Any top-5 hit",
                "test": "Cochran's Q",
                "n_subjects": n_subjects,
                "n_conditions": n_conditions,
                "statistic": q_statistic,
                "df": n_conditions - 1,
                "kendalls_w": np.nan,
                "p_unc": p_value,
                "significant": p_value < ALPHA,
            }
        ]
    )

    if not bool(omnibus["significant"].iloc[0]):
        return omnibus, pd.DataFrame()

    rows = []
    for algorithm_a, algorithm_b in combinations(wide.columns, 2):
        values_a = wide[algorithm_a]
        values_b = wide[algorithm_b]

        a_hit_b_miss = int(((values_a == 1) & (values_b == 0)).sum())
        a_miss_b_hit = int(((values_a == 0) & (values_b == 1)).sum())
        discordant = a_hit_b_miss + a_miss_b_hit

        if discordant == 0:
            statistic = 0.0
            pair_p = 1.0
        else:
            statistic = float(min(a_hit_b_miss, a_miss_b_hit))
            pair_p = float(
                binomtest(
                    min(a_hit_b_miss, a_miss_b_hit),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )

        rows.append(
            {
                "family": "exploratory_sensitivity",
                "metric": "any_top5_hit",
                "algorithm_a": algorithm_a,
                "algorithm_b": algorithm_b,
                "statistic": statistic,
                "p_unc": pair_p,
                "a_hit_b_miss": a_hit_b_miss,
                "a_miss_b_hit": a_miss_b_hit,
                "n_discordant": discordant,
                "exact": True,
            }
        )

    posthoc = pd.DataFrame(rows)
    reject, corrected = holm_correction(
        posthoc["p_unc"].to_numpy(),
        alpha=ALPHA,
    )
    posthoc["p_holm_within_metric"] = corrected
    posthoc["significant"] = reject
    return omnibus, posthoc.sort_values("p_holm_within_metric")


# =============================================================================
# 9. CONSERVATIVE POWER APPROXIMATION
# =============================================================================


def run_power_analysis(n_algorithms: int) -> pd.DataFrame:
    """
    Calculate a conservative repeated-measures ANOVA planning approximation.

    This is not an exact Friedman-test power function. The result must therefore
    be described as an ANOVA-based approximation for planning a future
    confirmatory study.
    """
    n_primary_tests = len(PRIMARY_METRICS)
    adjusted_alpha = ALPHA / n_primary_tests
    f_squared = POWER_COHENS_F**2
    eta_squared = f_squared / (1 + f_squared)
    minimum_epsilon = 1 / (n_algorithms - 1)

    if not minimum_epsilon <= POWER_EPSILON <= 1:
        raise ValueError(
            f"POWER_EPSILON must lie between {minimum_epsilon:.4f} and 1."
        )

    def achieved_power_for_n(sample_size: float) -> float:
        f_squared_internal = eta_squared / (1 - eta_squared)
        noncentrality = (
            f_squared_internal
            * sample_size
            * n_algorithms
            * POWER_EPSILON
            / (1 - POWER_AVERAGE_CORRELATION)
        )
        df_numerator = (n_algorithms - 1) * POWER_EPSILON
        df_denominator = (sample_size - 1) * df_numerator
        critical_value = f_distribution.ppf(
            1 - adjusted_alpha,
            df_numerator,
            df_denominator,
        )
        return float(
            ncf.sf(
                critical_value,
                df_numerator,
                df_denominator,
                noncentrality,
            )
        )

    calculated_n = brentq(
        lambda sample_size: achieved_power_for_n(sample_size) - POWER_TARGET,
        2.01,
        100000.0,
    )
    complete_n = math.ceil(calculated_n)
    recruitment_n = math.ceil(complete_n / (1 - POWER_DROPOUT_RATE))
    achieved_power = achieved_power_for_n(complete_n)

    return pd.DataFrame(
        [
            {
                "method": "Repeated-measures ANOVA approximation",
                "cohens_f": POWER_COHENS_F,
                "cohens_f_squared": f_squared,
                "eta_squared": eta_squared,
                "repeated_measurements": n_algorithms,
                "target_power": POWER_TARGET,
                "family_wise_alpha": ALPHA,
                "number_primary_omnibus_tests": n_primary_tests,
                "planning_alpha": adjusted_alpha,
                "average_correlation": POWER_AVERAGE_CORRELATION,
                "epsilon": POWER_EPSILON,
                "minimum_possible_epsilon": minimum_epsilon,
                "calculated_complete_n": calculated_n,
                "required_complete_n": complete_n,
                "dropout_rate": POWER_DROPOUT_RATE,
                "recruitment_target": recruitment_n,
                "achieved_power_after_rounding": achieved_power,
            }
        ]
    )


# =============================================================================
# 10. LATEX TABLE EXPORTS USING PANDAS
# =============================================================================


def export_latex(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    longtable: bool = False,
    column_format: str | None = None,
) -> None:
    """Export a documentation-ready LaTeX table using pandas.to_latex."""
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(
        index=False,
        escape=True,
        na_rep="--",
        caption=caption,
        label=label,
        position=None if longtable else "htbp",
        longtable=longtable,
        column_format=column_format,
    )
    path.write_text(latex, encoding="utf-8")


def make_latex_tables(
    tables_dir: Path,
    algorithm_summary: pd.DataFrame,
    primary_omnibus: pd.DataFrame,
    primary_posthoc: pd.DataFrame,
    preference_sources: pd.DataFrame,
    novelty_distribution: pd.DataFrame,
    overall: pd.DataFrame,
    ranking: pd.DataFrame,
    serendipity_omnibus: pd.DataFrame,
    any_hit_omnibus: pd.DataFrame,
    power: pd.DataFrame,
) -> list[Path]:
    """Build compact display copies and export every report table."""
    created: list[Path] = []

    # Algorithm descriptive table.
    rows = []
    for row in algorithm_summary.itertuples(index=False):
        rows.append(
            {
                "Algorithm": display_algorithm(row.algorithm),
                "Relevance M (SD)": (
                    f"{row.mean_relevance:.2f} ({row.sd_relevance:.2f})"
                ),
                "Preference M (SD)": (
                    f"{row.mean_preference_score:.2f} "
                    f"({row.sd_preference_score:.2f})"
                ),
                "Novelty M (SD)": (
                    f"{row.mean_novelty_score:.2f} "
                    f"({row.sd_novelty_score:.2f})"
                ),
                "Precision@5 M (SD)": (
                    f"{row.mean_precision_at_5:.2f} "
                    f"({row.sd_precision_at_5:.2f})"
                ),
                "Any hit n/N (%)": (
                    f"{int(row.evaluators_with_any_top5_hit)}/"
                    f"{int(row.n_evaluators)} "
                    f"({100 * row.proportion_evaluators_with_any_top5_hit:.1f})"
                ),
                "Serendipity M (SD)": (
                    f"{row.mean_serendipity_score:.3f} "
                    f"({row.sd_serendipity_score:.3f})"
                ),
            }
        )
    display = pd.DataFrame(rows)
    path = tables_dir / "table_algorithm_descriptives.tex"
    export_latex(
        display,
        path,
        caption=(
            "Evaluator-level descriptive statistics by recommendation algorithm."
        ),
        label="tab:algorithm-descriptives",
    )
    created.append(path)

    # Primary omnibus table.
    display = primary_omnibus.copy()
    display["Outcome"] = display["metric_label"]
    display["Test"] = display["test"]
    display["Statistic"] = display["statistic"].map(lambda x: format_number(x, 2))
    display["df"] = display["df"].map(lambda x: format_number(x, 0))
    display["Kendall's W"] = display["kendalls_w"].map(
        lambda x: format_number(x, 3)
    )
    display["p"] = display["p_unc"].map(format_p)
    display["Holm p"] = display["p_holm_across_primary_metrics"].map(format_p)
    display["Significant"] = display["significant"].map({True: "Yes", False: "No"})
    display = display[
        ["Outcome", "Test", "Statistic", "df", "Kendall's W", "p", "Holm p", "Significant"]
    ]
    path = tables_dir / "table_primary_omnibus.tex"
    export_latex(
        display,
        path,
        caption=(
            "Primary repeated-measures omnibus tests with Holm correction "
            "across the four outcomes."
        ),
        label="tab:primary-omnibus",
    )
    created.append(path)

    # Pairwise tables are exported per metric only when the gated post-hoc
    # procedure actually produced comparisons.
    if not primary_posthoc.empty:
        for metric_name, group in primary_posthoc.groupby("metric", observed=True):
            display = group.copy()
            display["A"] = display["algorithm_a"].map(display_algorithm)
            display["B"] = display["algorithm_b"].map(display_algorithm)
            display["W"] = display["w_statistic"].map(lambda x: format_number(x, 1))
            display["p"] = display["p_unc"].map(format_p)
            display["Holm p"] = display["p_holm_within_metric"].map(format_p)
            display["RBC"] = display["rank_biserial_correlation"].map(
                lambda x: format_number(x, 3)
            )
            display["CLES"] = display["cles"].map(lambda x: format_number(x, 3))
            display["Mean diff."] = display[
                "mean_difference_a_minus_b"
            ].map(lambda x: format_number(x, 3))
            display["n non-zero"] = display["n_nonzero"].astype(int)
            display["Significant"] = display["significant"].map(
                {True: "Yes", False: "No"}
            )
            display = display[
                [
                    "A",
                    "B",
                    "W",
                    "p",
                    "Holm p",
                    "RBC",
                    "CLES",
                    "Mean diff.",
                    "n non-zero",
                    "Significant",
                ]
            ]
            path = tables_dir / f"table_posthoc_{metric_name}.tex"
            export_latex(
                display,
                path,
                caption=(
                    f"Pairwise Wilcoxon signed-rank comparisons for "
                    f"{PRIMARY_METRICS[metric_name]['label']}."
                ),
                label=f"tab:posthoc-{metric_name.replace('_', '-')}",
                longtable=True,
            )
            created.append(path)

    # Preference source composition.
    display = preference_sources.copy()
    display["Algorithm"] = display["algorithm"].map(display_algorithm)
    display["Rating n"] = display["rating"].astype(int)
    display["Watch likelihood n"] = display["watch_likelihood"].astype(int)
    display["Rating %"] = (100 * display["rating_share"]).map(
        lambda x: f"{x:.1f}"
    )
    display["Watch likelihood %"] = (
        100 * display["watch_likelihood_share"]
    ).map(lambda x: f"{x:.1f}")
    display = display[
        ["Algorithm", "Rating n", "Watch likelihood n", "Rating %", "Watch likelihood %"]
    ]
    path = tables_dir / "table_preference_source.tex"
    export_latex(
        display,
        path,
        caption="Composition of the combined preference score by algorithm.",
        label="tab:preference-source",
    )
    created.append(path)

    # Novelty categories.
    display = novelty_distribution.copy()
    display["Algorithm"] = display["algorithm"].map(display_algorithm)
    selected = ["Algorithm"]
    for label in NOVELTY_LABELS.values():
        count_name = f"{label.replace('_', ' ').title()} n"
        share_name = f"{label.replace('_', ' ').title()} %"
        display[count_name] = display[f"{label}_count"].astype(int)
        display[share_name] = (100 * display[f"{label}_share"]).map(
            lambda x: f"{x:.1f}"
        )
        selected.extend([count_name, share_name])
    display = display[selected]
    path = tables_dir / "table_novelty_distribution.tex"
    export_latex(
        display,
        path,
        caption="Familiarity-derived novelty categories by algorithm.",
        label="tab:novelty-distribution",
    )
    created.append(path)

    # Overall ratings.
    if not overall.empty:
        display = overall.copy()
        display["Construct"] = display["construct"].str.replace("_", " ").str.title()
        display["n"] = display["n"].astype(int)
        for column, label in (
            ("mean", "Mean"),
            ("median", "Median"),
            ("sd", "SD"),
        ):
            display[label] = display[column].map(lambda x: format_number(x, 2))
        display = display[["Construct", "n", "Mean", "Median", "SD"]]
        path = tables_dir / "table_overall_ratings.tex"
        export_latex(
            display,
            path,
            caption="Descriptive statistics for session-level overall ratings.",
            label="tab:overall-ratings",
        )
        created.append(path)

    # Decision ranking with uncertainty for the primary scenario.
    scenario = PRIMARY_DECISION_SCENARIO
    display = ranking.copy()
    display["Algorithm"] = display["algorithm"].map(display_algorithm)
    display["Score"] = display[f"score_{scenario}"].map(
        lambda x: format_number(x, 3)
    )
    display["Rank"] = display[f"rank_{scenario}"].map(
        lambda x: format_number(x, 1)
    )
    columns = ["Algorithm", "Score", "Rank"]
    if f"score_ci95_low_{scenario}" in display.columns:
        display["Bootstrap 95% CI"] = display.apply(
            lambda row: (
                f"[{row[f'score_ci95_low_{scenario}']:.3f}, "
                f"{row[f'score_ci95_high_{scenario}']:.3f}]"
            ),
            axis=1,
        )
        display["Probability best"] = display[
            f"probability_best_{scenario}"
        ].map(lambda x: f"{x:.3f}")
        columns.extend(["Bootstrap 95% CI", "Probability best"])
    display = display[columns]
    path = tables_dir / "table_decision_ranking.tex"
    export_latex(
        display,
        path,
        caption=(
            f"Descriptive {scenario.replace('_', ' ')} decision ranking. "
            "This is not an inferential test."
        ),
        label="tab:decision-ranking",
    )
    created.append(path)

    # Exploratory omnibus table.
    exploratory = pd.concat(
        [serendipity_omnibus, any_hit_omnibus],
        ignore_index=True,
        sort=False,
    )
    display = exploratory.copy()
    display["Outcome"] = display["metric_label"]
    display["Test"] = display["test"]
    display["Statistic"] = display["statistic"].map(lambda x: format_number(x, 2))
    display["df"] = display["df"].map(lambda x: format_number(x, 0))
    display["Kendall's W"] = display["kendalls_w"].map(
        lambda x: format_number(x, 3)
    )
    display["p"] = display["p_unc"].map(format_p)
    display["Significant"] = display["significant"].map({True: "Yes", False: "No"})
    display = display[
        ["Outcome", "Test", "Statistic", "df", "Kendall's W", "p", "Significant"]
    ]
    path = tables_dir / "table_exploratory_omnibus.tex"
    export_latex(
        display,
        path,
        caption="Exploratory serendipity and any-hit sensitivity analyses.",
        label="tab:exploratory-omnibus",
    )
    created.append(path)

    if not power.empty:
        row = power.iloc[0]
        display = pd.DataFrame(
            [
                {
                    "Cohen's f": f"{row['cohens_f']:.2f}",
                    "Measurements": int(row["repeated_measurements"]),
                    "Target power": f"{row['target_power']:.2f}",
                    "Planning alpha": f"{row['planning_alpha']:.4f}",
                    "Correlation": f"{row['average_correlation']:.2f}",
                    "Epsilon": f"{row['epsilon']:.2f}",
                    "Complete N": int(row["required_complete_n"]),
                    "Recruitment target": int(row["recruitment_target"]),
                }
            ]
        )
        path = tables_dir / "table_power_analysis.tex"
        export_latex(
            display,
            path,
            caption=(
                "Conservative repeated-measures ANOVA power approximation "
                "for planning a future confirmatory study."
            ),
            label="tab:power-analysis",
        )
        created.append(path)

    # Convenience file that can be included once from the main document.
    all_tables_path = tables_dir / "all_tables.tex"
    all_tables_path.write_text(
        "% Generated by evaluation_pipeline_complete.py\n"
        "% Required packages: booktabs, longtable\n\n"
        + "\n\n".join(
            f"\\input{{tables/{path.name}}}" for path in created
        )
        + "\n",
        encoding="utf-8",
    )
    created.append(all_tables_path)

    return created


# =============================================================================
# 11. TEXT REPORT, VALIDATION REPORT, AND MANIFEST
# =============================================================================


def build_validation_report(
    per_movie: pd.DataFrame,
    evaluator_algorithm: pd.DataFrame,
    warnings: list[str],
) -> str:
    """Create a human-readable audit trail for the processed dataset."""
    evaluators = per_movie["evaluator_id"].nunique()
    algorithms = sorted(per_movie["algorithm"].unique())
    unique_responses = per_movie[["evaluator_id", "movie_id"]].drop_duplicates()
    shared_unique = per_movie.loc[
        per_movie["is_shared_recommendation"] == 1,
        ["evaluator_id", "movie_id"],
    ].drop_duplicates()
    cell_sizes = per_movie.groupby(
        ["evaluator_id", "algorithm"], observed=True
    ).size()

    lines = [
        "EVALUATION PIPELINE VALIDATION REPORT",
        "=" * 78,
        "",
        f"Evaluators: {evaluators}",
        f"Algorithms ({len(algorithms)}): {', '.join(algorithms)}",
        f"Unique evaluator × movie responses: {len(unique_responses)}",
        f"Algorithm-attributed recommendation rows: {len(per_movie)}",
        f"Shared evaluator × movie responses: {len(shared_unique)}",
        f"Evaluator × algorithm rows: {len(evaluator_algorithm)}",
        (
            "Recommendations per evaluator × algorithm cell: "
            f"min={cell_sizes.min()}, max={cell_sizes.max()}, "
            f"mean={cell_sizes.mean():.2f}"
        ),
        "",
        "Interpretation notes",
        "-" * 78,
        "- Inferential tests use evaluator-level aggregates.",
        "- CSV analytical values are not rounded before testing.",
        "- Algorithm-level SDs describe between-evaluator variation.",
        "- precision@5 is the primary top-5 outcome.",
        "- any-top-5-hit and serendipity analyses are exploratory.",
        "- Shared movies remain attributed to every algorithm that produced them.",
        "- Decision rankings are descriptive and include bootstrap uncertainty.",
        "",
        "Warnings",
        "-" * 78,
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def build_analysis_report(
    primary_omnibus: pd.DataFrame,
    primary_posthoc: pd.DataFrame,
    serendipity_omnibus: pd.DataFrame,
    any_hit_omnibus: pd.DataFrame,
    power: pd.DataFrame,
) -> str:
    """Write a concise result summary for checking and thesis drafting."""
    lines = [
        "STATISTICAL ANALYSIS REPORT",
        "=" * 78,
        "",
        "Primary omnibus family",
        "-" * 78,
    ]

    for row in primary_omnibus.itertuples(index=False):
        lines.append(
            f"{row.metric_label}: Friedman chi-square({int(row.df)}) = "
            f"{row.statistic:.4f}, p = {row.p_unc:.6f}, "
            f"Holm p = {row.p_holm_across_primary_metrics:.6f}, "
            f"Kendall's W = {row.kendalls_w:.4f}, "
            f"significant = {bool(row.significant)}"
        )

    lines.extend(["", "Gated post-hoc comparisons", "-" * 78])
    if primary_posthoc.empty:
        lines.append("No primary post-hoc comparisons were run.")
    else:
        for metric_name, group in primary_posthoc.groupby("metric", observed=True):
            n_significant = int(group["significant"].sum())
            lines.append(
                f"{metric_name}: {n_significant} of {len(group)} pairs "
                "significant after within-metric Holm correction."
            )

    ser = serendipity_omnibus.iloc[0]
    any_hit = any_hit_omnibus.iloc[0]
    lines.extend(
        [
            "",
            "Exploratory analyses",
            "-" * 78,
            (
                "Serendipity: Friedman chi-square"
                f"({int(ser['df'])}) = {ser['statistic']:.4f}, "
                f"p = {ser['p_unc']:.6f}, "
                f"Kendall's W = {ser['kendalls_w']:.4f}."
            ),
            (
                "Any top-5 hit: Cochran's Q"
                f"({int(any_hit['df'])}) = {any_hit['statistic']:.4f}, "
                f"p = {any_hit['p_unc']:.6f}."
            ),
        ]
    )

    if not power.empty:
        row = power.iloc[0]
        lines.extend(
            [
                "",
                "Power-planning approximation",
                "-" * 78,
                (
                    f"Required complete evaluators: "
                    f"{int(row['required_complete_n'])}"
                ),
                (
                    f"Recruitment target with "
                    f"{100 * row['dropout_rate']:.0f}% allowance: "
                    f"{int(row['recruitment_target'])}"
                ),
                (
                    "This is a conservative repeated-measures ANOVA "
                    "approximation, not an exact Friedman-test calculation."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Software",
            "-" * 78,
            f"Python: {sys.version.split()[0]}",
            f"pandas: {package_version('pandas')}",
            f"numpy: {package_version('numpy')}",
            f"scipy: {package_version('scipy')}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_manifest(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Create an index describing every generated file."""
    return pd.DataFrame(records)


# =============================================================================
# 12. MAIN PIPELINE
# =============================================================================


def main() -> None:
    args = parse_arguments()
    validate_configuration()

    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "evaluation_outputs"
    )
    csv_dir = output_dir / "csv"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"

    for directory in (csv_dir, tables_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    manifest_records: list[dict[str, Any]] = []

    print("=" * 78)
    print("COMPLETE RECOMMENDATION-EVALUATION PIPELINE")
    print("=" * 78)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # -------------------------------------------------------------------------
    # Stage 1: load either JSON source files or an existing per-movie CSV.
    # -------------------------------------------------------------------------
    if args.resume_per_movie is not None:
        print("Stage 1: loading existing recommendation-level CSV")
        per_movie, overall_df, mapping_df = load_resume_csv(
            args.resume_per_movie.resolve(),
            args.overall_csv.resolve() if args.overall_csv else None,
        )
    else:
        print("Stage 1: loading and validating JSON evaluation files")
        evaluations = load_json_evaluations(input_dir, warnings)
        if not evaluations:
            raise FileNotFoundError(
                f"No JSON evaluation files found in {input_dir}."
            )
        per_movie, overall_df, mapping_df = json_to_dataframes(
            evaluations, warnings
        )

    per_movie = validate_and_enrich_recommendations(per_movie, warnings)

    path = csv_dir / "01_recommendation_level.csv"
    write_csv(per_movie, path)
    manifest_records.append(
        {
            "file": str(path.relative_to(output_dir)),
            "type": "CSV",
            "rows": len(per_movie),
            "description": (
                "One row per evaluator × movie × algorithm, including raw "
                "responses, normalized components, overlap flags, and "
                "serendipity."
            ),
        }
    )

    if not mapping_df.empty:
        path = csv_dir / "02_evaluator_mapping.csv"
        write_csv(mapping_df, path)
        manifest_records.append(
            {
                "file": str(path.relative_to(output_dir)),
                "type": "CSV",
                "rows": len(mapping_df),
                "description": "Anonymous evaluator ID to source-file mapping.",
            }
        )

    if not overall_df.empty:
        path = csv_dir / "03_overall_ratings_raw.csv"
        write_csv(overall_df, path)
        manifest_records.append(
            {
                "file": str(path.relative_to(output_dir)),
                "type": "CSV",
                "rows": len(overall_df),
                "description": "One session-level overall-rating row per evaluator.",
            }
        )

    # -------------------------------------------------------------------------
    # Stage 2: aggregate to the repeated-measures unit.
    # -------------------------------------------------------------------------
    print("Stage 2: aggregating to evaluator × algorithm")
    evaluator_algorithm = aggregate_evaluator_algorithm(per_movie)
    path = csv_dir / "04_evaluator_algorithm.csv"
    write_csv(evaluator_algorithm, path)
    manifest_records.append(
        {
            "file": str(path.relative_to(output_dir)),
            "type": "CSV",
            "rows": len(evaluator_algorithm),
            "description": (
                "One row per evaluator × algorithm; main repeated-measures "
                "analysis dataset."
            ),
        }
    )

    analysis_long = build_analysis_long(evaluator_algorithm)
    path = csv_dir / "05_analysis_long.csv"
    write_csv(analysis_long, path)
    manifest_records.append(
        {
            "file": str(path.relative_to(output_dir)),
            "type": "CSV",
            "rows": len(analysis_long),
            "description": "Long-format evaluator × algorithm × metric dataset.",
        }
    )

    analysis_wide = build_analysis_wide(evaluator_algorithm, overall_df)
    path = csv_dir / "06_analysis_wide.csv"
    write_csv(analysis_wide, path)
    manifest_records.append(
        {
            "file": str(path.relative_to(output_dir)),
            "type": "CSV",
            "rows": len(analysis_wide),
            "description": "Wide-format dataset with one row per evaluator.",
        }
    )

    # -------------------------------------------------------------------------
    # Stage 3: descriptive summaries and diagnostics.
    # -------------------------------------------------------------------------
    print("Stage 3: creating descriptive summaries and diagnostics")
    algorithm_summary = summarize_algorithm_metrics(evaluator_algorithm)
    preference_sources = preference_source_summary(per_movie)
    novelty_distribution = novelty_distribution_summary(per_movie)
    overlap_by_algorithm, overlap_matrix = overlap_summaries(per_movie)
    overall = overall_summary(overall_df)
    ranking = decision_ranking(
        evaluator_algorithm,
        skip_bootstrap=args.skip_bootstrap,
    )

    descriptive_outputs = [
        (
            "07_algorithm_descriptives.csv",
            algorithm_summary,
            "Algorithm-level descriptive statistics across evaluator-level values.",
        ),
        (
            "08_preference_source_by_algorithm.csv",
            preference_sources,
            "Rating versus watch-likelihood composition by algorithm.",
        ),
        (
            "09_novelty_distribution_by_algorithm.csv",
            novelty_distribution,
            "Watched, heard-of, and unknown recommendation counts and shares.",
        ),
        (
            "10_overlap_by_algorithm.csv",
            overlap_by_algorithm,
            "Share of algorithm-attributed rows originating from shared movies.",
        ),
        (
            "11_algorithm_overlap_matrix.csv",
            overlap_matrix,
            "Counts of evaluator-specific movies shared by each algorithm pair.",
        ),
        (
            "12_overall_ratings_summary.csv",
            overall,
            "Descriptive session-level overall ratings.",
        ),
        (
            "13_decision_ranking.csv",
            ranking,
            "Descriptive quality/discovery rankings with bootstrap uncertainty.",
        ),
    ]

    for filename, dataframe, description in descriptive_outputs:
        path = csv_dir / filename
        write_csv(dataframe, path)
        manifest_records.append(
            {
                "file": str(path.relative_to(output_dir)),
                "type": "CSV",
                "rows": len(dataframe),
                "description": description,
            }
        )

    # -------------------------------------------------------------------------
    # Stage 4: inferential tests.
    # -------------------------------------------------------------------------
    print("Stage 4: running primary and exploratory statistical analyses")
    primary_omnibus, primary_posthoc = run_primary_analysis(evaluator_algorithm)
    serendipity_omnibus, serendipity_posthoc = run_serendipity_analysis(
        evaluator_algorithm
    )
    any_hit_omnibus, any_hit_posthoc = run_any_hit_sensitivity(
        evaluator_algorithm
    )

    statistical_outputs = [
        (
            "14_primary_omnibus_tests.csv",
            primary_omnibus,
            "Four primary Friedman tests with Holm correction across outcomes.",
        ),
        (
            "15_primary_posthoc_tests.csv",
            primary_posthoc,
            "Gated Wilcoxon pairwise tests with within-outcome Holm correction.",
        ),
        (
            "16_exploratory_serendipity_omnibus.csv",
            serendipity_omnibus,
            "Exploratory Friedman test for the serendipity composite.",
        ),
        (
            "17_exploratory_serendipity_posthoc.csv",
            serendipity_posthoc,
            "Exploratory serendipity pairwise tests, when omnibus significant.",
        ),
        (
            "18_any_hit_sensitivity_omnibus.csv",
            any_hit_omnibus,
            "Exploratory Cochran's Q sensitivity analysis for any top-5 hit.",
        ),
        (
            "19_any_hit_sensitivity_posthoc.csv",
            any_hit_posthoc,
            "Exploratory pairwise McNemar tests, when omnibus significant.",
        ),
    ]

    for filename, dataframe, description in statistical_outputs:
        path = csv_dir / filename
        write_csv(dataframe, path)
        manifest_records.append(
            {
                "file": str(path.relative_to(output_dir)),
                "type": "CSV",
                "rows": len(dataframe),
                "description": description,
            }
        )

    # -------------------------------------------------------------------------
    # Stage 5: power analysis for future planning.
    # -------------------------------------------------------------------------
    print("Stage 5: calculating power-planning approximation")
    power = (
        pd.DataFrame()
        if args.skip_power
        else run_power_analysis(per_movie["algorithm"].nunique())
    )
    if not power.empty:
        path = csv_dir / "20_power_analysis.csv"
        write_csv(power, path)
        manifest_records.append(
            {
                "file": str(path.relative_to(output_dir)),
                "type": "CSV",
                "rows": len(power),
                "description": (
                    "Conservative repeated-measures ANOVA sample-size "
                    "approximation for future planning."
                ),
            }
        )

    # -------------------------------------------------------------------------
    # Stage 6: LaTeX tables.
    # -------------------------------------------------------------------------
    print("Stage 6: exporting LaTeX tables with pandas.to_latex")
    latex_paths = make_latex_tables(
        tables_dir=tables_dir,
        algorithm_summary=algorithm_summary,
        primary_omnibus=primary_omnibus,
        primary_posthoc=primary_posthoc,
        preference_sources=preference_sources,
        novelty_distribution=novelty_distribution,
        overall=overall,
        ranking=ranking,
        serendipity_omnibus=serendipity_omnibus,
        any_hit_omnibus=any_hit_omnibus,
        power=power,
    )
    for path in latex_paths:
        manifest_records.append(
            {
                "file": str(path.relative_to(output_dir)),
                "type": "LaTeX",
                "rows": "",
                "description": (
                    "Documentation-ready table generated with pandas.to_latex."
                    if path.name != "all_tables.tex"
                    else "Convenience file that inputs all generated LaTeX tables."
                ),
            }
        )

    requirements_path = tables_dir / "latex_requirements.txt"
    requirements_path.write_text(
        "Required LaTeX packages:\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{longtable}\n\n"
        "Include every generated table with:\n"
        "\\input{tables/all_tables.tex}\n",
        encoding="utf-8",
    )
    manifest_records.append(
        {
            "file": str(requirements_path.relative_to(output_dir)),
            "type": "Text",
            "rows": "",
            "description": "Required packages and LaTeX include instruction.",
        }
    )

    # -------------------------------------------------------------------------
    # Stage 7: reports and manifest.
    # -------------------------------------------------------------------------
    print("Stage 7: writing reports and manifest")
    validation_report = build_validation_report(
        per_movie, evaluator_algorithm, warnings
    )
    validation_path = reports_dir / "validation_report.txt"
    validation_path.write_text(validation_report, encoding="utf-8")
    manifest_records.append(
        {
            "file": str(validation_path.relative_to(output_dir)),
            "type": "Text",
            "rows": "",
            "description": "Data validation, structure, overlap, and warning report.",
        }
    )

    analysis_report = build_analysis_report(
        primary_omnibus,
        primary_posthoc,
        serendipity_omnibus,
        any_hit_omnibus,
        power,
    )
    analysis_path = reports_dir / "analysis_report.txt"
    analysis_path.write_text(analysis_report, encoding="utf-8")
    manifest_records.append(
        {
            "file": str(analysis_path.relative_to(output_dir)),
            "type": "Text",
            "rows": "",
            "description": "Concise statistical results and software-version report.",
        }
    )

    manifest = build_manifest(manifest_records)
    manifest_path = output_dir / "output_manifest.csv"
    write_csv(manifest, manifest_path)

    print()
    print("Pipeline completed successfully.")
    print(f"Open the manifest for a file-by-file guide: {manifest_path}")
    print(
        "Primary inference uses relevance, preference, novelty, and "
        "precision@5. Serendipity and any-hit results are exploratory."
    )


if __name__ == "__main__":
    main()
