"""
Statistical comparison of recommendation algorithms.

Place this script in the same folder as:

    evaluation_results_per_movie.csv

Then run:

    python statistical_comparison.py

The script performs:

Continuous metrics:
    - relevance
    - preference_score
    - novelty_score

Tests:
    - Friedman omnibus test
    - Kendall's W effect size
    - Pairwise Wilcoxon signed-rank post-hoc tests
    - Matched-pairs rank-biserial correlation (RBC)
    - Holm correction across pairwise comparisons

Binary metric:
    - in_top5

Tests:
    - Cochran's Q omnibus test
    - Pairwise McNemar post-hoc tests
    - Holm correction across pairwise comparisons

Multiple testing:
    - Holm correction is first applied across the four omnibus tests.
    - Post-hoc tests are run only for metrics whose omnibus test remains
      significant after that across-metric correction.

Requirements:

    pip install pandas numpy pingouin scipy statsmodels
"""

import sys

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg

try:
    from statsmodels.stats.contingency_tables import mcnemar

    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

INPUT_FILENAME = "evaluation_results_per_movie.csv"
OUTPUT_FILENAME = "statistical_comparison_report.txt"

CONTINUOUS_METRICS = (
    "relevance",
    "preference_score",
    "novelty_score",
)

BINARY_METRIC = "in_top5"

ALPHA = 0.05

# Used both across omnibus tests and within post-hoc test families.
PADJUST_METHOD = "holm"


# -------------------------------------------------------------------
# Reporting
# -------------------------------------------------------------------

class Reporter:
    """Print output to the console and collect it for a text report."""

    def __init__(self):
        self.lines = []

    def write(self, text=""):
        print(text)
        self.lines.append(str(text))

    def section(self, title):
        self.write()
        self.write("=" * 78)
        self.write(title)
        self.write("=" * 78)

    def subsection(self, title):
        self.write()
        self.write("-" * 78)
        self.write(title)
        self.write("-" * 78)

    def save(self, path):
        path.write_text(
            "\n".join(self.lines),
            encoding="utf-8",
        )


# -------------------------------------------------------------------
# Data loading and validation
# -------------------------------------------------------------------

def load_data(folder):
    """Load and validate the per-movie evaluation CSV."""
    path = folder / INPUT_FILENAME

    if not path.exists():
        sys.exit(
            f"ERROR: could not find '{INPUT_FILENAME}' in:\n"
            f"  {folder}\n\n"
            "Place this script in the same folder as the CSV and rerun it."
        )

    df = pd.read_csv(path)

    required_columns = {
        "evaluator_id",
        "algorithm",
        "relevance",
        "preference_score",
        "novelty_score",
        "in_top5",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        sys.exit(
            "ERROR: input file is missing expected columns:\n"
            f"  {sorted(missing_columns)}"
        )

    # Ensure the outcome columns are numeric.
    numeric_columns = [
        *CONTINUOUS_METRICS,
        BINARY_METRIC,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    invalid_binary = df[
        df[BINARY_METRIC].notna()
        & ~df[BINARY_METRIC].isin([0, 1])
    ]

    if not invalid_binary.empty:
        sys.exit(
            f"ERROR: '{BINARY_METRIC}' contains values other than 0 or 1."
        )

    return df


# -------------------------------------------------------------------
# Subject-by-condition aggregation
# -------------------------------------------------------------------

def aggregate_to_subject_by_condition(df, reporter):
    """
    Collapse movies into one observation per evaluator and algorithm.

    Continuous metrics:
        Mean across the recommended movies.

    Binary in_top5:
        Maximum across the recommended movies, meaning 1 when at least
        one recommendation from the algorithm was in the evaluator's
        top-five selection.

    Returns:
        wide_frames:
            Dictionary mapping each metric to a subject-by-condition
            DataFrame.

        algorithms:
            Sorted list of algorithm names.
    """
    reporter.section("1. DATA AGGREGATION")

    n_evaluators = df["evaluator_id"].nunique()
    algorithms = sorted(df["algorithm"].dropna().unique())
    n_algorithms = len(algorithms)

    reporter.write(
        f"Input rows (per recommendation): {len(df)}"
    )
    reporter.write(
        f"Evaluators: {n_evaluators}"
    )
    reporter.write(
        f"Algorithms ({n_algorithms}): {', '.join(algorithms)}"
    )

    rows_per_cell = (
        df.groupby(["evaluator_id", "algorithm"])
        .size()
    )

    reporter.write(
        "Movies per (evaluator, algorithm) cell - "
        f"min: {rows_per_cell.min()}, "
        f"max: {rows_per_cell.max()}, "
        f"mean: {rows_per_cell.mean():.1f}"
    )

    if rows_per_cell.min() != rows_per_cell.max():
        reporter.write(
            "NOTE: the number of movies differs across some evaluator/"
            "algorithm cells. Means are therefore based on different "
            "numbers of observations."
        )

    wide_frames = {}

    # Continuous metrics: evaluator-level mean
    for metric in CONTINUOUS_METRICS:
        aggregated = (
            df.groupby(
                ["evaluator_id", "algorithm"],
                as_index=False,
            )[metric]
            .mean()
        )

        wide = aggregated.pivot(
            index="evaluator_id",
            columns="algorithm",
            values=metric,
        )

        # Apply a consistent algorithm-column order.
        wide = wide.reindex(columns=algorithms)

        n_missing_cells = int(
            wide.isna().sum().sum()
        )

        if n_missing_cells:
            reporter.write(
                f"WARNING: {metric} contains {n_missing_cells} missing "
                "evaluator-algorithm cells after aggregation. "
                "Evaluators with incomplete data will be removed "
                "listwise for that metric."
            )

        wide_frames[metric] = wide

    # Binary metric: whether any recommendation was a top-five hit
    aggregated_binary = (
        df.groupby(
            ["evaluator_id", "algorithm"],
            as_index=False,
        )[BINARY_METRIC]
        .max()
    )

    wide_binary = aggregated_binary.pivot(
        index="evaluator_id",
        columns="algorithm",
        values=BINARY_METRIC,
    )

    wide_binary = wide_binary.reindex(
        columns=algorithms
    )

    n_missing_binary = int(
        wide_binary.isna().sum().sum()
    )

    if n_missing_binary:
        reporter.write(
            f"WARNING: {BINARY_METRIC} contains "
            f"{n_missing_binary} missing evaluator-algorithm cells."
        )

    wide_frames[BINARY_METRIC] = wide_binary

    reporter.write()
    reporter.write(
        "Aggregation method: "
        f"MEAN for {', '.join(CONTINUOUS_METRICS)}; "
        f"ANY-hit (maximum) for {BINARY_METRIC}."
    )
    reporter.write(
        "Result: one subject-by-condition matrix per metric "
        f"({n_evaluators} evaluators × {n_algorithms} algorithms)."
    )

    return wide_frames, algorithms


# -------------------------------------------------------------------
# Friedman omnibus test
# -------------------------------------------------------------------

def run_friedman(wide, metric_name, reporter):
    """
    Run the Friedman test and report Kendall's W.

    The returned wide_complete DataFrame contains only evaluators with
    observations for every algorithm.
    """
    wide_complete = wide.dropna()

    n_original = len(wide)
    n_subjects = len(wide_complete)
    n_dropped = n_original - n_subjects
    n_conditions = wide_complete.shape[1]

    reporter.subsection(
        f"Friedman test: {metric_name}"
    )

    if n_subjects < 2:
        reporter.write(
            "ERROR: fewer than two complete evaluators remain. "
            "The Friedman test cannot be calculated."
        )

        return {
            "metric": metric_name,
            "test": "Friedman",
            "statistic": np.nan,
            "p_unc": np.nan,
            "kendalls_w": np.nan,
            "n_subjects": n_subjects,
            "wide_complete": wide_complete,
        }

    long_df = (
        wide_complete
        .reset_index()
        .melt(
            id_vars="evaluator_id",
            var_name="algorithm",
            value_name=metric_name,
        )
    )

    result = pg.friedman(
        data=long_df,
        dv=metric_name,
        within="algorithm",
        subject="evaluator_id",
    )

    chi_square = float(
        result["Q"].iloc[0]
    )

    degrees_of_freedom = int(
        result["ddof1"].iloc[0]
    )

    kendalls_w = float(
        result["W"].iloc[0]
    )

    p_value = float(
        result["p_unc"].iloc[0]
    )

    subject_line = (
        f"Subjects (evaluators) used: {n_subjects}"
    )

    if n_dropped:
        subject_line += (
            f" ({n_dropped} dropped for missing data)"
        )

    reporter.write(subject_line)
    reporter.write(
        f"Conditions (algorithms): {n_conditions}"
    )
    reporter.write(
        f"Chi-square (Q): {chi_square:.4f}"
    )
    reporter.write(
        f"Degrees of freedom: {degrees_of_freedom}"
    )
    reporter.write(
        f"Uncorrected p-value: {p_value:.6f}"
    )
    reporter.write(
        f"Kendall's W: {kendalls_w:.4f}"
    )
    reporter.write(
        interpret_kendalls_w(kendalls_w)
    )

    return {
        "metric": metric_name,
        "test": "Friedman",
        "statistic": chi_square,
        "p_unc": p_value,
        "kendalls_w": kendalls_w,
        "n_subjects": n_subjects,
        "wide_complete": wide_complete,
    }


def interpret_kendalls_w(value):
    """Return a cautious rule-of-thumb interpretation of Kendall's W."""
    if value < 0.10:
        label = "negligible"
    elif value < 0.30:
        label = "small"
    elif value < 0.50:
        label = "moderate"
    else:
        label = "large"

    return (
        f"  -> interpreted as a {label} effect "
        "(rule of thumb, not a strict cutoff)"
    )


# -------------------------------------------------------------------
# Cochran's Q omnibus test
# -------------------------------------------------------------------

def run_cochran(wide, reporter):
    """Run Cochran's Q test for the binary top-five outcome."""
    wide_complete = wide.dropna()

    n_original = len(wide)
    n_subjects = len(wide_complete)
    n_dropped = n_original - n_subjects

    reporter.subsection(
        f"Cochran's Q test: {BINARY_METRIC}"
    )

    if n_subjects < 2:
        reporter.write(
            "ERROR: fewer than two complete evaluators remain. "
            "Cochran's Q cannot be calculated."
        )

        return {
            "metric": BINARY_METRIC,
            "test": "Cochran's Q",
            "statistic": np.nan,
            "p_unc": np.nan,
            "kendalls_w": None,
            "n_subjects": n_subjects,
            "wide_complete": wide_complete,
        }

    long_df = (
        wide_complete
        .reset_index()
        .melt(
            id_vars="evaluator_id",
            var_name="algorithm",
            value_name=BINARY_METRIC,
        )
    )

    result = pg.cochran(
        data=long_df,
        dv=BINARY_METRIC,
        within="algorithm",
        subject="evaluator_id",
    )

    q_statistic = float(
        result["Q"].iloc[0]
    )

    degrees_of_freedom = int(
        result["dof"].iloc[0]
    )

    p_value = float(
        result["p_unc"].iloc[0]
    )

    subject_line = (
        f"Subjects (evaluators) used: {n_subjects}"
    )

    if n_dropped:
        subject_line += (
            f" ({n_dropped} dropped for missing data)"
        )

    reporter.write(subject_line)
    reporter.write(
        f"Conditions (algorithms): {wide_complete.shape[1]}"
    )
    reporter.write(
        f"Q statistic: {q_statistic:.4f}"
    )
    reporter.write(
        f"Degrees of freedom: {degrees_of_freedom}"
    )
    reporter.write(
        f"Uncorrected p-value: {p_value:.6f}"
    )

    return {
        "metric": BINARY_METRIC,
        "test": "Cochran's Q",
        "statistic": q_statistic,
        "p_unc": p_value,
        "kendalls_w": None,
        "n_subjects": n_subjects,
        "wide_complete": wide_complete,
    }


# -------------------------------------------------------------------
# Holm correction across omnibus tests
# -------------------------------------------------------------------

def correct_omnibus_across_metrics(
    omnibus_results,
    reporter,
):
    """
    Apply Holm correction across all omnibus tests.

    The corrected significance flag is added to each result dictionary.
    """
    reporter.section(
        "3. OMNIBUS TEST SUMMARY AND ACROSS-METRIC CORRECTION"
    )

    valid_results = [
        result
        for result in omnibus_results
        if np.isfinite(result["p_unc"])
    ]

    if not valid_results:
        reporter.write(
            "No valid omnibus p-values were available."
        )
        return pd.DataFrame()

    p_values = np.array([
        result["p_unc"]
        for result in valid_results
    ])

    reject, corrected_p_values = pg.multicomp(
        p_values,
        alpha=ALPHA,
        method=PADJUST_METHOD,
    )

    summary_rows = []

    for result, rejected, corrected_p in zip(
        valid_results,
        reject,
        corrected_p_values,
    ):
        result["p_corr_across_metrics"] = float(
            corrected_p
        )

        result["significant_after_correction"] = bool(
            rejected
        )

        summary_rows.append({
            "metric": result["metric"],
            "test": result["test"],
            "statistic": round(
                result["statistic"],
                4,
            ),
            "n_subjects": result["n_subjects"],
            "kendalls_w": (
                round(result["kendalls_w"], 4)
                if result["kendalls_w"] is not None
                else "n/a"
            ),
            "p_unc": round(
                result["p_unc"],
                6,
            ),
            (
                f"p_corr_{PADJUST_METHOD}_"
                "across_metrics"
            ): round(
                corrected_p,
                6,
            ),
            "significant": bool(rejected),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    reporter.write(
        f"{PADJUST_METHOD.capitalize()} correction applied across "
        f"{len(valid_results)} omnibus tests."
    )
    reporter.write(
        "Post-hoc comparisons will be run only for metrics whose "
        "omnibus test remains significant after this correction."
    )
    reporter.write()
    reporter.write(
        summary_df.to_string(
            index=False
        )
    )

    return summary_df


# -------------------------------------------------------------------
# Wilcoxon post-hoc tests
# -------------------------------------------------------------------

def get_result_value(result, possible_columns):
    """
    Return the first matching value from a Pingouin result DataFrame.

    This supports minor column-name differences between Pingouin
    versions, such as W-val versus W_val.
    """
    for column in possible_columns:
        if column in result.columns:
            return float(result[column].iloc[0])

    raise KeyError(
        "None of the expected columns were found: "
        f"{possible_columns}"
    )


def run_wilcoxon_posthoc(
    wide_complete,
    metric_name,
    reporter,
):
    """
    Run all pairwise Wilcoxon signed-rank tests.

    Effect sizes:
        RBC:
            Matched-pairs rank-biserial correlation.

        CLES:
            Common-language effect size.

    The sign of RBC follows A minus B:
        positive -> A generally scores higher;
        negative -> B generally scores higher.
    """
    reporter.subsection(
        f"Wilcoxon signed-rank post-hoc: "
        f"{metric_name} ({PADJUST_METHOD.capitalize()}-corrected)"
    )

    algorithms = list(
        wide_complete.columns
    )

    results = []

    for algorithm_a, algorithm_b in combinations(
        algorithms,
        2,
    ):
        paired_data = (
            wide_complete[
                [algorithm_a, algorithm_b]
            ]
            .dropna()
        )

        values_a = (
            paired_data[algorithm_a]
            .to_numpy(dtype=float)
        )

        values_b = (
            paired_data[algorithm_b]
            .to_numpy(dtype=float)
        )

        # Avoid tiny floating-point differences affecting tie handling.
        values_a = np.round(
            values_a,
            decimals=10,
        )

        values_b = np.round(
            values_b,
            decimals=10,
        )

        differences = np.round(
            values_a - values_b,
            decimals=10,
        )

        nonzero_differences = differences[
            differences != 0
        ]

        n_pairs = len(paired_data)
        n_nonzero = len(nonzero_differences)

        if n_nonzero == 0:
            # Every evaluator gave exactly the same score to both
            # algorithms.
            w_value = 0.0
            p_value = 1.0
            rank_biserial = 0.0
            common_language = 0.5

        else:
            result = pg.wilcoxon(
                values_a,
                values_b,
                alternative="two-sided",
                correction=False,
            )

            w_value = get_result_value(
                result,
                ["W-val", "W_val"],
            )

            p_value = get_result_value(
                result,
                ["p-val", "p_val"],
            )

            rank_biserial = get_result_value(
                result,
                ["RBC"],
            )

            common_language = get_result_value(
                result,
                ["CLES"],
            )

        results.append({
            "A": algorithm_a,
            "B": algorithm_b,
            "W_val": w_value,
            "p_unc": p_value,
            "RBC": rank_biserial,
            "CLES": common_language,
            "mean_difference": float(
                np.mean(differences)
            ),
            "median_difference": float(
                np.median(differences)
            ),
            "n_pairs": n_pairs,
            "n_nonzero": n_nonzero,
        })

    posthoc = pd.DataFrame(
        results
    )

    reject, corrected_p_values = pg.multicomp(
        posthoc["p_unc"].to_numpy(),
        alpha=ALPHA,
        method=PADJUST_METHOD,
    )

    posthoc["p_corr"] = corrected_p_values
    posthoc["significant"] = reject

    posthoc = posthoc.sort_values(
        by=[
            "p_corr",
            "p_unc",
            "A",
            "B",
        ]
    )

    display_columns = [
        "A",
        "B",
        "W_val",
        "p_unc",
        "p_corr",
        "RBC",
        "CLES",
        "mean_difference",
        "median_difference",
        "n_pairs",
        "n_nonzero",
        "significant",
    ]

    reporter.write(
        posthoc[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    significant_pairs = posthoc[
        posthoc["significant"]
    ]

    reporter.write(
        f"\nSignificant pairs at alpha={ALPHA} "
        f"({PADJUST_METHOD.capitalize()}-corrected): "
        f"{len(significant_pairs)} / {len(posthoc)}"
    )

    for _, row in significant_pairs.iterrows():
        direction = (
            f"{row['A']} higher"
            if row["RBC"] > 0
            else f"{row['B']} higher"
            if row["RBC"] < 0
            else "no directional difference"
        )

        reporter.write(
            f"  {row['A']} vs {row['B']}: "
            f"p-corr = {row['p_corr']:.4f}, "
            f"RBC = {row['RBC']:.4f} "
            f"({direction})"
        )

    reporter.write()
    reporter.write(
        "RBC is the matched-pairs rank-biserial correlation. "
        "Its sign is based on A minus B."
    )
    reporter.write(
        "CLES is the estimated probability that a randomly selected "
        "A score exceeds a randomly selected B score."
    )

    return posthoc


# -------------------------------------------------------------------
# McNemar post-hoc tests
# -------------------------------------------------------------------

def run_mcnemar_posthoc(
    wide_complete,
    reporter,
):
    """
    Run pairwise McNemar tests for the binary top-five outcome.
    """
    reporter.subsection(
        f"McNemar post-hoc: "
        f"{BINARY_METRIC} "
        f"({PADJUST_METHOD.capitalize()}-corrected)"
    )

    if not HAVE_STATSMODELS:
        reporter.write(
            "statsmodels is not installed, so McNemar post-hoc "
            "comparisons were skipped."
        )
        reporter.write(
            "Install it with:\n"
            "    pip install statsmodels"
        )

        return None

    algorithms = list(
        wide_complete.columns
    )

    results = []

    for algorithm_a, algorithm_b in combinations(
        algorithms,
        2,
    ):
        values_a = wide_complete[
            algorithm_a
        ]

        values_b = wide_complete[
            algorithm_b
        ]

        both_miss = int(
            ((values_a == 0) & (values_b == 0)).sum()
        )

        a_miss_b_hit = int(
            ((values_a == 0) & (values_b == 1)).sum()
        )

        a_hit_b_miss = int(
            ((values_a == 1) & (values_b == 0)).sum()
        )

        both_hit = int(
            ((values_a == 1) & (values_b == 1)).sum()
        )

        contingency_table = [
            [both_hit, a_hit_b_miss],
            [a_miss_b_hit, both_miss],
        ]

        n_discordant = (
            a_miss_b_hit
            + a_hit_b_miss
        )

        # Exact McNemar is preferable when the number of discordant
        # pairs is small.
        use_exact = n_discordant < 25

        result = mcnemar(
            contingency_table,
            exact=use_exact,
            correction=not use_exact,
        )

        results.append({
            "A": algorithm_a,
            "B": algorithm_b,
            "statistic": float(result.statistic),
            "p_unc": float(result.pvalue),
            "A_hit_B_miss": a_hit_b_miss,
            "A_miss_B_hit": a_miss_b_hit,
            "n_discordant": n_discordant,
            "exact": use_exact,
        })

    posthoc = pd.DataFrame(
        results
    )

    reject, corrected_p_values = pg.multicomp(
        posthoc["p_unc"].to_numpy(),
        alpha=ALPHA,
        method=PADJUST_METHOD,
    )

    posthoc["p_corr"] = corrected_p_values
    posthoc["significant"] = reject

    posthoc = posthoc.sort_values(
        by=[
            "p_corr",
            "p_unc",
            "A",
            "B",
        ]
    )

    reporter.write(
        posthoc.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    significant_pairs = posthoc[
        posthoc["significant"]
    ]

    reporter.write(
        f"\nSignificant pairs at alpha={ALPHA} "
        f"({PADJUST_METHOD.capitalize()}-corrected): "
        f"{len(significant_pairs)} / {len(posthoc)}"
    )

    for _, row in significant_pairs.iterrows():
        reporter.write(
            f"  {row['A']} vs {row['B']}: "
            f"p-corr = {row['p_corr']:.4f}"
        )

    return posthoc


# -------------------------------------------------------------------
# Descriptive rankings
# -------------------------------------------------------------------

def print_descriptive_ranking(
    wide_frames,
    reporter,
):
    """Print algorithm means for descriptive context."""
    reporter.section(
        "5. DESCRIPTIVE MEANS PER ALGORITHM"
    )

    for metric_name, wide in wide_frames.items():
        reporter.subsection(
            metric_name
        )

        means = (
            wide.mean()
            .sort_values(
                ascending=False
            )
        )

        means_df = (
            means
            .reset_index()
        )

        means_df.columns = [
            "algorithm",
            f"mean_{metric_name}",
        ]

        means_df.index = range(
            1,
            len(means_df) + 1,
        )

        means_df.index.name = "rank"

        reporter.write(
            means_df.to_string(
                float_format=lambda value: f"{value:.4f}",
            )
        )


# -------------------------------------------------------------------
# Main program
# -------------------------------------------------------------------

def main():
    folder = Path(__file__).resolve().parent
    reporter = Reporter()

    reporter.write(
        "STATISTICAL COMPARISON OF RECOMMENDATION ALGORITHMS"
    )
    reporter.write(
        "Friedman + Wilcoxon for continuous metrics"
    )
    reporter.write(
        "Cochran's Q + McNemar for in_top5"
    )
    reporter.write(
        f"alpha = {ALPHA}, "
        f"multiple-comparison correction = {PADJUST_METHOD}"
    )

    df = load_data(
        folder
    )

    wide_frames, algorithms = (
        aggregate_to_subject_by_condition(
            df,
            reporter,
        )
    )

    # ---------------------------------------------------------------
    # Run all omnibus tests first
    # ---------------------------------------------------------------

    reporter.section(
        "2. UNCORRECTED OMNIBUS TEST RESULTS"
    )

    omnibus_results = []

    for metric in CONTINUOUS_METRICS:
        result = run_friedman(
            wide=wide_frames[metric],
            metric_name=metric,
            reporter=reporter,
        )

        omnibus_results.append(
            result
        )

    cochran_result = run_cochran(
        wide=wide_frames[BINARY_METRIC],
        reporter=reporter,
    )

    omnibus_results.append(
        cochran_result
    )

    # ---------------------------------------------------------------
    # Correct omnibus tests across metrics
    # ---------------------------------------------------------------

    correct_omnibus_across_metrics(
        omnibus_results=omnibus_results,
        reporter=reporter,
    )

    # ---------------------------------------------------------------
    # Run only post-hoc tests that pass corrected omnibus testing
    # ---------------------------------------------------------------

    reporter.section(
        "4. POST-HOC COMPARISONS"
    )

    for result in omnibus_results:
        metric = result["metric"]

        is_significant = result.get(
            "significant_after_correction",
            False,
        )

        corrected_p = result.get(
            "p_corr_across_metrics",
            np.nan,
        )

        if not is_significant:
            reporter.subsection(
                f"Post-hoc decision: {metric}"
            )

            reporter.write(
                f"No post-hoc tests were run because the "
                f"{result['test']} omnibus result was not significant "
                "after across-metric Holm correction "
                f"(corrected p = {corrected_p:.6f})."
            )

            continue

        if result["test"] == "Friedman":
            run_wilcoxon_posthoc(
                wide_complete=result[
                    "wide_complete"
                ],
                metric_name=metric,
                reporter=reporter,
            )

        elif result["test"] == "Cochran's Q":
            run_mcnemar_posthoc(
                wide_complete=result[
                    "wide_complete"
                ],
                reporter=reporter,
            )

    # ---------------------------------------------------------------
    # Descriptive context
    # ---------------------------------------------------------------

    print_descriptive_ranking(
        wide_frames=wide_frames,
        reporter=reporter,
    )

    # ---------------------------------------------------------------
    # Save report
    # ---------------------------------------------------------------

    reporter.section(
        "DONE"
    )

    output_path = (
        folder / OUTPUT_FILENAME
    )

    reporter.write(
        f"Full report written to: {output_path}"
    )

    reporter.save(
        output_path
    )


if __name__ == "__main__":
    main()