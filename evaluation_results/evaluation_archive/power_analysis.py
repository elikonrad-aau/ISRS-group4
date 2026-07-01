import math

import pingouin as pg


# -------------------------------------------------------------------
# Study design assumptions
# -------------------------------------------------------------------

# Conventional medium effect size for a repeated-measures ANOVA.
# This is an assumption used for planning, not calculated from results.
COHENS_F = 0.25

# Every evaluator assesses all eight recommendation algorithms.
N_REPEATED_MEASUREMENTS = 8

# Desired probability of detecting the assumed effect when it exists.
TARGET_POWER = 0.90

# Overall family-wise significance level.
FAMILY_WISE_ALPHA = 0.05

# Four planned omnibus tests:
#   1. relevance
#   2. preference score
#   3. novelty score
#   4. in_top5
N_OMNIBUS_TESTS = 4

# Assumed average correlation between repeated algorithm measurements.
# This should ideally come from pilot data. A value of 0.30 is a
# cautious planning assumption.
AVERAGE_CORRELATION = 0.30

# Assumed sphericity correction.
# 1.00 means perfect sphericity; 0.50 represents a substantial violation.
# This is relevant to the ANOVA approximation used by Pingouin.
EPSILON = 0.50

# Expected proportion of recruited evaluators whose data may be
# incomplete or unusable.
DROPOUT_RATE = 0.15


# -------------------------------------------------------------------
# Helper for formatted output
# -------------------------------------------------------------------

def print_value(label, value, description):
    """Print one value together with a short explanation."""
    print(f"{label:<30} {value}")
    print(f"{'':<30} {description}")
    print()


# -------------------------------------------------------------------
# Power analysis
# -------------------------------------------------------------------

def main():
    # ---------------------------------------------------------------
    # Calculate values derived from the study assumptions
    # ---------------------------------------------------------------

    # Cohen's f squared.
    f_squared = COHENS_F ** 2

    # Pingouin expects eta-squared rather than Cohen's f.
    #
    # eta² = f² / (1 + f²)
    eta_squared = (
        f_squared
        / (1 + f_squared)
    )

    # Conservative Bonferroni-equivalent threshold for four omnibus
    # tests. This is also the strictest first threshold used by Holm.
    #
    # adjusted alpha = family-wise alpha / number of tests
    adjusted_alpha = (
        FAMILY_WISE_ALPHA
        / N_OMNIBUS_TESTS
    )

    # Theoretical minimum epsilon for eight repeated measurements.
    #
    # minimum epsilon = 1 / (m - 1)
    minimum_epsilon = (
        1
        / (N_REPEATED_MEASUREMENTS - 1)
    )

    if not minimum_epsilon <= EPSILON <= 1:
        raise ValueError(
            "EPSILON must be between "
            f"{minimum_epsilon:.4f} and 1.0 "
            f"for {N_REPEATED_MEASUREMENTS} measurements."
        )

    if not 0 <= AVERAGE_CORRELATION < 1:
        raise ValueError(
            "AVERAGE_CORRELATION must be at least 0 "
            "and smaller than 1."
        )

    if not 0 <= DROPOUT_RATE < 1:
        raise ValueError(
            "DROPOUT_RATE must be at least 0 "
            "and smaller than 1."
        )

    # ---------------------------------------------------------------
    # Calculate the required complete sample size
    # ---------------------------------------------------------------

    calculated_sample_size = pg.power_rm_anova(
        eta_squared=eta_squared,
        m=N_REPEATED_MEASUREMENTS,
        n=None,
        power=TARGET_POWER,
        alpha=adjusted_alpha,
        corr=AVERAGE_CORRELATION,
        epsilon=EPSILON,
    )

    # A fraction of a participant is impossible, so round upward.
    required_complete_sample = math.ceil(
        calculated_sample_size
    )

    # ---------------------------------------------------------------
    # Add the dropout allowance
    # ---------------------------------------------------------------

    # recruitment target = complete sample / expected retention rate
    retention_rate = 1 - DROPOUT_RATE

    recruitment_target = math.ceil(
        required_complete_sample
        / retention_rate
    )

    expected_number_lost = (
        recruitment_target
        - required_complete_sample
    )

    # ---------------------------------------------------------------
    # Verify power after rounding the sample size
    # ---------------------------------------------------------------

    achieved_power = pg.power_rm_anova(
        eta_squared=eta_squared,
        m=N_REPEATED_MEASUREMENTS,
        n=required_complete_sample,
        power=None,
        alpha=adjusted_alpha,
        corr=AVERAGE_CORRELATION,
        epsilon=EPSILON,
    )

    beta = 1 - achieved_power

    # ---------------------------------------------------------------
    # Display the assumptions and results
    # ---------------------------------------------------------------

    print("=" * 78)
    print("CONSERVATIVE REPEATED-MEASURES POWER ANALYSIS")
    print("=" * 78)
    print()

    print("PLANNING ASSUMPTIONS")
    print("-" * 78)
    print()

    print_value(
        "Cohen's f:",
        f"{COHENS_F:.4f}",
        (
            "Assumed standardized omnibus effect size. "
            "A value of 0.25 is conventionally described as medium."
        ),
    )

    print_value(
        "Cohen's f²:",
        f"{f_squared:.4f}",
        (
            "Calculated as f². This is an intermediate value used "
            "to convert Cohen's f to eta-squared."
        ),
    )

    print_value(
        "Eta-squared:",
        f"{eta_squared:.4f}",
        (
            "Calculated as f² / (1 + f²). Pingouin requires "
            "eta-squared as its effect-size input."
        ),
    )

    print_value(
        "Repeated measurements:",
        N_REPEATED_MEASUREMENTS,
        (
            "The number of algorithms evaluated by each participant. "
            "Every evaluator provides measurements for all eight algorithms."
        ),
    )

    print_value(
        "Target power:",
        f"{TARGET_POWER:.3f}",
        (
            "The desired probability of detecting the assumed effect "
            "when that effect truly exists."
        ),
    )

    print_value(
        "Family-wise alpha:",
        f"{FAMILY_WISE_ALPHA:.4f}",
        (
            "The overall maximum probability of making at least one "
            "false-positive decision across the omnibus test family."
        ),
    )

    print_value(
        "Number of omnibus tests:",
        N_OMNIBUS_TESTS,
        (
            "The planned tests for relevance, preference score, "
            "novelty score, and in_top5."
        ),
    )

    print_value(
        "Adjusted alpha:",
        f"{adjusted_alpha:.4f}",
        (
            "Calculated as family-wise alpha divided by the number "
            "of omnibus tests: "
            f"{FAMILY_WISE_ALPHA} / {N_OMNIBUS_TESTS}."
        ),
    )

    print_value(
        "Average correlation:",
        f"{AVERAGE_CORRELATION:.2f}",
        (
            "Assumed average correlation between repeated algorithm "
            "measurements from the same evaluator."
        ),
    )

    print_value(
        "Epsilon:",
        f"{EPSILON:.2f}",
        (
            "Assumed sphericity correction. Smaller values are more "
            "conservative and increase the required sample size."
        ),
    )

    print_value(
        "Minimum possible epsilon:",
        f"{minimum_epsilon:.4f}",
        (
            "Calculated as 1 / (number of measurements - 1). "
            "For eight measurements, the minimum is 1/7."
        ),
    )

    print_value(
        "Dropout rate:",
        f"{DROPOUT_RATE:.1%}",
        (
            "The expected proportion of recruited evaluators whose "
            "data may be incomplete or unusable."
        ),
    )

    print_value(
        "Retention rate:",
        f"{retention_rate:.1%}",
        (
            "Calculated as 1 minus the dropout rate. This is the "
            "expected proportion of recruited participants retained."
        ),
    )

    print("POWER-ANALYSIS RESULTS")
    print("-" * 78)
    print()

    print_value(
        "Calculated complete N:",
        f"{calculated_sample_size:.2f}",
        (
            "The numerical sample-size solution returned by Pingouin "
            "before rounding to a whole number."
        ),
    )

    print_value(
        "Required complete N:",
        required_complete_sample,
        (
            "The calculated sample size rounded upward. At least this "
            "many evaluators must provide complete, usable data."
        ),
    )

    print_value(
        "Recruitment target:",
        recruitment_target,
        (
            "Calculated as required complete N divided by the expected "
            "retention rate, then rounded upward."
        ),
    )

    print_value(
        "Expected number lost:",
        expected_number_lost,
        (
            "The difference between the recruitment target and the "
            "required number of complete evaluators."
        ),
    )

    print_value(
        "Achieved power:",
        f"{achieved_power:.3f}",
        (
            "The estimated power after rounding the required complete "
            "sample size upward to a whole participant."
        ),
    )

    print_value(
        "Type-II error rate:",
        f"{beta:.3f}",
        (
            "Calculated as 1 minus achieved power. This is the estimated "
            "probability of failing to detect the assumed effect."
        ),
    )

    print("=" * 78)
    print(
        f"Recruit {recruitment_target} evaluators to obtain approximately "
        f"{required_complete_sample} complete evaluations."
    )
    print("=" * 78)
    print()
    print(
        "Note: Pingouin performs a repeated-measures ANOVA power analysis. "
        "This is a conservative planning approximation for the Friedman "
        "tests used in the final analysis, not an exact Friedman-test "
        "power calculation."
    )


if __name__ == "__main__":
    main()