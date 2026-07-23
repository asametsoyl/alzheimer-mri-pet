"""Statistical comparison tests for multi-experiment evaluation.

Wilcoxon signed-rank test with Bonferroni correction.
Generates formatted Markdown comparison tables.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.stats import wilcoxon, ttest_rel

logger = logging.getLogger(__name__)


def compare_experiments(
    experiments: dict[str, list[dict[str, float]]],
    reference: str,
    metrics: list[str] | None = None,
    test: str = "wilcoxon",
    correction: str = "bonferroni",
    alternative: str = "two-sided",
) -> dict:
    """Compare multiple experiments against a reference using statistical tests.

    Args:
        experiments: Dict mapping experiment_name → list of per-subject metric dicts.
        reference: Name of the reference experiment in `experiments`.
        metrics: List of metrics to compare. Defaults to all keys in first record.
        test: 'wilcoxon' or 'ttest'.
        correction: 'bonferroni', 'fdr', or 'none'.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        Dict with comparison results and formatted Markdown table.
    """
    if reference not in experiments:
        raise ValueError(f"Reference experiment '{reference}' not in experiments dict.")

    ref_records = experiments[reference]
    if metrics is None:
        metrics = list(ref_records[0].keys())

    n_comparisons = (len(experiments) - 1) * len(metrics)
    alpha = 0.05
    corrected_alpha = alpha / max(n_comparisons, 1) if correction == "bonferroni" else alpha

    results: dict = {"reference": reference, "test": test, "correction": correction,
                     "corrected_alpha": corrected_alpha, "comparisons": {}}
    table_rows = []

    for exp_name, exp_records in experiments.items():
        if exp_name == reference:
            continue
        comp: dict[str, dict] = {}
        for metric in metrics:
            ref_vals = np.array([d[metric] for d in ref_records])
            exp_vals = np.array([d[metric] for d in exp_records])

            if len(ref_vals) != len(exp_vals):
                logger.warning(
                    "Unequal sample sizes for %s vs %s (%s): %d vs %d. Skipping.",
                    reference, exp_name, metric, len(ref_vals), len(exp_vals),
                )
                continue

            try:
                if test == "wilcoxon":
                    stat, pval = wilcoxon(ref_vals, exp_vals, alternative=alternative)
                else:
                    stat, pval = ttest_rel(ref_vals, exp_vals, alternative=alternative)
            except ValueError as e:
                logger.warning("Statistical test failed for %s/%s: %s", exp_name, metric, e)
                stat, pval = float("nan"), float("nan")

            significant = bool(pval < corrected_alpha)
            comp[metric] = {
                "ref_mean": float(np.mean(ref_vals)),
                "exp_mean": float(np.mean(exp_vals)),
                "delta": float(np.mean(exp_vals) - np.mean(ref_vals)),
                "p_value": float(pval),
                "significant": significant,
            }
            table_rows.append({
                "experiment": exp_name,
                "metric": metric,
                **comp[metric],
            })

        results["comparisons"][exp_name] = comp

    # Build Markdown table
    md_lines = [
        "",
        "## Statistical Comparison Table",
        f"**Reference**: {reference} | **Test**: {test} | "
        f"**Correction**: {correction} (α={corrected_alpha:.4f}) | "
        f"**Alternative**: {alternative}",
        "",
        "| Experiment | Metric | Ref Mean | Exp Mean | Δ | p-value | Significant |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in table_rows:
        sig = "✓" if row["significant"] else "✗"
        md_lines.append(
            f"| {row['experiment']} | {row['metric']} "
            f"| {row['ref_mean']:.4f} | {row['exp_mean']:.4f} "
            f"| {row['delta']:+.4f} | {row['p_value']:.4f} | {sig} |"
        )

    results["markdown_table"] = "\n".join(md_lines)
    print(results["markdown_table"])
    return results
