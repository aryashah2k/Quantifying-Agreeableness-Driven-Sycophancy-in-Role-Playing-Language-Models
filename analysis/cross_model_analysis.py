#!/usr/bin/env python3
"""
Cross-Model Analysis for Sycophancy Benchmark
==============================================
Aggregate and compare results across all evaluated models.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from typing import Optional

from config import (
    MODELS,
    ANALYSIS_OUTPUT_DIR,
    MODEL_DISPLAY_NAMES,
    ALPHA,
)
from data_loader import (
    load_agreeableness_scores,
    load_baseline_sycophancy,
    load_persona_sycophancy_summary,
    merge_agreeableness_sycophancy,
)
from metrics import (
    sycophancy_rate,
    agreeableness_sycophancy_correlation,
    sycophancy_effect_size,
    hedges_g,
    split_by_agreeableness,
)


def compute_model_summary(model_name: str) -> Optional[dict]:
    """
    Compute summary statistics for a single model.
    
    Args:
        model_name: Short name of the model
        
    Returns:
        Dictionary with summary statistics or None if data unavailable
    """
    agreeableness_df = load_agreeableness_scores(model_name)
    baseline_df = load_baseline_sycophancy(model_name)
    persona_summary_df = load_persona_sycophancy_summary(model_name)
    
    if agreeableness_df is None or persona_summary_df is None:
        return None
    
    merged_df = merge_agreeableness_sycophancy(agreeableness_df, persona_summary_df)
    
    if len(merged_df) == 0:
        return None
    
    agreeableness = merged_df["overall_agreeableness"].values
    sycophancy = merged_df["mean_sycophancy_score"].values
    
    # Baseline sycophancy rate
    baseline_sr = np.nan
    if baseline_df is not None and "sycophancy_score" in baseline_df.columns:
        baseline_sr = sycophancy_rate(baseline_df["sycophancy_score"].values)
    
    # Correlation
    r_pearson, p_pearson = agreeableness_sycophancy_correlation(agreeableness, sycophancy, "pearson")
    r_spearman, p_spearman = agreeableness_sycophancy_correlation(agreeableness, sycophancy, "spearman")
    
    # Group comparison
    high_df, low_df = split_by_agreeableness(merged_df, method="median")
    high_syc = high_df["mean_sycophancy_score"].values
    low_syc = low_df["mean_sycophancy_score"].values
    
    d = sycophancy_effect_size(high_syc, low_syc)
    g = hedges_g(high_syc, low_syc)
    
    # t-test
    t_stat, p_ttest = stats.ttest_ind(high_syc, low_syc, equal_var=False, alternative="greater")
    
    return {
        "model": model_name,
        "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        "n_personas": len(merged_df),
        "agreeableness_mean": float(np.nanmean(agreeableness)),
        "agreeableness_std": float(np.nanstd(agreeableness)),
        "sycophancy_mean": float(np.nanmean(sycophancy)),
        "sycophancy_std": float(np.nanstd(sycophancy)),
        "baseline_sycophancy": float(baseline_sr) if not np.isnan(baseline_sr) else None,
        "sycophancy_shift": float(np.nanmean(sycophancy) - baseline_sr) if not np.isnan(baseline_sr) else None,
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "pearson_significant": p_pearson / 2 < ALPHA and r_pearson > 0,
        "spearman_rho": float(r_spearman),
        "spearman_p": float(p_spearman),
        "spearman_significant": p_spearman / 2 < ALPHA and r_spearman > 0,
        "high_agree_mean": float(np.nanmean(high_syc)),
        "low_agree_mean": float(np.nanmean(low_syc)),
        "group_difference": float(np.nanmean(high_syc) - np.nanmean(low_syc)),
        "cohens_d": float(d),
        "hedges_g": float(g),
        "ttest_t": float(t_stat),
        "ttest_p": float(p_ttest),
        "ttest_significant": p_ttest < ALPHA,
    }


def compute_cross_model_summary(models: list[str]) -> pd.DataFrame:
    """
    Compute summary statistics across all models.
    
    Args:
        models: List of model short names
        
    Returns:
        DataFrame with one row per model
    """
    summaries = []
    
    for model in models:
        print(f"Processing {model}...")
        summary = compute_model_summary(model)
        if summary is not None:
            summaries.append(summary)
        else:
            print(f"  Skipping {model}: Data not available")
    
    if not summaries:
        return pd.DataFrame()
    
    return pd.DataFrame(summaries)


def compute_meta_analysis(df: pd.DataFrame) -> dict:
    """
    Perform meta-analysis across models.
    
    Args:
        df: DataFrame with per-model summaries
        
    Returns:
        Dictionary with meta-analysis results
    """
    if len(df) == 0:
        return {"error": "No data available"}
    
    # Count significant results
    n_models = len(df)
    n_pearson_sig = df["pearson_significant"].sum()
    n_spearman_sig = df["spearman_significant"].sum()
    n_ttest_sig = df["ttest_significant"].sum()
    
    # Average effect sizes
    mean_r = df["pearson_r"].mean()
    mean_d = df["cohens_d"].mean()
    
    # Weighted average correlation (by sample size)
    weights = df["n_personas"].values
    weighted_r = np.average(df["pearson_r"].values, weights=weights)
    
    # Fisher's z transformation for combining correlations
    z_values = np.arctanh(df["pearson_r"].values)
    z_mean = np.average(z_values, weights=weights - 3)  # n-3 weighting
    combined_r = np.tanh(z_mean)
    
    # Combined p-value using Fisher's method
    p_values = df["pearson_p"].values
    # Avoid log(0)
    p_values = np.clip(p_values, 1e-300, 1)
    chi2_stat = -2 * np.sum(np.log(p_values))
    combined_p = 1 - stats.chi2.cdf(chi2_stat, 2 * n_models)
    
    # Effect size heterogeneity (I² statistic)
    d_values = df["cohens_d"].values
    d_mean = np.mean(d_values)
    Q = np.sum((d_values - d_mean)**2)
    df_Q = n_models - 1
    I_squared = max(0, (Q - df_Q) / Q * 100) if Q > 0 else 0
    
    return {
        "n_models": n_models,
        "n_pearson_significant": int(n_pearson_sig),
        "n_spearman_significant": int(n_spearman_sig),
        "n_ttest_significant": int(n_ttest_sig),
        "proportion_significant": float(n_ttest_sig / n_models),
        "mean_correlation": float(mean_r),
        "weighted_correlation": float(weighted_r),
        "combined_correlation_fisher": float(combined_r),
        "combined_p_value_fisher": float(combined_p),
        "mean_cohens_d": float(mean_d),
        "effect_heterogeneity_I2": float(I_squared),
        "overall_conclusion": (
            "SUPPORT" if n_ttest_sig > n_models / 2 else "MIXED"
        ),
        "interpretation": (
            f"{n_ttest_sig}/{n_models} models ({n_ttest_sig/n_models*100:.1f}%) show significant "
            f"evidence that high agreeableness personas lead to higher sycophancy. "
            f"Mean effect size: d = {mean_d:.3f} ({_interpret_d(mean_d)}). "
            f"Mean correlation: r = {mean_r:.3f}."
        ),
    }


def _interpret_d(d: float) -> str:
    """Interpret Cohen's d."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def generate_cross_model_report(df: pd.DataFrame, meta: dict, output_dir: Path) -> str:
    """
    Generate cross-model comparison report.
    
    Args:
        df: DataFrame with per-model summaries
        meta: Meta-analysis results
        output_dir: Output directory
        
    Returns:
        Path to generated report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# Cross-Model Sycophancy Analysis",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Meta-Analysis Summary",
        "",
        f"- **Models Analyzed:** {meta['n_models']}",
        f"- **Models with Significant Effect:** {meta['n_ttest_significant']} ({meta['proportion_significant']*100:.1f}%)",
        f"- **Mean Correlation (r):** {meta['mean_correlation']:.3f}",
        f"- **Combined Correlation (Fisher's z):** {meta['combined_correlation_fisher']:.3f}",
        f"- **Combined p-value:** {meta['combined_p_value_fisher']:.6f}",
        f"- **Mean Effect Size (Cohen's d):** {meta['mean_cohens_d']:.3f}",
        f"- **Effect Heterogeneity (I²):** {meta['effect_heterogeneity_I2']:.1f}%",
        "",
        f"**Overall Conclusion:** {meta['overall_conclusion']}",
        "",
        meta['interpretation'],
        "",
        "---",
        "",
        "## Per-Model Results",
        "",
        "| Model | n | r | d | Δ Sycophancy | Significant |",
        "|-------|---|---|---|--------------|-------------|",
    ]
    
    for _, row in df.iterrows():
        sig = "✓" if row["ttest_significant"] else "✗"
        shift = f"{row['sycophancy_shift']:.3f}" if row['sycophancy_shift'] is not None else "-"
        lines.append(
            f"| {row['display_name']} | {row['n_personas']} | "
            f"{row['pearson_r']:.3f} | {row['cohens_d']:.3f} | {shift} | {sig} |"
        )
    
    lines.extend([
        "",
        "*r = Pearson correlation, d = Cohen's d effect size, Δ = shift from baseline*",
        "",
        "---",
        "",
        "## Detailed Statistics",
        "",
    ])
    
    # Add detailed table
    for _, row in df.iterrows():
        lines.extend([
            f"### {row['display_name']}",
            "",
            f"- **Sample:** {row['n_personas']} personas",
            f"- **Agreeableness:** M = {row['agreeableness_mean']:.3f}, SD = {row['agreeableness_std']:.3f}",
            f"- **Sycophancy:** M = {row['sycophancy_mean']:.3f}, SD = {row['sycophancy_std']:.3f}",
            f"- **Baseline Sycophancy:** {row['baseline_sycophancy']:.3f}" if row['baseline_sycophancy'] else "- **Baseline Sycophancy:** N/A",
            f"- **Pearson r:** {row['pearson_r']:.3f} (p = {row['pearson_p']:.4f})",
            f"- **Spearman ρ:** {row['spearman_rho']:.3f} (p = {row['spearman_p']:.4f})",
            f"- **High Agreeableness Mean:** {row['high_agree_mean']:.3f}",
            f"- **Low Agreeableness Mean:** {row['low_agree_mean']:.3f}",
            f"- **Group Difference:** {row['group_difference']:.3f}",
            f"- **Cohen's d:** {row['cohens_d']:.3f}",
            f"- **Welch's t-test:** t = {row['ttest_t']:.3f}, p = {row['ttest_p']:.4f}",
            "",
        ])
    
    report_path = output_dir / "cross_model_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Report saved to: {report_path}")
    return str(report_path)


def run_cross_model_analysis(models: list[str] = None) -> tuple:
    """
    Run complete cross-model analysis.
    
    Args:
        models: List of models to analyze (default: all)
        
    Returns:
        Tuple of (DataFrame, meta-analysis dict)
    """
    if models is None:
        models = MODELS
    
    print("=" * 60)
    print("CROSS-MODEL SYCOPHANCY ANALYSIS")
    print("=" * 60)
    
    # Compute summaries
    df = compute_cross_model_summary(models)
    
    if len(df) == 0:
        print("\nNo models with complete data found.")
        return pd.DataFrame(), {}
    
    # Meta-analysis
    meta = compute_meta_analysis(df)
    
    # Save results
    output_dir = ANALYSIS_OUTPUT_DIR / "cross_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / "model_summaries.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummaries saved to: {csv_path}")
    
    # Save JSON
    json_path = output_dir / "meta_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta-analysis saved to: {json_path}")
    
    # Generate report
    generate_cross_model_report(df, meta, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nModels analyzed: {meta['n_models']}")
    print(f"Significant effects: {meta['n_ttest_significant']}/{meta['n_models']}")
    print(f"Mean correlation: r = {meta['mean_correlation']:.3f}")
    print(f"Mean effect size: d = {meta['mean_cohens_d']:.3f}")
    print(f"\nConclusion: {meta['overall_conclusion']}")
    
    return df, meta


if __name__ == "__main__":
    df, meta = run_cross_model_analysis()
