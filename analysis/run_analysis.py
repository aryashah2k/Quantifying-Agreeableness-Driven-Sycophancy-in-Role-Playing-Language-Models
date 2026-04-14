#!/usr/bin/env python3
"""
Main Analysis Runner for Sycophancy Benchmark
==============================================
Orchestrates all analysis components: data loading, hypothesis testing,
visualization, and report generation.

Usage:
    python run_analysis.py                    # Run all analyses for all models
    python run_analysis.py --model qwen3-0-6b # Run for specific model
    python run_analysis.py --tests-only       # Run only hypothesis tests
    python run_analysis.py --plots-only       # Generate only plots
    python run_analysis.py --report           # Generate summary report
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODELS,
    ANALYSIS_OUTPUT_DIR,
    ALPHA,
    MODEL_DISPLAY_NAMES,
)
from data_loader import load_all_model_data
from hypothesis_tests import HypothesisTestSuite, run_hypothesis_tests_all_models
from visualizations import SycophancyVisualizer, plot_cross_model_comparison


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def check_data_availability(models: list[str]) -> dict:
    """
    Check which models have data available.
    
    Returns:
        Dictionary with model status
    """
    status = {}
    
    for model in models:
        suite = HypothesisTestSuite(model)
        status[model] = {
            "has_agreeableness": suite.agreeableness_df is not None,
            "has_baseline": suite.baseline_df is not None,
            "has_persona_summary": suite.persona_summary_df is not None,
            "has_merged": suite.merged_df is not None,
            "ready": suite.is_data_available(),
        }
    
    return status


def run_hypothesis_tests(models: list[str], output_dir: Path) -> dict:
    """
    Run hypothesis tests for specified models.
    
    Args:
        models: List of model names
        output_dir: Directory to save results
        
    Returns:
        Dictionary with test results
    """
    print("\n" + "=" * 60)
    print("RUNNING HYPOTHESIS TESTS")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    
    for model in models:
        suite = HypothesisTestSuite(model, alpha=ALPHA)
        
        if suite.is_data_available():
            results = suite.run_all_tests()
            suite.print_summary()
            
            # Save individual results
            model_path = output_dir / f"{model}_hypothesis_tests.json"
            with open(model_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            
            all_results[model] = results
        else:
            print(f"\nSkipping {model}: Data not available")
            all_results[model] = {"error": "Data not available"}
    
    # Save combined results
    combined_path = output_dir / "all_models_hypothesis_tests.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to: {output_dir}")
    
    return all_results


def generate_visualizations(models: list[str], output_dir: Path):
    """
    Generate visualizations for specified models.
    
    Args:
        models: List of model names
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    available_models = []
    
    for model in models:
        viz = SycophancyVisualizer(model, output_dir=output_dir)
        
        if viz.is_data_available():
            viz.generate_all_plots()
            available_models.append(model)
        else:
            print(f"Skipping {model}: Data not available")
    
    # Cross-model comparison
    if len(available_models) > 1:
        print("\nGenerating cross-model comparison...")
        plot_cross_model_comparison(available_models, save=True)
    
    print(f"\nPlots saved to: {output_dir}")


def generate_summary_report(test_results: dict, output_dir: Path) -> str:
    """
    Generate a markdown summary report of all analyses.
    
    Args:
        test_results: Dictionary with hypothesis test results
        output_dir: Directory to save report
        
    Returns:
        Path to generated report
    """
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY REPORT")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# Sycophancy Benchmark Analysis Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report presents the analysis of the relationship between persona agreeableness ",
        "and model sycophancy across multiple language models.",
        "",
        "### Research Hypothesis",
        "",
        "- **H₀ (Null):** High Agreeableness personas do NOT cause higher sycophancy",
        "- **H₁ (Alternative):** High Agreeableness personas cause higher sycophancy (μ_high > μ_low)",
        "",
        "---",
        "",
        "## Results by Model",
        "",
    ]
    
    # Summary table
    report_lines.extend([
        "| Model | n | r (Pearson) | Cohen's d | Welch t-test | Conclusion |",
        "|-------|---|-------------|-----------|--------------|------------|",
    ])
    
    for model, results in test_results.items():
        if "error" in results:
            report_lines.append(f"| {MODEL_DISPLAY_NAMES.get(model, model)} | - | - | - | - | No data |")
            continue
        
        n = results.get("descriptive_stats", {}).get("n_personas", "-")
        
        corr = results.get("correlation_tests", {}).get("pearson", {})
        r = corr.get("r", float("nan"))
        r_str = f"{r:.3f}" if not np.isnan(r) else "-"
        if corr.get("significant"):
            r_str += "*"
        
        gd = results.get("group_difference_tests", {})
        d = gd.get("effect_sizes", {}).get("cohens_d", float("nan"))
        d_str = f"{d:.3f}" if not np.isnan(d) else "-"
        
        welch = gd.get("welch_ttest", {})
        p_welch = welch.get("p_value", float("nan"))
        p_str = f"p={p_welch:.4f}" if not np.isnan(p_welch) else "-"
        if welch.get("significant"):
            p_str += "*"
        
        conclusion = results.get("conclusion", {})
        verdict = "**REJECT H₀**" if conclusion.get("reject_null") else "Fail to reject H₀"
        
        report_lines.append(
            f"| {MODEL_DISPLAY_NAMES.get(model, model)} | {n} | {r_str} | {d_str} | {p_str} | {verdict} |"
        )
    
    report_lines.extend([
        "",
        "*\\* = significant at α=0.05*",
        "",
        "---",
        "",
    ])
    
    # Detailed results per model
    for model, results in test_results.items():
        if "error" in results:
            continue
        
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        report_lines.extend([
            f"## {display_name}",
            "",
        ])
        
        # Descriptive stats
        ds = results.get("descriptive_stats", {})
        if ds:
            report_lines.extend([
                "### Descriptive Statistics",
                "",
                f"- **Sample Size:** {ds.get('n_personas', '-')} personas",
                f"- **Agreeableness:** M = {ds.get('agreeableness', {}).get('mean', 0):.3f}, "
                f"SD = {ds.get('agreeableness', {}).get('std', 0):.3f}",
                f"- **Sycophancy:** M = {ds.get('sycophancy', {}).get('mean', 0):.3f}, "
                f"SD = {ds.get('sycophancy', {}).get('std', 0):.3f}",
            ])
            if ds.get("baseline_sycophancy_rate"):
                report_lines.append(f"- **Baseline Sycophancy Rate:** {ds['baseline_sycophancy_rate']:.3f}")
            report_lines.append("")
        
        # Correlation results
        corr = results.get("correlation_tests", {})
        if corr:
            pearson = corr.get("pearson", {})
            spearman = corr.get("spearman", {})
            report_lines.extend([
                "### Correlation Analysis",
                "",
                f"- **Pearson:** r = {pearson.get('r', 0):.3f}, p = {pearson.get('p_value_one_sided', 1):.4f} "
                f"({pearson.get('interpretation', '')})",
                f"- **Spearman:** ρ = {spearman.get('rho', 0):.3f}, p = {spearman.get('p_value_one_sided', 1):.4f} "
                f"({spearman.get('interpretation', '')})",
                "",
            ])
        
        # Group comparison
        gd = results.get("group_difference_tests", {})
        if gd:
            gs = gd.get("group_statistics", {})
            es = gd.get("effect_sizes", {})
            report_lines.extend([
                "### Group Comparison (Median Split)",
                "",
                f"- **High Agreeableness:** n = {gs.get('high_agreeableness', {}).get('n', '-')}, "
                f"M = {gs.get('high_agreeableness', {}).get('mean_sycophancy', 0):.3f}",
                f"- **Low Agreeableness:** n = {gs.get('low_agreeableness', {}).get('n', '-')}, "
                f"M = {gs.get('low_agreeableness', {}).get('mean_sycophancy', 0):.3f}",
                f"- **Difference:** {gs.get('difference', 0):.3f}",
                f"- **Effect Size:** Cohen's d = {es.get('cohens_d', 0):.3f} ({es.get('interpretation', '')})",
                "",
                "### Statistical Tests",
                "",
                f"- **Welch's t-test:** t = {gd.get('welch_ttest', {}).get('t_statistic', 0):.3f}, "
                f"p = {gd.get('welch_ttest', {}).get('p_value', 1):.4f}",
                f"- **Mann-Whitney U:** U = {gd.get('mann_whitney_u', {}).get('u_statistic', 0):.1f}, "
                f"p = {gd.get('mann_whitney_u', {}).get('p_value', 1):.4f}",
                f"- **Permutation Test:** p = {gd.get('permutation_test', {}).get('p_value', 1):.4f}",
                "",
            ])
        
        # Regression
        reg = results.get("regression_test", {})
        if reg:
            report_lines.extend([
                "### Regression Analysis",
                "",
                f"- **Equation:** {reg.get('equation', '')}",
                f"- **R²:** {reg.get('r_squared', 0):.3f}",
                f"- **p-value:** {reg.get('p_value_one_sided', 1):.4f}",
                "",
            ])
        
        # Conclusion
        conclusion = results.get("conclusion", {})
        if conclusion:
            report_lines.extend([
                "### Conclusion",
                "",
                f"**{conclusion.get('summary', '')}**",
                "",
                f"Significant tests: {', '.join(conclusion.get('significant_tests', [])) or 'None'}",
                "",
                "---",
                "",
            ])
    
    # Overall conclusions
    report_lines.extend([
        "## Overall Conclusions",
        "",
    ])
    
    # Count models that rejected H0
    reject_count = sum(
        1 for r in test_results.values() 
        if r.get("conclusion", {}).get("reject_null", False)
    )
    total_count = sum(1 for r in test_results.values() if "error" not in r)
    
    if total_count > 0:
        report_lines.extend([
            f"Out of {total_count} models analyzed:",
            "",
            f"- **{reject_count}** models showed significant evidence that high agreeableness "
            "personas lead to higher sycophancy (rejected H₀)",
            f"- **{total_count - reject_count}** models did not show significant evidence "
            "(failed to reject H₀)",
            "",
        ])
        
        if reject_count > total_count / 2:
            report_lines.extend([
                "**Overall Finding:** The majority of models support the hypothesis that ",
                "adopting high agreeableness personas causes models to exhibit more sycophantic behavior.",
                "",
            ])
        else:
            report_lines.extend([
                "**Overall Finding:** The evidence is mixed across models. The relationship between ",
                "persona agreeableness and sycophancy may be model-dependent.",
                "",
            ])
    
    report_lines.extend([
        "---",
        "",
        "*Report generated by Sycophancy Benchmark Analysis Suite*",
    ])
    
    # Write report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"Report saved to: {report_path}")
    
    return str(report_path)


def main():
    """Main entry point for analysis."""
    parser = argparse.ArgumentParser(
        description="Sycophancy Benchmark Analysis Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Run analysis for specific model only",
    )
    parser.add_argument(
        "--tests-only",
        action="store_true",
        help="Run only hypothesis tests (no plots)",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate only plots (no tests)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate summary report",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check data availability only",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(ANALYSIS_OUTPUT_DIR),
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Determine which models to analyze
    if args.model:
        models = [args.model]
    else:
        models = MODELS
    
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("SYCOPHANCY BENCHMARK ANALYSIS SUITE")
    print("=" * 60)
    print(f"Models: {len(models)}")
    print(f"Output: {output_dir}")
    print(f"Alpha: {ALPHA}")
    
    # Check data availability
    print("\nChecking data availability...")
    status = check_data_availability(models)
    
    available_models = [m for m, s in status.items() if s["ready"]]
    unavailable_models = [m for m, s in status.items() if not s["ready"]]
    
    print(f"\nAvailable: {len(available_models)}")
    for m in available_models:
        print(f"  ✓ {m}")
    
    if unavailable_models:
        print(f"\nUnavailable: {len(unavailable_models)}")
        for m in unavailable_models:
            s = status[m]
            missing = []
            if not s["has_agreeableness"]:
                missing.append("agreeableness")
            if not s["has_baseline"]:
                missing.append("baseline")
            if not s["has_persona_summary"]:
                missing.append("persona")
            print(f"  ✗ {m} (missing: {', '.join(missing)})")
    
    if args.check:
        return
    
    if not available_models:
        print("\nNo models with complete data. Please run evaluation scripts first.")
        return
    
    test_results = {}
    
    # Run hypothesis tests
    if not args.plots_only:
        test_results = run_hypothesis_tests(
            available_models,
            output_dir / "hypothesis_tests",
        )
    
    # Generate visualizations
    if not args.tests_only:
        generate_visualizations(
            available_models,
            output_dir / "plots",
        )
    
    # Generate report
    if args.report or (not args.tests_only and not args.plots_only):
        if not test_results:
            # Load existing results
            results_path = output_dir / "hypothesis_tests" / "all_models_hypothesis_tests.json"
            if results_path.exists():
                with open(results_path, "r", encoding="utf-8") as f:
                    test_results = json.load(f)
        
        if test_results:
            generate_summary_report(test_results, output_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
