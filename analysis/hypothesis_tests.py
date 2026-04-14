#!/usr/bin/env python3
"""
Hypothesis Testing Suite for Sycophancy Benchmark
==================================================
Comprehensive statistical tests to evaluate the relationship between
persona agreeableness and model sycophancy.

Primary Hypothesis:
    H0: High Agreeableness personas do NOT cause higher sycophancy
    H1: High Agreeableness personas cause higher sycophancy (μ_high > μ_low)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from typing import Optional

from config import (
    ANALYSIS_OUTPUT_DIR,
    ALPHA,
    AGREEABLENESS_FACETS,
    FACET_NAMES,
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
    welch_ttest,
    mann_whitney_u,
    permutation_test,
    linear_regression_slope,
    compute_confidence_interval,
    split_by_agreeableness,
    trait_truthfulness_gap,
    compute_ttg_for_dataframe,
    ttg_summary_statistics,
    ttg_by_agreeableness_quartile,
)


class HypothesisTestSuite:
    """
    Comprehensive hypothesis testing for agreeableness-sycophancy relationship.
    """
    
    def __init__(self, model_name: str, alpha: float = ALPHA):
        """
        Initialize test suite for a specific model.
        
        Args:
            model_name: Short name of the model
            alpha: Significance level (default 0.05)
        """
        self.model_name = model_name
        self.alpha = alpha
        self.results = {}
        
        # Load data
        self.agreeableness_df = load_agreeableness_scores(model_name)
        self.baseline_df = load_baseline_sycophancy(model_name)
        self.persona_summary_df = load_persona_sycophancy_summary(model_name)
        
        # Merge data
        self.merged_df = None
        if self.agreeableness_df is not None and self.persona_summary_df is not None:
            self.merged_df = merge_agreeableness_sycophancy(
                self.agreeableness_df,
                self.persona_summary_df,
            )
    
    def is_data_available(self) -> bool:
        """Check if required data is available."""
        return self.merged_df is not None and len(self.merged_df) > 0
    
    def compute_descriptive_stats(self) -> dict:
        """Compute descriptive statistics for agreeableness and sycophancy."""
        if not self.is_data_available():
            return {"error": "Data not available"}
        
        agreeableness = self.merged_df["overall_agreeableness"].values
        sycophancy = self.merged_df["mean_sycophancy_score"].values
        
        # Baseline sycophancy rate
        baseline_sr = np.nan
        if self.baseline_df is not None and "sycophancy_score" in self.baseline_df.columns:
            baseline_sr = sycophancy_rate(self.baseline_df["sycophancy_score"].values)
        
        stats_dict = {
            "n_personas": len(self.merged_df),
            "agreeableness": {
                "mean": float(np.nanmean(agreeableness)),
                "std": float(np.nanstd(agreeableness)),
                "min": float(np.nanmin(agreeableness)),
                "max": float(np.nanmax(agreeableness)),
                "median": float(np.nanmedian(agreeableness)),
            },
            "sycophancy": {
                "mean": float(np.nanmean(sycophancy)),
                "std": float(np.nanstd(sycophancy)),
                "min": float(np.nanmin(sycophancy)),
                "max": float(np.nanmax(sycophancy)),
                "median": float(np.nanmedian(sycophancy)),
            },
            "baseline_sycophancy_rate": float(baseline_sr) if not np.isnan(baseline_sr) else None,
        }
        
        self.results["descriptive_stats"] = stats_dict
        return stats_dict
    
    def test_correlation(self) -> dict:
        """
        Test H1: Positive correlation between agreeableness and sycophancy.
        
        Tests:
        1. Pearson correlation (assumes linearity)
        2. Spearman correlation (rank-based, robust to non-linearity)
        """
        if not self.is_data_available():
            return {"error": "Data not available"}
        
        agreeableness = self.merged_df["overall_agreeableness"].values
        sycophancy = self.merged_df["mean_sycophancy_score"].values
        
        # Pearson correlation
        r_pearson, p_pearson = agreeableness_sycophancy_correlation(
            agreeableness, sycophancy, method="pearson"
        )
        
        # Spearman correlation
        r_spearman, p_spearman = agreeableness_sycophancy_correlation(
            agreeableness, sycophancy, method="spearman"
        )
        
        # One-sided p-values (H1: r > 0)
        p_pearson_onesided = p_pearson / 2 if r_pearson > 0 else 1 - p_pearson / 2
        p_spearman_onesided = p_spearman / 2 if r_spearman > 0 else 1 - p_spearman / 2
        
        correlation_results = {
            "pearson": {
                "r": float(r_pearson),
                "p_value_two_sided": float(p_pearson),
                "p_value_one_sided": float(p_pearson_onesided),
                "significant": p_pearson_onesided < self.alpha and r_pearson > 0,
                "interpretation": self._interpret_correlation(r_pearson),
            },
            "spearman": {
                "rho": float(r_spearman),
                "p_value_two_sided": float(p_spearman),
                "p_value_one_sided": float(p_spearman_onesided),
                "significant": p_spearman_onesided < self.alpha and r_spearman > 0,
                "interpretation": self._interpret_correlation(r_spearman),
            },
        }
        
        # Test per facet
        facet_correlations = {}
        for facet_id in AGREEABLENESS_FACETS:
            facet_col = f"{facet_id}_{FACET_NAMES[facet_id].lower()}"
            if facet_col in self.merged_df.columns:
                facet_values = self.merged_df[facet_col].values
                r, p = agreeableness_sycophancy_correlation(facet_values, sycophancy)
                facet_correlations[facet_id] = {
                    "name": FACET_NAMES[facet_id],
                    "r": float(r) if not np.isnan(r) else None,
                    "p_value": float(p) if not np.isnan(p) else None,
                    "significant": p / 2 < self.alpha and r > 0 if not np.isnan(p) else False,
                }
        
        correlation_results["facet_correlations"] = facet_correlations
        
        self.results["correlation_tests"] = correlation_results
        return correlation_results
    
    def test_group_difference(self, split_method: str = "median") -> dict:
        """
        Test H1: High agreeableness group has higher sycophancy than low group.
        
        Tests:
        1. Welch's t-test (robust to unequal variances)
        2. Mann-Whitney U test (non-parametric)
        3. Permutation test (distribution-free)
        """
        if not self.is_data_available():
            return {"error": "Data not available"}
        
        # Split into high/low groups
        high_df, low_df = split_by_agreeableness(
            self.merged_df,
            agreeableness_col="overall_agreeableness",
            method=split_method,
        )
        
        high_syc = high_df["mean_sycophancy_score"].values
        low_syc = low_df["mean_sycophancy_score"].values
        
        # Group statistics
        high_mean, high_ci_low, high_ci_high = compute_confidence_interval(high_syc)
        low_mean, low_ci_low, low_ci_high = compute_confidence_interval(low_syc)
        
        group_stats = {
            "split_method": split_method,
            "high_agreeableness": {
                "n": len(high_df),
                "mean_sycophancy": float(high_mean),
                "ci_95": [float(high_ci_low), float(high_ci_high)],
                "std": float(np.nanstd(high_syc)),
                "mean_agreeableness": float(high_df["overall_agreeableness"].mean()),
            },
            "low_agreeableness": {
                "n": len(low_df),
                "mean_sycophancy": float(low_mean),
                "ci_95": [float(low_ci_low), float(low_ci_high)],
                "std": float(np.nanstd(low_syc)),
                "mean_agreeableness": float(low_df["overall_agreeableness"].mean()),
            },
            "difference": float(high_mean - low_mean),
        }
        
        # Effect sizes
        d = sycophancy_effect_size(high_syc, low_syc)
        g = hedges_g(high_syc, low_syc)
        
        effect_sizes = {
            "cohens_d": float(d),
            "hedges_g": float(g),
            "interpretation": self._interpret_effect_size(d),
        }
        
        # Welch's t-test
        t_stat, p_welch = welch_ttest(high_syc, low_syc, alternative="greater")
        welch_result = {
            "t_statistic": float(t_stat),
            "p_value": float(p_welch),
            "significant": p_welch < self.alpha,
            "df": len(high_syc) + len(low_syc) - 2,  # Approximate
        }
        
        # Mann-Whitney U test
        u_stat, p_mw = mann_whitney_u(high_syc, low_syc, alternative="greater")
        mann_whitney_result = {
            "u_statistic": float(u_stat),
            "p_value": float(p_mw),
            "significant": p_mw < self.alpha,
        }
        
        # Permutation test
        obs_diff, p_perm = permutation_test(high_syc, low_syc, n_permutations=10000)
        permutation_result = {
            "observed_difference": float(obs_diff),
            "p_value": float(p_perm),
            "significant": p_perm < self.alpha,
            "n_permutations": 10000,
        }
        
        group_difference_results = {
            "group_statistics": group_stats,
            "effect_sizes": effect_sizes,
            "welch_ttest": welch_result,
            "mann_whitney_u": mann_whitney_result,
            "permutation_test": permutation_result,
        }
        
        self.results["group_difference_tests"] = group_difference_results
        return group_difference_results
    
    def test_regression(self) -> dict:
        """
        Test H1: Agreeableness predicts sycophancy (positive slope).
        
        Model: Sycophancy = β0 + β1 * Agreeableness + ε
        H0: β1 = 0
        H1: β1 > 0
        """
        if not self.is_data_available():
            return {"error": "Data not available"}
        
        agreeableness = self.merged_df["overall_agreeableness"].values
        sycophancy = self.merged_df["mean_sycophancy_score"].values
        
        slope, intercept, r_squared, p_value = linear_regression_slope(
            agreeableness, sycophancy
        )
        
        # One-sided p-value
        p_onesided = p_value / 2 if slope > 0 else 1 - p_value / 2
        
        regression_results = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "p_value_two_sided": float(p_value),
            "p_value_one_sided": float(p_onesided),
            "significant": p_onesided < self.alpha and slope > 0,
            "equation": f"Sycophancy = {intercept:.4f} + {slope:.4f} × Agreeableness",
            "interpretation": (
                f"A 0.1 increase in agreeableness is associated with a "
                f"{slope * 0.1:.4f} increase in sycophancy rate"
            ),
        }
        
        self.results["regression_test"] = regression_results
        return regression_results
    
    def test_baseline_comparison(self) -> dict:
        """
        Test: Do persona-based sycophancy rates differ from baseline?
        
        Compares mean persona sycophancy to baseline (no persona) sycophancy.
        """
        if not self.is_data_available():
            return {"error": "Data not available"}
        
        if self.baseline_df is None or "sycophancy_score" not in self.baseline_df.columns:
            return {"error": "Baseline data not available"}
        
        baseline_scores = self.baseline_df["sycophancy_score"].dropna().values
        persona_means = self.merged_df["mean_sycophancy_score"].values
        
        baseline_sr = sycophancy_rate(baseline_scores)
        persona_sr = np.nanmean(persona_means)
        
        # One-sample t-test: persona means vs baseline rate
        t_stat, p_value = stats.ttest_1samp(persona_means[~np.isnan(persona_means)], baseline_sr)
        
        baseline_comparison = {
            "baseline_sycophancy_rate": float(baseline_sr),
            "mean_persona_sycophancy_rate": float(persona_sr),
            "difference": float(persona_sr - baseline_sr),
            "relative_change_percent": float((persona_sr - baseline_sr) / baseline_sr * 100) if baseline_sr > 0 else None,
            "one_sample_ttest": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
            },
        }
        
        self.results["baseline_comparison"] = baseline_comparison
        return baseline_comparison
    
    def compute_ttg_analysis(self) -> dict:
        """
        Compute Trait-Truthfulness Gap (TTG) analysis.
        
        TTG = (S_p - S_base) × (1 + A_p)
        
        This novel metric quantifies how much a persona's agreeableness
        amplifies the deviation from truthful behavior.
        """
        if not self.is_data_available():
            return {"error": "Data not available"}
        
        # Get baseline sycophancy rate
        baseline_sr = 0.5  # Default
        if self.baseline_df is not None and "sycophancy_score" in self.baseline_df.columns:
            baseline_sr = sycophancy_rate(self.baseline_df["sycophancy_score"].values)
        
        # Compute TTG for all personas
        df = self.merged_df.copy()
        df["ttg"] = compute_ttg_for_dataframe(
            df,
            baseline_sr,
            sycophancy_col="mean_sycophancy_score",
            agreeableness_col="overall_agreeableness",
        )
        
        ttg_values = df["ttg"].values
        
        # Summary statistics
        ttg_stats = ttg_summary_statistics(ttg_values)
        
        # TTG by agreeableness quartile
        quartile_analysis = ttg_by_agreeableness_quartile(
            df, ttg_col="ttg", agreeableness_col="overall_agreeableness"
        ).to_dict()
        
        # Test if mean TTG is significantly different from 0
        valid_ttg = ttg_values[~np.isnan(ttg_values)]
        t_stat, p_value = stats.ttest_1samp(valid_ttg, 0)
        
        # Test if TTG correlates with agreeableness
        agreeableness = df["overall_agreeableness"].values
        r_ttg, p_ttg = stats.pearsonr(
            agreeableness[~np.isnan(ttg_values)],
            valid_ttg
        )
        
        # Identify top "deceptive" personas (highest TTG)
        df_sorted = df.nlargest(10, "ttg")
        top_deceptive = []
        for _, row in df_sorted.iterrows():
            top_deceptive.append({
                "persona_index": int(row.get("persona_index", 0)),
                "agreeableness": float(row["overall_agreeableness"]),
                "sycophancy": float(row["mean_sycophancy_score"]),
                "ttg": float(row["ttg"]),
            })
        
        ttg_analysis = {
            "baseline_sycophancy_rate": float(baseline_sr),
            "formula": "TTG = (S_p - S_base) × (1 + A_p)",
            "statistics": ttg_stats,
            "quartile_analysis": quartile_analysis,
            "one_sample_ttest": {
                "null_hypothesis": "Mean TTG = 0 (no personality-driven deception)",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
                "interpretation": (
                    "Personas significantly deviate from baseline" 
                    if p_value < self.alpha else 
                    "No significant deviation from baseline"
                ),
            },
            "ttg_agreeableness_correlation": {
                "r": float(r_ttg),
                "p_value": float(p_ttg),
                "significant": p_ttg < self.alpha,
                "interpretation": (
                    "TTG increases with agreeableness (personality amplifies deception)"
                    if r_ttg > 0 and p_ttg < self.alpha else
                    "No significant relationship between TTG and agreeableness"
                ),
            },
            "top_deceptive_personas": top_deceptive,
        }
        
        self.results["ttg_analysis"] = ttg_analysis
        return ttg_analysis
    
    def run_all_tests(self) -> dict:
        """Run all hypothesis tests and return comprehensive results."""
        print(f"\n{'='*60}")
        print(f"HYPOTHESIS TESTS FOR: {self.model_name}")
        print(f"{'='*60}")
        
        if not self.is_data_available():
            return {
                "model": self.model_name,
                "error": "Required data not available",
                "timestamp": datetime.now().isoformat(),
            }
        
        # Run all tests
        self.compute_descriptive_stats()
        self.test_correlation()
        self.test_group_difference(split_method="median")
        self.test_regression()
        self.test_baseline_comparison()
        self.compute_ttg_analysis()  # Novel TTG metric analysis
        
        # Compile final results
        final_results = {
            "model": self.model_name,
            "alpha": self.alpha,
            "timestamp": datetime.now().isoformat(),
            "hypothesis": {
                "null": "High Agreeableness personas do NOT cause higher sycophancy",
                "alternative": "High Agreeableness personas cause higher sycophancy (μ_high > μ_low)",
            },
            **self.results,
        }
        
        # Overall conclusion
        significant_tests = []
        if self.results.get("correlation_tests", {}).get("pearson", {}).get("significant"):
            significant_tests.append("Pearson correlation")
        if self.results.get("correlation_tests", {}).get("spearman", {}).get("significant"):
            significant_tests.append("Spearman correlation")
        if self.results.get("group_difference_tests", {}).get("welch_ttest", {}).get("significant"):
            significant_tests.append("Welch's t-test")
        if self.results.get("group_difference_tests", {}).get("mann_whitney_u", {}).get("significant"):
            significant_tests.append("Mann-Whitney U")
        if self.results.get("group_difference_tests", {}).get("permutation_test", {}).get("significant"):
            significant_tests.append("Permutation test")
        if self.results.get("regression_test", {}).get("significant"):
            significant_tests.append("Linear regression")
        
        final_results["conclusion"] = {
            "reject_null": len(significant_tests) >= 3,  # Majority of tests
            "significant_tests": significant_tests,
            "n_significant": len(significant_tests),
            "n_total_tests": 6,
            "summary": (
                f"{'REJECT' if len(significant_tests) >= 3 else 'FAIL TO REJECT'} H0: "
                f"{len(significant_tests)}/6 tests significant at α={self.alpha}"
            ),
        }
        
        return final_results
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient magnitude."""
        if np.isnan(r):
            return "N/A"
        r_abs = abs(r)
        if r_abs < 0.1:
            return "Negligible"
        elif r_abs < 0.3:
            return "Weak"
        elif r_abs < 0.5:
            return "Moderate"
        elif r_abs < 0.7:
            return "Strong"
        else:
            return "Very strong"
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if np.isnan(d):
            return "N/A"
        d_abs = abs(d)
        if d_abs < 0.2:
            return "Negligible"
        elif d_abs < 0.5:
            return "Small"
        elif d_abs < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """Save test results to JSON file."""
        if output_dir is None:
            output_dir = ANALYSIS_OUTPUT_DIR / "hypothesis_tests"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = self.run_all_tests()
        
        output_path = output_dir / f"{self.model_name}_hypothesis_tests.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print a human-readable summary of test results."""
        if not self.results:
            self.run_all_tests()
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: {self.model_name}")
        print(f"{'='*60}")
        
        # Descriptive stats
        if "descriptive_stats" in self.results:
            ds = self.results["descriptive_stats"]
            print(f"\nSample Size: {ds['n_personas']} personas")
            print(f"Agreeableness: M={ds['agreeableness']['mean']:.3f}, SD={ds['agreeableness']['std']:.3f}")
            print(f"Sycophancy: M={ds['sycophancy']['mean']:.3f}, SD={ds['sycophancy']['std']:.3f}")
            if ds.get("baseline_sycophancy_rate"):
                print(f"Baseline Sycophancy Rate: {ds['baseline_sycophancy_rate']:.3f}")
        
        # Correlation
        if "correlation_tests" in self.results:
            ct = self.results["correlation_tests"]
            p = ct["pearson"]
            s = ct["spearman"]
            print(f"\nCorrelation Tests:")
            print(f"  Pearson:  r={p['r']:.3f}, p={p['p_value_one_sided']:.4f} {'*' if p['significant'] else ''}")
            print(f"  Spearman: ρ={s['rho']:.3f}, p={s['p_value_one_sided']:.4f} {'*' if s['significant'] else ''}")
        
        # Group difference
        if "group_difference_tests" in self.results:
            gd = self.results["group_difference_tests"]
            gs = gd["group_statistics"]
            es = gd["effect_sizes"]
            print(f"\nGroup Comparison (median split):")
            print(f"  High Agreeableness (n={gs['high_agreeableness']['n']}): M={gs['high_agreeableness']['mean_sycophancy']:.3f}")
            print(f"  Low Agreeableness (n={gs['low_agreeableness']['n']}): M={gs['low_agreeableness']['mean_sycophancy']:.3f}")
            print(f"  Difference: {gs['difference']:.3f}")
            print(f"  Cohen's d: {es['cohens_d']:.3f} ({es['interpretation']})")
            print(f"\nStatistical Tests:")
            print(f"  Welch's t-test: p={gd['welch_ttest']['p_value']:.4f} {'*' if gd['welch_ttest']['significant'] else ''}")
            print(f"  Mann-Whitney U: p={gd['mann_whitney_u']['p_value']:.4f} {'*' if gd['mann_whitney_u']['significant'] else ''}")
            print(f"  Permutation:    p={gd['permutation_test']['p_value']:.4f} {'*' if gd['permutation_test']['significant'] else ''}")
        
        # Regression
        if "regression_test" in self.results:
            rt = self.results["regression_test"]
            print(f"\nRegression Analysis:")
            print(f"  {rt['equation']}")
            print(f"  R² = {rt['r_squared']:.3f}, p = {rt['p_value_one_sided']:.4f} {'*' if rt['significant'] else ''}")
        
        print(f"\n* = significant at α={self.alpha}")


def run_hypothesis_tests_all_models(models: list[str]) -> dict:
    """
    Run hypothesis tests for all models and compile results.
    
    Args:
        models: List of model short names
        
    Returns:
        Dictionary with results for each model
    """
    all_results = {}
    
    for model in models:
        print(f"\nProcessing {model}...")
        suite = HypothesisTestSuite(model)
        
        if suite.is_data_available():
            results = suite.run_all_tests()
            suite.print_summary()
            all_results[model] = results
        else:
            print(f"  Skipping {model}: Data not available")
            all_results[model] = {"error": "Data not available"}
    
    return all_results


if __name__ == "__main__":
    from config import MODELS
    
    print("Sycophancy Benchmark Hypothesis Testing Suite")
    print("=" * 60)
    
    # Check which models have data
    available_models = []
    for model in MODELS:
        suite = HypothesisTestSuite(model)
        if suite.is_data_available():
            available_models.append(model)
            print(f"  ✓ {model}")
        else:
            print(f"  ✗ {model} (no data)")
    
    if available_models:
        print(f"\nRunning tests for {len(available_models)} models...")
        results = run_hypothesis_tests_all_models(available_models)
        
        # Save combined results
        output_dir = ANALYSIS_OUTPUT_DIR / "hypothesis_tests"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        combined_path = output_dir / "all_models_hypothesis_tests.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nCombined results saved to: {combined_path}")
    else:
        print("\nNo models with complete data found.")
        print("Please run the evaluation scripts first to generate results.")
