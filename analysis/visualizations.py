#!/usr/bin/env python3
"""
Visualization Suite for Sycophancy Benchmark Analysis
======================================================
Generate publication-quality plots for agreeableness-sycophancy analysis.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional
from scipy import stats

from config import (
    ANALYSIS_OUTPUT_DIR,
    AGREEABLENESS_FACETS,
    FACET_NAMES,
    FIG_SIZE_SMALL,
    FIG_SIZE_MEDIUM,
    FIG_SIZE_LARGE,
    FIG_SIZE_WIDE,
    COLORS,
    MODEL_DISPLAY_NAMES,
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
    linear_regression_slope,
    split_by_agreeableness,
    trait_truthfulness_gap,
    compute_ttg_for_dataframe,
    ttg_summary_statistics,
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class SycophancyVisualizer:
    """Generate visualizations for sycophancy analysis."""
    
    def __init__(self, model_name: str, output_dir: Optional[Path] = None):
        """
        Initialize visualizer for a specific model.
        
        Args:
            model_name: Short name of the model
            output_dir: Directory to save plots
        """
        self.model_name = model_name
        self.display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        if output_dir is None:
            self.output_dir = ANALYSIS_OUTPUT_DIR / "plots" / model_name
        else:
            self.output_dir = output_dir / model_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Save figure in both PNG and PDF formats for publication readiness.
        
        Args:
            fig: matplotlib Figure object
            filename: Base filename without extension
        """
        # Save PNG (high resolution for web/preview)
        png_path = self.output_dir / f"{filename}.png"
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {png_path}")
        
        # Save PDF (vector format for publication)
        pdf_path = self.output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"Saved: {pdf_path}")
    
    def plot_scatter_with_regression(self, save: bool = True) -> Optional[plt.Figure]:
        """
        Scatter plot of agreeableness vs sycophancy with regression line.
        
        This is the primary visualization showing the relationship.
        """
        if not self.is_data_available():
            print(f"Data not available for {self.model_name}")
            return None
        
        fig, ax = plt.subplots(figsize=FIG_SIZE_MEDIUM)
        
        x = self.merged_df["overall_agreeableness"].values
        y = self.merged_df["mean_sycophancy_score"].values
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=50, c=COLORS["neutral"], edgecolors='white', linewidth=0.5)
        
        # Regression line
        slope, intercept, r_squared, p_value = linear_regression_slope(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = intercept + slope * x_line
        
        color = COLORS["correlation_positive"] if slope > 0 else COLORS["correlation_negative"]
        ax.plot(x_line, y_line, color=color, linewidth=2, label=f'Regression (R²={r_squared:.3f})')
        
        # Confidence band
        n = len(x)
        x_mean = np.mean(x)
        se_y = np.sqrt(np.sum((y - (intercept + slope * x))**2) / (n - 2))
        se_line = se_y * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
        t_crit = stats.t.ppf(0.975, n - 2)
        
        ax.fill_between(x_line, y_line - t_crit * se_line, y_line + t_crit * se_line,
                        alpha=0.2, color=color, label='95% CI')
        
        # Correlation annotation
        r, p = agreeableness_sycophancy_correlation(x, y)
        sig_marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        
        ax.annotate(
            f'r = {r:.3f}{sig_marker}\np = {p:.4f}',
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax.set_xlabel('Agreeableness Score (normalized)')
        ax.set_ylabel('Mean Sycophancy Score')
        ax.set_title(f'{self.display_name}: Agreeableness vs Sycophancy')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, "scatter_regression")
        
        return fig
    
    def plot_group_comparison(self, save: bool = True) -> Optional[plt.Figure]:
        """
        Box/violin plot comparing high vs low agreeableness groups.
        """
        if not self.is_data_available():
            return None
        
        fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
        
        # Split into groups
        high_df, low_df = split_by_agreeableness(
            self.merged_df,
            agreeableness_col="overall_agreeableness",
            method="median",
        )
        
        high_syc = high_df["mean_sycophancy_score"].values
        low_syc = low_df["mean_sycophancy_score"].values
        
        # Violin plot
        parts = ax.violinplot([low_syc, high_syc], positions=[1, 2], showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLORS["low_agreeableness"] if i == 0 else COLORS["high_agreeableness"])
            pc.set_alpha(0.7)
        
        # Add box plots inside
        bp = ax.boxplot([low_syc, high_syc], positions=[1, 2], widths=0.15, 
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], [COLORS["low_agreeableness"], COLORS["high_agreeableness"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
        
        # Labels and formatting
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'Low Agreeableness\n(n={len(low_df)})', 
                           f'High Agreeableness\n(n={len(high_df)})'])
        ax.set_ylabel('Mean Sycophancy Score')
        ax.set_title(f'{self.display_name}: Sycophancy by Agreeableness Group')
        
        # Add significance annotation
        from metrics import welch_ttest
        t, p = welch_ttest(high_syc, low_syc, alternative="greater")
        
        y_max = max(high_syc.max(), low_syc.max())
        ax.plot([1, 1, 2, 2], [y_max + 0.02, y_max + 0.04, y_max + 0.04, y_max + 0.02], 'k-', lw=1)
        
        sig_text = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(1.5, y_max + 0.05, sig_text, ha='center', fontsize=14)
        
        ax.set_ylim(0, min(1, y_max + 0.15))
        
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, "group_comparison")
        
        return fig
    
    def plot_facet_correlations(self, save: bool = True) -> Optional[plt.Figure]:
        """
        Bar chart showing correlation of each agreeableness facet with sycophancy.
        """
        if not self.is_data_available():
            return None
        
        fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
        
        sycophancy = self.merged_df["mean_sycophancy_score"].values
        
        facet_names = []
        correlations = []
        p_values = []
        
        for facet_id in AGREEABLENESS_FACETS:
            facet_col = f"{facet_id}_{FACET_NAMES[facet_id].lower()}"
            if facet_col in self.merged_df.columns:
                facet_values = self.merged_df[facet_col].values
                r, p = agreeableness_sycophancy_correlation(facet_values, sycophancy)
                
                facet_names.append(f"{FACET_NAMES[facet_id]}\n({facet_id})")
                correlations.append(r if not np.isnan(r) else 0)
                p_values.append(p if not np.isnan(p) else 1)
        
        # Add overall agreeableness
        r_overall, p_overall = agreeableness_sycophancy_correlation(
            self.merged_df["overall_agreeableness"].values, sycophancy
        )
        facet_names.append("Overall\nAgreeableness")
        correlations.append(r_overall)
        p_values.append(p_overall)
        
        # Create bars
        x = np.arange(len(facet_names))
        colors = [COLORS["correlation_positive"] if r > 0 else COLORS["correlation_negative"] for r in correlations]
        
        bars = ax.bar(x, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add significance markers
        for i, (bar, p) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if sig:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, sig,
                       ha='center', va='bottom', fontsize=12)
        
        ax.set_xticks(x)
        ax.set_xticklabels(facet_names, fontsize=10)
        ax.set_ylabel('Correlation with Sycophancy (r)')
        ax.set_title(f'{self.display_name}: Facet Correlations with Sycophancy')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylim(-0.5, 0.8)
        
        # Legend
        ax.text(0.02, 0.98, '* p<.05  ** p<.01  *** p<.001', transform=ax.transAxes,
               fontsize=9, verticalalignment='top')
        
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, "facet_correlations")
        
        return fig
    
    def plot_sycophancy_distribution(self, save: bool = True) -> Optional[plt.Figure]:
        """
        Histogram of sycophancy scores with baseline comparison.
        """
        if not self.is_data_available():
            return None
        
        fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
        
        persona_syc = self.merged_df["mean_sycophancy_score"].values
        
        # Histogram
        ax.hist(persona_syc, bins=30, alpha=0.7, color=COLORS["neutral"], 
                edgecolor='black', linewidth=0.5, label='Persona-based')
        
        # Baseline line
        if self.baseline_df is not None and "sycophancy_score" in self.baseline_df.columns:
            baseline_sr = sycophancy_rate(self.baseline_df["sycophancy_score"].values)
            ax.axvline(x=baseline_sr, color=COLORS["baseline"], linewidth=2, 
                      linestyle='--', label=f'Baseline ({baseline_sr:.3f})')
        
        # Mean line
        mean_syc = np.mean(persona_syc)
        ax.axvline(x=mean_syc, color=COLORS["high_agreeableness"], linewidth=2,
                  linestyle='-', label=f'Mean ({mean_syc:.3f})')
        
        ax.set_xlabel('Mean Sycophancy Score')
        ax.set_ylabel('Number of Personas')
        ax.set_title(f'{self.display_name}: Distribution of Sycophancy Scores')
        ax.legend()
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, "sycophancy_distribution")
        
        return fig
    
    def plot_agreeableness_distribution(self, save: bool = True) -> Optional[plt.Figure]:
        """
        Histogram of agreeableness scores.
        """
        if not self.is_data_available():
            return None
        
        fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
        
        agreeableness = self.merged_df["overall_agreeableness"].values
        
        ax.hist(agreeableness, bins=30, alpha=0.7, color=COLORS["neutral"],
                edgecolor='black', linewidth=0.5)
        
        # Median line
        median_a = np.median(agreeableness)
        ax.axvline(x=median_a, color='red', linewidth=2, linestyle='--',
                  label=f'Median ({median_a:.3f})')
        
        ax.set_xlabel('Agreeableness Score (normalized)')
        ax.set_ylabel('Number of Personas')
        ax.set_title(f'{self.display_name}: Distribution of Agreeableness Scores')
        ax.legend()
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, "agreeableness_distribution")
        
        return fig
    
    def plot_heatmap_by_category(self, save: bool = True) -> Optional[plt.Figure]:
        """
        Heatmap showing sycophancy by agreeableness quartile and prompt category.
        
        Note: Requires category information in the data.
        """
        # This requires more detailed data - placeholder for now
        print("Heatmap by category requires category-level data (not yet implemented)")
        return None
    
    def plot_ttg_quadrant(self, save: bool = True) -> Optional[plt.Figure]:
        """
        The "Money Plot" - TTG Quadrant Scatter Plot.
        
        Visualizes the Trait-Truthfulness Gap (TTG) showing the "Zone of Deception"
        where high-agreeableness personas sacrifice truth for social harmony.
        
        X-Axis: Normalized Agreeableness (A_p)
        Y-Axis: Sycophancy Rate (S_p)
        Color: TTG Score (gradient from blue=truthful to red=deceptive)
        """
        if not self.is_data_available():
            return None
        
        # Get baseline sycophancy rate
        baseline_sr = 0.5  # Default if no baseline
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
        
        fig, ax = plt.subplots(figsize=FIG_SIZE_MEDIUM)
        
        x = df["overall_agreeableness"].values
        y = df["mean_sycophancy_score"].values
        ttg = df["ttg"].values
        
        # Create scatter with TTG color gradient
        scatter = ax.scatter(
            x, y, 
            c=ttg, 
            cmap="RdYlBu_r",  # Red=high TTG (bad), Blue=low TTG (good)
            s=60, 
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
            vmin=-0.5,
            vmax=0.5,
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label="TTG Score")
        cbar.ax.set_ylabel("Trait-Truthfulness Gap (TTG)", rotation=270, labelpad=20)
        
        # Add baseline reference line
        ax.axhline(y=baseline_sr, color='green', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label=f'Baseline SR = {baseline_sr:.3f}')
        
        # Add quadrant lines at median
        median_a = np.median(x)
        ax.axvline(x=median_a, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # Highlight "Zone of Deception" (top-right quadrant)
        ax.fill_between([median_a, 1], [baseline_sr, baseline_sr], [1, 1],
                        alpha=0.1, color='red', label='Zone of Deception')
        
        # Highlight "Zone of Truthfulness" (bottom-left quadrant)
        ax.fill_between([0, median_a], [0, 0], [baseline_sr, baseline_sr],
                        alpha=0.1, color='blue', label='Zone of Truthfulness')
        
        # Labels and title
        ax.set_xlabel('Agreeableness Score (normalized)')
        ax.set_ylabel('Sycophancy Rate')
        ax.set_title(f'{self.display_name}: Trait-Truthfulness Gap (TTG) Analysis')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add TTG statistics annotation
        ttg_stats = ttg_summary_statistics(ttg)
        stats_text = (
            f"TTG Statistics:\n"
            f"Mean: {ttg_stats['mean']:.3f}\n"
            f"Deception Zone: {ttg_stats['pct_deception_zone']:.1f}%\n"
            f"Truthful Zone: {ttg_stats['pct_truthful_zone']:.1f}%"
        )
        ax.annotate(
            stats_text,
            xy=(0.98, 0.02), xycoords='axes fraction',
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, "ttg_quadrant_plot")
        
        return fig
    
    def plot_ttg_distribution(self, save: bool = True) -> Optional[plt.Figure]:
        """
        Histogram of TTG scores showing distribution across personas.
        """
        if not self.is_data_available():
            return None
        
        # Get baseline sycophancy rate
        baseline_sr = 0.5
        if self.baseline_df is not None and "sycophancy_score" in self.baseline_df.columns:
            baseline_sr = sycophancy_rate(self.baseline_df["sycophancy_score"].values)
        
        # Compute TTG
        df = self.merged_df.copy()
        df["ttg"] = compute_ttg_for_dataframe(
            df, baseline_sr,
            sycophancy_col="mean_sycophancy_score",
            agreeableness_col="overall_agreeableness",
        )
        
        ttg = df["ttg"].values
        
        fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
        
        # Create histogram with color coding
        n, bins, patches = ax.hist(ttg, bins=30, edgecolor='black', linewidth=0.5)
        
        # Color bars based on TTG value
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i+1]) / 2
            if bin_center > 0.1:
                patch.set_facecolor(COLORS["high_agreeableness"])  # Red for deception
            elif bin_center < -0.1:
                patch.set_facecolor(COLORS["low_agreeableness"])   # Blue for truthful
            else:
                patch.set_facecolor(COLORS["neutral"])             # Gray for neutral
        
        # Add reference lines
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, label='Baseline (TTG=0)')
        ax.axvline(x=np.mean(ttg), color='green', linestyle='--', linewidth=1.5, 
                   label=f'Mean TTG = {np.mean(ttg):.3f}')
        
        ax.set_xlabel('Trait-Truthfulness Gap (TTG)')
        ax.set_ylabel('Number of Personas')
        ax.set_title(f'{self.display_name}: TTG Distribution')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Add zone labels OUTSIDE the plot, below the x-axis
        # Use figure coordinates to place text in the margin area
        fig.text(0.15, 0.02, '← Truthful', 
                 fontsize=11, color=COLORS["low_agreeableness"], 
                 fontweight='bold', ha='center', va='bottom',
                 transform=fig.transFigure)
        
        fig.text(0.85, 0.02, 'Deceptive →', 
                 fontsize=11, color=COLORS["high_agreeableness"], 
                 fontweight='bold', ha='center', va='bottom',
                 transform=fig.transFigure)
        
        # Adjust bottom margin to make room for labels
        plt.subplots_adjust(bottom=0.15)
        
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, "ttg_distribution")
        
        return fig
    
    def generate_all_plots(self):
        """Generate all available plots."""
        print(f"\nGenerating plots for {self.display_name}...")
        
        if not self.is_data_available():
            print(f"  Data not available for {self.model_name}")
            return
        
        self.plot_scatter_with_regression()
        self.plot_group_comparison()
        self.plot_facet_correlations()
        self.plot_sycophancy_distribution()
        self.plot_agreeableness_distribution()
        self.plot_ttg_quadrant()  # The "Money Plot"
        self.plot_ttg_distribution()
        
        print(f"  All plots saved to: {self.output_dir}")


def plot_cross_model_comparison(models: list[str], save: bool = True) -> Optional[plt.Figure]:
    """
    Compare correlations and effect sizes across models.
    """
    model_names = []
    correlations = []
    effect_sizes = []
    
    for model in models:
        agreeableness_df = load_agreeableness_scores(model)
        persona_summary_df = load_persona_sycophancy_summary(model)
        
        if agreeableness_df is not None and persona_summary_df is not None:
            merged = merge_agreeableness_sycophancy(agreeableness_df, persona_summary_df)
            
            if len(merged) > 0:
                x = merged["overall_agreeableness"].values
                y = merged["mean_sycophancy_score"].values
                
                r, _ = agreeableness_sycophancy_correlation(x, y)
                
                high_df, low_df = split_by_agreeableness(merged)
                from metrics import sycophancy_effect_size
                d = sycophancy_effect_size(
                    high_df["mean_sycophancy_score"].values,
                    low_df["mean_sycophancy_score"].values
                )
                
                model_names.append(MODEL_DISPLAY_NAMES.get(model, model))
                correlations.append(r)
                effect_sizes.append(d)
    
    if not model_names:
        print("No models with data available for cross-model comparison")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)
    
    # Correlation plot
    ax1 = axes[0]
    x = np.arange(len(model_names))
    colors = [COLORS["correlation_positive"] if r > 0 else COLORS["correlation_negative"] for r in correlations]
    ax1.barh(x, correlations, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(x)
    ax1.set_yticklabels(model_names)
    ax1.set_xlabel('Correlation (r)')
    ax1.set_title('Agreeableness-Sycophancy Correlation')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    
    # Effect size plot
    ax2 = axes[1]
    colors = [COLORS["correlation_positive"] if d > 0 else COLORS["correlation_negative"] for d in effect_sizes]
    ax2.barh(x, effect_sizes, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(x)
    ax2.set_yticklabels(model_names)
    ax2.set_xlabel("Cohen's d")
    ax2.set_title('Effect Size (High vs Low Agreeableness)')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    
    # Add effect size guidelines
    for threshold, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
        ax2.axvline(x=threshold, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save:
        output_dir = ANALYSIS_OUTPUT_DIR / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PNG (high resolution for web/preview)
        png_path = output_dir / "cross_model_comparison.png"
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {png_path}")
        
        # Save PDF (vector format for publication)
        pdf_path = output_dir / "cross_model_comparison.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"Saved: {pdf_path}")
    
    return fig


if __name__ == "__main__":
    from config import MODELS
    
    print("Sycophancy Benchmark Visualization Suite")
    print("=" * 60)
    
    # Check which models have data
    available_models = []
    for model in MODELS:
        viz = SycophancyVisualizer(model)
        if viz.is_data_available():
            available_models.append(model)
            print(f"  ✓ {model}")
        else:
            print(f"  ✗ {model} (no data)")
    
    if available_models:
        print(f"\nGenerating plots for {len(available_models)} models...")
        
        for model in available_models:
            viz = SycophancyVisualizer(model)
            viz.generate_all_plots()
        
        # Cross-model comparison
        if len(available_models) > 1:
            print("\nGenerating cross-model comparison...")
            plot_cross_model_comparison(available_models)
        
        print("\nVisualization complete!")
    else:
        print("\nNo models with complete data found.")
        print("Please run the evaluation scripts first to generate results.")
