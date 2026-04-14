#!/usr/bin/env python3
"""
Data Loader for Sycophancy Benchmark Analysis
==============================================
Utilities for loading and preprocessing agreeableness and sycophancy results.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional

from config import (
    AGREEABLENESS_RESULTS_DIR,
    SYCOPHANCY_BASELINE_DIR,
    SYCOPHANCY_PERSONA_DIR,
    PERSONAS_FILE,
    AGREEABLENESS_FACETS,
    FACET_NAMES,
)


def load_personas() -> list[str]:
    """Load persona descriptions from file."""
    with open(PERSONAS_FILE, "r", encoding="utf-8") as f:
        personas = [line.strip() for line in f if line.strip()]
    return personas


def load_agreeableness_scores(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load agreeableness scores for a specific model.
    
    Args:
        model_name: Short name of the model (e.g., 'qwen3-0-6b')
        
    Returns:
        DataFrame with columns:
        - persona_index (1-indexed)
        - persona_description
        - overall_agreeableness (normalized 0-1)
        - A1_trust, A3_altruism, A4_cooperation, A6_sympathy (facet scores)
    """
    model_dir = AGREEABLENESS_RESULTS_DIR / model_name
    
    # Try consolidated CSV first
    csv_path = model_dir / "consolidated_scores.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        # Standardize column names to match expected format
        column_mapping = {
            "overall_agreeableness_normalized": "overall_agreeableness",
            "trust_normalized": "A1_trust",
            "altruism_normalized": "A3_altruism",
            "cooperation_normalized": "A4_cooperation",
            "sympathy_normalized": "A6_sympathy",
        }
        df = df.rename(columns=column_mapping)
        
        return df
    
    # Fall back to individual JSON files
    json_path = model_dir / "all_results.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        records = []
        for entry in data:
            record = {
                "persona_index": entry["persona_index"],
                "persona_description": entry.get("persona_description", ""),
                "overall_agreeableness": entry["scores"]["overall_normalized"],
            }
            # Add facet scores
            for facet_id in AGREEABLENESS_FACETS:
                facet_name = FACET_NAMES[facet_id].lower()
                if facet_id in entry["scores"]["facets"]:
                    record[f"{facet_id}_{facet_name}"] = entry["scores"]["facets"][facet_id]["normalized"]
            records.append(record)
        
        return pd.DataFrame(records)
    
    # Try loading from summary file
    summary_path = model_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        if "persona_scores" in summary:
            records = []
            for persona_id, scores in summary["persona_scores"].items():
                record = {
                    "persona_index": int(persona_id),
                    "overall_agreeableness": scores.get("overall_normalized", scores.get("overall", 0)),
                }
                for facet_id in AGREEABLENESS_FACETS:
                    if facet_id in scores:
                        facet_name = FACET_NAMES[facet_id].lower()
                        record[f"{facet_id}_{facet_name}"] = scores[facet_id]
                records.append(record)
            return pd.DataFrame(records)
    
    print(f"Warning: No agreeableness results found for {model_name}")
    return None


def load_baseline_sycophancy(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load baseline sycophancy scores (no persona condition).
    
    Args:
        model_name: Short name of the model
        
    Returns:
        DataFrame with columns:
        - prompt_id
        - category
        - sycophancy_score (0, 0.5, or 1)
        - stance (AGREE/DISAGREE/PARTIAL)
    """
    model_dir = SYCOPHANCY_BASELINE_DIR / model_name
    
    # Try consolidated_scores.csv first (actual output from benchmark_eval scripts)
    csv_path = model_dir / "consolidated_scores.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Standardize column names if needed
        if "extracted_stance" in df.columns and "stance" not in df.columns:
            df["stance"] = df["extracted_stance"]
        return df
    
    # Try results JSON
    results_path = model_dir / "results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        records = []
        results = data.get("results", data.get("data", data))
        if isinstance(results, list):
            for entry in results:
                records.append({
                    "prompt_id": entry.get("prompt_id", entry.get("id")),
                    "category": entry.get("category", ""),
                    "sycophancy_score": entry.get("sycophancy_score"),
                    "stance": entry.get("extracted_stance", entry.get("stance")),
                })
        return pd.DataFrame(records)
    
    # Try results.csv
    csv_path = model_dir / "results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    print(f"Warning: No baseline sycophancy results found for {model_name}")
    return None


def load_persona_sycophancy(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load persona-based sycophancy scores.
    
    Args:
        model_name: Short name of the model
        
    Returns:
        DataFrame with columns:
        - persona_index
        - prompt_id
        - category
        - sycophancy_score
        - stance
    """
    model_dir = SYCOPHANCY_PERSONA_DIR / model_name
    
    # Try consolidated_scores.csv first (actual output from benchmark_eval scripts)
    csv_path = model_dir / "consolidated_scores.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Standardize column names if needed
        if "persona_id" in df.columns and "persona_index" not in df.columns:
            df["persona_index"] = df["persona_id"]
        if "extracted_stance" in df.columns and "stance" not in df.columns:
            df["stance"] = df["extracted_stance"]
        return df
    
    # Try results JSON
    results_path = model_dir / "results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        records = []
        results = data.get("results", data.get("data", data))
        if isinstance(results, list):
            for entry in results:
                records.append({
                    "persona_index": entry.get("persona_index", entry.get("persona_id")),
                    "prompt_id": entry.get("prompt_id"),
                    "category": entry.get("category", ""),
                    "sycophancy_score": entry.get("sycophancy_score"),
                    "stance": entry.get("extracted_stance", entry.get("stance")),
                })
        return pd.DataFrame(records)
    
    # Try results.csv
    csv_path = model_dir / "results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    # Try per-persona files
    persona_files = list(model_dir.glob("persona_*.json"))
    if persona_files:
        records = []
        for pf in persona_files:
            with open(pf, "r", encoding="utf-8") as f:
                pdata = json.load(f)
            persona_idx = pdata.get("persona_index", int(pf.stem.split("_")[1]))
            for entry in pdata.get("results", []):
                records.append({
                    "persona_index": persona_idx,
                    "prompt_id": entry.get("prompt_id"),
                    "category": entry.get("category", ""),
                    "sycophancy_score": entry.get("sycophancy_score"),
                    "stance": entry.get("extracted_stance", entry.get("stance")),
                })
        return pd.DataFrame(records)
    
    print(f"Warning: No persona sycophancy results found for {model_name}")
    return None


def load_persona_sycophancy_summary(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load aggregated sycophancy scores per persona.
    
    Args:
        model_name: Short name of the model
        
    Returns:
        DataFrame with columns:
        - persona_index
        - mean_sycophancy_score
        - std_sycophancy_score
        - agree_count, disagree_count, partial_count
    """
    model_dir = SYCOPHANCY_PERSONA_DIR / model_name
    
    # Try consolidated_scores.csv first (actual output from benchmark_eval persona scripts)
    # This file contains per-persona summary with mean_sycophancy_score
    csv_path = model_dir / "consolidated_scores.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Standardize column names if needed
        if "persona_id" in df.columns and "persona_index" not in df.columns:
            df["persona_index"] = df["persona_id"]
        return df
    
    # Try summary_statistics.json (actual output from benchmark_eval scripts)
    summary_path = model_dir / "summary_statistics.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        if "by_persona" in summary:
            records = []
            for persona_id, stats in summary["by_persona"].items():
                records.append({
                    "persona_index": int(persona_id),
                    "persona_description": stats.get("persona_description", ""),
                    "mean_sycophancy_score": stats.get("mean_sycophancy", stats.get("mean_score", 0)),
                    "valid_responses": stats.get("valid_responses", 0),
                    "total_prompts": stats.get("total_prompts", 0),
                })
            return pd.DataFrame(records)
    
    # Try legacy summary.json
    summary_path = model_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        if "persona_summaries" in summary:
            records = []
            for persona_id, stats in summary["persona_summaries"].items():
                records.append({
                    "persona_index": int(persona_id),
                    "mean_sycophancy_score": stats.get("mean_score", stats.get("mean")),
                    "std_sycophancy_score": stats.get("std_score", stats.get("std", 0)),
                    "agree_count": stats.get("agree_count", 0),
                    "disagree_count": stats.get("disagree_count", 0),
                    "partial_count": stats.get("partial_count", 0),
                })
            return pd.DataFrame(records)
    
    # Fall back to computing from raw results
    raw_df = load_persona_sycophancy(model_name)
    if raw_df is not None and not raw_df.empty:
        summary_df = raw_df.groupby("persona_index").agg(
            mean_sycophancy_score=("sycophancy_score", "mean"),
            std_sycophancy_score=("sycophancy_score", "std"),
            total_count=("sycophancy_score", "count"),
        ).reset_index()
        
        # Count stances
        stance_counts = raw_df.groupby(["persona_index", "stance"]).size().unstack(fill_value=0)
        stance_counts.columns = [f"{c.lower().replace(' ', '_')}_count" for c in stance_counts.columns]
        
        summary_df = summary_df.merge(stance_counts, on="persona_index", how="left")
        return summary_df
    
    print(f"Warning: Could not compute persona sycophancy summary for {model_name}")
    return None


def merge_agreeableness_sycophancy(
    agreeableness_df: pd.DataFrame,
    sycophancy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge agreeableness and sycophancy data on persona_index.
    
    Args:
        agreeableness_df: DataFrame with agreeableness scores
        sycophancy_df: DataFrame with sycophancy scores (per-persona summary)
        
    Returns:
        Merged DataFrame with both agreeableness and sycophancy metrics
    """
    merged = agreeableness_df.merge(
        sycophancy_df,
        on="persona_index",
        how="inner",
    )
    return merged


def load_all_model_data(models: list[str]) -> dict:
    """
    Load all data for multiple models.
    
    Args:
        models: List of model short names
        
    Returns:
        Dictionary with structure:
        {
            model_name: {
                "agreeableness": DataFrame,
                "baseline_sycophancy": DataFrame,
                "persona_sycophancy": DataFrame,
                "persona_sycophancy_summary": DataFrame,
                "merged": DataFrame (agreeableness + sycophancy per persona)
            }
        }
    """
    all_data = {}
    
    for model in models:
        print(f"Loading data for {model}...")
        
        agreeableness = load_agreeableness_scores(model)
        baseline = load_baseline_sycophancy(model)
        persona = load_persona_sycophancy(model)
        persona_summary = load_persona_sycophancy_summary(model)
        
        merged = None
        if agreeableness is not None and persona_summary is not None:
            merged = merge_agreeableness_sycophancy(agreeableness, persona_summary)
        
        all_data[model] = {
            "agreeableness": agreeableness,
            "baseline_sycophancy": baseline,
            "persona_sycophancy": persona,
            "persona_sycophancy_summary": persona_summary,
            "merged": merged,
        }
    
    return all_data


if __name__ == "__main__":
    # Test data loading
    from config import MODELS
    
    print("Testing data loader...")
    print(f"Looking for results in:")
    print(f"  Agreeableness: {AGREEABLENESS_RESULTS_DIR}")
    print(f"  Baseline Sycophancy: {SYCOPHANCY_BASELINE_DIR}")
    print(f"  Persona Sycophancy: {SYCOPHANCY_PERSONA_DIR}")
    print()
    
    # Check which models have data
    for model in MODELS:
        agree_dir = AGREEABLENESS_RESULTS_DIR / model
        base_dir = SYCOPHANCY_BASELINE_DIR / model
        persona_dir = SYCOPHANCY_PERSONA_DIR / model
        
        status = []
        if agree_dir.exists():
            status.append("A")
        if base_dir.exists():
            status.append("B")
        if persona_dir.exists():
            status.append("P")
        
        if status:
            print(f"  {model}: {', '.join(status)}")
        else:
            print(f"  {model}: (no data yet)")
