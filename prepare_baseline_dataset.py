#!/usr/bin/env python3
"""
Prepare Baseline Dataset for Sycophancy Evaluation
===================================================
Creates a dataset where each sycophancy prompt is paired with a 
generic "You are a helpful assistant" system instruction.

This serves as the control condition to measure baseline sycophancy
rates without persona influence.

Output: datasets/baseline_sycophancy_dataset.json
"""

import json
import csv
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
PROMPTS_FILE = BASE_DIR / "sycophancy_prompts.csv"
OUTPUT_DIR = BASE_DIR / "datasets"
OUTPUT_FILE = OUTPUT_DIR / "baseline_sycophancy_dataset.json"

# Baseline system instruction
BASELINE_SYSTEM_INSTRUCTION = "You are a helpful assistant."

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def load_sycophancy_prompts(filepath: Path) -> list[dict]:
    """Load sycophancy prompts from CSV file."""
    prompts = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            prompts.append({
                "prompt_id": idx,
                "prompt_text": row["prompt"].strip(),
                "category": row["category"].strip()
            })
    return prompts


def create_baseline_dataset(prompts: list[dict]) -> dict:
    """
    Create baseline dataset with generic system instruction.
    
    Format:
    {
        "metadata": {...},
        "data": [
            {
                "id": "baseline_001",
                "prompt_id": 1,
                "category": "Technology and digital life",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "<sycophancy prompt>"}
                ]
            },
            ...
        ]
    }
    """
    dataset = {
        "metadata": {
            "name": "Baseline Sycophancy Evaluation Dataset",
            "description": "Sycophancy prompts with generic helpful assistant instruction (control condition)",
            "system_instruction": BASELINE_SYSTEM_INSTRUCTION,
            "total_prompts": len(prompts),
            "created_at": datetime.now().isoformat(),
            "source_file": str(PROMPTS_FILE.name)
        },
        "data": []
    }
    
    # Get unique categories for metadata
    categories = sorted(set(p["category"] for p in prompts))
    dataset["metadata"]["categories"] = categories
    dataset["metadata"]["num_categories"] = len(categories)
    
    # Create data entries
    for prompt in prompts:
        entry = {
            "id": f"baseline_{prompt['prompt_id']:04d}",
            "prompt_id": prompt["prompt_id"],
            "category": prompt["category"],
            "messages": [
                {"role": "system", "content": BASELINE_SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt["prompt_text"]}
            ]
        }
        dataset["data"].append(entry)
    
    return dataset


def main():
    """Main execution function."""
    print("=" * 70)
    print("Preparing Baseline Sycophancy Dataset")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    print(f"\nLoading prompts from: {PROMPTS_FILE}")
    prompts = load_sycophancy_prompts(PROMPTS_FILE)
    print(f"Loaded {len(prompts)} sycophancy prompts")
    
    # Create dataset
    print("\nCreating baseline dataset...")
    dataset = create_baseline_dataset(prompts)
    
    # Save dataset
    print(f"\nSaving dataset to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total entries: {len(dataset['data'])}")
    print(f"Categories: {dataset['metadata']['num_categories']}")
    print(f"System instruction: \"{BASELINE_SYSTEM_INSTRUCTION}\"")
    print(f"\nCategories breakdown:")
    
    # Count per category
    category_counts = {}
    for entry in dataset["data"]:
        cat = entry["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat in sorted(category_counts.keys()):
        print(f"  - {cat}: {category_counts[cat]} prompts")
    
    print(f"\nDataset saved to: {OUTPUT_FILE}")
    print("=" * 70)
    
    # Show sample entry
    print("\nSample entry:")
    print(json.dumps(dataset["data"][0], indent=2))


if __name__ == "__main__":
    main()
