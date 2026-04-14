#!/usr/bin/env python3
"""
Prepare Persona-Based Dataset for Sycophancy Evaluation
========================================================
Creates a dataset where each sycophancy prompt is paired with each persona
as the system instruction.

This creates the experimental condition to measure how persona agreeableness
affects sycophancy rates.

Output: datasets/persona_sycophancy_dataset.json

Total combinations: 275 personas × ~4951 prompts = ~1,361,525 entries
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
PERSONAS_FILE = BASE_DIR / "personas.txt"
OUTPUT_DIR = BASE_DIR / "datasets"
OUTPUT_FILE = OUTPUT_DIR / "persona_sycophancy_dataset.json"

# Template for persona system instruction
PERSONA_INSTRUCTION_TEMPLATE = "You are {persona_description}"

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


def load_personas(filepath: Path) -> list[dict]:
    """Load personas from text file, one per line."""
    personas = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                personas.append({
                    "persona_id": idx,
                    "persona_description": line
                })
    return personas


def create_persona_instruction(persona_description: str) -> str:
    """
    Create system instruction from persona description.
    
    Handles articles (a/an) that may already be in the persona description.
    """
    # Most personas already start with "A" or "An", so we use them directly
    # The template adds "You are " prefix
    return PERSONA_INSTRUCTION_TEMPLATE.format(persona_description=persona_description.lower())


def create_persona_dataset(prompts: list[dict], personas: list[dict]) -> dict:
    """
    Create persona-based dataset with each persona × prompt combination.
    
    Format:
    {
        "metadata": {...},
        "data": [
            {
                "id": "persona_001_prompt_0001",
                "persona_id": 1,
                "persona_description": "...",
                "prompt_id": 1,
                "category": "Technology and digital life",
                "messages": [
                    {"role": "system", "content": "You are a veteran private equity partner..."},
                    {"role": "user", "content": "<sycophancy prompt>"}
                ]
            },
            ...
        ]
    }
    """
    total_combinations = len(personas) * len(prompts)
    
    dataset = {
        "metadata": {
            "name": "Persona-Based Sycophancy Evaluation Dataset",
            "description": "Sycophancy prompts paired with persona-based system instructions (experimental condition)",
            "instruction_template": PERSONA_INSTRUCTION_TEMPLATE,
            "total_personas": len(personas),
            "total_prompts": len(prompts),
            "total_combinations": total_combinations,
            "created_at": datetime.now().isoformat(),
            "source_files": {
                "prompts": str(PROMPTS_FILE.name),
                "personas": str(PERSONAS_FILE.name)
            }
        },
        "data": []
    }
    
    # Get unique categories for metadata
    categories = sorted(set(p["category"] for p in prompts))
    dataset["metadata"]["categories"] = categories
    dataset["metadata"]["num_categories"] = len(categories)
    
    # Create data entries for each persona × prompt combination
    print(f"Creating {total_combinations:,} persona × prompt combinations...")
    
    entry_count = 0
    for persona in personas:
        persona_instruction = create_persona_instruction(persona["persona_description"])
        
        for prompt in prompts:
            entry = {
                "id": f"persona_{persona['persona_id']:03d}_prompt_{prompt['prompt_id']:04d}",
                "persona_id": persona["persona_id"],
                "persona_description": persona["persona_description"],
                "prompt_id": prompt["prompt_id"],
                "category": prompt["category"],
                "messages": [
                    {"role": "system", "content": persona_instruction},
                    {"role": "user", "content": prompt["prompt_text"]}
                ]
            }
            dataset["data"].append(entry)
            entry_count += 1
            
            # Progress indicator
            if entry_count % 100000 == 0:
                print(f"  Created {entry_count:,} / {total_combinations:,} entries...")
    
    return dataset


def main():
    """Main execution function."""
    print("=" * 70)
    print("Preparing Persona-Based Sycophancy Dataset")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    print(f"\nLoading prompts from: {PROMPTS_FILE}")
    prompts = load_sycophancy_prompts(PROMPTS_FILE)
    print(f"Loaded {len(prompts)} sycophancy prompts")
    
    # Load personas
    print(f"\nLoading personas from: {PERSONAS_FILE}")
    personas = load_personas(PERSONAS_FILE)
    print(f"Loaded {len(personas)} personas")
    
    # Calculate total combinations
    total = len(personas) * len(prompts)
    print(f"\nTotal combinations to create: {total:,}")
    
    # Create dataset
    print("\nCreating persona-based dataset...")
    dataset = create_persona_dataset(prompts, personas)
    
    # Save dataset
    print(f"\nSaving dataset to: {OUTPUT_FILE}")
    print("(This may take a moment due to file size...)")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total entries: {len(dataset['data']):,}")
    print(f"Personas: {dataset['metadata']['total_personas']}")
    print(f"Prompts: {dataset['metadata']['total_prompts']}")
    print(f"Categories: {dataset['metadata']['num_categories']}")
    print(f"Instruction template: \"{PERSONA_INSTRUCTION_TEMPLATE}\"")
    
    # Count per category
    print(f"\nPrompts per category:")
    category_counts = {}
    for prompt in prompts:
        cat = prompt["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat in sorted(category_counts.keys()):
        print(f"  - {cat}: {category_counts[cat]} prompts")
    
    # File size estimate
    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\nDataset file size: {file_size:.1f} MB")
    print(f"Dataset saved to: {OUTPUT_FILE}")
    print("=" * 70)
    
    # Show sample entries
    print("\nSample entry (first persona, first prompt):")
    print(json.dumps(dataset["data"][0], indent=2))
    
    print("\nSample entry (last persona, last prompt):")
    print(json.dumps(dataset["data"][-1], indent=2))


if __name__ == "__main__":
    main()
