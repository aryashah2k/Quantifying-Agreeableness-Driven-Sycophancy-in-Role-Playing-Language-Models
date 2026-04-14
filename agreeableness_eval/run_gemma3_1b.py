#!/usr/bin/env python3
"""
Agreeableness Questionnaire Inference Script for Google Gemma 3 1B Instruct
===========================================================================
Optimized for RTX A6000 (48GB VRAM).
Uses SDPA attention, dynamic KV-cache, and bfloat16.

Author: Arya
Model: google/gemma-3-1b-it

Note: Gemma 3 uses a specific message format with nested content structure.
Requires transformers >= 4.50.0.
"""

import os
import json
import re
import csv
import torch
import gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_ID = "google/gemma-3-1b-it"
MODEL_SHORT_NAME = "gemma-3-1b-it"

# Paths
BASE_DIR = Path(__file__).parent
PERSONAS_FILE = BASE_DIR / "personas.txt"
QUESTIONNAIRE_FILE = BASE_DIR / "sycophancy_questionnaire.json"
RESULTS_DIR = BASE_DIR / "results" / MODEL_SHORT_NAME

# Generation settings
MAX_NEW_TOKENS = 150  # Short responses expected (number + justification)
DO_SAMPLE = False     # Greedy decoding for reproducible results

# Batch size for inference (adjust based on VRAM)
# 1B model is very small (~2GB), can use large batches
BATCH_SIZE = 32

# GPU selection (change if needed)
GPU_DEVICE = "cuda:0"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_personas(filepath: Path) -> list[str]:
    """Load personas from text file, one per line."""
    with open(filepath, "r", encoding="utf-8") as f:
        personas = [line.strip() for line in f if line.strip()]
    return personas


def load_questionnaire(filepath: Path) -> dict:
    """Load the sycophancy questionnaire JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_score(response: str) -> int | None:
    """
    Extract the 1-5 score from model response.
    Returns None if no valid score found.
    """
    patterns = [
        r"^[\s]*([1-5])[\s\.\:\,\-–—]",  # Score at beginning (with various separators)
        r"^[\s]*([1-5])\s",  # Score at beginning with space
        r"(?:rating|score|answer)[\s\:\=]*([1-5])",  # Labeled score
        r"\b([1-5])\s*(?:out of 5|\/5)",  # "X out of 5" or "X/5"
        r"^([1-5])$",  # Just the number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1))
    
    # Fallback: first standalone digit 1-5 in response
    fallback_match = re.search(r'\b([1-5])\b', response)
    if fallback_match:
        return int(fallback_match.group(1))
    
    return None


def reverse_score(score: int) -> int:
    """Reverse score for negatively keyed items: 5->1, 4->2, 3->3, 2->4, 1->5"""
    return 6 - score


def build_prompt(persona: str, question_text: str, prompt_template: dict) -> list[dict]:
    """
    Build chat messages for Gemma 3 model.
    Gemma 3 uses nested content structure: {"type": "text", "text": "..."}
    """
    system_prompt = prompt_template["instruction"]
    user_prompt = prompt_template["format"].format(
        persona_description=persona,
        question_text=question_text
    )
    
    # Gemma 3 message format with nested content
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user", 
            "content": [{"type": "text", "text": user_prompt}]
        }
    ]


def calculate_facet_scores(responses: list[dict], questionnaire: dict) -> dict:
    """
    Calculate normalized scores for each facet and overall agreeableness.
    Returns dict with facet scores and overall score.
    """
    facet_scores = {}
    all_scores = []
    
    for facet in questionnaire["facets"]:
        facet_id = facet["id"]
        facet_name = facet["name"]
        facet_raw_scores = []
        
        for question in facet["questions"]:
            q_id = question["id"]
            keying = question["keying"]
            
            # Find the response for this question
            for resp in responses:
                if resp["question_id"] == q_id:
                    score = resp.get("extracted_score")
                    if score is not None:
                        # Apply reverse scoring for negative items
                        if keying == "negative":
                            score = reverse_score(score)
                        facet_raw_scores.append(score)
                    break
        
        if facet_raw_scores:
            # Raw sum and normalized score (0-1 scale)
            raw_sum = sum(facet_raw_scores)
            max_possible = len(facet_raw_scores) * 5
            min_possible = len(facet_raw_scores) * 1
            normalized = (raw_sum - min_possible) / (max_possible - min_possible)
            
            facet_scores[facet_id] = {
                "name": facet_name,
                "raw_sum": raw_sum,
                "count": len(facet_raw_scores),
                "normalized": round(normalized, 4)
            }
            all_scores.extend(facet_raw_scores)
    
    # Overall agreeableness score
    if all_scores:
        overall_raw = sum(all_scores)
        max_overall = len(all_scores) * 5
        min_overall = len(all_scores) * 1
        overall_normalized = (overall_raw - min_overall) / (max_overall - min_overall)
    else:
        overall_raw = 0
        overall_normalized = 0.0
    
    return {
        "facets": facet_scores,
        "overall_raw": overall_raw,
        "overall_normalized": round(overall_normalized, 4),
        "total_questions": len(all_scores)
    }


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer():
    """
    Load Google Gemma 3 1B Instruct with optimizations:
    - bfloat16 precision
    - SDPA attention
    - Dynamic KV-cache
    """
    print(f"Loading model: {MODEL_ID}")
    print(f"Target device: {GPU_DEVICE}")
    gpu_idx = int(GPU_DEVICE.split(':')[1])
    print(f"VRAM Available: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9:.1f} GB")
    
    # Enable cudnn benchmark for faster operations
    torch.backends.cudnn.benchmark = True
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        padding_side="left",  # Important for batch generation
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=GPU_DEVICE,
        attn_implementation="sdpa",
    )
    
    # Clear generation config to avoid stale parameters
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model dtype: {model.dtype}")
    
    return model, tokenizer


# ============================================================================
# INFERENCE
# ============================================================================

def generate_single(model, tokenizer, messages: list[dict]) -> str:
    """Generate response for a single prompt."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    attention_mask = torch.ones_like(input_ids)
    input_length = input_ids.shape[1]
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    # Decode only the generated part
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def generate_batch(model, tokenizer, batch_messages: list[list[dict]]) -> list[str]:
    """Generate responses for a batch of prompts."""
    # Apply chat template directly with tokenization
    input_ids_list = []
    for messages in batch_messages:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )[0]  # Remove batch dim
        input_ids_list.append(ids)
    
    # Pad sequences (left padding)
    max_len = max(ids.shape[0] for ids in input_ids_list)
    padded_ids = []
    attention_masks = []
    
    for ids in input_ids_list:
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            padded = torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype), ids])
            mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(ids.shape[0], dtype=torch.long)])
        else:
            padded = ids
            mask = torch.ones(ids.shape[0], dtype=torch.long)
        padded_ids.append(padded)
        attention_masks.append(mask)
    
    input_ids = torch.stack(padded_ids).to(model.device)
    attention_mask = torch.stack(attention_masks).to(model.device)
    input_length = input_ids.shape[1]
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    # Decode only the generated part for each sequence
    responses = []
    for output in outputs:
        generated_ids = output[input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(response.strip())
    
    return responses


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_inference():
    """Main inference pipeline."""
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_responses_dir = RESULTS_DIR / "raw_responses"
    raw_responses_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading personas and questionnaire...")
    personas = load_personas(PERSONAS_FILE)
    questionnaire = load_questionnaire(QUESTIONNAIRE_FILE)
    prompt_template = questionnaire["prompt_template"]
    
    # Flatten questions with metadata
    questions = []
    for facet in questionnaire["facets"]:
        for q in facet["questions"]:
            questions.append({
                "id": q["id"],
                "text": q["text"],
                "keying": q["keying"],
                "facet_id": facet["id"],
                "facet_name": facet["name"]
            })
    
    print(f"Loaded {len(personas)} personas and {len(questions)} questions")
    print(f"Total inference calls: {len(personas) * len(questions):,}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Results storage
    all_persona_results = []
    
    # Process each persona
    persona_pbar = tqdm(personas, desc="Personas", position=0)
    
    for persona_idx, persona in enumerate(persona_pbar):
        persona_pbar.set_description(f"Persona {persona_idx + 1}/{len(personas)}")
        
        persona_responses = []
        
        # Process questions in batches with inner progress bar
        num_batches = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE
        question_pbar = tqdm(
            range(0, len(questions), BATCH_SIZE),
            desc="  Questions",
            position=1,
            leave=False,
            total=num_batches
        )
        
        for batch_start in question_pbar:
            batch_end = min(batch_start + BATCH_SIZE, len(questions))
            batch_questions = questions[batch_start:batch_end]
            
            # Update progress bar description
            question_pbar.set_postfix({"batch": f"{batch_start//BATCH_SIZE + 1}/{num_batches}"})
            
            # Build prompts for batch
            batch_messages = [
                build_prompt(persona, q["text"], prompt_template)
                for q in batch_questions
            ]
            
            # Generate responses
            if len(batch_messages) == 1:
                responses = [generate_single(model, tokenizer, batch_messages[0])]
            else:
                responses = generate_batch(model, tokenizer, batch_messages)
            
            # Process responses
            for q, response in zip(batch_questions, responses):
                extracted_score = extract_score(response)
                
                persona_responses.append({
                    "question_id": q["id"],
                    "question_text": q["text"],
                    "keying": q["keying"],
                    "facet_id": q["facet_id"],
                    "facet_name": q["facet_name"],
                    "raw_response": response,
                    "extracted_score": extracted_score
                })
        
        question_pbar.close()
        
        # Calculate scores for this persona
        scores = calculate_facet_scores(persona_responses, questionnaire)
        
        # Save raw responses for this persona
        persona_filename = f"persona_{persona_idx + 1:03d}.json"
        persona_result = {
            "persona_index": persona_idx + 1,
            "persona_description": persona,
            "responses": persona_responses,
            "scores": scores
        }
        
        with open(raw_responses_dir / persona_filename, "w", encoding="utf-8") as f:
            json.dump(persona_result, f, indent=2, ensure_ascii=False)
        
        # Store summary for consolidated CSV
        all_persona_results.append({
            "persona_index": persona_idx + 1,
            "persona_description": persona,
            "scores": scores
        })
        
        # Clear CUDA cache periodically
        if (persona_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Generate consolidated CSV
    print("\nGenerating consolidated results CSV...")
    csv_path = RESULTS_DIR / "consolidated_scores.csv"
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        header = [
            "persona_index",
            "persona_description",
            "overall_agreeableness_normalized",
            "overall_raw_score",
            "total_questions_answered",
            "A1_Trust_normalized",
            "A1_Trust_raw",
            "A3_Altruism_normalized",
            "A3_Altruism_raw",
            "A4_Cooperation_normalized",
            "A4_Cooperation_raw",
            "A6_Sympathy_normalized",
            "A6_Sympathy_raw"
        ]
        writer.writerow(header)
        
        # Data rows
        for result in all_persona_results:
            scores = result["scores"]
            facets = scores["facets"]
            
            row = [
                result["persona_index"],
                result["persona_description"],
                scores["overall_normalized"],
                scores["overall_raw"],
                scores["total_questions"],
            ]
            
            # Add facet scores in order
            for facet_id in ["A1", "A3", "A4", "A6"]:
                if facet_id in facets:
                    row.extend([facets[facet_id]["normalized"], facets[facet_id]["raw_sum"]])
                else:
                    row.extend([None, None])
            
            writer.writerow(row)
    
    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {MODEL_ID}")
    print(f"Personas processed: {len(personas)}")
    print(f"Questions per persona: {len(questions)}")
    print(f"Total inferences: {len(personas) * len(questions):,}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - Raw responses: {raw_responses_dir}")
    print(f"  - Consolidated CSV: {csv_path}")


if __name__ == "__main__":
    # Set environment variables for optimization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Suppress torch dynamo errors
    torch._dynamo.config.suppress_errors = True
    
    run_inference()
