#!/usr/bin/env python
"""
QA Evaluation Script

This script:
1. Loads a CSV file containing medical question-answer pairs
2. Generates system prompts on-the-fly using a prompt model
3. Has a separate LLM answer the questions based on these generated prompts
4. Outputs a CSV with question, correct answer, prompt info, and LLM answer
"""

import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional

from prompt_generation import PromptGenerator, ensure_hf_login
from config import PROMPT_MODEL_CONFIGS, ANSWER_MODEL_CONFIGS, PROMPT_TYPES

def load_qa_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset containing question-answer pairs.
    
    Args:
        dataset_path: Path to the CSV file
        
    Returns:
        DataFrame with at least 'question' and 'answer' columns
    """
    # Load the CSV file
    df = pd.read_csv(dataset_path)
    
    # Verify the necessary columns exist
    required_columns = ['question', 'answer']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset")
    
    print(f"Loaded {len(df)} question-answer pairs from {dataset_path}")
    
    return df

def generate_and_answer(
    df: pd.DataFrame,
    prompt_model_key: str,
    answering_model_key: str,
    prompt_types: List[str] = None,
    prompts_per_type: int = 1,
    num_samples: int = None,
    use_auth: bool = True
) -> pd.DataFrame:
    """
    Generate prompts and answers for questions using the specified models.
    
    Args:
        df: DataFrame containing question-answer pairs
        prompt_model_key: Key of the model to use for generating prompts
        answering_model_key: Key of the model to use for answering
        prompt_types: List of prompt types to use (None for all)
        prompts_per_type: Number of prompts to generate per type
        num_samples: Number of samples to process (None for all)
        use_auth: Whether to use Hugging Face authentication
        
    Returns:
        DataFrame with original data and generated answers
    """
    # Limit the number of samples if specified
    if num_samples is not None and num_samples < len(df):
        df = df.sample(num_samples, random_state=42).reset_index(drop=True)
    
    # Use all prompt types if none specified
    if prompt_types is None:
        prompt_types = list(PROMPT_TYPES.keys())
    
    # Initialize the prompt generator model
    print(f"Initializing {prompt_model_key} for prompt generation...")
    prompt_generator = PromptGenerator(
        model_key=prompt_model_key,
        use_auth=use_auth
    )
    
    # Initialize the answering model
    print(f"Initializing {answering_model_key} for answer generation...")
    answer_generator = PromptGenerator(
        model_key=answering_model_key,
        use_auth=use_auth
    )
    
    # Override the MODEL_CONFIGS for the answer generator to use ANSWER_MODEL_CONFIGS
    answer_generator.model_config = ANSWER_MODEL_CONFIGS[answering_model_key]
    
    # Create a list to store results
    results = []
    
    # Process each question-answer pair
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = row['question']
        correct_answer = row['answer']
        
        # Generate and use prompts for each type
        for prompt_type in tqdm(prompt_types, desc=f"Generating prompts for Q{idx}", leave=False):
            print(f"\nGenerating {prompts_per_type} prompts for question {idx+1}/{len(df)} using {prompt_type}...")
            
            # Generate prompts for this question and prompt type
            system_prompts = prompt_generator.generate_prompt(
                prompt_type=prompt_type,
                num_prompts=prompts_per_type
            )
            
            # Use each generated prompt to answer the question
            for i, system_prompt in enumerate(system_prompts):
                print(f"Using prompt {i+1}/{len(system_prompts)} to answer the question...")
                
                # Create the full prompt by combining system prompt and question
                full_prompt = f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"
                
                # Tokenize the prompt
                inputs = answer_generator.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(answer_generator.device)
                
                # Generate the answer with ANSWER_MODEL_CONFIGS parameters
                generation_config = {
                    "max_new_tokens": ANSWER_MODEL_CONFIGS[answering_model_key]["max_new_tokens"],
                    "do_sample": True,
                    "temperature": ANSWER_MODEL_CONFIGS[answering_model_key]["temperature"],
                    "top_p": ANSWER_MODEL_CONFIGS[answering_model_key]["top_p"],
                    "top_k": ANSWER_MODEL_CONFIGS[answering_model_key]["top_k"],
                    "repetition_penalty": ANSWER_MODEL_CONFIGS[answering_model_key]["repetition_penalty"],
                    "pad_token_id": answer_generator.tokenizer.pad_token_id,
                    "eos_token_id": answer_generator.tokenizer.eos_token_id
                }
                
                # Generate answer
                with torch.no_grad():
                    outputs = answer_generator.model.generate(**inputs, **generation_config)
                
                # Decode the answer
                generated_text = answer_generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the answer (everything after "Answer:")
                llm_answer = generated_text.split("Answer:", 1)[-1].strip()
                
                # Store the result
                results.append({
                    "question": question,
                    "correct_answer": correct_answer,
                    "prompt_llm": PROMPT_MODEL_CONFIGS[prompt_model_key]["name"],
                    "prompt_type": prompt_type,
                    "system_prompt": system_prompt,
                    "answering_llm": ANSWER_MODEL_CONFIGS[answering_model_key]["name"],
                    "llm_answer": llm_answer
                })
                
                # Print a preview of the answer
                print(f"Answer preview: {llm_answer[:100]}..." if len(llm_answer) > 100 else f"Answer: {llm_answer}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate question answering with on-the-fly prompt generation')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the CSV dataset with question-answer pairs')
    parser.add_argument('--prompt-model', type=str, default='phi-2',
                        choices=list(PROMPT_MODEL_CONFIGS.keys()),
                        help='Model to use for generating prompts')
    parser.add_argument('--answer-model', type=str, default='mistral-7b',
                        choices=list(ANSWER_MODEL_CONFIGS.keys()),
                        help='Model to use for answering questions')
    parser.add_argument('--prompt-types', type=str, nargs='+',
                        choices=list(PROMPT_TYPES.keys()), default=None,
                        help='Prompt types to use (default: all)')
    parser.add_argument('--prompts-per-type', type=int, default=1,
                        help='Number of prompts to generate per type')
    parser.add_argument('--output', type=str, default='qa_results.csv',
                        help='Path for the output CSV file')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of question samples to process (default: all)')
    parser.add_argument('--no-auth', action='store_true',
                        help='Do not use Hugging Face authentication')
    
    args = parser.parse_args()
    
    # Ensure Hugging Face login if authentication is enabled
    if not args.no_auth:
        ensure_hf_login()
    
    # Load the dataset
    df = load_qa_dataset(args.dataset)
    
    # Generate prompts and answers
    results_df = generate_and_answer(
        df=df,
        prompt_model_key=args.prompt_model,
        answering_model_key=args.answer_model,
        prompt_types=args.prompt_types,
        prompts_per_type=args.prompts_per_type,
        num_samples=args.samples,
        use_auth=not args.no_auth
    )
    
    # Save results to CSV
    results_df.to_csv(args.output, index=False)
    
    # Calculate some statistics
    question_count = len(df) if args.samples is None else min(args.samples, len(df))
    prompt_type_count = len(args.prompt_types) if args.prompt_types else len(PROMPT_TYPES)
    expected_rows = question_count * prompt_type_count * args.prompts_per_type
    
    print(f"\nEvaluation complete! Results saved to {args.output}")
    print(f"Processed {question_count} questions with {prompt_type_count} prompt types")
    print(f"Generated {len(results_df)} rows (expected: {expected_rows})")
    print(f"Used {args.prompt_model} for prompts and {args.answer_model} for answers")

if __name__ == "__main__":
    main() 