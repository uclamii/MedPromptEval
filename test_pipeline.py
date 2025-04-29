"""
Test pipeline for the prompt generation and answer generation system.
This script provides testing functionality for:
1. Generating prompts using various models and configurations
2. Testing the full pipeline by generating a prompt and then answering a question
"""

import argparse
import pandas as pd
import random
from pathlib import Path
import os

from prompt_generation import PromptGenerator, ensure_hf_login
from answer_generation import AnswerGenerator
from config import PROMPT_MODEL_CONFIGS, ANSWER_MODEL_CONFIGS, PROMPT_TYPES

def test_prompt_generation(
    prompt_model_key: str,
    output_dir: str,
    num_prompts_per_type: int = 1,
    use_auth: bool = True
):
    """
    Test the prompt generation functionality
    
    Args:
        prompt_model_key: Key of the model to use for prompt generation
        output_dir: Directory to save the generated prompts
        num_prompts_per_type: Number of prompts to generate per type
        use_auth: Whether to use Hugging Face authentication
    """
    # Initialize the prompt generator
    generator = PromptGenerator(
        model_key=prompt_model_key,
        use_auth=use_auth
    )
    
    # Generate all prompts and save to files
    generator.generate_all_prompts(
        output_dir=output_dir,
        num_prompts_per_type=num_prompts_per_type
    )
    
    print(f"\nPrompt generation complete! Files saved to {output_dir} directory.")

def test_full_pipeline(
    prompt_model_key: str,
    answer_model_key: str,
    test_csv_path: str,
    prompt_type: str = None,
    num_samples: int = 1,
    prompts_per_type: int = 1,
    use_auth: bool = True
):
    """
    Test the full pipeline by generating a prompt and then answering a question
    
    Args:
        prompt_model_key: Key of the model to use for prompt generation
        answer_model_key: Key of the model to use for answering
        test_csv_path: Path to the CSV file with test questions
        prompt_type: Type of prompt to use (if None, a random one will be selected)
        num_samples: Number of questions to sample from the CSV
        prompts_per_type: Number of different prompts to generate for each prompt type
        use_auth: Whether to use Hugging Face authentication
    """
    # Load the test questions
    try:
        df = pd.read_csv(test_csv_path)
        if 'question' not in df.columns:
            raise ValueError("The CSV file must have a 'question' column")
        
        # Check for answer column to display expected answer
        has_answer = 'answer' in df.columns
    except Exception as e:
        print(f"Error loading test CSV: {e}")
        return
    
    # Sample questions
    if num_samples < len(df):
        sampled_df = df.sample(num_samples, random_state=42)
    else:
        sampled_df = df
        
    print(f"Loaded {len(sampled_df)} test questions from {test_csv_path}")
    
    # Select prompt type if not specified
    if prompt_type is None:
        prompt_type = random.choice(list(PROMPT_TYPES.keys()))
    
    print(f"Using prompt type: {prompt_type}")
    print(f"Generating {prompts_per_type} different prompt(s) per question")
    print(f"Prompt model: {PROMPT_MODEL_CONFIGS[prompt_model_key]['name']}")
    print(f"Answer model: {ANSWER_MODEL_CONFIGS[answer_model_key]['name']}")
    
    # Initialize the models
    prompt_generator = PromptGenerator(
        model_key=prompt_model_key,
        use_auth=use_auth
    )
    
    answer_generator = AnswerGenerator(
        model_key=answer_model_key,
        use_auth=use_auth
    )
    
    # Process each question
    for i, row in enumerate(sampled_df.itertuples(), 1):
        question = row.question
        expected_answer = row.answer if has_answer else "N/A"
        
        print(f"\n{'#'*100}")
        print(f"PROCESSING QUESTION {i}/{len(sampled_df)}")
        print(f"{'#'*100}")
        print(f"QUESTION: {question}")
        if has_answer:
            print(f"\n{'='*40} EXPECTED ANSWER {'='*40}")
            print(expected_answer)
            print(f"{'='*90}")
        
        # Generate prompts and answer one at a time
        for j in range(1, prompts_per_type + 1):
            print(f"\n{'*'*100}")
            print(f"PROMPT VARIATION {j}/{prompts_per_type} using {prompt_type} methodology")
            print(f"{'*'*100}")
            
            # Generate a single prompt
            print(f"Generating system prompt {j}...")
            system_prompt = prompt_generator.generate_prompt(
                prompt_type=prompt_type,
                num_prompts=1
            )[0]  # Take the first (and only) prompt
            
            # Immediately use this prompt to answer the question
            # The answer_generator will print the full system prompt, question, and answer
            print(f"\nUsing generated prompt to answer the question...")
            answer = answer_generator.generate_answer(
                system_prompt=system_prompt,
                question=question
            )
    
    print(f"\n{'#'*100}")
    print("FULL PIPELINE TEST COMPLETE!")
    print(f"{'#'*100}")

def main():
    parser = argparse.ArgumentParser(description='Test the prompt generation and answer generation pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Prompt generation testing
    prompt_parser = subparsers.add_parser('prompts', help='Generate prompts for medical QA')
    prompt_parser.add_argument('--model', type=str, default='phi-2',
                       choices=list(PROMPT_MODEL_CONFIGS.keys()),
                       help='Model to use for generation')
    prompt_parser.add_argument('--output_dir', type=str, default='prompts',
                       help='Directory to save generated prompts')
    prompt_parser.add_argument('--num_prompts', type=int, default=1,
                       help='Number of prompts to generate per type')
    prompt_parser.add_argument('--no_auth', action='store_true',
                       help='Do not use authentication with Hugging Face')
    
    # Full pipeline testing (prompt generation + answering)
    pipeline_parser = subparsers.add_parser('pipeline', help='Test full pipeline with prompt generation and answering')
    pipeline_parser.add_argument('--prompt_model', type=str, default='phi-2',
                         choices=list(PROMPT_MODEL_CONFIGS.keys()),
                         help='Model to use for prompt generation')
    pipeline_parser.add_argument('--answer_model', type=str, default='mistral-7b',
                         choices=list(ANSWER_MODEL_CONFIGS.keys()),
                         help='Model to use for answering')
    pipeline_parser.add_argument('--test_csv', type=str, default='datasets/test_data/medquad_test.csv',
                         help='Path to CSV with test questions')
    pipeline_parser.add_argument('--prompt_type', type=str, default=None,
                         choices=list(PROMPT_TYPES.keys()),
                         help='Type of prompt to use (default: random)')
    pipeline_parser.add_argument('--num_samples', type=int, default=1,
                         help='Number of questions to sample from CSV')
    pipeline_parser.add_argument('--prompts_per_type', type=int, default=1,
                         help='Number of different prompts to generate per prompt type')
    pipeline_parser.add_argument('--no_auth', action='store_true',
                         help='Do not use authentication with Hugging Face')
    
    args = parser.parse_args()
    
    # Ensure Hugging Face login if needed
    if not getattr(args, 'no_auth', False):
        ensure_hf_login()
    
    # Handle commands
    if args.command == 'prompts':
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Test prompt generation
        test_prompt_generation(
            prompt_model_key=args.model,
            output_dir=args.output_dir,
            num_prompts_per_type=args.num_prompts,
            use_auth=not args.no_auth
        )
    elif args.command == 'pipeline':
        # Test full pipeline
        test_full_pipeline(
            prompt_model_key=args.prompt_model,
            answer_model_key=args.answer_model,
            test_csv_path=args.test_csv,
            prompt_type=args.prompt_type,
            num_samples=args.num_samples,
            prompts_per_type=args.prompts_per_type,
            use_auth=not args.no_auth
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 