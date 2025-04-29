#!/usr/bin/env python
"""
Medical QA Evaluation Pipeline

This module provides a comprehensive end-to-end pipeline for:
1. Loading a CSV file containing medical question-answer pairs
2. Generating system prompts on-the-fly using a prompt generation model
3. Using an answer generation model to answer questions based on these prompts
4. Storing results in a CSV with detailed information about the entire process

The output CSV includes:
- question
- correct_answer (from the CSV)
- prompt_model (model used for generating the prompt)
- prompt_type
- system_prompt (the generated system prompt)
- answer_model (model used for answering)
- model_answer (answer from the LLM)
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from prompt_generation import PromptGenerator
from answer_generation import AnswerGenerator
from config import PROMPT_MODEL_CONFIGS, ANSWER_MODEL_CONFIGS, PROMPT_TYPES, ensure_hf_login

class MedicalQAPipeline:
    """
    End-to-end pipeline for medical question answering evaluation.
    """
    
    def __init__(
        self, 
        prompt_model: str = 'phi-2',
        answer_model: str = 'mistral-7b',
        prompt_types: List[str] = None,
        prompts_per_type: int = 1,
        use_auth: bool = True
    ):
        """
        Initialize the pipeline with specified models and parameters.
        
        Args:
            prompt_model: Model to use for generating system prompts
            answer_model: Model to use for answering questions
            prompt_types: List of prompt types to use (None for all)
            prompts_per_type: Number of prompts to generate per type
            use_auth: Whether to use Hugging Face authentication
        """
        # Set configuration
        self.prompt_model = prompt_model
        self.answer_model = answer_model
        self.prompt_types = prompt_types if prompt_types else list(PROMPT_TYPES.keys())
        self.prompts_per_type = prompts_per_type
        self.use_auth = use_auth
        
        # Ensure authentication if needed
        if self.use_auth:
            ensure_hf_login()
        
        # Initialize models
        print(f"Initializing {self.prompt_model} for prompt generation...")
        self.prompt_generator = PromptGenerator(
            model_key=self.prompt_model,
            use_auth=self.use_auth
        )
        
        print(f"Initializing {self.answer_model} for answer generation...")
        self.answer_generator = AnswerGenerator(
            model_key=self.answer_model,
            use_auth=self.use_auth
        )
        
    def load_dataset(self, dataset_path: str, num_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load a dataset containing question-answer pairs.
        
        Args:
            dataset_path: Path to the CSV file
            num_samples: Number of samples to process (None for all)
            
        Returns:
            DataFrame with questions and answers
        """
        # Load the CSV file
        df = pd.read_csv(dataset_path)
        
        # Verify the necessary columns exist
        required_columns = ['question', 'answer']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the dataset")
        
        # Sample if requested
        if num_samples is not None and num_samples < len(df):
            df = df.sample(num_samples, random_state=42).reset_index(drop=True)
        
        print(f"Loaded {len(df)} question-answer pairs from {dataset_path}")
        return df
    
    def run(self, dataset_path: str, output_path: str, num_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Execute the full pipeline.
        
        Args:
            dataset_path: Path to the CSV dataset
            output_path: Path for the output CSV file
            num_samples: Number of samples to process (None for all)
            
        Returns:
            DataFrame with evaluation results
        """
        # Print configuration
        self._print_config(dataset_path, output_path, num_samples)
        
        # Load the dataset
        df = self.load_dataset(dataset_path, num_samples)
        
        # Process data and generate results
        results = self._process_data(df)
        
        # Create output directory if it doesn't exist
        output_path_obj = Path(output_path)
        if not output_path_obj.parent.exists():
            os.makedirs(output_path_obj.parent, exist_ok=True)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        
        # Print summary
        self._print_summary(df, results_df, output_path)
        
        return results_df
    
    def _process_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        Process each question-answer pair in the dataset.
        
        Args:
            df: DataFrame containing question-answer pairs
            
        Returns:
            List of dictionaries with results
        """
        results = []
        
        # Process each question-answer pair
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
            question = row['question']
            correct_answer = row['answer']
            
            # Print question info
            print(f"\n{'#'*100}")
            print(f"PROCESSING QUESTION {idx+1}/{len(df)}")
            print(f"{'#'*100}")
            print(f"QUESTION: {question}")
            print(f"\n{'='*40} CORRECT ANSWER {'='*40}")
            print(correct_answer)
            print(f"{'='*90}")
            
            # Process each prompt type
            for prompt_type in tqdm(self.prompt_types, desc=f"Prompt types for Q{idx+1}", leave=False):
                print(f"\n{'*'*100}")
                print(f"GENERATING PROMPTS FOR TYPE: {prompt_type}")
                print(f"{'*'*100}")
                
                # Generate system prompts
                system_prompts = self.prompt_generator.generate_prompt(
                    prompt_type=prompt_type,
                    num_prompts=self.prompts_per_type
                )
                
                # Use each prompt to answer the question
                for i, system_prompt in enumerate(system_prompts):
                    print(f"\n{'-'*100}")
                    print(f"PROCESSING PROMPT VARIATION {i+1}/{len(system_prompts)} FOR {prompt_type}")
                    print(f"{'-'*100}")
                    
                    # Generate answer
                    print(f"\nGenerating answer using prompt variation {i+1}...")
                    model_answer = self.answer_generator.generate_answer(
                        system_prompt=system_prompt,
                        question=question
                    )
                    
                    # Store result
                    results.append({
                        "question": question,
                        "correct_answer": correct_answer,
                        "prompt_model": PROMPT_MODEL_CONFIGS[self.prompt_model]["name"],
                        "prompt_type": prompt_type,
                        "system_prompt": system_prompt,
                        "answer_model": ANSWER_MODEL_CONFIGS[self.answer_model]["name"],
                        "model_answer": model_answer
                    })
        
        return results
    
    def _print_config(self, dataset_path: str, output_path: str, num_samples: Optional[int]) -> None:
        """Print the pipeline configuration."""
        print(f"\n{'='*100}")
        print("MEDICAL QA EVALUATION PIPELINE CONFIGURATION")
        print(f"{'='*100}")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_path}")
        print(f"Prompt generation model: {self.prompt_model}")
        print(f"Answer generation model: {self.answer_model}")
        print(f"Prompt types: {self.prompt_types}")
        print(f"Prompts per type: {self.prompts_per_type}")
        print(f"Number of samples: {num_samples if num_samples else 'All'}")
        print(f"Using authentication: {self.use_auth}")
        print(f"{'='*100}\n")
    
    def _print_summary(self, df: pd.DataFrame, results_df: pd.DataFrame, output_path: str) -> None:
        """Print a summary of the pipeline results."""
        prompt_type_count = len(self.prompt_types)
        expected_rows = len(df) * prompt_type_count * self.prompts_per_type
        
        print(f"\n{'#'*100}")
        print("PIPELINE EXECUTION COMPLETE!")
        print(f"{'#'*100}")
        print(f"Results saved to: {output_path}")
        print(f"Processed {len(df)} questions")
        print(f"Used {prompt_type_count} prompt types ({', '.join(self.prompt_types)})")
        print(f"Generated {len(results_df)} result rows (expected: {expected_rows})")
        print(f"Used {self.prompt_model} for prompts and {self.answer_model} for answers")
        print(f"{'#'*100}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Medical QA Evaluation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to CSV dataset with question-answer pairs')
    parser.add_argument('--output', type=str, default='results/qa_results.csv',
                        help='Path for the output CSV file')
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
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of question samples to process (default: all)')
    parser.add_argument('--no-auth', action='store_true',
                        help='Do not use Hugging Face authentication')
    
    args = parser.parse_args()
    
    # Initialize and run the pipeline
    pipeline = MedicalQAPipeline(
        prompt_model=args.prompt_model,
        answer_model=args.answer_model,
        prompt_types=args.prompt_types,
        prompts_per_type=args.prompts_per_type,
        use_auth=not args.no_auth
    )
    
    # Execute the pipeline
    pipeline.run(
        dataset_path=args.dataset,
        output_path=args.output,
        num_samples=args.samples
    )

if __name__ == "__main__":
    main() 