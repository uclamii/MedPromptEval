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
from typing import Dict, List, Optional, Tuple, Union

from prompt_generation import PromptGenerator
from answer_generation import AnswerGenerator
from config import PROMPT_MODEL_CONFIGS, ANSWER_MODEL_CONFIGS, PROMPT_TYPES, ensure_hf_login

class MedicalQAPipeline:
    """
    End-to-end pipeline for medical question answering evaluation.
    """
    
    def __init__(
        self, 
        prompt_models: Union[str, List[str]] = 'phi-2',
        answer_models: Union[str, List[str]] = 'mistral-7b',
        prompt_types: List[str] = None,
        prompts_per_type: int = 1,
        use_auth: bool = True
    ):
        """
        Initialize the pipeline with specified models and parameters.
        
        Args:
            prompt_models: Model(s) to use for generating system prompts (single or list)
            answer_models: Model(s) to use for answering questions (single or list)
            prompt_types: List of prompt types to use (None for all)
            prompts_per_type: Number of prompts to generate per type
            use_auth: Whether to use Hugging Face authentication
        """
        # Convert single models to lists for consistent handling
        if isinstance(prompt_models, str):
            prompt_models = [prompt_models]
        if isinstance(answer_models, str):
            answer_models = [answer_models]
            
        # Set configuration
        self.prompt_models = prompt_models
        self.answer_models = answer_models
        self.prompt_types = prompt_types if prompt_types else list(PROMPT_TYPES.keys())
        self.prompts_per_type = prompts_per_type
        self.use_auth = use_auth
        
        # Ensure authentication if needed
        if self.use_auth:
            ensure_hf_login()
        
        # Initialize models
        self.prompt_generators = {}
        for model_key in self.prompt_models:
            print(f"Initializing {model_key} for prompt generation...")
            self.prompt_generators[model_key] = PromptGenerator(
                model_key=model_key,
                use_auth=self.use_auth
            )
        
        self.answer_generators = {}
        for model_key in self.answer_models:
            print(f"Initializing {model_key} for answer generation...")
            self.answer_generators[model_key] = AnswerGenerator(
                model_key=model_key,
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
        
        # Create a total progress bar for all combinations
        total_iterations = len(df) * len(self.prompt_types) * self.prompts_per_type * len(self.prompt_models) * len(self.answer_models)
        progress_bar = tqdm(total=total_iterations, desc="Overall progress")
        
        # Counter for prompt numbering across all questions/models
        prompt_num = 0
        
        # Process each question-answer pair
        for idx, row in enumerate(df.itertuples(), 1):
            question = row.question
            correct_answer = row.answer
            
            # Print question info
            print(f"\n{'#'*100}")
            print(f"PROCESSING QUESTION {idx}/{len(df)}")
            print(f"{'#'*100}")
            print(f"QUESTION: {question}")
            print(f"\n{'='*40} CORRECT ANSWER {'='*40}")
            print(correct_answer)
            print(f"{'='*90}")
            
            # Loop through prompt models
            for prompt_model_key in self.prompt_models:
                prompt_generator = self.prompt_generators[prompt_model_key]
                prompt_model_name = PROMPT_MODEL_CONFIGS[prompt_model_key]["name"]
                
                print(f"\n{'@'*100}")
                print(f"USING PROMPT MODEL: {prompt_model_name}")
                print(f"{'@'*100}")
                
                # Process each prompt type
                for prompt_type in self.prompt_types:
                    print(f"\n{'*'*100}")
                    print(f"GENERATING PROMPTS FOR TYPE: {prompt_type}")
                    print(f"{'*'*100}")
                    
                    # Get prompt type definition from config
                    prompt_definition = PROMPT_TYPES[prompt_type]
                    
                    # Construct the actual system prompt used in the prompt_generation.py
                    generation_system_prompt = f"""You are an expert prompt engineer. Your task is to create a single system prompt for a chatbot that answers medical questions using the {prompt_type} methodology: {prompt_definition}. Ensure that the system prompt is clear and instructs the chatbot effectively. Only provide the system prompt as the output, do not provide any other text besides the prompt. Do not generate any code, just text for a system prompt."""
                    
                    # Print the actual system prompt used for generation
                    print(f"\n{'='*40} SYSTEM PROMPT TEMPLATE {'='*40}")
                    print(generation_system_prompt)
                    print(f"{'='*90}")
                    
                    # Generate prompts and answer immediately for each one
                    for i in range(self.prompts_per_type):
                        # Increment the prompt number
                        prompt_num += 1
                        
                        print(f"\n{'-'*100}")
                        print(f"GENERATING PROMPT VARIATION {i+1}/{self.prompts_per_type} FOR {prompt_type} (PROMPT #{prompt_num})")
                        print(f"{'-'*100}")
                        
                        # Generate a single system prompt - suppressing the detailed output
                        system_prompt = prompt_generator.generate_prompt(
                            prompt_type=prompt_type,
                            num_prompts=1,
                            verbose=False
                        )[0]  # Get the first (and only) prompt
                        
                        # Loop through answer models
                        for answer_model_key in self.answer_models:
                            answer_generator = self.answer_generators[answer_model_key]
                            answer_model_name = ANSWER_MODEL_CONFIGS[answer_model_key]["name"]
                            
                            print(f"\n{'-'*40} USING ANSWER MODEL: {answer_model_name} {'-'*40}")
                            
                            # Generate answer immediately using this prompt
                            print(f"Generating answer using prompt variation {i+1}...")
                            model_answer = answer_generator.generate_answer(
                                system_prompt=system_prompt,
                                question=question
                            )
                            
                            # Store result
                            results.append({
                                "prompt_num": prompt_num,
                                "question": question,
                                "correct_answer": correct_answer,
                                "prompt_model": prompt_model_name,
                                "prompt_model_key": prompt_model_key,
                                "prompt_type": prompt_type,
                                "prompt_variation": i+1,
                                "system_prompt": system_prompt,
                                "answer_model": answer_model_name,
                                "answer_model_key": answer_model_key,
                                "model_answer": model_answer
                            })
                            
                            # Update progress bar
                            progress_bar.update(1)
        
        # Close the progress bar
        progress_bar.close()
        
        return results
    
    def _print_config(self, dataset_path: str, output_path: str, num_samples: Optional[int]) -> None:
        """Print the pipeline configuration."""
        print(f"\n{'='*100}")
        print("MEDICAL QA EVALUATION PIPELINE CONFIGURATION")
        print(f"{'='*100}")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_path}")
        print(f"Prompt generation models: {', '.join(self.prompt_models)}")
        print(f"Answer generation models: {', '.join(self.answer_models)}")
        print(f"Prompt types: {', '.join(self.prompt_types)}")
        print(f"Prompts per type: {self.prompts_per_type}")
        print(f"Number of samples: {num_samples if num_samples else 'All'}")
        print(f"Using authentication: {self.use_auth}")
        
        # Calculate total number of combinations
        total_combinations = len(self.prompt_models) * len(self.answer_models) * len(self.prompt_types) * self.prompts_per_type
        total_results = total_combinations * (num_samples if num_samples else len(pd.read_csv(dataset_path)))
        print(f"Total model combinations: {total_combinations}")
        print(f"Expected total results: {total_results}")
        print(f"{'='*100}\n")
    
    def _print_summary(self, df: pd.DataFrame, results_df: pd.DataFrame, output_path: str) -> None:
        """Print a summary of the pipeline results."""
        prompt_model_count = len(self.prompt_models)
        answer_model_count = len(self.answer_models)
        prompt_type_count = len(self.prompt_types)
        expected_rows = len(df) * prompt_model_count * answer_model_count * prompt_type_count * self.prompts_per_type
        
        print(f"\n{'#'*100}")
        print("PIPELINE EXECUTION COMPLETE!")
        print(f"{'#'*100}")
        print(f"Results saved to: {output_path}")
        print(f"Processed {len(df)} questions")
        print(f"Used {prompt_model_count} prompt models ({', '.join(self.prompt_models)})")
        print(f"Used {answer_model_count} answer models ({', '.join(self.answer_models)})")
        print(f"Used {prompt_type_count} prompt types ({', '.join(self.prompt_types)})")
        print(f"Generated {len(results_df)} result rows (expected: {expected_rows})")
        
        # Print a breakdown of results by model combination
        print(f"\nResults breakdown by model combination:")
        for prompt_model in self.prompt_models:
            for answer_model in self.answer_models:
                combo_df = results_df[
                    (results_df['prompt_model_key'] == prompt_model) & 
                    (results_df['answer_model_key'] == answer_model)
                ]
                print(f"  - {prompt_model} (prompt) + {answer_model} (answer): {len(combo_df)} results")
        
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
    parser.add_argument('--prompt-models', type=str, nargs='+', default=['phi-2'],
                        choices=list(PROMPT_MODEL_CONFIGS.keys()),
                        help='Models to use for generating prompts (multiple allowed)')
    parser.add_argument('--answer-models', type=str, nargs='+', default=['mistral-7b'],
                        choices=list(ANSWER_MODEL_CONFIGS.keys()),
                        help='Models to use for answering questions (multiple allowed)')
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
        prompt_models=args.prompt_models,
        answer_models=args.answer_models,
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