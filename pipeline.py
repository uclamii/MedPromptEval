#!/usr/bin/env python
"""
Medical QA Evaluation Pipeline

This module provides a comprehensive end-to-end pipeline for:
1. Loading a CSV file containing medical question-answer pairs
2. Generating system prompts on-the-fly using a prompt generation model
3. Using an answer generation model to answer questions based on these prompts
4. Evaluating generated answers using various metrics
5. Storing results in a CSV with detailed information about the entire process

The output CSV includes:
- question
- correct_answer (from the CSV)
- prompt_model (model used for generating the prompt)
- prompt_type
- system_prompt (the generated system prompt)
- answer_model (model used for answering)
- model_answer (answer from the LLM)
- Multiple evaluation metrics, including:
  - NLP Metrics: semantic_similarity, answer_similarity, answer_length, 
    flesch_reading_ease, flesch_kincaid_grade, sentiment_polarity, sentiment_subjectivity
  - Text Comparison: rouge1_f, rouge2_f, rougeL_f, bleu_score, 
    bertscore_precision, bertscore_recall, bertscore_f1
  - DeepEval Metrics: bias_score, hallucination_score, toxicity_score, 
    relevancy_score, prompt_alignment_score, correctness_score
  - Explanatory metrics: relevancy_reason, prompt_alignment_reason
  - Reference metrics: correct_semantic_similarity, correct_answer_length, etc.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from prompt_generation import PromptGenerator
from answer_generation import AnswerGenerator
from metrics_evaluator import MetricsEvaluator
from config import PROMPT_MODEL_CONFIGS, ANSWER_MODEL_CONFIGS, PROMPT_TYPES, ensure_hf_login

# List of all expected metrics categories from the metrics_evaluator
METRIC_CATEGORIES = {
    'NLP Metrics': [
        'semantic_similarity', 'answer_similarity', 'answer_length',
        'flesch_reading_ease', 'flesch_kincaid_grade', 
        'sentiment_polarity', 'sentiment_subjectivity'
    ],
    'Text Comparison': [
        'rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu_score',
        'bertscore_precision', 'bertscore_recall', 'bertscore_f1'
    ],
    'DeepEval Metrics': [
        'bias_score', 'hallucination_score', 'toxicity_score',
        'relevancy_score', 'prompt_alignment_score', 'correctness_score'
    ],
    'Explanatory Metrics': [
        'relevancy_reason', 'prompt_alignment_reason'
    ],
    'Reference Metrics': [
        'correct_semantic_similarity', 'correct_answer_length',
        'correct_flesch_reading_ease', 'correct_flesch_kincaid_grade',
        'correct_sentiment_polarity', 'correct_sentiment_subjectivity'
    ]
}

# Flattened list of all expected metrics
ALL_EXPECTED_METRICS = [
    metric for category in METRIC_CATEGORIES.values() for metric in category
]

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
        enable_metrics: bool = True,
        metrics_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_auth: bool = True,
        exclude_long_text: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the pipeline with specified models and parameters.
        
        Args:
            prompt_models: Model(s) to use for generating system prompts (single or list)
            answer_models: Model(s) to use for answering questions (single or list)
            prompt_types: List of prompt types to use (None for all)
            prompts_per_type: Number of prompts to generate per type
            enable_metrics: Whether to enable metrics evaluation
            metrics_model: Model to use for metrics evaluation
            use_auth: Whether to use Hugging Face authentication
            exclude_long_text: Whether to exclude long text fields (reasons) from CSV output
            verbose: Whether to print detailed metrics in the terminal
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
        self.enable_metrics = enable_metrics
        self.metrics_model = metrics_model
        self.use_auth = use_auth
        self.exclude_long_text = exclude_long_text
        self.verbose = verbose
        
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
        
        # Initialize metrics evaluator if enabled
        self.metrics_evaluator = None
        if self.enable_metrics:
            print(f"Initializing metrics evaluator with model {metrics_model}...")
            self.metrics_evaluator = MetricsEvaluator(
                eval_model_name=metrics_model,
                verbose=self.verbose
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
        
        # Remove long text fields if requested
        if self.exclude_long_text and results:
            # Fields that might contain long text
            long_text_fields = ['relevancy_reason', 'prompt_alignment_reason']
            results_processed = []
            
            for result in results:
                # Create a copy without the specified fields
                processed_result = {k: v for k, v in result.items() if k not in long_text_fields}
                results_processed.append(processed_result)
            
            results_df = pd.DataFrame(results_processed)
        else:
            results_df = pd.DataFrame(results)
        
        # Save results to CSV
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
            question_id = str(getattr(row, 'id', idx))  # Use ID if available, otherwise use index
            
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
                            
                            # Prepare base result dictionary
                            result = {
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
                            }
                            
                            # Evaluate metrics if enabled
                            if self.enable_metrics and self.metrics_evaluator:
                                print(f"Evaluating answer metrics...")
                                
                                # Run evaluation metrics
                                metrics = self.metrics_evaluator.evaluate_answer(
                                    question=question,
                                    model_answer=model_answer,
                                    correct_answer=correct_answer,
                                    system_prompt=system_prompt,
                                    question_id=question_id
                                )
                                
                                # Check for any missing expected metrics
                                for expected_metric in ALL_EXPECTED_METRICS:
                                    if expected_metric not in metrics and not expected_metric.startswith("correct_"):
                                        print(f"Warning: Expected metric '{expected_metric}' not found in evaluation results")
                                
                                # Add all metrics to the result dictionary
                                for metric_name, metric_value in metrics.items():
                                    # Format float values to 4 decimal places for readability in CSV
                                    if isinstance(metric_value, float):
                                        result[metric_name] = round(metric_value, 6)
                                    else:
                                        result[metric_name] = metric_value
                                
                                # Print key metrics for visibility
                                print(f"Key metrics: similarity={metrics.get('answer_similarity', 'N/A'):.4f}, "
                                      f"relevancy={metrics.get('relevancy_score', 'N/A'):.4f}, "
                                      f"correctness={metrics.get('correctness_score', 'N/A'):.4f}")
                                
                                # Add detailed metrics breakdown
                                print("Metrics collected:")
                                for category, category_metrics in METRIC_CATEGORIES.items():
                                    collected = [m for m in category_metrics if m in metrics]
                                    if collected:
                                        print(f"  - {category}: {len(collected)}/{len(category_metrics)} metrics")
                            
                            # Store result
                            results.append(result)
                            
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
        print(f"Metrics evaluation: {'Enabled' if self.enable_metrics else 'Disabled'}")
        if self.enable_metrics:
            print(f"Metrics model: {self.metrics_model}")
            print(f"Verbose metrics output: {self.verbose} (colorized formatting provided by metrics_evaluator)")
            # Print expected metrics by category
            print("Expected metrics by category:")
            for category, metrics in METRIC_CATEGORIES.items():
                print(f"  - {category}: {len(metrics)} metrics")
                if category == "Explanatory Metrics" and self.exclude_long_text:
                    print("    (Note: These will be excluded from CSV output)")
        print(f"Exclude long text fields: {self.exclude_long_text}")
        print(f"Using authentication: {self.use_auth}")
        
        # Calculate total number of combinations
        total_combinations = len(self.prompt_models) * len(self.answer_models) * len(self.prompt_types) * self.prompts_per_type
        
        # Get the number of rows in the dataset
        try:
            dataset_rows = num_samples if num_samples else len(pd.read_csv(dataset_path))
        except Exception as e:
            dataset_rows = "Unknown (error reading dataset)"
            
        total_results = total_combinations * dataset_rows if isinstance(dataset_rows, int) else "Unknown"
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
        
        # Print metrics summary if available
        if self.enable_metrics:
            print(f"\nMetrics summary:")
            
            # Count the number of metrics collected by category
            if len(results_df) > 0:
                for category, category_metrics in METRIC_CATEGORIES.items():
                    existing_metrics = [m for m in category_metrics if m in results_df.columns]
                    if existing_metrics:
                        print(f"\n  {category} ({len(existing_metrics)}/{len(category_metrics)} metrics):")
                        
                        # Report average values for numeric metrics
                        for metric in existing_metrics:
                            if metric in results_df.columns and pd.api.types.is_numeric_dtype(results_df[metric]):
                                avg_value = results_df[metric].mean()
                                print(f"    - Average {metric}: {avg_value:.4f}")
            else:
                print("  No metrics data available in results.")
        
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
    parser.add_argument('--no-metrics', action='store_true',
                        help='Disable metrics evaluation')
    parser.add_argument('--metrics-model', type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help='Model to use for metrics evaluation')
    parser.add_argument('--exclude-long-text', action='store_true',
                        help='Exclude long text fields from CSV output')
    parser.add_argument('--list-metrics', action='store_true',
                        help='List all available metrics with descriptions and exit')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Disable colorized metrics display in the terminal (formatting handled by metrics_evaluator)')
    
    args = parser.parse_args()
    
    # Handle special case to list metrics
    if args.list_metrics:
        # Create a metrics evaluator instance to get the documentation
        evaluator = MetricsEvaluator()
        metrics_docs = evaluator.get_metrics_documentation()
        
        print("\nAvailable metrics in the Medical QA Pipeline:\n")
        for category, metrics in metrics_docs.items():
            print(f"\n{category} ({len(metrics)} metrics):")
            print("=" * (len(category) + 12))
            for metric in metrics:
                print(f"  - {metric['name']}: {metric['description']}")
        
        return
    
    # Initialize and run the pipeline
    pipeline = MedicalQAPipeline(
        prompt_models=args.prompt_models,
        answer_models=args.answer_models,
        prompt_types=args.prompt_types,
        prompts_per_type=args.prompts_per_type,
        enable_metrics=not args.no_metrics,
        metrics_model=args.metrics_model,
        use_auth=not args.no_auth,
        exclude_long_text=args.exclude_long_text,
        verbose=not args.no_verbose
    )
    
    # Execute the pipeline
    pipeline.run(
        dataset_path=args.dataset,
        output_path=args.output,
        num_samples=args.samples
    )

if __name__ == "__main__":
    main() 