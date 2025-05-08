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
    bertscore_precision, bertscore_recall, bertscore_f1, entailment_score, entailment_label
  - Comparative Metrics: analysis of differences between model and reference answers
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
        'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
        'entailment_score', 'entailment_label'
    ],
    'Comparative Metrics': [
        'comparison_answer_length_delta', 'comparison_answer_length_pct_change', 'comparison_answer_length_analysis',
        'comparison_flesch_reading_ease_delta', 'comparison_flesch_reading_ease_pct_change', 'comparison_flesch_reading_ease_analysis',
        'comparison_flesch_kincaid_grade_delta', 'comparison_flesch_kincaid_grade_pct_change', 'comparison_flesch_kincaid_grade_analysis',
        'comparison_sentiment_polarity_delta', 'comparison_sentiment_polarity_pct_change', 'comparison_sentiment_polarity_analysis',
        'comparison_sentiment_subjectivity_delta', 'comparison_sentiment_subjectivity_pct_change', 'comparison_sentiment_subjectivity_analysis',
        'comparison_relevance_delta', 'comparison_relevance_analysis', 'comparison_summary'
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
            print("Initializing metrics evaluator...")
            self.metrics_evaluator = MetricsEvaluator(
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
    
    def run(self, dataset_path: str, output_path: str, num_samples: Optional[int] = None, resume_from: Optional[int] = None) -> pd.DataFrame:
        """
        Execute the full pipeline.
        
        Args:
            dataset_path: Path to the CSV dataset
            output_path: Path for the output CSV file
            num_samples: Number of samples to process (None for all)
            resume_from: Question index to resume from (1-based)
            
        Returns:
            DataFrame with evaluation results
        """
        # Print configuration
        self._print_config(dataset_path, output_path, num_samples)
        
        # Create output directory if it doesn't exist
        output_path_obj = Path(output_path)
        if not output_path_obj.parent.exists():
            os.makedirs(output_path_obj.parent, exist_ok=True)
        
        # Load the dataset
        df = self.load_dataset(dataset_path, num_samples)
        
        # Set up CSV file with headers
        self.output_path = output_path
        self.results_df = None
        self.csv_initialized = False
        self.long_text_fields = ['relevancy_reason', 'prompt_alignment_reason']
        
        # Process data and generate results (this now writes to CSV directly)
        results = self._process_data(df, resume_from)
        
        # Print summary 
        self._print_summary(df, self.results_df, output_path)
        
        return self.results_df
    
    def _process_data(self, df: pd.DataFrame, resume_from: Optional[int] = None) -> List[Dict]:
        """
        Process each question-answer pair in the dataset.
        
        Args:
            df: DataFrame containing question-answer pairs
            resume_from: Question index to resume from (1-based)
            
        Returns:
            List of dictionaries with results (for backward compatibility)
        """
        results = []
        
        # Create a total progress bar for all combinations
        total_iterations = len(df) * len(self.prompt_types) * self.prompts_per_type * len(self.prompt_models) * len(self.answer_models)
        progress_bar = tqdm(total=total_iterations, desc="Overall progress")
        
        # Counter for prompt numbering across all questions/models
        prompt_num = 0
        
        # If resuming, load existing results to get the last prompt number
        if resume_from is not None:
            try:
                existing_df = pd.read_csv(self.output_path)
                if len(existing_df) > 0:
                    prompt_num = existing_df['prompt_num'].max()
                    print(f"Resuming from prompt number {prompt_num}")
                    
                    # Calculate how many questions have been fully processed
                    questions_processed = len(existing_df) // (len(self.prompt_types) * self.prompts_per_type * len(self.prompt_models) * len(self.answer_models))
                    print(f"Questions fully processed: {questions_processed}")
                    
                    if questions_processed >= len(df):
                        print("All questions have been fully processed. No new processing needed.")
                        return results
            except Exception as e:
                print(f"Warning: Could not load existing results: {str(e)}")
        
        # Process each question-answer pair
        for idx, row in enumerate(df.itertuples(), 1):
            # Skip questions before resume point
            if resume_from is not None and idx < resume_from:
                continue
            
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
                    
                    # Generate prompts for this type
                    prompts = prompt_generator.generate_prompt(
                        prompt_type=prompt_type,
                        num_prompts=self.prompts_per_type,
                        verbose=False
                    )
                    
                    # Process each generated prompt
                    for i, system_prompt in enumerate(prompts):
                        # Increment the prompt number
                        prompt_num += 1
                        
                        print(f"\n{'-'*100}")
                        print(f"PROCESSING PROMPT VARIATION {i+1}/{self.prompts_per_type} FOR {prompt_type} (PROMPT #{prompt_num})")
                        print(f"{'-'*100}")
                        
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
                            
                            # Process and write to CSV immediately
                            if self.exclude_long_text:
                                processed_result = {k: v for k, v in result.items() if k not in self.long_text_fields}
                                self._write_to_csv(processed_result)
                            else:
                                self._write_to_csv(result)
                            
                            # Store result for backward compatibility
                            results.append(result)
                            
                            # Update progress bar
                            progress_bar.update(1)
                            
                            # Print concise confirmation of incremental save
                            print(f"✓ Result #{len(results)} saved: Q{idx}, {prompt_type} ({i+1}/{self.prompts_per_type}), {prompt_model_name} → {answer_model_name}")
        
        # Close the progress bar
        progress_bar.close()
        
        return results
    
    def _write_to_csv(self, result: Dict[str, Any]) -> None:
        """
        Write a single result to the CSV file.
        
        Args:
            result: Dictionary containing result data
        """
        # Convert single result to DataFrame
        result_df = pd.DataFrame([result])
        
        # Check if file exists and has content
        file_exists = os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0
        
        # If file doesn't exist or is empty, initialize with headers
        if not file_exists:
            result_df.to_csv(self.output_path, index=False, mode='w')
            self.csv_initialized = True
            self.results_df = result_df
        else:
            # Always append to existing file
            result_df.to_csv(self.output_path, index=False, mode='a', header=False)
            # Update in-memory DataFrame for the summary
            if self.results_df is None:
                self.results_df = pd.read_csv(self.output_path)
            else:
                self.results_df = pd.concat([self.results_df, result_df])
    
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
            print(f"Verbose metrics output: {self.verbose} (colorized formatting provided by metrics_evaluator)")
            # Print expected metrics by category
            print("Expected metrics by category:")
            for category, metrics in METRIC_CATEGORIES.items():
                print(f"  - {category}: {len(metrics)} metrics")
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
    
    def _print_summary(self, df: pd.DataFrame, results_df: Optional[pd.DataFrame], output_path: str) -> None:
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
        
        # Load the complete results from the CSV file
        try:
            complete_results = pd.read_csv(output_path)
            print(f"Total result rows in file: {len(complete_results)} (expected: {expected_rows})")
            
            # Print a breakdown of results by model combination
            print(f"\nResults breakdown by model combination:")
            for prompt_model in self.prompt_models:
                for answer_model in self.answer_models:
                    combo_df = complete_results[
                        (complete_results['prompt_model_key'] == prompt_model) & 
                        (complete_results['answer_model_key'] == answer_model)
                    ]
                    print(f"  - {prompt_model} (prompt) + {answer_model} (answer): {len(combo_df)} results")
            
            # Print metrics summary if available
            if self.enable_metrics:
                print(f"\nMetrics summary:")
                
                # Count the number of metrics collected by category
                if len(complete_results) > 0:
                    for category, category_metrics in METRIC_CATEGORIES.items():
                        existing_metrics = [m for m in category_metrics if m in complete_results.columns]
                        if existing_metrics:
                            print(f"\n  {category} ({len(existing_metrics)}/{len(category_metrics)} metrics):")
                            
                            # Report average values for numeric metrics
                            for metric in existing_metrics:
                                if metric in complete_results.columns and pd.api.types.is_numeric_dtype(complete_results[metric]):
                                    avg_value = complete_results[metric].mean()
                                    print(f"    - Average {metric}: {avg_value:.4f}")
                else:
                    print("  No metrics data available in results.")
        except Exception as e:
            print(f"Warning: Could not load complete results for summary: {str(e)}")
        
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
    parser.add_argument('--num-questions', type=int, default=None,
                        help='Number of question samples to process (default: all)')
    parser.add_argument('--no-auth', action='store_true',
                        help='Do not use Hugging Face authentication')
    parser.add_argument('--no-metrics', action='store_true',
                        help='Disable metrics evaluation')
    parser.add_argument('--exclude-long-text', action='store_true',
                        help='Exclude long text fields from CSV output')
    parser.add_argument('--list-metrics', action='store_true',
                        help='List all available metrics with descriptions and exit')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Disable colorized metrics display in the terminal')
    parser.add_argument('--resume-from', type=int, default=None,
                        help='Question index to resume from (1-based)')
    
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
        use_auth=not args.no_auth,
        exclude_long_text=args.exclude_long_text,
        verbose=not args.no_verbose
    )
    
    # Execute the pipeline with resume option
    pipeline.run(
        dataset_path=args.dataset,
        output_path=args.output,
        num_samples=args.num_questions,
        resume_from=args.resume_from
    )

if __name__ == "__main__":
    main() 