import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple
import os
from config import ConfigManager
from prompt_generator import BasePromptGenerator
from metrics_evaluator import BaseMetricsEvaluator
import torch
import argparse
from tqdm import tqdm

class PubMedQAEvaluator:
    def __init__(self, use_case: str = "IRB Documentation", output_dir: str = "pubmedqa_results"):
        """
        Initialize the PubMedQA evaluator.
        
        Args:
            use_case (str): Name of the use case to use for prompt generation
            output_dir (str): Directory to save evaluation results
        """
        self.use_case = use_case
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize config manager
        self.config_manager = ConfigManager()
        
        # Get use case configuration
        self.config = self.config_manager.get_use_case_config(use_case)
        
        # Load dataset
        self.dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        
        # Store results for each model
        self.model_results: Dict[str, pd.DataFrame] = {}
    
    def generate_answer(self, model_name: str, question: str, context: str, system_prompt: str) -> str:
        """
        Generate an answer using the model.
        
        Args:
            model_name (str): Name of the model to use
            question (str): The question to answer
            context (str): The context to use
            system_prompt (str): The system prompt to use
            
        Returns:
            str: Generated answer
        """
        # Format the prompt
        prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Generate answer using config manager
        answer = self.config_manager.generate_text(model_name, prompt)
        
        # Extract answer from response
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        return answer
    
    def evaluate_model(self, model_name: str, num_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Evaluate a model on the PubMedQA dataset.
        
        Args:
            model_name (str): Name of the model to evaluate
            num_samples (int, optional): Number of samples to evaluate. If None, evaluate all samples.
            
        Returns:
            pd.DataFrame: Evaluation results
        """
        print(f"\nEvaluating model: {model_name}")
        
        # Initialize prompt generator
        prompt_generator = BasePromptGenerator(self.config, model_name=model_name)
        
        # Get system prompt
        system_prompt = prompt_generator.get_system_prompt_template()
        
        # Initialize metrics evaluator
        metrics_evaluator = BaseMetricsEvaluator(self.config)
        
        # Prepare results
        results = []
        
        # Get samples to evaluate
        samples = self.dataset['train']
        if num_samples:
            samples = samples.select(range(min(num_samples, len(samples))))
        
        # Evaluate each sample
        for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
            # Generate answer
            answer = self.generate_answer(
                model_name,
                sample['question'],
                sample['context'],
                system_prompt
            )
            
            # Evaluate answer
            metrics = metrics_evaluator.evaluate_answer(
                question=sample['question'],
                answer=answer,
                ground_truth=sample['long_answer']
            )
            
            results.append({
                'question': sample['question'],
                'context': sample['context'],
                'model_answer': answer,
                'ground_truth': sample['long_answer'],
                **metrics
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(
            os.path.join(self.output_dir, f"{model_name}_results.csv"),
            index=False
        )
        
        self.model_results[model_name] = results_df
        return results_df
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare metrics across all evaluated models.
        
        Returns:
            pd.DataFrame: Comparison results
        """
        metrics = ['Relevance', 'Readability', 'Bias_Score', 'Hallucination_Score']
        comparison_data = []
        
        for model_name, results_df in self.model_results.items():
            for metric in metrics:
                if metric in results_df.columns:
                    mean_value = results_df[metric].mean()
                    std_value = results_df[metric].std()
                    comparison_data.append({
                        'Model': model_name,
                        'Metric': metric,
                        'Mean': mean_value,
                        'Std': std_value
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_df.to_csv(
            os.path.join(self.output_dir, 'model_comparison.csv'),
            index=False
        )
        
        return comparison_df

def main():
    """Run model evaluation on PubMedQA dataset."""
    parser = argparse.ArgumentParser(description="Evaluate models on PubMedQA dataset")
    parser.add_argument("--use-case", default="IRB Documentation",
                      help="Use case to use for prompt generation")
    parser.add_argument("--output-dir", default="pubmedqa_results",
                      help="Directory to save evaluation results")
    parser.add_argument("--num-samples", type=int, default=None,
                      help="Number of samples to evaluate (default: all)")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PubMedQAEvaluator(args.use_case, args.output_dir)
    
    # Get available models from use case config
    if not evaluator.config.models:
        print("No models configured for this use case")
        return
    
    # Evaluate each model
    for model in evaluator.config.models:
        try:
            evaluator.evaluate_model(model.name, args.num_samples)
        except Exception as e:
            print(f"Error evaluating model {model.name}: {str(e)}")
            continue
    
    # Compare models
    comparison_df = evaluator.compare_models()
    print("\nModel Comparison Results:")
    print(comparison_df)
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()