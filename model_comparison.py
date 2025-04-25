import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
from config import UseCaseConfigManager
from metrics_evaluator import BaseMetricsEvaluator
import numpy as np
from scipy import stats
import argparse

class ModelComparison:
    def __init__(self, use_case: str, output_dir: str = "model_comparison_results"):
        """
        Initialize the model comparison framework.
        
        Args:
            use_case (str): Name of the use case to compare models for
            output_dir (str): Directory to save comparison results
        """
        self.use_case = use_case
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize configuration manager
        self.config_manager = UseCaseConfigManager()
        self.config = self.config_manager.get_config(use_case)
        
        # Store results for each model
        self.model_results: Dict[str, pd.DataFrame] = {}
        
    def load_model_results(self, model_name: str, results_file: str) -> pd.DataFrame:
        """
        Load evaluation results for a specific model.
        
        Args:
            model_name (str): Name of the model
            results_file (str): Path to the results CSV file
            
        Returns:
            pd.DataFrame: Evaluation results
        """
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        results_df = pd.read_csv(results_file)
        self.model_results[model_name] = results_df
        return results_df
    
    def compare_metrics(self) -> pd.DataFrame:
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
        return comparison_df
    
    def plot_metric_comparison(self, metric: str):
        """
        Create a box plot comparing a specific metric across models.
        
        Args:
            metric (str): Name of the metric to plot
        """
        plt.figure(figsize=(10, 6))
        data = []
        labels = []
        
        for model_name, results_df in self.model_results.items():
            if metric in results_df.columns:
                data.append(results_df[metric])
                labels.append(model_name)
        
        plt.boxplot(data, labels=labels)
        plt.title(f'{metric} Comparison Across Models')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'{metric}_comparison.png'))
        plt.close()
    
    def plot_metric_heatmap(self):
        """Create a heatmap of mean metric values across models."""
        comparison_df = self.compare_metrics()
        pivot_df = comparison_df.pivot(index='Model', columns='Metric', values='Mean')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Metric Comparison Heatmap')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'metric_heatmap.png'))
        plt.close()
    
    def perform_statistical_test(self, metric: str) -> pd.DataFrame:
        """
        Perform statistical tests to compare models for a specific metric.
        
        Args:
            metric (str): Name of the metric to compare
            
        Returns:
            pd.DataFrame: Statistical test results
        """
        model_names = list(self.model_results.keys())
        n_models = len(model_names)
        test_results = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model1 = model_names[i]
                model2 = model_names[j]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(
                    self.model_results[model1][metric],
                    self.model_results[model2][metric]
                )
                
                test_results.append({
                    'Model1': model1,
                    'Model2': model2,
                    'T-statistic': t_stat,
                    'P-value': p_value,
                    'Significant': p_value < 0.05
                })
        
        return pd.DataFrame(test_results)
    
    def generate_report(self):
        """Generate a comprehensive comparison report."""
        # Compare metrics
        comparison_df = self.compare_metrics()
        comparison_df.to_csv(
            os.path.join(self.output_dir, 'metric_comparison.csv'),
            index=False
        )
        
        # Create plots for each metric
        metrics = ['Relevance', 'Readability', 'Bias_Score', 'Hallucination_Score']
        for metric in metrics:
            self.plot_metric_comparison(metric)
        
        # Create heatmap
        self.plot_metric_heatmap()
        
        # Perform statistical tests
        for metric in metrics:
            test_results = self.perform_statistical_test(metric)
            test_results.to_csv(
                os.path.join(self.output_dir, f'{metric}_statistical_tests.csv'),
                index=False
            )
        
        # Generate summary report
        with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w') as f:
            f.write(f"Model Comparison Report for {self.use_case}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. Overall Performance\n")
            f.write("-" * 20 + "\n")
            for metric in metrics:
                f.write(f"\n{metric}:\n")
                metric_data = comparison_df[comparison_df['Metric'] == metric]
                for _, row in metric_data.iterrows():
                    f.write(f"{row['Model']}: {row['Mean']:.3f} (±{row['Std']:.3f})\n")
            
            f.write("\n2. Statistical Significance\n")
            f.write("-" * 20 + "\n")
            for metric in metrics:
                f.write(f"\n{metric}:\n")
                test_results = self.perform_statistical_test(metric)
                for _, row in test_results.iterrows():
                    significance = "Significant" if row['Significant'] else "Not Significant"
                    f.write(f"{row['Model1']} vs {row['Model2']}: {significance} (p={row['P-value']:.3f})\n")

def main():
    """Run model comparison for a specific use case."""
    parser = argparse.ArgumentParser(description="Compare different models for a use case")
    parser.add_argument("--use-case", required=True, help="Use case to compare models for")
    parser.add_argument("--output-dir", default="model_comparison_results",
                      help="Directory to save comparison results")
    parser.add_argument("--results-dir", required=True,
                      help="Directory containing model evaluation results")
    args = parser.parse_args()
    
    # Initialize comparison framework
    comparison = ModelComparison(args.use_case, args.output_dir)
    
    # Get available models
    config_manager = UseCaseConfigManager()
    config = config_manager.get_config(args.use_case)
    
    if not config.models:
        print("No models configured for this use case")
        return
    
    # Load results for each model
    for model in config.models:
        results_file = os.path.join(args.results_dir, f"{model.name}_results.csv")
        try:
            comparison.load_model_results(model.name, results_file)
            print(f"Loaded results for model: {model.name}")
        except FileNotFoundError:
            print(f"Warning: Results file not found for model {model.name}")
            continue
    
    if not comparison.model_results:
        print("No model results found to compare")
        return
    
    # Generate comparison report
    comparison.generate_report()
    print(f"\nComparison results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 