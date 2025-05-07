#!/usr/bin/env python
"""
Visualization Module for Medical QA Evaluation Results

This module provides visualization capabilities for analyzing the results
from the Medical QA Evaluation Pipeline. It includes various plots and charts
to help understand the performance of different models and prompt types.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path


class ResultsVisualizer:
    """
    A class for visualizing and analyzing results from the Medical QA evaluation pipeline.
    """
    
    def __init__(
        self, 
        results_path: str,
        output_dir: str = "results/visualizations",
        subfolder: str = None,
        palette: str = "viridis"
    ):
        """
        Initialize the visualizer with the results file.
        
        Args:
            results_path: Path to the CSV file containing results
            output_dir: Base directory to save visualizations (default: results/visualizations)
            subfolder: Optional subfolder within output_dir to organize visualizations
            palette: Color palette for plots (seaborn palette name)
        """
        self.results_path = results_path
        self.base_output_dir = output_dir
        self.subfolder = subfolder
        self.palette = palette
        self.df = None
        
        # Set up the full output directory path
        if subfolder:
            self.output_dir = os.path.join(output_dir, subfolder)
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the results
        self._load_results()
        
        # Configure plot style
        self._set_plot_style()
    
    def _load_results(self) -> None:
        """Load the results CSV file into a pandas DataFrame."""
        print(f"Loading results from {self.results_path}...")
        self.df = pd.read_csv(self.results_path)
        
        # Basic data cleaning
        print(f"Loaded {len(self.df)} results.")
        print(f"Available prompt models: {self.df['prompt_model'].unique()}")
        print(f"Available answer models: {self.df['answer_model'].unique()}")
        print(f"Available prompt types: {self.df['prompt_type'].unique()}")
    
    def _set_plot_style(self) -> None:
        """Set the default plot style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(self.palette)
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a string to be used as a filename by replacing invalid characters.
        
        Args:
            filename: The string to sanitize
            
        Returns:
            Sanitized string safe for use in filenames
        """
        # Replace forward slashes and backslashes with underscores
        filename = filename.replace('/', '_').replace('\\', '_')
        # Replace other potentially problematic characters
        filename = filename.replace(':', '_').replace('*', '_').replace('?', '_')
        filename = filename.replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        return filename

    def _save_figure(self, filename: str, tight_layout: bool = True) -> None:
        """
        Save the current figure to the output directory.
        
        Args:
            filename: Name of the file to save (without extension)
            tight_layout: Whether to apply tight layout to the figure
        """
        if tight_layout:
            plt.tight_layout()
        
        # Sanitize the filename
        safe_filename = self._sanitize_filename(filename)
        output_path = os.path.join(self.output_dir, f"{safe_filename}.png")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    def plot_model_comparison(
        self, 
        metric: str = "answer_similarity", 
        group_by: str = "answer_model",
        sort_values: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Create a bar chart comparing models based on a specific metric.
        
        Args:
            metric: The metric to compare (default: answer_similarity)
            group_by: Column to group by (answer_model or prompt_model)
            sort_values: Whether to sort values in descending order
            figsize: Figure size as (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Group by the specified column and calculate mean of the metric
        grouped = self.df.groupby(group_by)[metric].mean().reset_index()
        
        # Sort if requested
        if sort_values:
            grouped = grouped.sort_values(metric, ascending=False)
        
        # Create the bar chart
        ax = sns.barplot(x=group_by, y=metric, data=grouped)
        
        # Add value labels on top of bars
        for i, v in enumerate(grouped[metric]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        # Set title and labels
        plt.title(f"Comparison of {group_by.replace('_', ' ').title()} by {metric.replace('_', ' ').title()}")
        plt.xlabel(group_by.replace('_', ' ').title())
        plt.ylabel(metric.replace('_', ' ').title())
        
        # Rotate x-labels if there are many models
        if len(grouped) > 5:
            plt.xticks(rotation=45, ha='right')
        
        self._save_figure(f"{group_by}_comparison_by_{metric}")
        plt.close()
    
    def plot_prompt_type_comparison(
        self, 
        metrics: List[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> None:
        """
        Create a bar chart comparing prompt types across multiple metrics.
        
        Args:
            metrics: List of metrics to compare (defaults to a common set if None)
            figsize: Figure size as (width, height)
        """
        # Default metrics if none provided
        if metrics is None:
            metrics = ["answer_similarity", "semantic_similarity", "entailment_score", "bertscore_f1"]
        
        plt.figure(figsize=figsize)
        
        # Group by prompt type and calculate mean of the metrics
        grouped = self.df.groupby('prompt_type')[metrics].mean().reset_index()
        
        # Melt the DataFrame for easier plotting with seaborn
        melted = pd.melt(
            grouped, 
            id_vars=['prompt_type'], 
            value_vars=metrics,
            var_name='Metric', 
            value_name='Score'
        )
        
        # Create the grouped bar chart
        ax = sns.barplot(x='prompt_type', y='Score', hue='Metric', data=melted)
        
        # Set title and labels
        plt.title('Comparison of Prompt Types Across Key Metrics')
        plt.xlabel('Prompt Type')
        plt.ylabel('Score')
        
        # Adjust legend position
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-labels for readability
        plt.xticks(rotation=45, ha='right')
        
        self._save_figure("prompt_type_comparison_multi_metric")
        plt.close()
    
    def plot_heatmap(
        self, 
        metric: str = "answer_similarity",
        figsize: Tuple[int, int] = (14, 12)
    ) -> None:
        """
        Create a heatmap showing performance across prompt models and answer models.
        
        Args:
            metric: The metric to visualize in the heatmap
            figsize: Figure size as (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Create a pivot table with prompt models as rows and answer models as columns
        pivot = self.df.pivot_table(
            index='prompt_model', 
            columns='answer_model',
            values=metric,
            aggfunc='mean'
        )
        
        # Create the heatmap
        ax = sns.heatmap(
            pivot, 
            annot=True, 
            fmt=".3f", 
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={'label': metric.replace('_', ' ').title()}
        )
        
        # Set title and labels
        plt.title(f'Heatmap of {metric.replace("_", " ").title()} by Model Combination')
        plt.xlabel('Answer Model')
        plt.ylabel('Prompt Model')
        
        self._save_figure(f"model_combination_heatmap_{metric}")
        plt.close()
    
    def plot_metric_distributions(
        self, 
        metrics: List[str] = None,
        by_column: Optional[str] = None,
        n_cols: int = 2,
        figsize: Tuple[int, int] = (16, 12)
    ) -> None:
        """
        Create histograms or KDE plots of metric distributions.
        
        Args:
            metrics: List of metrics to plot (defaults to a common set if None)
            by_column: Column to group distributions by (e.g., 'answer_model')
            n_cols: Number of columns in the grid
            figsize: Figure size as (width, height)
        """
        # Default metrics if none provided
        if metrics is None:
            metrics = [
                "answer_similarity", "semantic_similarity", 
                "entailment_score", "rouge1_f", "bleu_score", "bertscore_f1"
            ]
        
        # Calculate number of rows needed
        n_rows = (len(metrics) + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Create a distribution plot for each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if by_column:
                # If grouping by a column, create a KDE plot for each group
                for name, group in self.df.groupby(by_column):
                    sns.kdeplot(
                        data=group, 
                        x=metric, 
                        label=name,
                        ax=ax,
                        fill=True,
                        alpha=0.3
                    )
                ax.legend(title=by_column.replace('_', ' ').title())
            else:
                # Otherwise, create a simple histogram
                sns.histplot(data=self.df, x=metric, kde=True, ax=ax)
            
            # Set title and labels
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Score')
            ax.set_ylabel('Density' if by_column else 'Count')
        
        # Hide any unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        # Set overall title
        fig.suptitle(
            f'Distribution of Metrics{" by " + by_column.replace("_", " ").title() if by_column else ""}',
            fontsize=16
        )
        
        suffix = f"_by_{by_column}" if by_column else ""
        self._save_figure(f"metric_distributions{suffix}", tight_layout=False)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.subplots_adjust(top=0.92)  # Adjust for suptitle
        plt.savefig(os.path.join(self.output_dir, f"metric_distributions{suffix}.png"), dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {os.path.join(self.output_dir, f'metric_distributions{suffix}.png')}")
        plt.close()
    
    def plot_correlation_matrix(
        self, 
        metrics: List[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Create a correlation matrix heatmap for metrics.
        
        Args:
            metrics: List of metrics to include (defaults to all numeric metrics if None)
            figsize: Figure size as (width, height)
        """
        # Select numeric columns if no metrics provided
        if metrics is None:
            # Get numeric columns that are likely metrics
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            # Filter out non-metric columns
            metrics = [col for col in numeric_cols if not col.startswith('prompt_') 
                      and not col in ['prompt_num', 'prompt_variation']]
        
        # Create a correlation matrix
        corr_matrix = self.df[metrics].corr()
        
        plt.figure(figsize=figsize)
        
        # Create the heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        ax = sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm",
            vmin=-1, 
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'}
        )
        
        # Set title
        plt.title('Correlation Matrix of Metrics')
        
        # Rotate the y-axis labels for better readability
        plt.yticks(rotation=0)
        
        self._save_figure("metrics_correlation_matrix")
        plt.close()
    
    def plot_question_difficulty(
        self, 
        metric: str = "answer_similarity",
        top_n: int = 20,
        figsize: Tuple[int, int] = (14, 8)
    ) -> None:
        """
        Visualize questions by difficulty based on average metric performance.
        
        Args:
            metric: The metric to use for difficulty assessment
            top_n: Number of easiest/hardest questions to show
            figsize: Figure size as (width, height)
        """
        # Group by question and calculate average metric value
        question_difficulty = self.df.groupby('question')[metric].mean().reset_index()
        
        # Sort by difficulty
        question_difficulty = question_difficulty.sort_values(metric)
        
        # Get the hardest and easiest questions
        hardest = question_difficulty.head(top_n)
        easiest = question_difficulty.tail(top_n).iloc[::-1]  # Reverse to show highest first
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot hardest questions
        ax1 = sns.barplot(x=metric, y='question', data=hardest, ax=ax1, color='salmon')
        ax1.set_title(f'Top {top_n} Most Difficult Questions')
        ax1.set_xlabel(metric.replace('_', ' ').title())
        ax1.set_ylabel('')
        
        # Truncate long questions
        hardest['short_question'] = hardest['question'].str.slice(0, 50) + '...'
        easiest['short_question'] = easiest['question'].str.slice(0, 50) + '...'
        
        # Recreate plots with shortened questions
        ax1.clear()
        sns.barplot(x=metric, y='short_question', data=hardest, ax=ax1, color='salmon')
        ax1.set_title(f'Top {top_n} Most Difficult Questions')
        ax1.set_xlabel(metric.replace('_', ' ').title())
        ax1.set_ylabel('')
        
        # Plot easiest questions
        ax2 = sns.barplot(x=metric, y='short_question', data=easiest, ax=ax2, color='skyblue')
        ax2.set_title(f'Top {top_n} Easiest Questions')
        ax2.set_xlabel(metric.replace('_', ' ').title())
        ax2.set_ylabel('')
        
        plt.tight_layout()
        self._save_figure("question_difficulty_analysis", tight_layout=False)
        plt.close()
    
    def generate_model_report(
        self, 
        model: str,
        model_type: str = 'answer_model',
        metrics: List[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> None:
        """
        Generate a comprehensive report for a specific model.
        
        Args:
            model: The model name to analyze
            model_type: The type of model ('answer_model' or 'prompt_model')
            metrics: List of metrics to include (defaults to a common set if None)
            figsize: Figure size as (width, height)
        """
        if model_type not in ['answer_model', 'prompt_model']:
            raise ValueError("model_type must be 'answer_model' or 'prompt_model'")
        
        # Default metrics if none provided
        if metrics is None:
            metrics = [
                "answer_similarity", "semantic_similarity", 
                "rouge1_f", "bertscore_f1", "entailment_score"
            ]
        
        # Filter data for the specified model
        model_data = self.df[self.df[model_type] == model]
        
        if len(model_data) == 0:
            print(f"No data found for {model_type} = '{model}'")
            return
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=figsize)
        
        # Setup grid layout
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        # 1. Metrics comparison with other models
        other_models_data = self.df[self.df[model_type] != model]
        this_model_avg = model_data[metrics].mean()
        other_models_avg = other_models_data[metrics].mean()
        
        comparison_df = pd.DataFrame({
            'Metric': metrics,
            f'{model}': this_model_avg.values,
            'Other Models (Avg)': other_models_avg.values
        })
        comparison_df = pd.melt(
            comparison_df, 
            id_vars=['Metric'], 
            var_name='Model', 
            value_name='Score'
        )
        
        sns.barplot(x='Metric', y='Score', hue='Model', data=comparison_df, ax=ax1)
        ax1.set_title(f'{model} vs Other Models')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.legend(title='')
        
        # 2. Performance across prompt types
        if model_type == 'answer_model':
            prompt_performance = model_data.groupby('prompt_type')[metrics].mean().reset_index()
            prompt_performance = pd.melt(
                prompt_performance, 
                id_vars=['prompt_type'], 
                value_vars=metrics,
                var_name='Metric', 
                value_name='Score'
            )
            sns.boxplot(x='prompt_type', y='Score', hue='Metric', data=prompt_performance, ax=ax2)
            ax2.set_title(f'{model} Performance by Prompt Type')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            answer_performance = model_data.groupby('answer_model')[metrics].mean().reset_index()
            answer_performance = pd.melt(
                answer_performance, 
                id_vars=['answer_model'], 
                value_vars=metrics,
                var_name='Metric', 
                value_name='Score'
            )
            sns.boxplot(x='answer_model', y='Score', hue='Metric', data=answer_performance, ax=ax2)
            ax2.set_title(f'{model} Performance by Answer Model')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Distribution of key metrics
        for metric in metrics[:3]:  # Only show first 3 metrics to avoid crowding
            sns.kdeplot(
                data=model_data, 
                x=metric, 
                ax=ax3,
                label=metric.replace('_', ' ').title(),
                fill=True,
                alpha=0.3
            )
        ax3.set_title(f'Distribution of Key Metrics for {model}')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Density')
        ax3.legend()
        
        plt.tight_layout()
        self._save_figure(f"{model_type.split('_')[0]}_model_report_{model}", tight_layout=False)
        plt.close()
    
    def generate_comprehensive_report(self) -> None:
        """
        Generate a comprehensive set of visualizations for the results.
        This runs all major visualization methods to create a complete report.
        """
        print("Generating comprehensive report...")
        
        # Model comparisons
        self.plot_model_comparison(metric="answer_similarity", group_by="answer_model")
        self.plot_model_comparison(metric="answer_similarity", group_by="prompt_model")
        
        # Compare additional metrics
        for metric in ["semantic_similarity", "entailment_score", "bertscore_f1"]:
            self.plot_model_comparison(metric=metric, group_by="answer_model")
        
        # Prompt type comparison
        self.plot_prompt_type_comparison()
        
        # Heatmaps
        self.plot_heatmap(metric="answer_similarity")
        self.plot_heatmap(metric="entailment_score")
        
        # Distributions
        self.plot_metric_distributions()
        self.plot_metric_distributions(by_column="answer_model")
        
        # Correlation matrix
        self.plot_correlation_matrix()
        
        # Question difficulty
        self.plot_question_difficulty()
        
        # Model reports for each unique model
        for model in self.df['answer_model'].unique():
            self.generate_model_report(model=model, model_type='answer_model')
        
        for model in self.df['prompt_model'].unique():
            self.generate_model_report(model=model, model_type='prompt_model')
        
        print(f"Comprehensive report generated and saved to {self.output_dir}/")

# Example usage if run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate visualizations for Medical QA evaluation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results CSV file')
    parser.add_argument('--output-dir', type=str, default='results/visualizations',
                       help='Base directory to save visualizations')
    parser.add_argument('--subfolder', type=str, default=None,
                       help='Optional subfolder within output directory to organize visualizations')
    parser.add_argument('--report-type', type=str, default='comprehensive',
                       choices=['comprehensive', 'basic', 'model', 'metrics', 'prompts'],
                       help='Type of report to generate')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to analyze (used with report-type=model)')
    parser.add_argument('--model-type', type=str, default='answer_model',
                       choices=['answer_model', 'prompt_model'],
                       help='Type of model to analyze')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ResultsVisualizer(
        results_path=args.results,
        output_dir=args.output_dir,
        subfolder=args.subfolder
    )
    
    # Generate appropriate report based on type
    if args.report_type == 'comprehensive':
        visualizer.generate_comprehensive_report()
    elif args.report_type == 'basic':
        visualizer.plot_model_comparison()
        visualizer.plot_prompt_type_comparison()
        visualizer.plot_heatmap()
    elif args.report_type == 'model' and args.model:
        visualizer.generate_model_report(
            model=args.model,
            model_type=args.model_type
        )
    elif args.report_type == 'metrics':
        visualizer.plot_metric_distributions()
        visualizer.plot_correlation_matrix()
    elif args.report_type == 'prompts':
        visualizer.plot_prompt_type_comparison()
        visualizer.plot_question_difficulty() 