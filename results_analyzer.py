import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json

class ResultsAnalyzer:
    def __init__(self, csv_path: str):
        """
        Initialize the results analyzer with a CSV file path.
        
        Args:
            csv_path: Path to the CSV file containing pipeline results
        """
        self.df = pd.read_csv(csv_path)
        
        # Define metric categories and their weights
        self.metric_categories = {
            'semantic_similarity': 0.3,  # How well the answer matches the reference
            'answer_similarity': 0.2,    # How similar the answer is to the reference
            'rouge_scores': 0.15,        # Text overlap metrics
            'bertscore': 0.15,          # BERT-based semantic similarity
            'entailment': 0.2           # Logical consistency
        }
        
        # Define which metrics belong to each category
        self.metric_mapping = {
            'semantic_similarity': ['semantic_similarity'],
            'answer_similarity': ['answer_similarity'],
            'rouge_scores': ['rouge1_f', 'rouge2_f', 'rougeL_f'],
            'bertscore': ['bertscore_precision', 'bertscore_recall', 'bertscore_f1'],
            'entailment': ['entailment_score']
        }
    
    def normalize_metrics(self) -> pd.DataFrame:
        """
        Normalize all metrics to a 0-1 scale for fair comparison.
        """
        df = self.df.copy()
        
        # Get all metric columns
        metric_cols = [col for col in df.columns if any(metric in col for metric in 
                      ['semantic_similarity', 'answer_similarity', 'rouge', 'bertscore', 'entailment'])]
        
        # Normalize each metric
        for col in metric_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[f'{col}_normalized'] = 0.5  # If all values are the same
        
        return df
    
    def calculate_category_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weighted scores for each metric category.
        """
        df = df.copy()
        
        for category, weight in self.metric_categories.items():
            metrics = self.metric_mapping[category]
            normalized_metrics = [f'{metric}_normalized' for metric in metrics if f'{metric}_normalized' in df.columns]
            
            if normalized_metrics:
                df[f'{category}_score'] = df[normalized_metrics].mean(axis=1) * weight
        
        return df
    
    def calculate_overall_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the overall score by combining all category scores.
        """
        df = df.copy()
        category_scores = [f'{category}_score' for category in self.metric_categories.keys()]
        df['overall_score'] = df[category_scores].sum(axis=1)
        return df
    
    def find_best_combinations(self, top_n: int = 5) -> Dict:
        """
        Find the best combinations of models and prompt types.
        
        Args:
            top_n: Number of top combinations to return
            
        Returns:
            Dictionary containing the best combinations and their scores
        """
        # Normalize metrics and calculate scores
        df = self.normalize_metrics()
        df = self.calculate_category_scores(df)
        df = self.calculate_overall_score(df)
        
        # Group by model combinations and calculate average scores
        grouped = df.groupby(['prompt_model', 'answer_model', 'prompt_type']).agg({
            'overall_score': 'mean',
            'semantic_similarity_score': 'mean',
            'answer_similarity_score': 'mean',
            'rouge_scores_score': 'mean',
            'bertscore_score': 'mean',
            'entailment_score': 'mean'
        }).reset_index()
        
        # Sort by overall score
        grouped = grouped.sort_values('overall_score', ascending=False)
        
        # Get top N combinations
        top_combinations = grouped.head(top_n)
        
        # Format results
        results = {
            'best_combinations': [],
            'metric_weights': self.metric_categories,
            'analysis_summary': {
                'total_combinations': len(grouped),
                'best_overall_score': float(top_combinations.iloc[0]['overall_score']),
                'average_overall_score': float(grouped['overall_score'].mean())
            }
        }
        
        # Add detailed information for each top combination
        for _, row in top_combinations.iterrows():
            combination = {
                'prompt_model': row['prompt_model'],
                'answer_model': row['answer_model'],
                'prompt_type': row['prompt_type'],
                'scores': {
                    'overall': float(row['overall_score']),
                    'semantic_similarity': float(row['semantic_similarity_score']),
                    'answer_similarity': float(row['answer_similarity_score']),
                    'rouge_scores': float(row['rouge_scores_score']),
                    'bertscore': float(row['bertscore_score']),
                    'entailment': float(row['entailment_score'])
                },
                'reasoning': self._generate_reasoning(row)
            }
            results['best_combinations'].append(combination)
        
        return results
    
    def _generate_reasoning(self, row: pd.Series) -> str:
        """
        Generate reasoning for why a combination performed well.
        """
        reasoning = []
        
        # Check which metrics performed particularly well
        if row['semantic_similarity_score'] > 0.8:
            reasoning.append("Excellent semantic similarity with reference answers")
        if row['answer_similarity_score'] > 0.8:
            reasoning.append("High answer similarity indicating good content matching")
        if row['rouge_scores_score'] > 0.7:
            reasoning.append("Strong text overlap with reference answers")
        if row['bertscore_score'] > 0.8:
            reasoning.append("High BERT-based semantic similarity")
        if row['entailment_score'] > 0.8:
            reasoning.append("Strong logical consistency with reference answers")
        
        # Add model-specific reasoning
        reasoning.append(f"The {row['prompt_model']} model generated effective prompts for the {row['prompt_type']} methodology")
        reasoning.append(f"The {row['answer_model']} model produced high-quality answers based on these prompts")
        
        return " | ".join(reasoning)
    
    def save_analysis(self, output_path: str):
        """
        Save the analysis results to a JSON file.
        
        Args:
            output_path: Path to save the analysis results
        """
        results = self.find_best_combinations()
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis results saved to {output_path}")
        
        # Print summary
        print("\nTop 5 Best Combinations:")
        print("=" * 80)
        for i, combo in enumerate(results['best_combinations'], 1):
            print(f"\n{i}. {combo['prompt_model']} → {combo['answer_model']} ({combo['prompt_type']})")
            print(f"   Overall Score: {combo['scores']['overall']:.3f}")
            print(f"   Reasoning: {combo['reasoning']}")
            print("-" * 80) 