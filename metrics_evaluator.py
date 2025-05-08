#!/usr/bin/env python
"""
Metrics Evaluation Module for Medical QA Pipeline

This module provides metrics evaluation functionality for the Medical QA pipeline.
All evaluations are performed locally using lightweight NLP models on CPU.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Union, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import textstat
from textblob import TextBlob
import colorama
from colorama import Fore, Style

# Import evaluation metrics
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import BERTScorer
from transformers import pipeline

# Set environment variable to avoid parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize colorama for terminal coloring
colorama.init()

# Download necessary NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Metric categories for organization and display
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
    ]
}

class MetricsEvaluator:
    """
    A class for evaluating question-answer pairs using multiple metrics.
    All evaluations are performed locally using lightweight NLP models on CPU.
    Designed to be integrated into the medical QA pipeline.
    """
    
    def __init__(
        self,
        embedding_model_name: str = 'paraphrase-MiniLM-L6-v2',
        verbose: bool = True
    ):
        """
        Initialize the metrics evaluator with specified parameters.
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
            verbose: Whether to print detailed metrics in the terminal
        """
        self.device = "cpu"  # Always use CPU
        self.verbose = verbose
        self.correct_answer_metrics_cache = {}  # Cache for correct answer metrics
        
        # Initialize embedding model for semantic metrics
        print(f"Loading embedding model '{embedding_model_name}' on CPU...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize ROUGE
        self.rouge = Rouge()
        
        # Initialize BERTScore
        print("Loading BERTScore model...")
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        
        # Initialize NLI model as None - it'll be loaded on first use
        self._nli_model = None
        
        print("Metrics evaluator initialized successfully - using NLP metrics.")
    
    @property
    def nli_model(self):
        """Lazy loading of NLI model to improve startup performance"""
        if self._nli_model is None:
            print("Loading NLI model (facebook/bart-large-mnli)...")
            self._nli_model = pipeline(
                "text-classification", 
                model="facebook/bart-large-mnli", 
                device=self.device,
                max_length=512,  # Set maximum token length
                truncation=True  # Enable truncation for long inputs
            )
        return self._nli_model
    
    def check_entailment(self, pred_answer: str, true_answer: str) -> dict:
        """
        Check if the predicted answer entails (is consistent with) the true answer.
        
        Args:
            pred_answer: The model's generated answer
            true_answer: The ground truth answer
            
        Returns:
            Dictionary with entailment results
        """
        # Truncate very long inputs to prevent issues
        max_length = 500  # Characters, not tokens
        if len(pred_answer) > max_length:
            pred_answer = pred_answer[:max_length] + "..."
        if len(true_answer) > max_length:
            true_answer = true_answer[:max_length] + "..."
            
        # Create the input text
        input_text = f"{pred_answer} [SEP] {true_answer}"
        
        try:
            # Try direct BART-mnli approach
            result = self.nli_model(input_text)
            
            # Process the result based on its structure
            if isinstance(result, list):
                # Handle list-type result
                label = result[0]["label"]
                score = result[0]["score"]
                
                # Map BART-mnli label format: CONTRADICTION, NEUTRAL, ENTAILMENT
                label = label.lower()
                entailment_score = score if label == "entailment" else 0.0
                
                return {
                    "entailment_score": entailment_score,
                    "entailment_label": label
                }
            else:
                # Handle dict-type result
                label = result["label"]
                score = result["score"]
                
                # Map BART-mnli label format
                label = label.lower()
                entailment_score = score if label == "entailment" else 0.0
                
                return {
                    "entailment_score": entailment_score,
                    "entailment_label": label
                }
                
        except Exception as e:
            print(f"Error in entailment check: {str(e)}")
            # Provide detailed debugging info
            print(f"Input length: {len(input_text)} chars")
            
            # Return default values on failure
            return {
                "entailment_score": 0.5,
                "entailment_label": "unknown"
            }
    
    def _calculate_nlp_metrics(
        self,
        question: str,
        model_answer: str,
        correct_answer: str
    ) -> Dict[str, float]:
        """
        Calculate NLP-based metrics.
        
        Args:
            question: The question
            model_answer: The model's answer
            correct_answer: The correct answer
            
        Returns:
            Dictionary of NLP metrics
        """
        metrics = {}
        
        # Truncate long texts to prevent recursion errors
        MAX_TEXT_LENGTH = 1000  # characters
        model_answer = model_answer[:MAX_TEXT_LENGTH]
        correct_answer = correct_answer[:MAX_TEXT_LENGTH]
        
        try:
            # Calculate semantic similarity
            question_embedding = self.embedding_model.encode([question], convert_to_tensor=True)
            model_answer_embedding = self.embedding_model.encode([model_answer], convert_to_tensor=True)
            correct_answer_embedding = self.embedding_model.encode([correct_answer], convert_to_tensor=True)
            
            metrics['semantic_similarity'] = float(cosine_similarity(question_embedding, model_answer_embedding)[0][0])
            
            metrics['answer_similarity'] = float(cosine_similarity(model_answer_embedding, correct_answer_embedding)[0][0])
            
            # Calculate ROUGE scores with error handling
            try:
                rouge_scores = self.rouge.get_scores(model_answer, correct_answer)[0]
                metrics['rouge1_f'] = rouge_scores['rouge-1']['f']
                metrics['rouge2_f'] = rouge_scores['rouge-2']['f']
                metrics['rougeL_f'] = rouge_scores['rouge-l']['f']
            except RecursionError:
                print(f"Warning: ROUGE calculation failed due to text length. Skipping ROUGE metrics.")
                metrics['rouge1_f'] = 0.0
                metrics['rouge2_f'] = 0.0
                metrics['rougeL_f'] = 0.0
            except Exception as e:
                print(f"Warning: Error calculating ROUGE scores: {str(e)}")
                metrics['rouge1_f'] = 0.0
                metrics['rouge2_f'] = 0.0
                metrics['rougeL_f'] = 0.0
            
            # Calculate BLEU score
            try:
                reference = [correct_answer.split()]
                hypothesis = model_answer.split()
                
                # Use smoothing function to avoid score of 0 for short sentences
                smoothie = SmoothingFunction().method1
                metrics['bleu_score'] = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
            except Exception as e:
                print(f"Warning: Error calculating BLEU score: {str(e)}")
                metrics['bleu_score'] = 0.0
            
            # Calculate BERTScore
            try:
                precision, recall, f1 = self.bert_scorer.score([model_answer], [correct_answer])
                metrics['bertscore_precision'] = float(precision[0])
                metrics['bertscore_recall'] = float(recall[0])
                metrics['bertscore_f1'] = float(f1[0])
            except Exception as e:
                print(f"Warning: Error calculating BERTScore: {str(e)}")
                metrics['bertscore_precision'] = 0.0
                metrics['bertscore_recall'] = 0.0
                metrics['bertscore_f1'] = 0.0
            
            # Calculate readability metrics
            try:
                metrics['answer_length'] = len(model_answer.split())
                metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(model_answer)
                metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(model_answer)
            except Exception as e:
                print(f"Warning: Error calculating readability metrics: {str(e)}")
                metrics['answer_length'] = 0
                metrics['flesch_reading_ease'] = 0.0
                metrics['flesch_kincaid_grade'] = 0.0
            
            # Calculate sentiment
            try:
                sentiment = TextBlob(model_answer).sentiment
                metrics['sentiment_polarity'] = sentiment.polarity
                metrics['sentiment_subjectivity'] = sentiment.subjectivity
            except Exception as e:
                print(f"Warning: Error calculating sentiment: {str(e)}")
                metrics['sentiment_polarity'] = 0.0
                metrics['sentiment_subjectivity'] = 0.0
            
            # Calculate entailment
            try:
                entailment_results = self.check_entailment(model_answer, correct_answer)
                metrics.update(entailment_results)
            except Exception as e:
                print(f"Warning: Error calculating entailment: {str(e)}")
                metrics['entailment_score'] = 0.0
                metrics['entailment_label'] = 'neutral'
        
        except Exception as e:
            print(f"Error in NLP metrics calculation: {str(e)}")
            # Set default values for all metrics
            for metric in self.METRIC_CATEGORIES.values():
                for m in metric:
                    metrics[m] = 0.0
        
        return metrics
    
    def evaluate_answer(
        self,
        question: str,
        model_answer: str,
        correct_answer: str,
        system_prompt: Optional[str] = None,
        question_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model-generated answer against the correct answer.
        
        Args:
            question: The medical question
            model_answer: The answer generated by the model
            correct_answer: The ground truth answer
            system_prompt: The system prompt used to generate the answer (optional)
            question_id: Identifier for the question (for caching, optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Skip evaluation if answer is not available or empty
        if not model_answer or not model_answer.strip():
            return {"error": "No answer to evaluate"}
        
        # Use question_id for caching if provided, otherwise use the question text
        cache_key = question_id if question_id is not None else question
        
        # Check if we've already evaluated the correct answer for this question
        if cache_key in self.correct_answer_metrics_cache:
            # Retrieve cached metrics for the correct answer
            correct_metrics = self.correct_answer_metrics_cache[cache_key]
            
            # Add the cached correct answer metrics to the output
            for key, value in correct_metrics.items():
                metrics[f"correct_{key}"] = value
        else:
            # Evaluate the correct answer and cache the results
            correct_metrics = self._evaluate_correct_answer(question, correct_answer)
            self.correct_answer_metrics_cache[cache_key] = correct_metrics
            
            # Add the correct answer metrics to the output
            for key, value in correct_metrics.items():
                metrics[f"correct_{key}"] = value
        
        # Calculate semantic and NLP metrics for the model answer
        metrics.update(self._calculate_nlp_metrics(question, model_answer, correct_answer))
        
        # Generate comparative metrics
        comparison = self.compare_metrics(metrics)
        
        # Add comparative metrics to the output
        for key, value in comparison.items():
            if isinstance(value, (int, float, str, bool)):
                metrics[f"comparison_{key}"] = value
            elif isinstance(value, list):
                metrics[f"comparison_{key}"] = "; ".join(value)
        
        # Print metrics to terminal if verbose mode is enabled
        if self.verbose:
            self.print_metrics(metrics, question, model_answer, correct_answer)
            self.print_comparison(comparison, metrics)
        
        return metrics
    
    def print_metrics(
        self, 
        metrics: Dict[str, Any], 
        question: str, 
        model_answer: str, 
        correct_answer: str
    ) -> None:
        """
        Print metrics in a well-formatted way in the terminal.
        
        Args:
            metrics: Dictionary of evaluation metrics
            question: The question being evaluated
            model_answer: The model's answer
            correct_answer: The correct answer
        """
        # Print divider
        print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}METRICS EVALUATION SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        
        # Print basic information - show complete text without truncation
        print(f"{Fore.YELLOW}Question:{Style.RESET_ALL} {question}")
        print(f"{Fore.YELLOW}Model Answer:{Style.RESET_ALL} {model_answer}")
        print(f"{Fore.YELLOW}Correct Answer:{Style.RESET_ALL} {correct_answer}")
        
        # Print metrics by category
        for category, metric_names in METRIC_CATEGORIES.items():
            print(f"\n{Fore.GREEN}{category}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'-' * len(category)}{Style.RESET_ALL}")
            
            for metric_name in metric_names:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    
                    # Format based on value type
                    if isinstance(value, float):
                        # Color code based on score (higher is better for most metrics)
                        if value >= 0.8:
                            color = Fore.GREEN
                        elif value >= 0.6:
                            color = Fore.YELLOW
                        elif value >= 0.4:
                            color = Fore.YELLOW
                        else:
                            color = Fore.RED
                            
                        print(f"  {metric_name}: {color}{value:.4f}{Style.RESET_ALL}")
                    elif isinstance(value, int):
                        print(f"  {metric_name}: {value}")
                    elif isinstance(value, str):
                        # Special color coding for entailment_label
                        if metric_name == "entailment_label":
                            if value == "entailment":
                                print(f"  {metric_name}: {Fore.GREEN}{value}{Style.RESET_ALL}")
                            elif value == "neutral":
                                print(f"  {metric_name}: {Fore.YELLOW}{value}{Style.RESET_ALL}")
                            elif value == "contradiction":
                                print(f"  {metric_name}: {Fore.RED}{value}{Style.RESET_ALL}")
                            else:
                                print(f"  {metric_name}: {value}")
                        else:
                            # Display full string without truncation
                            print(f"  {metric_name}: {value}")
                    else:
                        print(f"  {metric_name}: {value}")
        
        # Print key comparative metrics
        print(f"\n{Fore.MAGENTA}KEY COMPARISONS{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'-' * 14}{Style.RESET_ALL}")
        
        # Compare model metrics to correct answer metrics
        for metric in ['answer_length', 'flesch_reading_ease', 'flesch_kincaid_grade']:
            if metric in metrics and f'correct_{metric}' in metrics:
                model_value = metrics[metric]
                correct_value = metrics[f'correct_{metric}']
                
                # Calculate and format difference
                diff = model_value - correct_value
                if abs(diff) < 0.01:  # For small floating point differences
                    diff_str = f"{Fore.GREEN}(≈ same){Style.RESET_ALL}"
                elif diff > 0:
                    diff_str = f"{Fore.YELLOW}(+{diff:.2f}){Style.RESET_ALL}"
                else:
                    diff_str = f"{Fore.YELLOW}({diff:.2f}){Style.RESET_ALL}"
                
                print(f"  {metric}: {model_value:.2f} vs {correct_value:.2f} {diff_str}")
        
        # Print overall evaluation scores
        print(f"\n{Fore.BLUE}OVERALL EVALUATION{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'-' * 18}{Style.RESET_ALL}")
        
        # Define key metrics to show in overall evaluation
        key_metrics = [
            ('answer_similarity', 'Content Similarity'),
            ('entailment_score', 'Factual Consistency')
        ]
        
        # Print key scores with color coding
        for metric_key, metric_label in key_metrics:
            if metric_key in metrics:
                value = metrics[metric_key]
                
                # Color code based on score (higher is better)
                if value >= 0.8:
                    color = Fore.GREEN
                elif value >= 0.6:
                    color = Fore.YELLOW
                elif value >= 0.4:
                    color = Fore.YELLOW
                else:
                    color = Fore.RED
                
                print(f"  {metric_label}: {color}{value:.4f}{Style.RESET_ALL}")
        
        # Print entailment label
        if "entailment_label" in metrics:
            label = metrics["entailment_label"]
            if label == "entailment":
                print(f"  Entailment: {Fore.GREEN}Model answer entails correct answer{Style.RESET_ALL}")
            elif label == "neutral":
                print(f"  Entailment: {Fore.YELLOW}Model answer is neutral to correct answer{Style.RESET_ALL}")
            elif label == "contradiction":
                print(f"  Entailment: {Fore.RED}Model answer contradicts correct answer{Style.RESET_ALL}")
        
        # Print bottom divider
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
    
    def _evaluate_correct_answer(
        self,
        question: str,
        correct_answer: str
    ) -> Dict[str, float]:
        """
        Evaluate the correct answer for baseline metrics.
        
        Args:
            question: The question
            correct_answer: The correct answer
        
        Returns:
            Dictionary of metrics for the correct answer
        """
        # Compute embeddings
        question_embedding = self.embedding_model.encode([question], convert_to_tensor=True)
        answer_embedding = self.embedding_model.encode([correct_answer], convert_to_tensor=True)
        
        # Convert to numpy for similarity calculations
        question_embedding_np = question_embedding.cpu().numpy()
        answer_embedding_np = answer_embedding.cpu().numpy()
        
        metrics = {}
        
        # Question-answer semantic similarity (relevance)
        metrics['semantic_similarity'] = float(cosine_similarity(question_embedding_np, answer_embedding_np)[0][0])
        
        # Basic statistics
        metrics['answer_length'] = len(correct_answer.split())
        
        # Readability metrics
        metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(correct_answer)
        metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(correct_answer)
        
        # Sentiment analysis
        sentiment = TextBlob(correct_answer).sentiment
        metrics['sentiment_polarity'] = sentiment.polarity
        metrics['sentiment_subjectivity'] = sentiment.subjectivity

        return metrics

    def get_metrics_documentation(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Get documentation for all metrics used in the evaluator.
        
        Returns:
            Dictionary with metric categories and descriptions
        """
        # Create a documentation dictionary for all metrics
        metrics_docs = {}
        
        # Terminal Output
        metrics_docs["Terminal Output"] = [
            {"name": "Colorized Display", "description": "When verbose mode is enabled, metrics are displayed in the terminal with color coding: green (good), yellow (moderate), red (poor)"},
            {"name": "Metrics Summary", "description": "Displays metrics grouped by category with clear section headers"},
            {"name": "Key Comparisons", "description": "Shows direct comparisons between model metrics and correct answer metrics"},
            {"name": "Overall Evaluation", "description": "Provides a summary of the most important metrics with clear visual feedback"}
        ]
        
        # NLP Metrics
        metrics_docs["NLP Metrics"] = [
            {"name": "semantic_similarity", "description": "Cosine similarity between question and model answer embeddings (0-1)"},
            {"name": "answer_similarity", "description": "Cosine similarity between model answer and correct answer embeddings (0-1)"},
            {"name": "answer_length", "description": "Word count of the model's answer"},
            {"name": "flesch_reading_ease", "description": "Readability score (higher = easier to read)"},
            {"name": "flesch_kincaid_grade", "description": "Reading grade level required to understand the text"},
            {"name": "sentiment_polarity", "description": "Sentiment score (-1 to 1, negative to positive)"},
            {"name": "sentiment_subjectivity", "description": "Subjectivity score (0-1, objective to subjective)"}
        ]
        
        # Text Comparison
        metrics_docs["Text Comparison"] = [
            {"name": "rouge1_f", "description": "ROUGE-1 F1 score comparing model and correct answers (unigram overlap)"},
            {"name": "rouge2_f", "description": "ROUGE-2 F1 score comparing model and correct answers (bigram overlap)"},
            {"name": "rougeL_f", "description": "ROUGE-L F1 score comparing model and correct answers (longest common sequence)"},
            {"name": "bleu_score", "description": "BLEU score measuring precision of model answer against reference"},
            {"name": "bertscore_precision", "description": "BERTScore precision score (semantic precision)"},
            {"name": "bertscore_recall", "description": "BERTScore recall score (semantic recall)"},
            {"name": "bertscore_f1", "description": "BERTScore F1 score (semantic similarity)"},
            {"name": "entailment_score", "description": "Score indicating how well the model answer entails the correct answer (higher is better)"},
            {"name": "entailment_label", "description": "Natural Language Inference label: entailment, neutral, or contradiction"}
        ]
        
        # Reference Metrics
        metrics_docs["Reference Metrics"] = [
            {"name": "correct_semantic_similarity", "description": "Semantic similarity between question and correct answer"},
            {"name": "correct_answer_length", "description": "Word count of the correct answer"},
            {"name": "correct_flesch_reading_ease", "description": "Readability score of the correct answer"},
            {"name": "correct_flesch_kincaid_grade", "description": "Reading grade level of the correct answer"},
            {"name": "correct_sentiment_polarity", "description": "Sentiment score of the correct answer"},
            {"name": "correct_sentiment_subjectivity", "description": "Subjectivity score of the correct answer"}
        ]
        
        # Comparative Metrics
        metrics_docs["Comparative Metrics"] = [
            {"name": "comparison_answer_length_delta", "description": "Absolute difference in word count between model and correct answers"},
            {"name": "comparison_answer_length_pct_change", "description": "Percentage difference in word count relative to correct answer"},
            {"name": "comparison_answer_length_analysis", "description": "Human-readable analysis of length difference"},
            
            {"name": "comparison_flesch_reading_ease_delta", "description": "Difference in readability score between model and correct answers"},
            {"name": "comparison_flesch_reading_ease_pct_change", "description": "Percentage difference in readability relative to correct answer"},
            {"name": "comparison_flesch_reading_ease_analysis", "description": "Human-readable analysis of readability difference"},
            
            {"name": "comparison_flesch_kincaid_grade_delta", "description": "Difference in grade level between model and correct answers"},
            {"name": "comparison_flesch_kincaid_grade_pct_change", "description": "Percentage difference in grade level relative to correct answer"},
            {"name": "comparison_flesch_kincaid_grade_analysis", "description": "Human-readable analysis of grade level difference"},
            
            {"name": "comparison_sentiment_polarity_delta", "description": "Difference in sentiment polarity between model and correct answers"},
            {"name": "comparison_sentiment_polarity_pct_change", "description": "Percentage difference in sentiment polarity relative to correct answer"},
            {"name": "comparison_sentiment_polarity_analysis", "description": "Human-readable analysis of sentiment polarity difference"},
            
            {"name": "comparison_sentiment_subjectivity_delta", "description": "Difference in subjectivity between model and correct answers"},
            {"name": "comparison_sentiment_subjectivity_pct_change", "description": "Percentage difference in subjectivity relative to correct answer"},
            {"name": "comparison_sentiment_subjectivity_analysis", "description": "Human-readable analysis of subjectivity difference"},
            
            {"name": "comparison_relevance_delta", "description": "Difference in question relevance between model and correct answers"},
            {"name": "comparison_relevance_analysis", "description": "Human-readable analysis of relevance difference"},
            
            {"name": "comparison_summary", "description": "Concise summary of key differences between model and correct answers"}
        ]
        
        return metrics_docs

    def compare_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare model metrics with correct answer metrics and generate a comparative analysis.
        
        Args:
            metrics: Dictionary containing both model and correct answer metrics
        
        Returns:
            Dictionary with comparative analysis metrics
        """
        comparison = {}
        
        # Define metrics to compare
        comparable_metrics = [
            ('answer_length', 'Length Delta', 'words'),
            ('flesch_reading_ease', 'Reading Ease Delta', 'points'),
            ('flesch_kincaid_grade', 'Grade Level Delta', 'grade levels'),
            ('sentiment_polarity', 'Sentiment Polarity Delta', 'points'),
            ('sentiment_subjectivity', 'Subjectivity Delta', 'points')
        ]
        
        # Calculate and format deltas
        for base_metric, label, unit in comparable_metrics:
            model_key = base_metric
            correct_key = f"correct_{base_metric}"
            
            if model_key in metrics and correct_key in metrics:
                model_value = metrics[model_key]
                correct_value = metrics[correct_key]
                
                # Calculate raw delta (model - correct)
                delta = model_value - correct_value
                
                # Calculate percentage change relative to correct answer
                if correct_value != 0:
                    pct_change = (delta / abs(correct_value)) * 100
                else:
                    pct_change = float('inf') if delta != 0 else 0
                
                # Add to comparison dictionary
                comparison[f"{base_metric}_delta"] = delta
                comparison[f"{base_metric}_pct_change"] = pct_change
                
                # Add formatted string for display
                if abs(pct_change) < float('inf'):
                    direction = "longer" if delta > 0 else "shorter"
                    if base_metric != "answer_length":
                        direction = "higher" if delta > 0 else "lower"
                        
                    comparison[f"{base_metric}_analysis"] = f"{abs(delta):.1f} {unit} {direction} ({pct_change:.1f}%)"
                else:
                    comparison[f"{base_metric}_analysis"] = f"{delta:.1f} {unit} difference (∞%)"
        
        # Special analysis for semantic metrics
        if "semantic_similarity" in metrics and "correct_semantic_similarity" in metrics:
            model_relevance = metrics["semantic_similarity"]
            correct_relevance = metrics["correct_semantic_similarity"]
            
            # Is the model answer more or less relevant to the question than the correct answer?
            relevance_delta = model_relevance - correct_relevance
            comparison["relevance_delta"] = relevance_delta
            
            if abs(relevance_delta) < 0.05:
                comparison["relevance_analysis"] = "Similar relevance to the question as the correct answer"
            elif relevance_delta > 0:
                comparison["relevance_analysis"] = f"More relevant to the question (+{relevance_delta:.2f})"
            else:
                comparison["relevance_analysis"] = f"Less relevant to the question ({relevance_delta:.2f})"
        
        # Overall comparison summary
        summary_points = []
        
        # Length comparison
        if "answer_length_delta" in comparison:
            delta = comparison["answer_length_delta"]
            if abs(delta) < 5:
                summary_points.append("Similar length to correct answer")
            elif delta > 0:
                pct = comparison["answer_length_pct_change"]
                summary_points.append(f"Longer than correct answer (+{delta:.0f} words, +{pct:.0f}%)")
            else:
                pct = comparison["answer_length_pct_change"]
                summary_points.append(f"Shorter than correct answer ({delta:.0f} words, {pct:.0f}%)")
        
        # Readability comparison
        if "flesch_reading_ease_delta" in comparison:
            delta = comparison["flesch_reading_ease_delta"]
            if abs(delta) < 10:
                summary_points.append("Similar readability to correct answer")
            elif delta > 0:
                summary_points.append("More readable than correct answer")
            else:
                summary_points.append("Less readable than correct answer")
        
        # Semantic comparison
        if "answer_similarity" in metrics:
            similarity = metrics["answer_similarity"]
            if similarity > 0.9:
                summary_points.append("Very similar content to correct answer")
            elif similarity > 0.75:
                summary_points.append("Similar content to correct answer")
            elif similarity > 0.5:
                summary_points.append("Moderately similar to correct answer")
            else:
                summary_points.append("Substantially different from correct answer")
        
        # Add summary
        comparison["summary"] = summary_points
        
        return comparison
        
    def print_comparison(self, comparison: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """
        Print a formatted comparison between model and correct answer metrics.
        
        Args:
            comparison: Dictionary with comparative metrics from compare_metrics()
            metrics: Original metrics dictionary
        """
        print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}MODEL VS. CORRECT ANSWER COMPARISON{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        
        # Print semantic similarity comparison
        print(f"\n{Fore.GREEN}Content Comparison{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'-'*17}{Style.RESET_ALL}")
        
        if "answer_similarity" in metrics:
            similarity = metrics["answer_similarity"]
            color = Fore.GREEN if similarity > 0.75 else (Fore.YELLOW if similarity > 0.5 else Fore.RED)
            print(f"  Content similarity: {color}{similarity:.2f}{Style.RESET_ALL}")
        
        if "entailment_label" in metrics and "entailment_score" in metrics:
            label = metrics["entailment_label"]
            score = metrics["entailment_score"]
            
            if label == "entailment":
                print(f"  Entailment: {Fore.GREEN}Model entails correct answer ({score:.2f}){Style.RESET_ALL}")
            elif label == "neutral":
                print(f"  Entailment: {Fore.YELLOW}Model is neutral to correct answer ({score:.2f}){Style.RESET_ALL}")
            elif label == "contradiction":
                print(f"  Entailment: {Fore.RED}Model contradicts correct answer ({score:.2f}){Style.RESET_ALL}")
        
        if "relevance_analysis" in comparison:
            print(f"  Relevance comparison: {comparison['relevance_analysis']}")
        
        # Print text statistics comparison
        print(f"\n{Fore.GREEN}Text Statistics Comparison{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'-'*25}{Style.RESET_ALL}")
        
        # Show detailed comparisons for different metrics
        for metric, label, _ in [
            ('answer_length', 'Length', 'words'),
            ('flesch_reading_ease', 'Reading ease', 'points'),
            ('flesch_kincaid_grade', 'Grade level', 'grade levels')
        ]:
            if f"{metric}_analysis" in comparison:
                print(f"  {label}: {comparison[f'{metric}_analysis']}")
            
            if metric in metrics and f"correct_{metric}" in metrics:
                model_val = metrics[metric]
                correct_val = metrics[f"correct_{metric}"]
                print(f"    Model: {model_val:.1f} vs. Correct: {correct_val:.1f}")
        
        # Print sentiment comparison
        if "sentiment_polarity_analysis" in comparison or "sentiment_subjectivity_analysis" in comparison:
            print(f"\n{Fore.GREEN}Sentiment Comparison{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'-'*19}{Style.RESET_ALL}")
            
            if "sentiment_polarity_analysis" in comparison:
                print(f"  Polarity: {comparison['sentiment_polarity_analysis']}")
                
                polarity = metrics["sentiment_polarity"]
                correct_polarity = metrics["correct_sentiment_polarity"]
                
                polarity_desc = "neutral"
                if polarity > 0.25: polarity_desc = "positive"
                elif polarity < -0.25: polarity_desc = "negative"
                
                correct_desc = "neutral"
                if correct_polarity > 0.25: correct_desc = "positive"
                elif correct_polarity < -0.25: correct_desc = "negative"
                
                print(f"    Model: {polarity:.2f} ({polarity_desc}) vs. Correct: {correct_polarity:.2f} ({correct_desc})")
            
            if "sentiment_subjectivity_analysis" in comparison:
                print(f"  Subjectivity: {comparison['sentiment_subjectivity_analysis']}")
                
                subj = metrics["sentiment_subjectivity"]
                correct_subj = metrics["correct_sentiment_subjectivity"]
                
                subj_desc = "neutral"
                if subj > 0.6: subj_desc = "subjective"
                elif subj < 0.4: subj_desc = "objective"
                
                correct_desc = "neutral"
                if correct_subj > 0.6: correct_desc = "subjective"
                elif correct_subj < 0.4: correct_desc = "objective"
                
                print(f"    Model: {subj:.2f} ({subj_desc}) vs. Correct: {correct_subj:.2f} ({correct_desc})")
        
        # Print summary
        if "summary" in comparison and comparison["summary"]:
            print(f"\n{Fore.BLUE}Summary{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{'-'*7}{Style.RESET_ALL}")
            
            for point in comparison["summary"]:
                print(f"  • {point}")
                
        print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}") 