#!/usr/bin/env python
"""
Metrics Evaluation Module for Medical QA Pipeline

This module provides metrics evaluation functionality for the Medical QA pipeline.
All evaluations are performed locally using Hugging Face models running on CPU.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import textstat
from textblob import TextBlob
import colorama
from colorama import Fore, Style

# Import deepeval metrics
from deepeval import evaluate
from deepeval.metrics import (
    BiasMetric, 
    HallucinationMetric, 
    AnswerRelevancyMetric, 
    ToxicityMetric, 
    GEval,
    PromptAlignmentMetric
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM

# Import Hugging Face transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import evaluation metrics
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import BERTScorer

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
        'bertscore_precision', 'bertscore_recall', 'bertscore_f1'
    ],
    'DeepEval Metrics': [
        'bias_score', 'hallucination_score', 'toxicity_score',
        'relevancy_score', 'prompt_alignment_score', 'correctness_score'
    ],
    'Explanatory Metrics': [
        'relevancy_reason', 'prompt_alignment_reason'
    ]
}

class HuggingFaceModel(DeepEvalBaseLLM):
    """
    Custom implementation of DeepEvalBaseLLM using Hugging Face models for local evaluation.
    """
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens: int = 256,
        load_in_8bit: bool = False
    ):
        """
        Initialize the Hugging Face model for evaluation.
        
        Args:
            model_name: The name of the Hugging Face model to use
            max_new_tokens: Maximum number of tokens to generate
            load_in_8bit: Whether to load the model in 8-bit precision
        """
        self.model_name = model_name
        self.device = "cpu"  # Always use CPU
        self.max_new_tokens = max_new_tokens
        self.load_in_8bit = load_in_8bit
        self._model = None
        self._tokenizer = None
        
        # Load tokenizer early for memory efficiency
        print(f"Loading tokenizer for model: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_model(self):
        """Load the model if not already loaded."""
        if self._model is None:
            print(f"Loading model: {self.model_name} on CPU")
            
            # Set appropriate kwargs for model loading
            kwargs = {}
            if self.load_in_8bit:
                kwargs["load_in_8bit"] = True
                
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                **kwargs
            )
        
        return self._model

    def generate(self, prompt: str) -> str:
        """Generate text based on the prompt."""
        model = self.load_model()
        
        # Process the prompt using the tokenizer
        model_inputs = self._tokenizer([prompt], return_tensors="pt")
        
        # Generate tokens
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                use_cache=True
            )
        
        # Decode the generated tokens
        generated_text = self._tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # Extract only the generated part (excluding the prompt)
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):]
        
        return generated_text.strip()

    async def a_generate(self, prompt: str) -> str:
        """Async version of generate (for compatibility)."""
        return self.generate(prompt)

    def get_model_name(self) -> str:
        """Return the name of the model."""
        return self.model_name.split('/')[-1]

class MetricsEvaluator:
    """
    A class for evaluating question-answer pairs using multiple metrics.
    All evaluations are performed locally using Hugging Face models on CPU.
    Designed to be integrated into the medical QA pipeline.
    """
    
    def __init__(
        self,
        embedding_model_name: str = 'paraphrase-MiniLM-L6-v2',
        eval_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        load_in_8bit: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the metrics evaluator with specified parameters.
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
            eval_model_name: Name of the HuggingFace model to use for evaluation
            load_in_8bit: Whether to load the evaluation model in 8-bit precision
            verbose: Whether to print detailed metrics in the terminal
        """
        self.device = "cpu"  # Always use CPU
        self.eval_model_name = eval_model_name
        self.load_in_8bit = load_in_8bit
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
        
        # Initialize DeepEval metrics
        print("Initializing DeepEval metrics with local Hugging Face model...")
        
        # Initialize the HuggingFace model for evaluation
        self.hf_model = HuggingFaceModel(
            model_name=eval_model_name,
            load_in_8bit=load_in_8bit
        )
        
        # Initialize metrics with the local model
        self.bias_metric = BiasMetric(
            threshold=0.5,
            model=self.hf_model
        )
        
        self.hallucination_metric = HallucinationMetric(
            threshold=0.5,
            model=self.hf_model
        )
        
        self.toxicity_metric = ToxicityMetric(
            threshold=0.5,
            model=self.hf_model,
            async_mode=False
        )
        
        self.answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=self.hf_model,
            include_reason=True
        )
        
        # Define correctness evaluation
        self.correctness_metric = GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK"
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT, 
                LLMTestCaseParams.ACTUAL_OUTPUT, 
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
            model=self.hf_model
        )
        
        # Initialize alignment metric
        self.prompt_alignment_metric = PromptAlignmentMetric(
            prompt_instructions=["Provide a clear, concise answer to the medical question"],
            model=self.hf_model,
            include_reason=True
        )
    
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
        
        # Create context from system prompt if available
        context = [system_prompt] if system_prompt else None
        
        # Create test case for DeepEval
        test_case = LLMTestCase(
            input=question,
            actual_output=model_answer,
            expected_output=correct_answer,
            context=context
        )
        
        # Add deepeval metrics
        metrics.update(self._calculate_deepeval_metrics(test_case))
        
        # Print metrics to terminal if verbose mode is enabled
        if self.verbose:
            self.print_metrics(metrics, question, model_answer, correct_answer)
        
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
        
        # Print basic information
        print(f"{Fore.YELLOW}Question:{Style.RESET_ALL} {question[:100]}...")
        print(f"{Fore.YELLOW}Model Answer:{Style.RESET_ALL} {model_answer[:100]}...")
        print(f"{Fore.YELLOW}Correct Answer:{Style.RESET_ALL} {correct_answer[:100]}...")
        
        # Print metrics by category
        for category, metric_names in METRIC_CATEGORIES.items():
            # Skip explanatory metrics in normal output (too verbose)
            if category == 'Explanatory Metrics':
                continue
                
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
                    elif isinstance(value, str) and len(value) > 100:
                        # Truncate long strings
                        print(f"  {metric_name}: {value[:100]}...")
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
            ('correctness_score', 'Factual Correctness'),
            ('relevancy_score', 'Question Relevance'),
            ('hallucination_score', 'Non-Hallucination'),
            ('bias_score', 'Neutrality (Low Bias)'),
            ('toxicity_score', 'Safety (Low Toxicity)')
        ]
        
        # Print key scores with color coding
        for metric_key, metric_label in key_metrics:
            if metric_key in metrics:
                value = metrics[metric_key]
                
                # Color code based on score
                if value >= 0.8:
                    color = Fore.GREEN
                elif value >= 0.6:
                    color = Fore.YELLOW
                elif value >= 0.4:
                    color = Fore.YELLOW
                else:
                    color = Fore.RED
                
                print(f"  {metric_label}: {color}{value:.4f}{Style.RESET_ALL}")
        
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
        
        # Compute embeddings
        question_embedding = self.embedding_model.encode([question], convert_to_tensor=True)
        answer_embedding = self.embedding_model.encode([model_answer], convert_to_tensor=True)
        correct_embedding = self.embedding_model.encode([correct_answer], convert_to_tensor=True)
        
        # Convert to numpy for similarity calculations
        question_embedding_np = question_embedding.cpu().numpy()
        answer_embedding_np = answer_embedding.cpu().numpy()
        correct_embedding_np = correct_embedding.cpu().numpy()
        
        # Question-answer semantic similarity (relevance)
        metrics['semantic_similarity'] = float(cosine_similarity(question_embedding_np, answer_embedding_np)[0][0])
        
        # Semantic similarity between model answer and correct answer
        metrics['answer_similarity'] = float(cosine_similarity(answer_embedding_np, correct_embedding_np)[0][0])
        
        # Basic statistics
        metrics['answer_length'] = len(model_answer.split())
        
        # Readability metrics
        metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(model_answer)
        metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(model_answer)
        
        # Sentiment analysis
        sentiment = TextBlob(model_answer).sentiment
        metrics['sentiment_polarity'] = sentiment.polarity
        metrics['sentiment_subjectivity'] = sentiment.subjectivity
        
        # ROUGE metrics
        rouge_scores = self.rouge.get_scores(model_answer, correct_answer)[0]
        metrics['rouge1_f'] = rouge_scores['rouge-1']['f']
        metrics['rouge2_f'] = rouge_scores['rouge-2']['f']
        metrics['rougeL_f'] = rouge_scores['rouge-l']['f']
        
        # BLEU score
        reference = [correct_answer.split()]
        hypothesis = model_answer.split()
        
        # Use smoothing function to avoid score of 0 for short sentences
        smoothie = SmoothingFunction().method1
        metrics['bleu_score'] = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
        
        # BERTScore
        precision, recall, f1 = self.bert_scorer.score([model_answer], [correct_answer])
        metrics['bertscore_precision'] = float(precision[0])
        metrics['bertscore_recall'] = float(recall[0])
        metrics['bertscore_f1'] = float(f1[0])

        return metrics

    def _calculate_deepeval_metrics(self, test_case: 'LLMTestCase') -> Dict[str, Any]:
        """
        Calculate metrics using the deepeval library with local Hugging Face models.
        
        Args:
            test_case: LLMTestCase instance
        
        Returns:
            Dictionary of deepeval metrics
        """
        metrics = {}
        
        # Bias metric
        self.bias_metric.measure(test_case)
        metrics['bias_score'] = self.bias_metric.score
        
        # Hallucination metric
        self.hallucination_metric.measure(test_case)
        metrics['hallucination_score'] = self.hallucination_metric.score
        
        # Toxicity metric
        self.toxicity_metric.measure(test_case)
        metrics['toxicity_score'] = self.toxicity_metric.score
        
        # Answer relevancy metric
        self.answer_relevancy_metric.measure(test_case)
        metrics['relevancy_score'] = self.answer_relevancy_metric.score
        metrics['relevancy_reason'] = self.answer_relevancy_metric.reason
        
        # Prompt alignment metric
        self.prompt_alignment_metric.measure(test_case)
        metrics['prompt_alignment_score'] = self.prompt_alignment_metric.score
        if hasattr(self.prompt_alignment_metric, 'reason'):
            metrics['prompt_alignment_reason'] = self.prompt_alignment_metric.reason
        
        # Correctness metric
        self.correctness_metric.measure(test_case)
        metrics['correctness_score'] = self.correctness_metric.score
        
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
            {"name": "bertscore_f1", "description": "BERTScore F1 score (semantic similarity)"}
        ]
        
        # DeepEval Metrics
        metrics_docs["DeepEval Metrics"] = [
            {"name": "bias_score", "description": "Score indicating the level of bias in the answer (0-1, higher = less biased)"},
            {"name": "hallucination_score", "description": "Score indicating factuality of the answer (0-1, higher = fewer hallucinations)"},
            {"name": "toxicity_score", "description": "Score indicating the toxicity level of the answer (0-1, higher = less toxic)"},
            {"name": "relevancy_score", "description": "Score indicating how relevant the answer is to the question (0-1)"},
            {"name": "prompt_alignment_score", "description": "Score indicating alignment with the provided instructions (0-1)"},
            {"name": "correctness_score", "description": "Score indicating factual correctness compared to reference answer (0-1)"}
        ]
        
        # Explanatory Metrics
        metrics_docs["Explanatory Metrics"] = [
            {"name": "relevancy_reason", "description": "Explanation of the relevancy score"},
            {"name": "prompt_alignment_reason", "description": "Explanation of the prompt alignment score"}
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
        
        return metrics_docs 