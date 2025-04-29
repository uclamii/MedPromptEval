import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import textstat
from textblob import TextBlob
from deepeval import evaluate
from deepeval.metrics import BiasMetric, HallucinationMetric, AnswerRelevancyMetric, ToxicityMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import os
from typing import Dict, Any, Optional, List
from .archive.config import UseCaseConfig

class BaseMetricsEvaluator:
    """Base class for evaluating answers."""
    
    def __init__(self, config: UseCaseConfig):
        """
        Initialize the metrics evaluator.
        
        Args:
            config (UseCaseConfig): Use case configuration
        """
        self.config = config
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single answer.
        
        Args:
            question (str): The question
            answer (str): The answer to evaluate
            ground_truth (str, optional): Ground truth answer for comparison
            
        Returns:
            Dict[str, float]: Dictionary of metric scores
        """
        metrics = {
            'Relevance': self._calculate_relevance(question, answer),
            'Readability': self._calculate_readability(answer),
            'Bias_Score': self._calculate_bias(answer),
            'Hallucination_Score': self._calculate_hallucination(answer, ground_truth)
        }
        
        # Apply thresholds if defined
        if self.config.metric_thresholds:
            for metric, threshold in self.config.metric_thresholds.items():
                if metric in metrics:
                    metrics[metric] = 1.0 if metrics[metric] >= threshold else 0.0
        
        return metrics
    
    def evaluate_answers(
        self,
        questions: List[str],
        answers: List[str],
        ground_truths: Optional[List[str]] = None,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate multiple answers.
        
        Args:
            questions (List[str]): List of questions
            answers (List[str]): List of answers to evaluate
            ground_truths (List[str], optional): List of ground truth answers
            output_csv (str, optional): Path to save the evaluation results
            
        Returns:
            pd.DataFrame: DataFrame containing evaluation results
        """
        if len(questions) != len(answers):
            raise ValueError("Number of questions must match number of answers")
        
        if ground_truths and len(ground_truths) != len(questions):
            raise ValueError("Number of ground truths must match number of questions")
        
        results = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            ground_truth = ground_truths[i] if ground_truths else None
            metrics = self.evaluate_answer(question, answer, ground_truth)
            
            results.append({
                'Question': question,
                'Answer': answer,
                'Ground_Truth': ground_truth,
                **metrics
            })
        
        df = pd.DataFrame(results)
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Evaluation results saved to {output_csv}")
        
        return df
    
    def _calculate_relevance(self, question: str, answer: str) -> float:
        """Calculate relevance score between question and answer."""
        # TODO: Implement relevance calculation
        return 0.5
    
    def _calculate_readability(self, answer: str) -> float:
        """Calculate readability score of the answer."""
        # TODO: Implement readability calculation
        return 0.5
    
    def _calculate_bias(self, answer: str) -> float:
        """Calculate bias score of the answer."""
        # TODO: Implement bias calculation
        return 0.5
    
    def _calculate_hallucination(self, answer: str, ground_truth: Optional[str]) -> float:
        """Calculate hallucination score of the answer."""
        # TODO: Implement hallucination calculation
        return 0.5

    def get_evaluation_criteria(self) -> str:
        """Get the evaluation criteria from the configuration."""
        return self.config.evaluation_criteria

    def get_evaluation_steps(self) -> List[str]:
        """Get the evaluation steps from the configuration."""
        return self.config.evaluation_steps

    def calculate_metrics(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Calculate all metrics for a question-answer pair.
        
        Args:
            question (str): The input question
            answer (str): The generated answer
        
        Returns:
            Dict[str, Any]: Dictionary containing all calculated metrics
        """
        metrics = {}
        if pd.notna(answer):
            # Calculate embeddings
            question_embedding = self.embedding_model.encode([question])
            answer_embedding = self.embedding_model.encode([answer])

            # Basic metrics
            metrics['Relevance'] = cosine_similarity(question_embedding, answer_embedding)[0][0]
            metrics['Length'] = len(answer.split())
            metrics['Readability'] = textstat.flesch_reading_ease(answer)
            metrics['Sentiment'] = TextBlob(answer).sentiment.polarity

            # Initialize correctness metric
            correctness_metric = GEval(
                name="Correctness",
                criteria=self.get_evaluation_criteria(),
                evaluation_steps=self.get_evaluation_steps(),
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
            )

            # Deep evaluation metrics
            test_case = LLMTestCase(input=question, actual_output=answer)
            
            try:
                self.bias_metric.measure(test_case)
                metrics['Bias_Score'] = self.bias_metric.score
            except Exception:
                metrics['Bias_Score'] = None

            try:
                hallucination_test_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    context=[question]
                )
                self.hallucination_metric.measure(hallucination_test_case)
                metrics['Hallucination_Score'] = self.hallucination_metric.score
            except Exception:
                metrics['Hallucination_Score'] = None

            try:
                self.toxicity_metric.measure(test_case)
                metrics['Toxicity_Score'] = self.toxicity_metric.score
            except Exception:
                metrics['Toxicity_Score'] = None

            try:
                self.answer_relevancy_metric.measure(test_case)
                metrics['Relevancy_Score'] = self.answer_relevancy_metric.score
            except Exception:
                metrics['Relevancy_Score'] = None

            try:
                correctness_metric.measure(test_case)
                metrics['GEval_Score'] = correctness_metric.score
            except Exception:
                metrics['GEval_Score'] = None

        return metrics

    def calculate_metrics_for_answers(
        self,
        questions: List[str],
        answers: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate metrics for a list of question-answer pairs.
        
        Args:
            questions (List[str]): List of questions
            answers (List[str]): List of answers
            ground_truths (List[str], optional): List of ground truth answers
        
        Returns:
            pd.DataFrame: DataFrame containing calculated metrics
        """
        results = []
        
        for question, answer in zip(questions, answers):
            ground_truth = ground_truths[questions.index(question)] if ground_truths else None
            metrics = self.calculate_metrics(question, answer)
            results.append({
                "Question": question,
                "Answer": answer,
                "Ground_Truth": ground_truth,
                **metrics
            })
        
        df = pd.DataFrame(results)
        return df 