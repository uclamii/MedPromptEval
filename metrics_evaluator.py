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
from .config import UseCaseConfig

class BaseMetricsEvaluator:
    def __init__(
        self,
        config: UseCaseConfig,
        embedding_model_name: str = 'paraphrase-MiniLM-L6-v2',
        bias_threshold: float = 0.5,
        hallucination_threshold: float = 0.5,
        relevancy_threshold: float = 0.7,
        toxicity_threshold: float = 0.5
    ):
        """
        Initialize the metrics evaluator.
        
        Args:
            config (UseCaseConfig): Configuration for the use case
            embedding_model_name (str): Name of the sentence transformer model to use
            bias_threshold (float): Threshold for bias detection
            hallucination_threshold (float): Threshold for hallucination detection
            relevancy_threshold (float): Threshold for answer relevancy
            toxicity_threshold (float): Threshold for toxicity detection
        """
        self.config = config
        
        # Override thresholds if specified in config
        if config.metric_thresholds:
            bias_threshold = config.metric_thresholds.get('bias', bias_threshold)
            hallucination_threshold = config.metric_thresholds.get('hallucination', hallucination_threshold)
            relevancy_threshold = config.metric_thresholds.get('relevancy', relevancy_threshold)
            toxicity_threshold = config.metric_thresholds.get('toxicity', toxicity_threshold)

        # Set environment variable to avoid parallelism issues
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Initialize metrics
        self.bias_metric = BiasMetric(threshold=bias_threshold)
        self.hallucination_metric = HallucinationMetric(threshold=hallucination_threshold)
        self.answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=relevancy_threshold,
            model="gpt-4",
            include_reason=True
        )
        self.toxicity_metric = ToxicityMetric(threshold=toxicity_threshold, async_mode=False)

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

    def evaluate_answers(
        self,
        questions: List[str],
        answers: List[str],
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate a list of question-answer pairs.
        
        Args:
            questions (List[str]): List of questions
            answers (List[str]): List of corresponding answers
            output_csv (str, optional): Path to save the evaluation results
        
        Returns:
            pd.DataFrame: DataFrame containing evaluation results
        """
        results = []
        
        for question, answer in zip(questions, answers):
            metrics = self.calculate_metrics(question, answer)
            results.append({
                "Question": question,
                "Answer": answer,
                **metrics
            })
        
        df = pd.DataFrame(results)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Evaluation results saved to {output_csv}")
        
        return df 