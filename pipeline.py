import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Union
import yaml
import os
from pathlib import Path
from tqdm import tqdm
import json
from metrics_evaluator import BaseMetricsEvaluator

class PromptGenerator:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the prompt generator with a specific model.
        
        Args:
            model_name (str): Name of the model to use (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
            device (str): Device to run the model on
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
    
    def generate_prompt(self, system_prompt: str, num_prompts: int = 1) -> List[str]:
        """
        Generate prompts using the system prompt.
        
        Args:
            system_prompt (str): The system prompt to use
            num_prompts (int): Number of prompts to generate
            
        Returns:
            List[str]: List of generated prompts
        """
        prompts = []
        for _ in range(num_prompts):
            inputs = self.tokenizer(system_prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompts.append(prompt)
        return prompts

class QuestionAnswerer:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the question answerer with a specific model.
        
        Args:
            model_name (str): Name of the model to use
            device (str): Device to run the model on
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
    
    def answer_question(self, question: str, system_prompt: str) -> str:
        """
        Generate an answer for a question using the system prompt.
        
        Args:
            question (str): The question to answer
            system_prompt (str): The system prompt to use
            
        Returns:
            str: The generated answer
        """
        prompt = f"{system_prompt}\n\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

class EvaluationPipeline:
    def __init__(
        self,
        config_path: str,
        prompt_model_name: str,
        answer_model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config_path (str): Path to the YAML configuration file
            prompt_model_name (str): Name of the model to use for prompt generation
            answer_model_name (str): Name of the model to use for answering questions
            device (str): Device to run the models on
        """
        self.config = self._load_config(config_path)
        self.prompt_generator = PromptGenerator(prompt_model_name, device)
        self.question_answerer = QuestionAnswerer(answer_model_name, device)
        self.metrics_evaluator = BaseMetricsEvaluator(self.config)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load the configuration from a YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_prompts_for_type(self, prompt_type: str, num_prompts: int = 1) -> List[str]:
        """
        Generate prompts for a specific prompt type.
        
        Args:
            prompt_type (str): The type of prompt to generate
            num_prompts (int): Number of prompts to generate per type
            
        Returns:
            List[str]: List of generated prompts
        """
        system_prompt = self.config['system_prompt_template'].format(
            prompt_type=prompt_type,
            definition=self.config['prompt_types'][prompt_type]
        )
        return self.prompt_generator.generate_prompt(system_prompt, num_prompts)
    
    def evaluate_questions(
        self,
        questions_csv: str,
        output_dir: str,
        num_prompts_per_type: int = 1
    ) -> None:
        """
        Run the complete evaluation pipeline.
        
        Args:
            questions_csv (str): Path to the CSV file containing questions
            output_dir (str): Directory to save the results
            num_prompts_per_type (int): Number of prompts to generate per type
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load questions
        questions_df = pd.read_csv(questions_csv)
        questions = questions_df['question'].tolist()
        
        # Generate prompts for each type
        prompt_results = {}
        for prompt_type in self.config['prompt_types']:
            prompts = self.generate_prompts_for_type(prompt_type, num_prompts_per_type)
            prompt_results[prompt_type] = prompts
            
            # Save prompts
            with open(os.path.join(output_dir, f"{prompt_type}_prompts.json"), 'w') as f:
                json.dump(prompts, f, indent=2)
        
        # Generate answers for each prompt type
        for prompt_type, prompts in prompt_results.items():
            answers = []
            for prompt in tqdm(prompts, desc=f"Generating answers for {prompt_type}"):
                type_answers = []
                for question in questions:
                    answer = self.question_answerer.answer_question(question, prompt)
                    type_answers.append(answer)
                answers.append(type_answers)
            
            # Evaluate answers
            for i, prompt_answers in enumerate(answers):
                results_df = self.metrics_evaluator.evaluate_answers(
                    questions=questions,
                    answers=prompt_answers,
                    output_csv=os.path.join(output_dir, f"{prompt_type}_prompt_{i+1}_results.csv")
                )
        
        # Generate summary
        self._generate_summary(output_dir)
    
    def _generate_summary(self, output_dir: str) -> None:
        """Generate a summary of all results."""
        summary = {
            "config": self.config,
            "results": {}
        }
        
        # Collect results from all CSV files
        for file in os.listdir(output_dir):
            if file.endswith("_results.csv"):
                prompt_type = file.split("_prompt_")[0]
                if prompt_type not in summary["results"]:
                    summary["results"][prompt_type] = []
                
                df = pd.read_csv(os.path.join(output_dir, file))
                summary["results"][prompt_type].append({
                    "file": file,
                    "metrics": df.mean().to_dict()
                })
        
        # Save summary
        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    # Example usage
    pipeline = EvaluationPipeline(
        config_path="configs/irb.yaml",
        prompt_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        answer_model_name="meta-llama/Llama-2-7b-chat-hf"
    )
    
    pipeline.evaluate_questions(
        questions_csv="datasets/questions.csv",
        output_dir="results",
        num_prompts_per_type=2
    ) 