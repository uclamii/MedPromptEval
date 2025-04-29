#!/usr/bin/env python
"""
Answer Generation Module

This module contains the AnswerGenerator class, which is responsible for
generating answers to medical questions using various LLMs and system prompts.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import os
from pathlib import Path
from huggingface_hub import login
from dotenv import load_dotenv

from config import ANSWER_MODEL_CONFIGS, ensure_hf_login

class AnswerGenerator:
    """
    Class for generating answers to medical questions using various LLMs and system prompts.
    Uses configurations optimized for accurate answer generation.
    """
    
    # Model configurations are imported from config.py
    ANSWER_MODEL_CONFIGS = ANSWER_MODEL_CONFIGS

    def __init__(
        self,
        model_key: str,
        use_auth: bool = True
    ):
        """
        Initialize the answer generator with a specific model.
        
        Args:
            model_key: Key for the model in ANSWER_MODEL_CONFIGS
            use_auth: Whether to use authentication with Hugging Face
        """
        # Enable CPU optimizations in PyTorch
        torch.set_num_threads(int(os.cpu_count()) if os.cpu_count() else 4)  # Use all available CPU cores
        
        # Ensure Hugging Face login if needed
        if use_auth:
            ensure_hf_login()
        
        self.model_config = self.ANSWER_MODEL_CONFIGS[model_key]
        self.model_key = model_key
        self.device = "cpu"  # Always use CPU for stability
        
        # Create offload directory if needed
        os.makedirs(self.model_config["model_kwargs"]["offload_folder"], exist_ok=True)
        
        # Configure model parameters for CPU
        model_kwargs = {
            "torch_dtype": torch.float32,
            "device_map": None
        }
        
        # Update with model-specific kwargs
        model_kwargs.update(self.model_config["model_kwargs"])
        
        # Initialize tokenizer and model
        print(f"\nLoading tokenizer for {self.model_config['name']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["name"],
            use_auth_token=True if use_auth else None,
            **self.model_config["tokenizer_kwargs"]
        )
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = "<|endoftext|>"
                self.tokenizer.add_special_tokens({'pad_token': "<|endoftext|>"})
        
        print(f"Loading model {self.model_config['name']} on CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config["name"],
            use_auth_token=True if use_auth else None,
            **model_kwargs
        ).to(self.device)
        
        # Update model config with pad token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Ensure the model is in evaluation mode
        self.model.eval()
        
        # Pre-compile the model with torch.compile if available
        if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
            try:
                print("Compiling model for faster CPU inference...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                print("Model compilation not available, continuing with standard model")
        
        print("Answer generation model loaded successfully!")
    
    def generate_answer(
        self,
        system_prompt: str,
        question: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        repetition_penalty: float = None,
    ) -> str:
        """
        Generate an answer to a question using the provided system prompt.
        
        Args:
            system_prompt: The system prompt to guide the model
            question: The question to answer
            max_new_tokens: Maximum number of tokens to generate (default: from config)
            temperature: Temperature for sampling (default: from config)
            top_p: Top-p sampling parameter (default: from config)
            top_k: Top-k sampling parameter (default: from config)
            repetition_penalty: Penalty for repetition (default: from config)
            
        Returns:
            Generated answer to the question
        """
        # Create the full prompt by combining system prompt and question
        full_prompt = f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"
        
        # Print full system prompt and question
        print(f"\n{'='*40} SYSTEM PROMPT {'='*40}")
        print(system_prompt)
        print(f"{'='*40} QUESTION {'='*40}")
        print(question)
        print(f"{'='*90}")
        
        # Tokenize the prompt
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Fixed context length
        ).to(self.device)
        
        # Set up generation config, using provided parameters or defaults from config
        generation_config = {
            "max_new_tokens": max_new_tokens or self.model_config["max_new_tokens"],
            "do_sample": True,
            "temperature": temperature or self.model_config["temperature"],
            "top_p": top_p or self.model_config["top_p"],
            "top_k": top_k or self.model_config["top_k"],
            "repetition_penalty": repetition_penalty or self.model_config["repetition_penalty"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        print(f"\nGenerating answer using {self.model_config['name']}...")
        
        # Generate the answer
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the answer (everything after "Answer:")
        answer = generated_text.split("Answer:", 1)[-1].strip()
        
        # Print full answer (no preview)
        print(f"\n{'='*40} FULL ANSWER {'='*40}")
        print(answer)
        print(f"{'='*90}")
        
        return answer
    
    def batch_generate_answers(
        self,
        system_prompts: List[str],
        questions: List[str]
    ) -> List[str]:
        """
        Generate answers for multiple system prompt / question pairs.
        
        Args:
            system_prompts: List of system prompts to use
            questions: Corresponding list of questions to answer
            
        Returns:
            List of generated answers
        """
        if len(system_prompts) != len(questions):
            raise ValueError("The number of system prompts must match the number of questions")
        
        answers = []
        for i, (system_prompt, question) in enumerate(zip(system_prompts, questions)):
            print(f"\nProcessing pair {i+1}/{len(questions)}")
            answer = self.generate_answer(
                system_prompt=system_prompt,
                question=question
            )
            answers.append(answer)
        
        return answers 