import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from typing import Dict, List
import os
from pathlib import Path
import json
from dotenv import load_dotenv
from config import PROMPT_MODEL_CONFIGS, PROMPT_TYPES, ensure_hf_login

class PromptGenerator:
    # Model and prompt configurations are imported from config.py
    MODEL_CONFIGS = PROMPT_MODEL_CONFIGS
    PROMPT_TYPES = PROMPT_TYPES

    def __init__(
        self,
        model_key: str,
        use_auth: bool = True
    ):
        """
        Initialize the prompt generator with a specific model.
        """
        # Enable CPU optimizations in PyTorch
        torch.set_num_threads(int(os.cpu_count()) if os.cpu_count() else 4)  # Use all available CPU cores
        
        # Ensure Hugging Face login if needed
        if use_auth:
            ensure_hf_login()
        
        self.model_config = self.MODEL_CONFIGS[model_key]
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
        
        print("Model loaded successfully!")
        
        # Cache for system prompts
        self._prompt_cache = {}
    
    def generate_prompt(
        self,
        prompt_type: str,
        num_prompts: int = 1,
        verbose: bool = True,
        max_retries: int = 3
    ) -> List[str]:
        """
        Generate prompts for a specific prompt type.
        
        Args:
            prompt_type: Type of prompt to generate
            num_prompts: Number of prompts to generate
            verbose: Whether to print detailed generation information
            max_retries: Maximum number of retries for empty prompts
            
        Returns:
            List of generated prompts
        """
        # Get prompt type definition
        prompt_definition = self.PROMPT_TYPES[prompt_type]
        
        # Simple, direct system prompt
        system_prompt = f"""Generate a system prompt for a medical chatbot that answers user questions using the {prompt_type} methodology, defined as: {prompt_definition}. The system prompt should clearly instruct the chatbot on how to respond according to this methodology. Return only the system prompt text—do not include explanations, code, or any additional output."""        
        
        # Print a clear header for prompt generation if verbose
        if verbose:
            print(f"\n{'='*40} PROMPT GENERATION {'='*40}")
            print(f"Generating a system prompt for '{prompt_type}' methodology")
            print(f"Definition: {prompt_definition}")
            print(f"{'='*90}")
        
        # Check cache for tokenized input
        cache_key = f"{prompt_type}"
        if cache_key not in self._prompt_cache:
            # Tokenize with padding
            self._prompt_cache[cache_key] = self.tokenizer(
                system_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Increased for better context
            ).to(self.device)
        
        # Get cached input
        inputs = self._prompt_cache[cache_key]
        
        prompts = []
        for i in range(num_prompts):
            # Generation configuration optimized for quality outputs
            generation_config = {
                "max_new_tokens": self.model_config["max_new_tokens"],
                "do_sample": True,
                "temperature": self.model_config["temperature"],
                "top_p": self.model_config["top_p"],
                "top_k": self.model_config["top_k"],
                "repetition_penalty": self.model_config["repetition_penalty"],
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            # Try generating with retries if needed
            prompt = ""
            retry_count = 0
            while not prompt.strip() and retry_count < max_retries:
                # Generate text
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **generation_config)
                
                # Process the generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the prompt by removing the system prompt template
                prompt = generated_text.replace(system_prompt, "").strip()
                
                # If prompt is empty, increase temperature for next retry
                if not prompt.strip():
                    retry_count += 1
                    if retry_count < max_retries:
                        generation_config["temperature"] = min(1.0, generation_config["temperature"] + 0.1)
                        if verbose:
                            print(f"Empty prompt generated, retrying with higher temperature ({generation_config['temperature']})...")
            
            # If still empty after retries, use a fallback prompt
            if not prompt.strip():
                prompt = f"You are a medical AI assistant. When answering medical questions, use the {prompt_type} methodology: {prompt_definition}. Provide clear, accurate, and helpful responses."
                if verbose:
                    print("Using fallback prompt after max retries")
            
            # Add the prompt
            prompts.append(prompt)
            
            # Print the generated prompt with a clear header if verbose
            if verbose:
                print(f"\n{'-'*40} GENERATED PROMPT ({i+1}/{num_prompts}) {'-'*40}")
                print(prompt)
                print(f"{'-'*90}")
        
        return prompts
    
    def generate_all_prompts(
        self,
        output_dir: str,
        num_prompts_per_type: int = 1
    ) -> Dict[str, List[str]]:
        """
        Generate prompts for all prompt types.
        """
        print("\nGenerating prompts for medical QA")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate prompts for each type
        all_prompts = {}
        
        # Add metadata
        metadata = {
            "model_info": {
                "name": self.model_config["name"],
                "description": self.model_config["description"]
            }
        }
        
        # Print minimal model info
        print(f"Model: {self.model_config['name']} on {self.device}")
        
        # Generate and store prompts
        for prompt_type in self.PROMPT_TYPES:
            prompts = self.generate_prompt(prompt_type, num_prompts_per_type)
            all_prompts[prompt_type] = prompts
        
        # Create final JSON structure
        output_json = {
            "metadata": metadata,
            "prompts": all_prompts
        }
        
        # Save output to JSON file
        json_path = os.path.join(output_dir, "prompts_medical_qa.json")
        with open(json_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        
        print(f"\nGeneration complete. File saved to {json_path}")
        
        return output_json 