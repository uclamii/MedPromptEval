import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Dict, List, Optional
from .config import UseCaseConfig, ModelConfig

class BasePromptGenerator:
    def __init__(
        self,
        config: UseCaseConfig,
        model_name: Optional[str] = None,
        num_prompts: int = 10,
        device: str = None
    ):
        """
        Initialize the prompt generator.
        
        Args:
            config (UseCaseConfig): Configuration for the use case
            model_name (str, optional): Name of the model to use. If None, uses the first model in the config.
            num_prompts (int): Number of prompts to generate per type
            device (str): Device to use for computation ('mps', 'cuda', 'cpu', or None for auto-detection)
        """
        self.config = config
        self.num_prompts = num_prompts
        
        # Select model configuration
        if not config.models:
            raise ValueError("No models configured for this use case")
        
        if model_name:
            self.model_config = next(
                (model for model in config.models if model.name == model_name),
                None
            )
            if not self.model_config:
                raise ValueError(
                    f"Model '{model_name}' not found. Available models: {[m.name for m in config.models]}"
                )
        else:
            self.model_config = config.models[0]
            print(f"Using default model: {self.model_config.name}")
        
        # Device configuration
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        print(f"Using device: {device.upper()}")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            torch_dtype=torch.float16 if device in ["mps", "cuda"] else torch.float32,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Generation configuration
        self.generation_config = {
            "do_sample": True,
            "temperature": self.model_config.temperature,
            "top_p": self.model_config.top_p,
            "top_k": self.model_config.top_k,
            "max_new_tokens": self.model_config.max_new_tokens,
            "repetition_penalty": self.model_config.repetition_penalty,
            "num_return_sequences": 1,
        }

        # Initialize generator
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def get_prompt_types(self) -> Dict[str, str]:
        """Get the prompt types from the configuration."""
        return self.config.prompt_types

    def get_system_prompt_template(self) -> str:
        """Get the system prompt template from the configuration."""
        return self.config.system_prompt_template

    def clean_output(self, text: str) -> str:
        """Extracts the generated prompt from the model output."""
        match = re.search(r'generate a system prompt for the chatbot:\s*(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "(No valid output generated)"

    def generate_prompts(self, prompt_type: str) -> List[str]:
        """Generates optimized prompts using batch processing."""
        prompt_types = self.get_prompt_types()
        if prompt_type not in prompt_types:
            raise ValueError(f"Invalid prompt type. Choose from: {', '.join(prompt_types.keys())}")
        
        definition = prompt_types[prompt_type]
        base_prompt = self.get_system_prompt_template().format(
            prompt_type=prompt_type,
            definition=definition
        )
        
        prompts = []
        for _ in range(self.num_prompts):
            completion = self.generator(
                base_prompt,
                **self.generation_config,
            )
            
            if completion and isinstance(completion, list) and "generated_text" in completion[0]:
                prompts.append(self.clean_output(completion[0]["generated_text"]))
            else:
                prompts.append("(No valid output generated)")
        
        return prompts

    def generate_all_prompts(self, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Generate prompts for all types and save to CSV if specified.
        
        Args:
            output_csv (str, optional): Path to save the generated prompts CSV file
        
        Returns:
            pd.DataFrame: DataFrame containing the generated prompts
        """
        prompt_types = self.get_prompt_types()
        all_prompts = {}
        
        for ptype in prompt_types:
            print(f"Generating prompts for: {ptype}")
            try:
                prompts = self.generate_prompts(ptype)
                all_prompts[ptype] = prompts
            except Exception as e:
                print(f"Error generating {ptype}: {str(e)}")
                all_prompts[ptype] = ["(Error generating prompt)"] * self.num_prompts

        # Convert results to DataFrame
        df = pd.DataFrame.from_dict(
            all_prompts, 
            orient='index', 
            columns=[f"Prompt {i+1}" for i in range(self.num_prompts)]
        )
        
        if output_csv:
            df.to_csv(output_csv, index=True)
            print(f"Prompts successfully saved to {output_csv}")
        
        return df 