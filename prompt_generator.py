import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Dict, List, Optional
from config import ConfigManager, UseCaseConfig, ModelConfig

class BasePromptGenerator:
    """Base class for generating prompts."""
    
    def __init__(self, config: UseCaseConfig, model_name: Optional[str] = None):
        """
        Initialize the prompt generator.
        
        Args:
            config (UseCaseConfig): Use case configuration
            model_name (str, optional): Name of the model to use. If None, uses the first model in the config.
        """
        self.config = config
        self.config_manager = ConfigManager()
        
        # Get model configuration
        if model_name:
            self.model_config = self.config_manager.get_model_config(model_name)
        elif config.models:
            self.model_config = config.models[0]
        else:
            raise ValueError("No model specified and no models in use case configuration")
    
    def get_system_prompt_template(self) -> str:
        """Get the system prompt template."""
        return self.config.system_prompt_template
    
    def generate_prompt(self, prompt_type: str) -> str:
        """
        Generate a prompt for a specific type.
        
        Args:
            prompt_type (str): Type of prompt to generate
            
        Returns:
            str: Generated prompt
        """
        if prompt_type not in self.config.prompt_types:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        
        definition = self.config.prompt_types[prompt_type]
        template = self.get_system_prompt_template()
        
        return template.format(
            prompt_type=prompt_type,
            definition=definition
        )
    
    def generate_all_prompts(self, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Generate prompts for all types.
        
        Args:
            output_csv (str, optional): Path to save the generated prompts CSV file
            
        Returns:
            pd.DataFrame: DataFrame containing the generated prompts
        """
        prompts = {}
        for ptype in self.config.prompt_types:
            prompts[ptype] = self.generate_prompt(ptype)
        
        df = pd.DataFrame.from_dict(prompts, orient='index', columns=['Prompt'])
        if output_csv:
            df.to_csv(output_csv, index=True)
            print(f"Prompts successfully saved to {output_csv}")
        
        return df 