from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dataclasses import dataclass
import os
import json
import logging
import yaml

@dataclass
class ModelConfig:
    """Configuration for a language model."""
    name: str
    description: str
    model_id: str
    model_type: str  # 'huggingface', 'openai', 'anthropic', etc.
    max_new_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.2
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    additional_params: Optional[Dict] = None

@dataclass
class UseCaseConfig:
    """Configuration for a specific use case."""
    name: str
    description: str
    domain: str
    prompt_types: Dict[str, str]
    system_prompt_template: str
    evaluation_criteria: str
    evaluation_steps: List[str]
    additional_considerations: Optional[List[str]] = None
    metric_thresholds: Optional[Dict[str, float]] = None
    models: Optional[List[ModelConfig]] = None

class ConfigManager:
    """Manages different types of language models and use cases."""
    
    def __init__(self, model_config_dir: str = "model_configs", use_case_config_dir: str = "use_case_configs"):
        """
        Initialize the config manager.
        
        Args:
            model_config_dir (str): Directory containing model configuration files
            use_case_config_dir (str): Directory containing use case configuration files
        """
        self.model_config_dir = model_config_dir
        self.use_case_config_dir = use_case_config_dir
        os.makedirs(model_config_dir, exist_ok=True)
        os.makedirs(use_case_config_dir, exist_ok=True)
        
        # Initialize configuration storage
        self.model_configs: Dict[str, ModelConfig] = {}
        self.use_case_configs: Dict[str, UseCaseConfig] = {}
        
        # Load configurations
        self._load_model_configs()
        self._load_use_case_configs()
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_model_configs(self):
        """Load all model configurations from the config directory."""
        for filename in os.listdir(self.model_config_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.model_config_dir, filename), 'r') as f:
                    config_data = json.load(f)
                    config = ModelConfig(**config_data)
                    self.model_configs[config.name] = config
    
    def _load_use_case_configs(self):
        """Load all use case configurations from the config directory."""
        for filename in os.listdir(self.use_case_config_dir):
            if filename.endswith('.yaml'):
                with open(os.path.join(self.use_case_config_dir, filename), 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                    # Convert model configurations to ModelConfig objects
                    if 'models' in config_data:
                        config_data['models'] = [
                            ModelConfig(**model_data)
                            for model_data in config_data['models']
                        ]
                    
                    config = UseCaseConfig(**config_data)
                    self.use_case_configs[config.name] = config
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get configuration for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            ModelConfig: Configuration for the model
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.model_configs.keys())}")
        return self.model_configs[model_name]
    
    def get_use_case_config(self, use_case_name: str) -> UseCaseConfig:
        """
        Get configuration for a specific use case.
        
        Args:
            use_case_name (str): Name of the use case
            
        Returns:
            UseCaseConfig: Configuration for the use case
        """
        if use_case_name not in self.use_case_configs:
            raise ValueError(f"Use case '{use_case_name}' not found. Available use cases: {list(self.use_case_configs.keys())}")
        return self.use_case_configs[use_case_name]
    
    def create_model_config(
        self,
        name: str,
        description: str,
        model_id: str,
        model_type: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 1.2,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        additional_params: Optional[Dict] = None
    ) -> ModelConfig:
        """
        Create a new model configuration.
        
        Args:
            name (str): Name of the model
            description (str): Description of the model
            model_id (str): Model identifier
            model_type (str): Type of model ('huggingface', 'openai', 'anthropic', etc.)
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            top_k (int): Top-k sampling parameter
            repetition_penalty (float): Repetition penalty
            api_key (str, optional): API key for the model
            api_base (str, optional): API base URL
            additional_params (Dict, optional): Additional model parameters
            
        Returns:
            ModelConfig: Created configuration
        """
        config = ModelConfig(
            name=name,
            description=description,
            model_id=model_id,
            model_type=model_type,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            api_key=api_key,
            api_base=api_base,
            additional_params=additional_params
        )
        
        # Save to JSON file
        config_path = os.path.join(self.model_config_dir, f"{name.lower().replace(' ', '_')}.json")
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        self.model_configs[name] = config
        return config
    
    def create_use_case_config(
        self,
        name: str,
        description: str,
        domain: str,
        prompt_types: Dict[str, str],
        system_prompt_template: str,
        evaluation_criteria: str,
        evaluation_steps: List[str],
        additional_considerations: Optional[List[str]] = None,
        metric_thresholds: Optional[Dict[str, float]] = None,
        models: Optional[List[ModelConfig]] = None
    ) -> UseCaseConfig:
        """
        Create a new use case configuration.
        
        Args:
            name (str): Name of the use case
            description (str): Description of the use case
            domain (str): Domain of the use case
            prompt_types (Dict[str, str]): Dictionary of prompt types and their descriptions
            system_prompt_template (str): Template for system prompts
            evaluation_criteria (str): Criteria for evaluating answers
            evaluation_steps (List[str]): Steps for evaluating answers
            additional_considerations (List[str], optional): Additional considerations for the use case
            metric_thresholds (Dict[str, float], optional): Custom thresholds for metrics
            models (List[ModelConfig], optional): List of available models for this use case
            
        Returns:
            UseCaseConfig: Created configuration
        """
        config = UseCaseConfig(
            name=name,
            description=description,
            domain=domain,
            prompt_types=prompt_types,
            system_prompt_template=system_prompt_template,
            evaluation_criteria=evaluation_criteria,
            evaluation_steps=evaluation_steps,
            additional_considerations=additional_considerations,
            metric_thresholds=metric_thresholds,
            models=models
        )
        
        # Save to YAML file
        config_path = os.path.join(self.use_case_config_dir, f"{name.lower().replace(' ', '_')}.yaml")
        with open(config_path, 'w') as f:
            # Convert ModelConfig objects to dictionaries for YAML serialization
            config_dict = config.__dict__.copy()
            if config_dict['models']:
                config_dict['models'] = [model.__dict__ for model in config_dict['models']]
            yaml.dump(config_dict, f, default_flow_style=False)
        
        self.use_case_configs[name] = config
        return config
    
    def load_model(self, model_name: str) -> Tuple[Union[AutoModelForCausalLM, Any], Union[AutoTokenizer, Any]]:
        """
        Load a language model based on its configuration.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            Tuple[Union[AutoModelForCausalLM, Any], Union[AutoTokenizer, Any]]: Model and tokenizer
        """
        config = self.get_model_config(model_name)
        self.logger.info(f"Loading model: {model_name}")
        
        if config.model_type == 'huggingface':
            return self._load_huggingface_model(config)
        elif config.model_type == 'openai':
            return self._load_openai_model(config)
        elif config.model_type == 'anthropic':
            return self._load_anthropic_model(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    def _load_huggingface_model(self, config: ModelConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a Hugging Face model."""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                **config.additional_params or {}
            )
            
            tokenizer = AutoTokenizer.from_pretrained(config.model_id)
            tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Error loading Hugging Face model: {str(e)}")
            raise
    
    def _load_openai_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Load an OpenAI model."""
        try:
            import openai
            openai.api_key = config.api_key
            if config.api_base:
                openai.api_base = config.api_base
            
            # OpenAI models don't need a tokenizer
            return None, None
        except Exception as e:
            self.logger.error(f"Error loading OpenAI model: {str(e)}")
            raise
    
    def _load_anthropic_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Load an Anthropic model."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=config.api_key)
            
            # Anthropic models don't need a tokenizer
            return client, None
        except Exception as e:
            self.logger.error(f"Error loading Anthropic model: {str(e)}")
            raise
    
    def generate_text(
        self,
        model_name: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None
    ) -> str:
        """
        Generate text using the specified model.
        
        Args:
            model_name (str): Name of the model to use
            prompt (str): Input prompt
            max_new_tokens (int, optional): Maximum number of new tokens to generate
            temperature (float, optional): Sampling temperature
            top_p (float, optional): Top-p sampling parameter
            top_k (int, optional): Top-k sampling parameter
            repetition_penalty (float, optional): Repetition penalty
            
        Returns:
            str: Generated text
        """
        config = self.get_model_config(model_name)
        model, tokenizer = self.load_model(model_name)
        
        # Use provided parameters or fall back to config defaults
        max_new_tokens = max_new_tokens or config.max_new_tokens
        temperature = temperature or config.temperature
        top_p = top_p or config.top_p
        top_k = top_k or config.top_k
        repetition_penalty = repetition_penalty or config.repetition_penalty
        
        if config.model_type == 'huggingface':
            return self._generate_huggingface_text(
                model, tokenizer, prompt,
                max_new_tokens, temperature, top_p, top_k, repetition_penalty
            )
        elif config.model_type == 'openai':
            return self._generate_openai_text(
                model, prompt,
                max_new_tokens, temperature, top_p, top_k, repetition_penalty
            )
        elif config.model_type == 'anthropic':
            return self._generate_anthropic_text(
                model, prompt,
                max_new_tokens, temperature, top_p, top_k, repetition_penalty
            )
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    def _generate_huggingface_text(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float
    ) -> str:
        """Generate text using a Hugging Face model."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _generate_openai_text(
        self,
        model: Any,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float
    ) -> str:
        """Generate text using an OpenAI model."""
        import openai
        
        response = openai.Completion.create(
            model=self.get_model_config(model).model_id,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=repetition_penalty
        )
        
        return response.choices[0].text.strip()
    
    def _generate_anthropic_text(
        self,
        model: Any,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float
    ) -> str:
        """Generate text using an Anthropic model."""
        response = model.messages.create(
            model=self.get_model_config(model).model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip() 