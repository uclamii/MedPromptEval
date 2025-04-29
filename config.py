"""
Configuration file for the prompt generation system.
Contains model configurations and prompt types.
"""

# Model configurations with optimized settings for quality outputs to generate a system prompt
PROMPT_MODEL_CONFIGS = {
    "phi-2": {
        "name": "microsoft/phi-2",
        "description": "Microsoft's Phi-2 model, good for concise answers",
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "model_kwargs": {
            "trust_remote_code": True,
            "offload_folder": "offload",
            "low_cpu_mem_usage": True
        },
        "tokenizer_kwargs": {
            "trust_remote_code": True
        }
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.1",
        "description": "Mistral 7B base model, good for general instruction following",
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "model_kwargs": {
            "trust_remote_code": True,
            "offload_folder": "offload",
            "low_cpu_mem_usage": True
        },
        "tokenizer_kwargs": {
            "trust_remote_code": True
        }
    },
    "llama-3-8b": {
        "name": "meta-llama/Meta-Llama-3-8B",
        "description": "Meta's Llama 3 8B model, good for detailed explanations",
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "model_kwargs": {
            "trust_remote_code": True,
            "offload_folder": "offload",
            "low_cpu_mem_usage": True
        },
        "tokenizer_kwargs": {
            "trust_remote_code": True
        }
    },
    "gemma-2-2b": {
        "name": "google/gemma-2-2b",
        "description": "Google's Gemma 2 2B model, good for efficient inference",
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "model_kwargs": {
            "trust_remote_code": True,
            "offload_folder": "offload",
            "low_cpu_mem_usage": True
        },
        "tokenizer_kwargs": {
            "trust_remote_code": True
        }
    },
    "openbiollm-8b": {
        "name": "aaditya/Llama3-OpenBioLLM-8B",
        "description": "OpenBioLLM 8B model, specialized for biomedical tasks",
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "model_kwargs": {
            "trust_remote_code": True,
            "offload_folder": "offload",
            "low_cpu_mem_usage": True
        },
        "tokenizer_kwargs": {
            "trust_remote_code": True
        }
    }
}

# Model configurations with optimized settings for answering medical questions
ANSWER_MODEL_CONFIGS = {
    "phi-2": {
        "name": "microsoft/phi-2",
        "description": "Microsoft's Phi-2 model, good for concise answers",
        "max_new_tokens": 1024,
        "temperature": 0.3,  # Lower temperature for more factual answers
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "model_kwargs": {
            "trust_remote_code": True,
            "offload_folder": "offload",
            "low_cpu_mem_usage": True
        },
        "tokenizer_kwargs": {
            "trust_remote_code": True
        }
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.1",
        "description": "Mistral 7B base model, good for general instruction following",
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "model_kwargs": {
            "trust_remote_code": True,
            "offload_folder": "offload",
            "low_cpu_mem_usage": True
        },
        "tokenizer_kwargs": {
            "trust_remote_code": True
        }
    },
    "llama-3-8b": {
        "name": "meta-llama/Meta-Llama-3-8B",
        "description": "Meta's Llama 3 8B model, good for detailed explanations",
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "model_kwargs": {
            "trust_remote_code": True,
            "offload_folder": "offload",
            "low_cpu_mem_usage": True
        },
        "tokenizer_kwargs": {
            "trust_remote_code": True
        }
    },
    "gemma-2-2b": {
        "name": "google/gemma-2-2b",
        "description": "Google's Gemma 2 2B model, good for efficient inference",
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "model_kwargs": {
            "trust_remote_code": True,
            "offload_folder": "offload",
            "low_cpu_mem_usage": True
        },
        "tokenizer_kwargs": {
            "trust_remote_code": True
        }
    },
    "openbiollm-8b": {
        "name": "aaditya/Llama3-OpenBioLLM-8B",
        "description": "OpenBioLLM 8B model, specialized for biomedical tasks",
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "model_kwargs": {
            "trust_remote_code": True,
            "offload_folder": "offload",
            "low_cpu_mem_usage": True
        },
        "tokenizer_kwargs": {
            "trust_remote_code": True
        }
    }
}

# Prompt types for medical QA
PROMPT_TYPES = {
    "chain of thought": "Break down your medical reasoning into clear, logical steps. Explain the medical concepts and thought process, ensuring transparency and improving comprehension",
    "trigger chain of thought": "Follow these steps: 1) Identify key elements in the data, 2) Apply relevant medical knowledge, 3) Make logical deductions, 4) Validate conclusions against the data context",
    "self consistency": "Generate multiple medical reasoning paths and cross-check them. Consider different interpretations of the data, evaluate their consistency, and converge on the most reliable conclusion",
    "prompt chaining": "Break down the analysis into sequential steps: 1) Data interpretation, 2) Clinical correlation, 3) Evidence synthesis, 4) Recommendation formulation",
    "react": "Implement an iterative process of: 1) Reasoning about the data, 2) Taking analytical actions, 3) Reflecting on findings, 4) Refining conclusions based on the specific context",
    "tree of thoughts": "Explore multiple analytical branches: 1) Consider various clinical interpretations, 2) Evaluate different diagnostic possibilities, 3) Assess multiple treatment approaches, all within the context of the data",
    "role based": "Assume the role of a specialist most relevant to the medical domain. Provide expert analysis and interpretation specific to this type of medical data",
    "metacognitive prompting": "Explicitly outline your reasoning process: 1) State your assumptions about the data, 2) Explain your analytical approach, 3) Justify your conclusions based on the specific context",
    "uncertainty based prompting": "Clearly indicate any uncertainties or limitations in the data. Specify when additional information would be needed for more confident conclusions",
    "guided prompting": "Structure your analysis through guided questions: What specific patterns are we looking for? What are the key clinical indicators? How does this data inform medical decision-making?"
} 