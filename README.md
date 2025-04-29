# Medical QA Prompt Generator

A system for generating diverse medical question-answering prompts using different reasoning methodologies and large language models.

## Overview

This project provides a framework for generating high-quality system prompts that guide language models in answering medical questions. It supports multiple reasoning approaches and can work with various Hugging Face models.

## Features

- Supports multiple LLMs (Phi-2, Mistral-7B, Llama-3-8B, Gemma-2-2B, OpenBioLLM)
- Generates prompts using various reasoning methodologies:
  - Chain of Thought
  - Trigger Chain of Thought
  - Self Consistency
  - Prompt Chaining
  - ReAct
  - Tree of Thoughts
  - Role-Based
  - Metacognitive Prompting
  - Uncertainty-Based Prompting
  - Guided Prompting
- Separate optimized configurations for:
  - Prompt generation (creative, diverse)
  - Answer generation (factual, precise)
- Command-line interface for easy experimentation
- JSON output format for further processing
- Evaluation pipeline for testing system prompts on real QA datasets

## Project Structure

- `config.py`: Contains model configurations and prompt type definitions
- `prompt_generation.py`: Core implementation of the `PromptGenerator` class
- `test_pipeline.py`: Command-line interface for running the generator
- `pipeline.py`: Comprehensive pipeline for evaluating generated prompts on question-answering datasets

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Hugging Face account with API token (for some models)
- dotenv
- pandas
- tqdm

## Installation

1. Clone this repository
2. Install requirements:
   ```
   pip install torch transformers huggingface_hub python-dotenv pandas tqdm
   ```
3. Create a `.env` file in the root directory with your Hugging Face token:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```

## Usage

### Basic Usage

Run the generator with default settings (Mistral-7B model):

```bash
python test_pipeline.py
```

### Custom Configuration

Specify different parameters:

```bash
python test_pipeline.py --model phi-2 --output_dir outputs/phi2 --num_prompts 3
```

### Evaluating QA Performance

Evaluate how well the generated system prompts perform on question-answering tasks:

```bash
python pipeline.py --dataset datasets/cleaned/medquad_cleaned.csv --output results/qa_results.csv --prompt-model phi-2 --answer-model mistral-7b
```

### Available Arguments

- `--model`: Choose the model to use (`phi-2`, `mistral-7b`, `llama-3-8b`, `gemma-2-2b`, `openbiollm-8b`)
- `--output_dir`: Directory to save generated prompts
- `--num_prompts`: Number of prompts to generate per type
- `--no_auth`: Run without Hugging Face authentication

### Available Arguments for Pipeline

- `--dataset`: Path to a CSV file containing question-answer pairs (required)
- `--output`: Path for the output CSV file (default: `results/qa_results.csv`)
- `--prompt-model`: Model to use for generating prompts (`phi-2`, `mistral-7b`, etc.)
- `--answer-model`: Model to use for answering questions (`phi-2`, `mistral-7b`, etc.)
- `--prompt-types`: Specific prompt types to use (default: all types)
- `--prompts-per-type`: Number of prompt variations to generate per type (default: 1)
- `--samples`: Number of question-answer pairs to process (default: all)
- `--no-auth`: Run without Hugging Face authentication

## Configuration Details

The system uses two separate model configurations to optimize for different tasks:

1. **PROMPT_MODEL_CONFIGS**: Models optimized for prompt generation
   - Higher temperature (0.7) for creativity
   - Balanced top_p and top_k for diverse suggestions
   - Shorter output (512 tokens) focused on system prompt creation

2. **ANSWER_MODEL_CONFIGS**: Models optimized for answering medical questions
   - Lower temperature (0.3) for factual, precise answers
   - Higher max_new_tokens (1024) for more detailed responses
   - Tuned repetition penalty for natural but focused answers

## Output

### Prompt Generation Output

The system generates a JSON file containing:

- Model metadata
- Generated prompts for each reasoning methodology

Example output structure:

```json
{
  "metadata": {
    "model_info": {
      "name": "mistralai/Mistral-7B-v0.1",
      "description": "Mistral 7B base model, good for general instruction following"
    }
  },
  "prompts": {
    "chain of thought": [
      "You are a medical AI assistant. When answering medical questions, break down your reasoning into clear, logical steps..."
    ],
    "role based": [
      "Assume the role of a medical specialist most relevant to the question being asked..."
    ],
    ...
  }
}
```

### Pipeline Output

The pipeline produces a CSV file with these columns:

- `question`: The original question
- `correct_answer`: The ground truth answer
- `prompt_model`: Which model generated the system prompt
- `prompt_type`: The type of reasoning (chain-of-thought, etc.)
- `system_prompt`: The full system prompt text
- `answer_model`: Which model generated the answer
- `model_answer`: The model's answer to the question

## Extending the System

### Adding New Models

Add new model configurations to the `PROMPT_MODEL_CONFIGS` and/or `ANSWER_MODEL_CONFIGS` dictionaries in `config.py`:

```python
"new-model": {
    "name": "organization/model-name",
    "description": "Description of the model",
    "max_new_tokens": 512,
    "temperature": 0.7,
    ...
}
```

### Adding New Prompt Types

Add new prompt types to the `PROMPT_TYPES` dictionary in `config.py`:

```python
"new prompt type": "Description of the reasoning methodology"
```

## Performance Considerations

- Models are configured to run on CPU by default for stability
- For better performance with larger models, consider using a machine with a GPU
- Adjust generation parameters in `config.py` to balance between quality and speed

## License

[MIT License](LICENSE) 