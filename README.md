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
- Multi-model evaluation capabilities for comprehensive performance comparisons

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
python pipeline.py --dataset datasets/cleaned/medquad_cleaned.csv --output results/qa_results.csv --prompt-models phi-2 --answer-models mistral-7b
```

For comprehensive multi-model evaluation, specify multiple models:

```bash
python pipeline.py --dataset datasets/cleaned/medquad_cleaned.csv --output results/multi_model_results.csv --prompt-models phi-2 mistral-7b --answer-models phi-2 mistral-7b llama-3-8b --prompt-types "chain of thought" "self consistency" --prompts-per-type 2 --samples 5
```

This example would test:
- 2 prompt models × 3 answer models × 2 prompt types × 2 prompts per type = 24 combinations per question
- Across 5 sample questions, resulting in 120 total evaluations

### Available Arguments

- `--model`: Choose the model to use (`phi-2`, `mistral-7b`, `llama-3-8b`, `gemma-2-2b`, `openbiollm-8b`)
- `--output_dir`: Directory to save generated prompts
- `--num_prompts`: Number of prompts to generate per type
- `--no_auth`: Run without Hugging Face authentication

### Available Arguments for Pipeline

- `--dataset`: Path to a CSV file containing question-answer pairs (required)
- `--output`: Path for the output CSV file (default: `results/qa_results.csv`)
- `--prompt-models`: Models to use for generating prompts, can specify multiple (default: `phi-2`)
- `--answer-models`: Models to use for answering questions, can specify multiple (default: `mistral-7b`)
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

### Answer Generation Output

The answer generator produces a JSON file containing:

- Model metadata (including optional prompt model information)
- Results for each question-answer pair

Example output structure:

```json
{
  "metadata": {
    "model_info": {
      "name": "microsoft/phi-2",
      "description": "Microsoft's Phi-2 model, good for concise answers"
    },
    "prompt_model": "mistralai/Mistral-7B-v0.1"
  },
  "results": [
    {
      "question": "What are the symptoms of diabetes?",
      "system_prompt": "You are a medical expert. When answering questions, break down your reasoning into clear, logical steps...",
      "answer": "Diabetes symptoms include increased thirst, frequent urination, unexplained weight loss...",
      "correct_answer": "The symptoms of diabetes include...",
      "prompt_type": "chain of thought"
    },
    {
      "question": "How is pneumonia diagnosed?",
      "system_prompt": "Assume the role of a pulmonologist when answering this medical question...",
      "answer": "Pneumonia is typically diagnosed through a combination of physical examination...",
      "correct_answer": "Pneumonia diagnosis involves...",
      "prompt_type": "role based"
    },
    ...
  ]
}
```

This JSON structure enables:
- Easy comparison between model-generated answers and correct answers
- Analysis of how different system prompts affect answer quality
- Evaluation of model performance across different question types
- Storage of complete question-answer-prompt triplets for later analysis

### Pipeline Output

The pipeline produces a CSV file with these columns:

- `prompt_num`: A sequential number for each prompt generated during execution
- `question`: The original question
- `correct_answer`: The ground truth answer
- `prompt_model`: Which model generated the system prompt
- `prompt_model_key`: The configuration key for the prompt model
- `prompt_type`: The type of reasoning (chain-of-thought, etc.)
- `prompt_variation`: The variation number for this prompt type
- `system_prompt`: The full system prompt text
- `answer_model`: Which model generated the answer
- `answer_model_key`: The configuration key for the answer model
- `model_answer`: The model's answer to the question

The multi-model evaluation capability allows for systematic comparison of:

1. Different prompt generation models
2. Different answer generation models
3. Various prompt types and variations
4. All combinations of the above

This enables comprehensive performance analysis to determine the most effective:
- Prompt generation model for creating system prompts
- Answer generation model for answering medical questions
- Prompt type for eliciting accurate medical reasoning
- Combinations of models and prompt types for specific medical domains

### Metrics Evaluation

The system includes a metrics evaluation component that analyzes the quality of generated answers using various NLP techniques and evaluation frameworks. This allows for quantitative assessment of model performance.

#### Available Metrics

The metrics evaluator provides the following metrics:

**Basic Metrics (Always Available):**
- `semantic_similarity`: Cosine similarity between question and answer embeddings (relevance)
- `answer_length`: Number of words in the answer
- `flesch_reading_ease`: Readability score (higher = easier to read)
- `flesch_kincaid_grade`: US grade level required to understand the text
- `smog_index`: Simple Measure of Gobbledygook readability score
- `sentiment_polarity`: Sentiment of the answer (-1 to +1)
- `sentiment_subjectivity`: Subjectivity of the answer (0 to 1)
- `answer_similarity`: Cosine similarity between model answer and correct answer (if available)

**DeepEval Metrics (Requires deepeval library):**
- `bias_score`: Measurement of bias in the answer
- `hallucination_score`: Detection of information not supported by context
- `toxicity_score`: Measurement of harmful content in the answer
- `relevancy_score`: Assessment of answer relevance to the question (requires OpenAI API)
- `medical_correctness_score`: Evaluation of medical accuracy (requires OpenAI API)

#### Using the Metrics Evaluator

The metrics evaluator can be used as a standalone tool or integrated with the pipeline:

```bash
python metrics_evaluator.py --input results/qa_results.csv --output results/metrics_evaluated.csv
```

**Available Arguments:**
- `--input`: Path to the CSV file with pipeline results (required)
- `--output`: Path to save the evaluation results
- `--embedding-model`: Name of the embedding model to use (default: 'paraphrase-MiniLM-L6-v2')
- `--use-deepeval`: Enable deepeval metrics (requires installation)
- `--openai-api-key`: OpenAI API key for advanced metrics
- `--use-gpu`: Use GPU for embedding computation if available

**Integration with Python Code:**

```python
from metrics_evaluator import integrate_with_pipeline

# Evaluate pipeline results
results_df = integrate_with_pipeline(
    pipeline_results_path="results/qa_results.csv",
    output_path="results/metrics_evaluated.csv",
    embedding_model="paraphrase-MiniLM-L6-v2",
    use_deepeval=True,
    openai_api_key="your-openai-api-key",  # Optional
    use_gpu=True  # Use GPU if available
)

# Analyze results
print(f"Average Semantic Similarity: {results_df['semantic_similarity'].mean():.4f}")
```

The evaluation results will include all original pipeline data plus the computed metrics, allowing for comprehensive analysis of model performance across different prompt types and models.

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
- When running multi-model evaluations, be aware that memory usage increases with each loaded model
- Consider running extensive multi-model evaluations on high-memory machines or in batches

## License

[MIT License](LICENSE) 