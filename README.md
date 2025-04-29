# Medical LLM Question and Answer Evaluation Framework

A comprehensive framework for evaluating and improving large language models for medical question answering through systematic prompt engineering, multi-model evaluation, and detailed metrics analysis.

## Overview

This project provides a complete framework for generating high-quality system prompts that guide language models in answering medical questions, evaluating their performance, and analyzing the results. It's designed to help researchers and practitioners understand which combinations of models and prompting strategies yield the most accurate, relevant, and unbiased medical answers.

### What Problem Does It Solve?

Medical question answering requires both accuracy and appropriate explanation. This framework allows you to:

1. Generate diverse system prompts using different reasoning methodologies
2. Test these prompts with various language models against medical QA datasets
3. Analyze answer quality through comprehensive metrics
4. Identify the most effective model and prompt combinations for medical QA

## Features

- Supports multiple LLMs (Phi-2, Mistral-7B, Llama-3-8B, Gemma-2-2B, OpenBioLLM)
- Generates prompts using various reasoning methodologies:
  - Chain of Thought: Step-by-step reasoning through medical concepts
  - Trigger Chain of Thought: Using prompts that elicit medical reasoning
  - Self Consistency: Generating multiple reasoning paths for verification
  - Prompt Chaining: Breaking complex medical questions into sub-prompts
  - ReAct: Reasoning and acting iteratively for clinical scenarios
  - Tree of Thoughts: Exploring multiple diagnostic or treatment branches
  - Role-Based: Assuming the persona of a relevant medical specialist
  - Metacognitive Prompting: Self-reflection on medical reasoning processes
  - Uncertainty-Based Prompting: Acknowledging knowledge limitations and providing confidence assessments
  - Guided Prompting: Using structured frameworks for medical explanations
- Separate optimized configurations for:
  - Prompt generation (creative, diverse)
  - Answer generation (factual, precise)
- Command-line interface for easy experimentation
- JSON output format for further processing
- Evaluation pipeline for testing system prompts on real QA datasets
- Multi-model evaluation capabilities for comprehensive performance comparisons
- **Incremental CSV writing** to save results as they're generated
- Memory-efficient evaluation with comprehensive NLP metrics
- **Advanced metrics** including entailment checking

## Project Structure

- `config.py`: Contains model configurations and prompt type definitions
- `prompt_generation.py`: Core implementation of the `PromptGenerator` class
- `answer_generation.py`: Handles question answering with different models
- `metrics_evaluator.py`: Evaluation module for comparing model answers with ground truth
- `pipeline.py`: Comprehensive pipeline for evaluating generated prompts on question-answering datasets
- `test_pipeline.py`: Command-line interface for running the generator
- `datasets/`: Directory containing medical QA datasets
- `results/`: Output directory for evaluation results

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Hugging Face account with API token (for some models)
- dotenv
- pandas
- tqdm
- colorama (for terminal output formatting)
- nltk, rouge, bert_score (for NLP metrics)
- sentence_transformers (for semantic embedding)
- textstat (for readability metrics)
- textblob (for sentiment analysis)

## Installation

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
   
   Alternatively, install individual packages:
   ```
   pip install torch transformers huggingface_hub python-dotenv pandas tqdm colorama nltk rouge bert_score sentence_transformers textstat textblob
   ```

3. Create a `.env` file in the root directory with your Hugging Face token:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```
   
   You can get your token from your [Hugging Face account settings](https://huggingface.co/settings/tokens).

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

Generate prompts for specific reasoning types:

```bash
python test_pipeline.py --model phi-2 --prompt_types "chain of thought" "role based" --num_prompts 2
```

### Evaluating QA Performance

Evaluate how well the generated system prompts perform on question-answering tasks:

```bash
python pipeline.py --dataset datasets/cleaned/medquad_cleaned.csv --output results/qa_results.csv --prompt-models phi-2 --answer-models mistral-7b
```

For comprehensive multi-model evaluation, specify multiple models:

```bash
python pipeline.py --dataset datasets/cleaned/medquad_cleaned.csv --output results/multi_model_results.csv --prompt-models phi-2 mistral-7b --answer-models phi-2 mistral-7b llama-3-8b --prompt-types "chain of thought" "self consistency" --prompts-per-type 2 --num-questions 5
```

This example would test:
- 2 prompt models × 3 answer models × 2 prompt types × 2 prompts per type = 24 combinations per question
- Across 5 sample questions, resulting in 120 total evaluations

### Memory Efficiency Options

For running on machines with limited memory, you can disable metrics evaluation entirely:

```bash
python pipeline.py --dataset datasets/cleaned/medquad_cleaned.csv --output results/no_metrics.csv --no-metrics
```

Or just disable the DeepEval metrics which are more memory-intensive:

```bash
python pipeline.py --dataset datasets/cleaned/medquad_cleaned.csv --output results/basic_metrics.csv --no-deepeval
```

### Workflow Examples

#### Example 1: Testing a Single Model with Different Prompt Types

```bash
# Step 1: Generate prompts with Phi-2
python test_pipeline.py --model phi-2 --num_prompts 1 --output_dir outputs/phi2_prompts

# Step 2: Run evaluation with all prompt types
python pipeline.py \
  --dataset datasets/cleaned/medquad_cleaned.csv \
  --output results/phi2_prompt_types.csv \
  --prompt-models phi-2 \
  --answer-models phi-2 \
  --num-questions 20
```

#### Example 2: Comparing Models' Medical QA Capabilities

```bash
# Evaluate all models on the same dataset with the same prompt type
python pipeline.py \
  --dataset datasets/cleaned/medquad_cleaned.csv \
  --output results/model_comparison.csv \
  --prompt-models phi-2 \
  --answer-models phi-2 mistral-7b llama-3-8b gemma-2-2b openbiollm-8b \
  --prompt-types "chain of thought" \
  --num-questions 25
```

#### Example 3: Optimizing for Low-Resource Environments

```bash
# Run with minimal memory usage
python pipeline.py \
  --dataset datasets/cleaned/medquad_cleaned.csv \
  --output results/low_resource.csv \
  --prompt-models phi-2 \
  --answer-models phi-2 \
  --prompt-types "chain of thought" \
  --num-questions 10 \
  --no-deepeval \
  --exclude-long-text \
  --no-verbose
```

### Available Arguments

#### For Prompt Generation (test_pipeline.py)

- `--model`: Choose the model to use (`phi-2`, `mistral-7b`, `llama-3-8b`, `gemma-2-2b`, `openbiollm-8b`)
- `--output_dir`: Directory to save generated prompts
- `--num_prompts`: Number of prompts to generate per type
- `--prompt_types`: Specific prompt types to generate (default: all types)
- `--no_auth`: Run without Hugging Face authentication

#### For Evaluation Pipeline (pipeline.py)

- `--dataset`: Path to a CSV file containing question-answer pairs (required)
- `--output`: Path for the output CSV file (default: `results/qa_results.csv`)
- `--prompt-models`: Models to use for generating prompts, can specify multiple (default: `phi-2`)
- `--answer-models`: Models to use for answering questions, can specify multiple (default: `mistral-7b`)
- `--prompt-types`: Specific prompt types to use (default: all types)
- `--prompts-per-type`: Number of prompt variations to generate per type (default: 1)
- `--num-questions`: Number of question-answer pairs to process (default: all)
- `--no-auth`: Run without Hugging Face authentication
- `--no-metrics`: Disable all metrics evaluation
- `--no-deepeval`: Disable DeepEval metrics for lower memory usage
- `--exclude-long-text`: Exclude long text fields from CSV output
- `--no-verbose`: Disable colorized metrics display in the terminal
- `--list-metrics`: List all available metrics with descriptions and exit

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
- Various metrics columns (see Metrics Evaluation section)

#### Incremental CSV Writing

The pipeline writes results to the CSV file incrementally after each question-answer evaluation, rather than saving everything at the end. This provides several benefits:

- **Crash resistance**: If the pipeline is interrupted or crashes, all processed results up to that point are already saved
- **Progress visibility**: You can open the CSV file while the pipeline is running to see current results
- **Reduced memory usage**: The pipeline doesn't need to keep all results in memory

After each result is processed, you'll see output like this:
```
✓ Result #42 saved: Q5, chain of thought (2/3), Phi-2 → Mistral-7B
```

This indicates that:
- Result #42 has been saved to the CSV file
- It's for question #5
- Using the "chain of thought" prompt type, variation 2 of 3
- Generated with Phi-2 as the prompt model and Mistral-7B as the answer model

### Metrics Evaluation

The system includes a metrics evaluation component that analyzes the quality of generated answers using various NLP techniques. This allows for quantitative assessment of model performance without requiring external APIs.

#### Available Metrics

The metrics evaluator provides the following metrics:

**NLP Metrics:**
- `semantic_similarity`: Cosine similarity between question and answer embeddings (relevance)
- `answer_similarity`: Cosine similarity between model answer and correct answer
- `answer_length`: Number of words in the answer
- `flesch_reading_ease`: Readability score (higher = easier to read)
- `flesch_kincaid_grade`: US grade level required to understand the text
- `sentiment_polarity`: Sentiment of the answer (-1 to +1)
- `sentiment_subjectivity`: Subjectivity of the answer (0 to 1)

**Text Comparison Metrics:**
- `rouge1_f`, `rouge2_f`, `rougeL_f`: ROUGE metrics for text overlap assessment
- `bleu_score`: BLEU score for measuring precision
- `bertscore_precision`, `bertscore_recall`, `bertscore_f1`: BERTScore metrics for semantic evaluation
- `entailment_score`: Score indicating whether the model answer entails (is consistent with) the correct answer
- `entailment_label`: Classification label: "entailment", "neutral", or "contradiction"

**Comparative Metrics:**
- `comparison_answer_length_delta`: Difference in length between model and correct answers
- `comparison_flesch_reading_ease_delta`: Difference in readability
- `comparison_flesch_kincaid_grade_delta`: Difference in grade level
- `comparison_sentiment_polarity_delta`: Difference in sentiment
- `comparison_sentiment_subjectivity_delta`: Difference in subjectivity
- `comparison_relevance_delta`: Difference in question relevance
- `comparison_summary`: Overall summary of key differences

**Reference Metrics:**
For each model metric, there's a corresponding `correct_` version that provides the same measurement for the reference answer, enabling direct comparison.

#### Colorized Terminal Output

When verbose mode is enabled (default), metrics are displayed with color coding in the terminal:
- Green: Good scores
- Yellow: Moderate scores
- Red: Poor scores

This provides immediate visual feedback on answer quality during evaluation.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Use the `--no-deepeval` flag to disable memory-intensive metrics
   - Run with fewer models loaded simultaneously
   - Reduce batch sizes or process fewer questions with `--num-questions`

2. **Model Loading Failures**
   - Ensure you have a valid Hugging Face token in your `.env` file
   - Check if you have sufficient disk space for model downloads
   - Try using the `--no-auth` flag if working with fully public models

3. **Slow Execution**
   - Use smaller models (e.g., phi-2 instead of llama-3-8b)
   - Reduce the number of prompt variations with `--prompts-per-type 1`
   - Sample fewer questions with `--num-questions`

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

### Adding New Metrics

Extend the `metrics_evaluator.py` file to include additional evaluation metrics:

1. Implement a new metric calculation method
2. Add the metric to the appropriate category in `METRIC_CATEGORIES`
3. Update the documentation in `get_metrics_documentation`

### Adding Custom Metrics

To add a new custom metric:

```python
# 1. Add the metric name to METRIC_CATEGORIES in metrics_evaluator.py
METRIC_CATEGORIES = {
    'Custom Metrics': [
        'my_new_metric',
        # ... other metrics
    ],
    # ... other categories
}

# 2. Implement the calculation in _calculate_nlp_metrics method
def _calculate_nlp_metrics(self, question, model_answer, correct_answer):
    # ... existing code ...
    
    # Calculate your custom metric
    metrics['my_new_metric'] = calculate_my_metric(model_answer, correct_answer)
    
    return metrics

# 3. Update documentation
def get_metrics_documentation(self):
    metrics_docs = {}
    # ... existing code ...
    
    metrics_docs["Custom Metrics"] = [
        {"name": "my_new_metric", "description": "Description of what this metric measures"}
    ]
    
    return metrics_docs
```

Access your new metric in the evaluation results CSV or through the metrics dictionary.

## Performance Considerations

- Models are configured to run on CPU by default for stability
- For better performance with larger models, consider using a machine with a GPU
- Adjust generation parameters in `config.py` to balance between quality and speed
- When running multi-model evaluations, be aware that memory usage increases with each loaded model
- Consider running extensive multi-model evaluations on high-memory machines or in batches

## Real-World Applications

This framework can be used for:

1. **Medical LLM Research**:
   - Benchmark different medical LLMs on standardized datasets
   - Identify optimal prompting strategies for different medical domains

2. **Medical Education**:
   - Evaluate LLMs for patient education content generation
   - Ensure medical explanations are accurate and at appropriate reading levels

3. **Clinical Decision Support**:
   - Test how well LLMs reason about medical cases
   - Identify and reduce bias in medical recommendations

4. **Healthcare Documentation**:
   - Assess models for medical summarization tasks
   - Evaluate factual consistency between source documents and model outputs

## License

[MIT License](LICENSE) 