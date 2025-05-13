# MedPromptEval - A Medical LLM Question and Answer Evaluation Framework

MedPromptEval is a comprehensive framework for evaluating and improving large language models for medical question answering through systematic prompt engineering, multi-model evaluation, and detailed metrics analysis.

## Overview

MedPromptEval provides a complete framework for generating high-quality system prompts that guide language models in answering medical questions, evaluating their performance, and analyzing the results. It's designed to help researchers and practitioners understand which combinations of models and prompting strategies yield the most accurate, relevant, and unbiased medical answers.

![image](https://github.com/user-attachments/assets/2946baf4-e414-44aa-81c9-baf0c3a9618d)

### What Problem Does It Solve?

Medical question answering requires both accuracy and appropriate explanation. This framework allows you to:

1. Generate diverse system prompts using different reasoning methodologies
2. Test these prompts with various language models against medical QA datasets
3. Analyze answer quality through comprehensive metrics
4. Identify the most effective model and prompt combinations for medical QA

## Real-World Applications

MedPromptEval can be used for:

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

## Features

- Supports multiple LLMs 
  - Phi-2
  - Mistral-7B
  - Llama-3-8B
  - Llama-3.2-1B
  - DeepSeek-R1-Distill-Qwen-1.5B
  - Qwen3-1.7B
  - Gemma-3-1B-IT
  - Granite-3.3-2B
  - Gemma-2-2B
  - OpenBioLLM-8B
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
- Memory-efficient evaluation with comprehensive NLP metrics
- **Multiple Prompt Types**: Supports various prompt engineering techniques including Chain of Thought, Self-Consistency, ReAct, and more
- **Comprehensive Metrics**: Evaluates answers using multiple metrics including semantic similarity, ROUGE scores, BLEU score, and BERTScore
- **Flexible Model Support**: Works with any Hugging Face model
- **Incremental Processing**: Results are written to CSV immediately after each evaluation, providing:
  - Crash resistance
  - Progress visibility
  - Memory efficiency
- **Resume Capability**: Continue processing from any question using the `--resume-from` argument
- **CSV Output Handling**:
  - Automatic directory creation
  - Safe append mode
  - Header management
  - Long text handling
- **Advanced Visualization**: Comprehensive analysis tools with:
  - Model comparisons
  - Prompt type analysis
  - Metric distributions
  - Correlation analysis
  - Best configurations analysis
  - Normalized metric scoring

## Project Structure

- `config.py`: Contains model configurations and prompt type definitions
- `prompt_generation.py`: Core implementation of the `PromptGenerator` class
- `answer_generation.py`: Handles question answering with different models
- `metrics_evaluator.py`: Evaluation module for comparing model answers with ground truth
- `pipeline.py`: Comprehensive pipeline for evaluating generated prompts on question-answering datasets
- `test_pipeline.py`: Command-line interface for running the generator
- `visualizer.py`: Visualization tools for analyzing evaluation results
- `summarize_experiments.py`: Automated analysis and visualization of experiment results.
- `datasets/`: Directory containing medical QA datasets
- `results/`: Output directory for evaluation results
- `visualizations/`: Generated charts and plots from analysis

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
- sentence-transformers (for semantic embedding)
- textstat (for readability metrics)
- textblob (for sentiment analysis)
- matplotlib, seaborn (for data visualization)
- scipy (for statistical analysis)
- numpy (for numerical operations)

## Installation

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
   
   Alternatively, install individual packages:
   ```
   pip install torch transformers huggingface_hub python-dotenv pandas tqdm colorama nltk rouge bert_score sentence_transformers textstat textblob matplotlib seaborn scipy numpy
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
python pipeline.py --dataset datasets/cleaned/medquad.csv --output results/multi_model_results.csv --prompt-models phi-2 mistral-7b --answer-models phi-2 mistral-7b llama-3-8b --prompt-types "chain of thought" "self consistency" --prompts-per-type 2 --num-questions 5
```

This example would test:
- 2 prompt models × 3 answer models × 2 prompt types × 2 prompts per type = 24 combinations per question
- Across 5 sample questions, resulting in 120 total evaluations

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
  --answer-models phi-2 mistral-7b llama-3-8b llama-3.2-1b deepseek-qwen-1.5b qwen3-1.7b gemma-3-1b-it granite-3.3-2b gemma-2-2b openbiollm-8b \
  --prompt-types "chain of thought" \
  --num-questions 25
```

#### Example 3: Comprehensive Evaluation Across All Models and Prompt Types

```bash
# Run with all available models and prompt types
python pipeline.py \
  --dataset datasets/cleaned/medquad_cleaned.csv \
  --output results/comprehensive_evaluation.csv \
  --prompt-models phi-2 mistral-7b llama-3-8b llama-3.2-1b deepseek-qwen-1.5b qwen3-1.7b gemma-3-1b-it granite-3.3-2b gemma-2-2b openbiollm-8b \
  --answer-models phi-2 mistral-7b llama-3-8b llama-3.2-1b deepseek-qwen-1.5b qwen3-1.7b gemma-3-1b-it granite-3.3-2b gemma-2-2b openbiollm-8b \
  --prompt-types "chain of thought" "trigger chain of thought" "self consistency" "prompt chaining" "react" "tree of thoughts" "role based" "metacognitive prompting" "uncertainty based prompting" "guided prompting" \
  --prompts-per-type 2 \
  --num-questions 10
```

#### Example 4: Optimizing for Low-Resource Environments

```bash
# Run with minimal memory usage
python pipeline.py \
  --dataset datasets/cleaned/medquad_cleaned.csv \
  --output results/low_resource.csv \
  --prompt-models phi-2 \
  --answer-models phi-2 \
  --prompt-types "chain of thought" \
  --num-questions 10 \
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
- `--no-deepeval`: Disable DeepEval metrics to reduce memory usage while still calculating basic NLP metrics
- `--exclude-long-text`: Exclude long text fields from CSV output
- `--no-verbose`: Disable colorized metrics display in the terminal
- `--list-metrics`: List all available metrics with descriptions and exit

## Configuration Details

The system uses two separate model configurations to optimize for different tasks:

1. **PROMPT_MODEL_CONFIGS**: Models optimized for prompt generation
   - Higher temperature (0.7) for creativity
   - Balanced top_p (0.9) and top_k (50) for diverse suggestions
   - Shorter output (512 tokens) focused on system prompt creation
   - Higher repetition penalty (1.2) to avoid repetitive patterns
   - Optimized for generating structured, clear instructions

2. **ANSWER_MODEL_CONFIGS**: Models optimized for answering medical questions
   - Lower temperature (0.3) for factual, precise answers
   - Higher max_new_tokens (1024) for more detailed responses
   - Lower top_p (0.7) and top_k (40) for more focused outputs
   - Balanced repetition penalty (1.1) for natural but focused answers
   - Optimized for generating comprehensive, accurate medical explanations

Each model configuration can be customized in `config.py` to fine-tune the generation parameters for specific use cases.

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
- **Reliable execution**: Even with large-scale evaluations across multiple models, your results are saved as they're generated

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

The system includes a comprehensive metrics evaluation component that analyzes the quality of generated answers using various NLP techniques. This allows for quantitative assessment of model performance without requiring external APIs.

#### Available Metrics

The metrics evaluator provides the following metrics:

**Semantic and Relevance Metrics:**
- `semantic_similarity`: Cosine similarity between question and answer embeddings (relevance)
- `answer_similarity`: Cosine similarity between model answer and correct answer
- `entailment_score`: Score indicating whether the model answer entails (is consistent with) the correct answer
- `entailment_label`: Classification label: "entailment", "neutral", or "contradiction"

**Text Comparison Metrics:**
- `rouge1_f`, `rouge2_f`, `rougeL_f`: ROUGE metrics for text overlap assessment
- `bleu_score`: BLEU score for measuring precision
- `bertscore_precision`, `bertscore_recall`, `bertscore_f1`: BERTScore metrics for semantic evaluation

**Readability and Style Metrics:**
- `answer_length`: Number of words in the answer
- `flesch_reading_ease`: Readability score (higher = easier to read)
- `flesch_kincaid_grade`: US grade level required to understand the text
- `sentiment_polarity`: Sentiment of the answer (-1 to +1)
- `sentiment_subjectivity`: Subjectivity of the answer (0 to 1)

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

## Analyzing Results

The evaluation results are saved to CSV files with comprehensive metrics. You can analyze these results using:

1. **Spreadsheet Applications**: Open in Excel, Google Sheets, etc. for basic filtering and visualization
2. **Data Analysis Libraries**: Use pandas, matplotlib, or other Python libraries for advanced analysis
3. **Built-in Visualization Tools**: Use the provided `visualizer.py` module for comprehensive visual analysis

### Using the Visualizer

The framework includes a powerful visualization module to help analyze and interpret evaluation results. By default, visualizations are saved in the `results/visualizations` directory, but you can customize the output location and organization:

```bash
# Generate a comprehensive report with all visualizations (default location)
python visualizer.py --results results/qa_results.csv

# Generate visualizations in a specific subfolder
python visualizer.py --results results/qa_results.csv --subfolder model_analysis

# Generate visualizations in a custom directory and subfolder
python visualizer.py --results results/qa_results.csv --output-dir results/analysis --subfolder prompt_analysis

# Generate specific visualization types
python visualizer.py --results results/qa_results.csv --report-type basic
python visualizer.py --results results/qa_results.csv --report-type metrics
python visualizer.py --results results/qa_results.csv --report-type prompts

# Analyze a specific model's performance
python visualizer.py --results results/qa_results.csv --report-type model --model mistral-7b
```

#### Output Organization

Visualizations are organized in the following structure:
```
results/
  visualizations/              # Default output directory
    [subfolder if specified]/  # Optional subfolder for organization
      model_comparison_by_answer_similarity.png
      prompt_type_comparison_multi_metric.png
      ...
```

#### Available Report Types

- `comprehensive`: Generate all visualizations (default)
- `basic`: Basic model and prompt type comparisons
- `metrics`: Focus on metric distributions and correlations
- `model`: Detailed analysis of a specific model
- `prompts`: Analysis of prompt types and question difficulty

### Visualization Capabilities

The visualizer creates multiple types of charts and analyses:

1. **Model Comparisons**: Bar charts comparing performance across different models
2. **Prompt Type Analysis**: Compare the effectiveness of different prompt methodologies
3. **Heatmaps**: Visualize performance across combinations of prompt and answer models
4. **Metric Distributions**: Understand the distribution of metric scores
5. **Correlation Matrices**: See relationships between different metrics
6. **Question Difficulty Analysis**: Identify which questions are hardest/easiest
7. **Per-Model Reports**: Detailed performance reports for each model

All visualizations are saved as high-quality PNG files in the specified output directory.

### Using the Visualizer in Python Code

You can also use the visualizer programmatically:

```python
from visualizer import ResultsVisualizer

# Initialize the visualizer with default settings
visualizer = ResultsVisualizer(
    results_path="results/qa_results.csv"
)

# Initialize with custom output organization
visualizer = ResultsVisualizer(
    results_path="results/qa_results.csv",
    output_dir="results/analysis",
    subfolder="model_comparison"
)

# Generate specific visualizations
visualizer.plot_model_comparison(metric="answer_similarity")
visualizer.plot_prompt_type_comparison()
visualizer.plot_heatmap(metric="entailment_score")
visualizer.plot_metric_distributions(by_column="answer_model")
visualizer.plot_correlation_matrix()
visualizer.plot_question_difficulty()

# Generate a comprehensive report
visualizer.generate_comprehensive_report()
```

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

## Results Analysis

After the pipeline completes, it automatically analyzes the results to find the best combinations of models and prompt types. The analysis:

1. Normalizes all metrics to a 0-1 scale for fair comparison
2. Groups metrics into weighted categories:
   - Semantic similarity (30%): How well the answer matches the reference
   - Answer similarity (20%): How similar the answer is to the reference
   - ROUGE scores (15%): Text overlap metrics
   - BERTScore (15%): BERT-based semantic similarity
   - Entailment (20%): Logical consistency
3. Calculates overall scores for each combination
4. Provides detailed reasoning for why each combination performed well

### Analysis Output

The analysis results are saved in two formats:

1. **CSV File** (`results/your_results.csv`):
   - Contains all raw results with individual metrics
   - Includes all prompt variations and model combinations
   - Preserves full text of prompts and answers

2. **Analysis JSON** (`results/your_results.analysis.json`):
   ```json
   {
     "best_combinations": [
       {
         "prompt_model": "model_name",
         "answer_model": "model_name",
         "prompt_type": "prompt_type",
         "scores": {
           "overall": 0.85,
           "semantic_similarity": 0.90,
           "answer_similarity": 0.85,
           "rouge_scores": 0.80,
           "bertscore": 0.88,
           "entailment": 0.82
         },
         "reasoning": "Detailed explanation of why this combination performed well"
       }
     ],
     "metric_weights": {
       "semantic_similarity": 0.3,
       "answer_similarity": 0.2,
       "rouge_scores": 0.15,
       "bertscore": 0.15,
       "entailment": 0.2
     },
     "analysis_summary": {
       "total_combinations": 100,
       "best_overall_score": 0.85,
       "average_overall_score": 0.75
     }
   }
   ```

### Command Line Options

The pipeline includes these additional options for results analysis:

- `--no-analysis`: Disable automatic results analysis after pipeline completion
- `--no-metrics`: Disable metrics evaluation (also disables analysis)
- `--exclude-long-text`: Exclude long text fields from CSV output

### Example Usage

```bash
# Run pipeline with automatic analysis
python pipeline.py \
  --dataset datasets/cleaned/your_dataset.csv \
  --output results/your_results.csv \
  --prompt-models phi-2 mistral-7b \
  --answer-models phi-2 mistral-7b \
  --prompt-types "chain of thought" "role based" \
  --prompts-per-type 2 \
  --num-questions 10

# Run pipeline without analysis
python pipeline.py \
  --dataset datasets/cleaned/your_dataset.csv \
  --output results/your_results.csv \
  --no-analysis
```

### Analysis Output Example

When the pipeline completes, you'll see a summary like this:

```
Top 5 Best Combinations:
================================================================================

1. mistral-7b → phi-2 (chain of thought)
   Overall Score: 0.853
   Reasoning: Excellent semantic similarity with reference answers | High answer similarity indicating good content matching | The mistral-7b model generated effective prompts for the chain of thought methodology | The phi-2 model produced high-quality answers based on these prompts
--------------------------------------------------------------------------------

2. phi-2 → mistral-7b (role based)
   Overall Score: 0.842
   Reasoning: Strong text overlap with reference answers | High BERT-based semantic similarity | The phi-2 model generated effective prompts for the role based methodology | The mistral-7b model produced high-quality answers based on these prompts
--------------------------------------------------------------------------------
```

This analysis helps you:
1. Identify the most effective model combinations
2. Understand which prompt types work best
3. See detailed metrics for each combination
4. Get explanations for why certain combinations performed well 

## Experiment Summarization and Visualization

The `summarize_experiments.py` script provides automated analysis and visualization of experiment results. It processes the output CSVs from all evaluation experiments and generates:

- A summary figure with bar charts and heatmaps comparing prompt types, models, and configurations across datasets.
- A best configuration summary table (`best_configurations_summary.csv`) that includes, for each experiment:
  - Dataset and experiment type
  - Best prompt type, prompt model, and answer model
  - System prompt used
  - All raw metric values for the best configuration
  - The overall weighted score
- Normalized versions of each experiment's results in the `normalized_results/` directory.

### Usage

To run the summarization and generate all outputs:

```bash
python summarize_experiments.py
```

This will create:
- `summary_figure.png`: A multi-panel figure visualizing the main findings
- `best_configurations_summary.csv`: A table of the best configuration and metrics for each experiment
- `normalized_results/`: Folder containing normalized CSVs for each experiment

You can use these outputs directly in your paper or for further analysis.

## Pipeline Features

### Incremental Processing and Resume Capability
The pipeline supports incremental processing and can resume from interruptions:

- **Incremental CSV Writing**: Results are written to CSV immediately after each evaluation, providing:
  - Crash resistance - no data loss if the process is interrupted
  - Progress visibility - results can be monitored in real-time
  - Memory efficiency - results don't need to be held in memory

- **Resume Functionality**: Use the `--resume-from` argument to continue from a specific question:
  ```bash
  python pipeline.py \
    --dataset datasets/cleaned/your_dataset.csv \
    --output results/your_results.csv \
    --prompt-models llama-3.2-1b \
    --answer-models llama-3.2-1b \
    --prompt-types "chain of thought" "guided prompting" \
    --prompts-per-type 5 \
    --num-questions 20 \
    --resume-from 19  # Resume from question #19
  ```

- **Progress Tracking**:
  - Shows overall progress with a progress bar
  - Displays current question being processed
  - Indicates which model combinations are being evaluated
  - Confirms each result being saved to CSV

### CSV Output Handling
The pipeline manages CSV output with the following features:

- **Automatic Directory Creation**: Creates output directories if they don't exist
- **Safe Append Mode**: Never overwrites existing results, always appends new data
- **Header Management**: Properly handles CSV headers for both new and existing files
- **Long Text Handling**: Option to exclude long text fields with `--exclude-long-text`

This ensures that results are saved reliably and efficiently, even when the pipeline is interrupted or when processing large datasets. 

## License

[MIT License](LICENSE)
