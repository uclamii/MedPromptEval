# Evaluation Method Framework

This framework provides a modular approach to evaluating prompt generation and answer quality for different use cases. It uses a configuration-based system to define use cases and models, making it easy to adapt to different domains and LLM models.

## Configuration System

The framework uses two types of configuration files:

### Model Configurations (JSON)
Located in `model_configs/`, these files define LLM models and their parameters:
```json
{
  "name": "Mistral-7B",
  "description": "Mistral 7B Instruct model",
  "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
  "model_type": "huggingface",
  "max_new_tokens": 1000,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 0,
  "repetition_penalty": 1.2
}
```

### Use Case Configurations (YAML)
Located in `use_case_configs/`, these files define use cases and their parameters:
```yaml
name: Your Use Case
description: Description of your use case
domain: Your Domain
prompt_types:
  type1: "Description of prompt type 1"
  type2: "Description of prompt type 2"
  # ...

system_prompt_template: |
  Your template here with {prompt_type} and {definition} placeholders

evaluation_criteria: "Your evaluation criteria"

evaluation_steps:
  - "Step 1"
  - "Step 2"
  # ...

additional_considerations:
  - "Consideration 1"
  - "Consideration 2"
  # ...

metric_thresholds:
  bias: 0.5
  hallucination: 0.5
  relevancy: 0.7
  toxicity: 0.5

models:
  - name: Mistral-7B
    description: "Mistral 7B Instruct model"
    model_id: "mistralai/Mistral-7B-Instruct-v0.3"
    model_type: huggingface
    # ... other model parameters
```

## Base Classes

### ConfigManager
The central configuration manager that handles:
- Model configurations (loading, saving, validation)
- Use case configurations (loading, saving, validation)
- Model loading and text generation
- Support for different model types (Hugging Face, OpenAI, Anthropic)

### BasePromptGenerator
The base class for generating prompts. It provides:
- Model initialization and configuration
- Prompt generation pipeline
- CSV output handling

### BaseMetricsEvaluator
The base class for evaluating answers. It provides:
- Metric calculation (relevance, readability, bias, etc.)
- Deep evaluation metrics (hallucination, toxicity, etc.)
- CSV output handling

## PubMedQA Evaluation

The framework includes a script for evaluating models on the PubMedQA dataset:

### Downloading the Dataset
```bash
python download_pubmedqa.py --output-dir datasets --split train
```

This will:
- Download the specified split of the PubMedQA dataset
- Save it as a CSV file
- Print dataset statistics and sample questions

### Running the Evaluation
```bash
python pubmedqa_evaluation.py --use-case "IRB Documentation" --output-dir "pubmedqa_results" --num-samples 100
```

This will:
- Load the specified use case configuration
- Evaluate each configured model on the dataset
- Generate answers using the system prompts
- Compare answers with ground truth
- Save detailed results and comparison metrics

## Usage

1. Create model configurations in the `model_configs` directory:
```json
{
  "name": "Your Model",
  "description": "Description of your model",
  "model_id": "your/model/id",
  "model_type": "huggingface",
  # ... other parameters
}
```

2. Create use case configurations in the `use_case_configs` directory:
```yaml
name: Your Use Case
description: Description of your use case
domain: Your Domain
prompt_types:
  # Define your prompt types
system_prompt_template: |
  # Define your template
evaluation_criteria: "Your criteria"
evaluation_steps:
  - "Your steps"
models:
  - name: Your Model
    # ... model parameters
```

3. Use the framework with your configurations:
```python
from eval_method.config import ConfigManager
from eval_method.prompt_generator import BasePromptGenerator
from eval_method.metrics_evaluator import BaseMetricsEvaluator

# Initialize the configuration manager
config_manager = ConfigManager()

# Get your use case configuration
config = config_manager.get_use_case_config("Your Use Case")

# Initialize the generators
prompt_generator = BasePromptGenerator(config)
metrics_evaluator = BaseMetricsEvaluator(config)

# Generate prompts
prompts_df = prompt_generator.generate_all_prompts(output_csv="your_prompts.csv")

# Evaluate answers
results_df = metrics_evaluator.evaluate_answers(
    questions=your_questions,
    answers=your_answers,
    output_csv="your_results.csv"
)
```

## Output Files

### Prompt Generation Output
The prompt generator creates a CSV file with:
- Index: Prompt type
- Prompt 1-N: Generated prompts for each type

### Metrics Evaluation Output
The metrics evaluator creates a CSV file with:
- Question: Input question
- Answer: Generated response
- Various metrics (relevance, readability, bias, etc.)

### Model Comparison Output
The PubMedQA evaluation creates:
- Individual model results CSV files
- Model comparison CSV file
- Statistical test results
- Visualizations (box plots, heatmap)

## Customization

You can customize the evaluation by:
1. Creating new model configurations in the `model_configs` directory
2. Creating new use case configurations in the `use_case_configs` directory
3. Defining your prompt types and descriptions
4. Setting your system prompt template
5. Specifying your evaluation criteria and steps
6. Adding domain-specific considerations
7. Adjusting metric thresholds

## Creating a New Use Case

1. Create a new YAML file in the `use_case_configs` directory
2. Define all required fields (name, description, domain, etc.)
3. Customize prompt types and templates for your domain
4. Set appropriate evaluation criteria and steps
5. Add any domain-specific considerations
6. Adjust metric thresholds if needed
7. Specify which models to use for this use case

The framework will automatically use your configurations to generate appropriate prompts and evaluate answers for your specific use case. 