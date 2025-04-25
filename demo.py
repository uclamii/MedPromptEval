import pandas as pd
from config import UseCaseConfigManager
from prompt_generator import BasePromptGenerator
from metrics_evaluator import BaseMetricsEvaluator
import os
import argparse

def load_dataset(use_case: str) -> tuple:
    """
    Load question-answer pairs from the appropriate dataset file.
    
    Args:
        use_case (str): Name of the use case
    
    Returns:
        tuple: (questions, answers) lists
    """
    # Map use case names to dataset files
    dataset_map = {
        "IRB Documentation": "irb_qa.csv",
        "Cancer Screening": "cancer_screening_qa.csv"
    }
    
    if use_case not in dataset_map:
        raise ValueError(f"No dataset available for use case: {use_case}")
    
    # Load the dataset
    dataset_path = os.path.join("datasets", dataset_map[use_case])
    df = pd.read_csv(dataset_path)
    
    return df["Question"].tolist(), df["Answer"].tolist()

def run_demo(use_case: str, model_name: str = None):
    """Run a demonstration of the framework for a specific use case."""
    print(f"\n{'='*80}")
    print(f"Running demo for: {use_case}")
    if model_name:
        print(f"Using model: {model_name}")
    print(f"{'='*80}\n")

    # Initialize configuration manager
    config_manager = UseCaseConfigManager()
    
    try:
        # Get configuration
        config = config_manager.get_config(use_case)
        print(f"Loaded configuration for: {config.name}")
        print(f"Domain: {config.domain}")
        print(f"Description: {config.description}\n")

        # List available models
        if config.models:
            print("Available models:")
            for model in config.models:
                print(f"- {model.name}: {model.description}")
            print()

        # Initialize generators
        prompt_generator = BasePromptGenerator(config, model_name=model_name)
        metrics_evaluator = BaseMetricsEvaluator(config)

        # Generate prompts
        print("Generating prompts...")
        prompts_df = prompt_generator.generate_all_prompts(
            output_csv=f"demo_{use_case.lower().replace(' ', '_')}_prompts.csv"
        )
        print(f"Generated {len(prompts_df)} prompt types\n")

        # Load questions and answers from dataset
        print("Loading dataset...")
        questions, answers = load_dataset(use_case)
        print(f"Loaded {len(questions)} question-answer pairs\n")

        # Evaluate answers
        print("Evaluating answers...")
        results_df = metrics_evaluator.evaluate_answers(
            questions=questions,
            answers=answers,
            output_csv=f"demo_{use_case.lower().replace(' ', '_')}_results.csv"
        )
        print(f"Evaluated {len(results_df)} question-answer pairs\n")

        # Display results summary
        print("Results Summary:")
        print("-" * 40)
        for metric in ['Relevance', 'Readability', 'Bias_Score', 'Hallucination_Score']:
            if metric in results_df.columns:
                mean_value = results_df[metric].mean()
                print(f"{metric}: {mean_value:.3f}")
        print("-" * 40)

        # Display example results
        print("\nExample Results:")
        print("-" * 40)
        for i, (question, answer) in enumerate(zip(questions[:3], answers[:3])):
            print(f"\nQuestion {i+1}:")
            print(f"Q: {question}")
            print(f"A: {answer}")
            print(f"Relevance: {results_df.iloc[i]['Relevance']:.3f}")
            print(f"Readability: {results_df.iloc[i]['Readability']:.3f}")
        print("-" * 40)

    except Exception as e:
        print(f"Error running demo: {str(e)}")

def main():
    """Run demos for all available use cases."""
    parser = argparse.ArgumentParser(description="Run the evaluation framework demo")
    parser.add_argument("--use-case", help="Specific use case to run (optional)")
    parser.add_argument("--model", help="Specific model to use (optional)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs("demo_output", exist_ok=True)
    os.chdir("demo_output")

    # Get available use cases
    config_manager = UseCaseConfigManager()
    use_cases = config_manager.list_use_cases()

    print(f"Available use cases: {', '.join(use_cases)}\n")

    # Run demo for specified use case or all use cases
    if args.use_case:
        if args.use_case not in use_cases:
            print(f"Error: Use case '{args.use_case}' not found")
            return
        run_demo(args.use_case, model_name=args.model)
    else:
        for use_case in use_cases:
            run_demo(use_case, model_name=args.model)

if __name__ == "__main__":
    main() 