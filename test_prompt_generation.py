import pandas as pd
import os
from pathlib import Path
from prompt_generation import PromptGenerator
import json
from tqdm import tqdm
import yaml

def test_prompt_generation(
    test_data_path: str,
    config_path: str,
    model_name: str,
    output_dir: str,
    num_prompts_per_type: int = 2
) -> None:
    """
    Test prompt generation with the test dataset.
    
    Args:
        test_data_path (str): Path to the test dataset CSV
        config_path (str): Path to the YAML configuration file
        model_name (str): Name of the model to use for prompt generation
        output_dir (str): Directory to save the generated prompts
        num_prompts_per_type (int): Number of prompts to generate per type
    """
    print(f"Testing prompt generation with {test_data_path}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    df = pd.read_csv(test_data_path)
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize prompt generator
    generator = PromptGenerator(
        model_name=model_name,
        config_path=config_path
    )
    
    # Generate prompts for each type
    all_prompts = generator.generate_all_prompts(
        output_dir=output_dir,
        num_prompts_per_type=num_prompts_per_type
    )
    
    # Generate a test summary
    test_summary = {
        "test_data": {
            "num_questions": len(questions),
            "questions": questions,
            "answers": answers
        },
        "config": {
            "name": config["name"],
            "description": config["description"],
            "domain": config["domain"],
            "prompt_types": list(config["prompt_types"].keys()),
            "evaluation_criteria": config["evaluation_criteria"],
            "evaluation_steps": config["evaluation_steps"]
        },
        "prompt_types": list(all_prompts.keys()),
        "num_prompts_per_type": num_prompts_per_type,
        "prompt_examples": {
            prompt_type: {
                "definition": config["prompt_types"][prompt_type],
                "prompts": all_prompts[prompt_type]
            }
            for prompt_type in all_prompts
        }
    }
    
    # Save test summary
    with open(os.path.join(output_dir, "test_summary.json"), 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    # Print test summary
    print("\nTest Summary:")
    print(f"Number of test questions: {len(questions)}")
    print(f"Number of prompt types: {len(all_prompts)}")
    print(f"Prompts per type: {num_prompts_per_type}")
    print("\nConfiguration:")
    print(f"Name: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Domain: {config['domain']}")
    print("\nExample prompts for each type:")
    for prompt_type, prompts in all_prompts.items():
        print(f"\n{prompt_type}:")
        print(f"Definition: {config['prompt_types'][prompt_type]}")
        print("\nGenerated prompts:")
        for i, prompt in enumerate(prompts, 1):
            print(f"\nPrompt {i}:")
            print(prompt)
    
    # Save individual prompt files
    for prompt_type, prompts in all_prompts.items():
        prompt_file = os.path.join(output_dir, f"{prompt_type}_prompts.txt")
        with open(prompt_file, 'w') as f:
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"Definition: {config['prompt_types'][prompt_type]}\n\n")
            for i, prompt in enumerate(prompts, 1):
                f.write(f"Prompt {i}:\n{prompt}\n\n")

def main():
    # Test configuration
    test_data_path = "datasets/test_data/medquad_test.csv"
    config_path = "configs/medquad_test.yml"
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    output_dir = "datasets/test_data/prompts"
    
    if os.path.exists(test_data_path):
        test_prompt_generation(
            test_data_path=test_data_path,
            config_path=config_path,
            model_name=model_name,
            output_dir=output_dir,
            num_prompts_per_type=2
        )
    else:
        print(f"Warning: Test dataset not found at {test_data_path}")

if __name__ == "__main__":
    main() 