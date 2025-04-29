"""
Test pipeline for the prompt generation system.
This script runs the prompt generator on different models and configurations.
"""

from prompt_generation import PromptGenerator, ensure_hf_login
from config import PROMPT_MODEL_CONFIGS
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate prompts for medical QA')
    parser.add_argument('--model', type=str, default='mistral-7b',
                        choices=list(PROMPT_MODEL_CONFIGS.keys()),
                        help='Model to use for generation')
    parser.add_argument('--output_dir', type=str, default='prompts',
                        help='Directory to save generated prompts')
    parser.add_argument('--num_prompts', type=int, default=1,
                        help='Number of prompts to generate per type')
    parser.add_argument('--no_auth', action='store_true',
                        help='Do not use authentication with Hugging Face')
    
    args = parser.parse_args()
    
    # Ensure logged into Hugging Face if auth is enabled
    if not args.no_auth:
        ensure_hf_login()
    
    # Initialize the prompt generator
    generator = PromptGenerator(
        model_key=args.model,
        use_auth=not args.no_auth
    )
    
    # Generate all prompts and save to files
    generator.generate_all_prompts(
        output_dir=args.output_dir,
        num_prompts_per_type=args.num_prompts
    )
    
    print(f"\nGeneration complete! Files saved to {args.output_dir} directory.")

if __name__ == "__main__":
    main() 