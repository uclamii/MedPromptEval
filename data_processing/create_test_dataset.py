import pandas as pd
import os
from pathlib import Path
import random

def create_test_dataset(
    input_path: str,
    output_path: str,
    num_questions: int = 5,
    random_seed: int = 42
) -> None:
    """
    Create a small test dataset from the cleaned MedQuad dataset.
    
    Args:
        input_path (str): Path to the cleaned MedQuad CSV file
        output_path (str): Path to save the test dataset
        num_questions (int): Number of questions to include
        random_seed (int): Random seed for reproducibility
    """
    print(f"Creating test dataset from {input_path}...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Read the cleaned dataset
    df = pd.read_csv(input_path)
    
    # Randomly sample questions
    df_test = df.sample(n=num_questions, random_state=random_seed)
    
    # Save the test dataset
    df_test.to_csv(output_path, index=False)
    print(f"Test dataset saved to {output_path}")
    print(f"Number of questions: {len(df_test)}")
    
    # Print the questions for verification
    print("\nTest dataset questions:")
    for i, row in df_test.iterrows():
        print(f"\nQuestion {i+1}:")
        print(f"Q: {row['question']}")
        print(f"A: {row['answer']}")

def main():    
    # Create test dataset from cleaned MedQuad
    medquad_input = "datasets/cleaned/medquad_cleaned.csv"
    medquad_output = "datasets/test_data/medquad_test.csv" 

    if os.path.exists(medquad_input):
        create_test_dataset(
            input_path=medquad_input,
            output_path=medquad_output,
            num_questions=5
        )
    else:
        print(f"Warning: Cleaned MedQuad dataset not found at {medquad_input}")

if __name__ == "__main__":
    main() 