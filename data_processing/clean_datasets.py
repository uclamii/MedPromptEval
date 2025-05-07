import pandas as pd
import os
from pathlib import Path

def clean_medquad(input_path: str, output_path: str) -> None:
    """
    Clean the MedQuad dataset to have only question and answer columns.
    
    Args:
        input_path (str): Path to the input MedQuad CSV file
        output_path (str): Path to save the cleaned CSV file
    """
    print(f"Cleaning MedQuad dataset from {input_path}...")
    
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Print column names for debugging
    print("Available columns:", df.columns.tolist())
    
    # Map possible column names to our target names
    column_mapping = {
        'question': ['Question', 'question', 'QUESTION'],
        'answer': ['Answer', 'answer', 'ANSWER']
    }
    
    # Find the actual column names
    question_col = next((col for col in df.columns if col in column_mapping['question']), None)
    answer_col = next((col for col in df.columns if col in column_mapping['answer']), None)
    
    if not question_col or not answer_col:
        raise ValueError(f"Could not find question/answer columns. Available columns: {df.columns.tolist()}")
    
    # Select only the required columns
    df_cleaned = df[[question_col, answer_col]]
    
    # Rename columns to lowercase
    df_cleaned.columns = ['question', 'answer']
    
    # Save the cleaned dataset
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned MedQuad dataset saved to {output_path}")
    print(f"Number of samples: {len(df_cleaned)}")

def clean_pubmedqa(input_path: str, output_path: str) -> None:
    """
    Clean the PubMedQA dataset to have only question and answer columns.
    For PubMedQA, we'll use the Long_answer as the answer.
    
    Args:
        input_path (str): Path to the input PubMedQA CSV file
        output_path (str): Path to save the cleaned CSV file
    """
    print(f"Cleaning PubMedQA dataset from {input_path}...")
    
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Print column names for debugging
    print("Available columns:", df.columns.tolist())
    
    # Map possible column names to our target names
    column_mapping = {
        'question': ['Question', 'question', 'QUESTION'],
        'answer': ['Long_answer', 'long_answer', 'LONG_ANSWER']
    }
    
    # Find the actual column names
    question_col = next((col for col in df.columns if col in column_mapping['question']), None)
    answer_col = next((col for col in df.columns if col in column_mapping['answer']), None)
    
    if not question_col or not answer_col:
        raise ValueError(f"Could not find question/answer columns. Available columns: {df.columns.tolist()}")
    
    # Select only the required columns
    df_cleaned = df[[question_col, answer_col]]
    
    # Rename columns to lowercase
    df_cleaned.columns = ['question', 'answer']
    
    # Save the cleaned dataset
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned PubMedQA dataset saved to {output_path}")
    print(f"Number of samples: {len(df_cleaned)}")

def clean_healthcare_magic(input_path: str, output_path: str) -> None:
    """
    Clean the HealthCareMagic-100k dataset to have only question and answer columns.
    For this dataset:
    - 'input' will be mapped to 'question'
    - 'output' will be mapped to 'answer'
    - 'instruction' will be removed
    
    Args:
        input_path (str): Path to the input HealthCareMagic CSV file
        output_path (str): Path to save the cleaned CSV file
    """
    print(f"Cleaning HealthCareMagic dataset from {input_path}...")
    
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Print column names for debugging
    print("Available columns:", df.columns.tolist())
    
    # Check if required columns exist
    required_columns = ['input', 'output']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset. Available columns: {df.columns.tolist()}")
    
    # Select only the required columns
    df_cleaned = df[['input', 'output']]
    
    # Rename columns to match pipeline expectations
    df_cleaned.columns = ['question', 'answer']
    
    # Save the cleaned dataset
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned HealthCareMagic dataset saved to {output_path}")
    print(f"Number of samples: {len(df_cleaned)}")

def main():
    # Create cleaned datasets directory if it doesn't exist
    cleaned_dir = Path("datasets/cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # Create raw directory if it doesn't exist
    raw_dir = Path("datasets/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean MedQuad dataset
    medquad_input = "datasets/raw/medquad.csv"
    medquad_output = cleaned_dir / "medquad_cleaned.csv"
    if os.path.exists(medquad_input):
        clean_medquad(medquad_input, medquad_output)
    else:
        print(f"Warning: MedQuad dataset not found at {medquad_input}")
    
    # Clean PubMedQA dataset
    pubmedqa_input = "datasets/raw/pubmedqa_train.csv"
    pubmedqa_output = cleaned_dir / "pubmedqa_cleaned.csv"
    if os.path.exists(pubmedqa_input):
        clean_pubmedqa(pubmedqa_input, pubmedqa_output)
    else:
        print(f"Warning: PubMedQA dataset not found at {pubmedqa_input}")
    
    # Clean HealthCareMagic dataset
    healthcare_magic_input = "datasets/raw/Doctor-HealthCare-100k.csv"
    healthcare_magic_output = cleaned_dir / "DrHCM_cleaned.csv"
    if os.path.exists(healthcare_magic_input):
        clean_healthcare_magic(healthcare_magic_input, healthcare_magic_output)
    else:
        print(f"Warning: HealthCareMagic dataset not found at {healthcare_magic_input}")

if __name__ == "__main__":
    main() 



