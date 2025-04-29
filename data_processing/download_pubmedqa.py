import pandas as pd
from datasets import load_dataset
import os
import argparse
from tqdm import tqdm

def download_and_save_pubmedqa(output_dir: str = "datasets", split: str = "train"):
    """
    Download the PubMedQA dataset and save it as a CSV file.
    
    Args:
        output_dir (str): Directory to save the dataset
        split (str): Dataset split to download ('train', 'validation', 'test')
    """
    print(f"Downloading PubMedQA {split} dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"pubmedqa_{split}.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Number of samples: {len(df)}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    # Print sample questions
    print("\nSample Questions:")
    for i, row in df.head(3).iterrows():
        print(f"\nQuestion {i+1}:")
        print(f"Question: {row['question']}")
        print(f"Answer: {row['long_answer']}")
        print(f"Final Decision: {row['final_decision']}")

def main():
    """Download PubMedQA dataset."""
    parser = argparse.ArgumentParser(description="Download PubMedQA dataset")
    parser.add_argument("--output-dir", default="datasets",
                      help="Directory to save the dataset")
    parser.add_argument("--split", default="train",
                      choices=["train", "validation", "test"],
                      help="Dataset split to download")
    args = parser.parse_args()
    
    download_and_save_pubmedqa(args.output_dir, args.split)

if __name__ == "__main__":
    main() 