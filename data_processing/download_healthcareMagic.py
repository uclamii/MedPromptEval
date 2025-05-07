from datasets import load_dataset
import pandas as pd
import os

# Create the datasets directory if it doesn't exist
os.makedirs("datasets/raw", exist_ok=True)

print("Downloading HealthCareMagic-100k-en dataset...")
ds = load_dataset("wangrongsheng/HealthCareMagic-100k-en")

# Convert to DataFrame for easier manipulation
print("Converting dataset to DataFrame...")
train_df = pd.DataFrame(ds['train'])
test_df = pd.DataFrame(ds['test'])
valid_df = pd.DataFrame(ds['validation'])

# Display information about the dataset
print(f"Train set size: {len(train_df)} rows")
print(f"Test set size: {len(test_df)} rows")
print(f"Validation set size: {len(valid_df)} rows")
print(f"Columns in dataset: {train_df.columns.tolist()}")

# The dataset should have 'instruction', 'input', and 'output' columns
# For our pipeline, we'll keep 'input' and 'output' columns
# (later the clean_datasets.py script will rename these to 'question' and 'answer')

# Save to CSV
print("Saving datasets to CSV files...")
train_csv_path = "datasets/raw/healthcareMagic_train.csv"
test_csv_path = "datasets/raw/healthcareMagic_test.csv"
valid_csv_path = "datasets/raw/healthcareMagic_valid.csv"
combined_csv_path = "datasets/raw/healthcareMagic_combined.csv"

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)
valid_df.to_csv(valid_csv_path, index=False)

# Combine all splits into one file
combined_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)
combined_df.to_csv(combined_csv_path, index=False)

print(f"Saved CSV files:")
print(f"- Train: {train_csv_path} ({len(train_df)} rows)")
print(f"- Test: {test_csv_path} ({len(test_df)} rows)")
print(f"- Validation: {valid_csv_path} ({len(valid_df)} rows)")
print(f"- Combined: {combined_csv_path} ({len(combined_df)} rows)")
print("\nNext steps:")
print("1. Run `python data_processing/clean_datasets.py` to convert the dataset to the proper format")
print("2. The cleaned dataset will be available at 'datasets/cleaned/healthcareMagic_cleaned.csv'") 