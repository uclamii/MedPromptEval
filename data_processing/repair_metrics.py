# import pandas as pd
# from metrics_evaluator import MetricsEvaluator  # Ensure this is your working version!
# from tqdm import tqdm

# # Path to your original CSV and the output CSV
# INPUT_CSV = "DrHCM_all_prompt_types_llama_3.2_1.csv"
# OUTPUT_CSV = "DrHCM_all_prompt_types_llama_3.2_fixed.csv"

# # Load the CSV
# df = pd.read_csv(INPUT_CSV)

# # Initialize the metrics evaluator (use the same config as your pipeline)
# evaluator = MetricsEvaluator(verbose=False)

# # List of columns to recalculate (add/remove as needed)
# metric_columns = [
#     'semantic_similarity', 'answer_similarity', 'answer_length',
#     'flesch_reading_ease', 'flesch_kincaid_grade', 
#     'sentiment_polarity', 'sentiment_subjectivity',
#     'rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu_score',
#     'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
#     'entailment_score', 'entailment_label'
# ]

# # Only recalculate metrics for rows 912 and onward
# start_row = 909

# for idx in tqdm(range(start_row, len(df))):
#     row = df.iloc[idx]
#     try:
#         metrics = evaluator.evaluate_answer(
#             question=row['question'],
#             model_answer=row['model_answer'],
#             correct_answer=row['correct_answer'],
#             system_prompt=row.get('system_prompt', None),
#             question_id=row.get('question_id', None)
#         )
#         # Update the DataFrame with new metrics
#         for col in metric_columns:
#             if col in metrics:
#                 df.at[idx, col] = metrics[col]
#     except Exception as e:
#         print(f"Error recalculating metrics for row {idx}: {e}")

# # Save the fixed CSV
# df.to_csv(OUTPUT_CSV, index=False)
# print(f"Fixed CSV written to {OUTPUT_CSV}")


import pandas as pd
from metrics_evaluator import MetricsEvaluator  # Ensure this is your working version!
from tqdm import tqdm

# Path to your original CSV and the output CSV
INPUT_CSV = "medquad_self_consistency_5_models.csv"
OUTPUT_CSV = "medquad_self_consistency_5_models_2.csv"

# Load the CSV
df = pd.read_csv(INPUT_CSV)

# Ensure string columns have the correct dtype
df['entailment_label'] = df['entailment_label'].astype('object')

# Helper to safely convert to string
def safe_str(x):
    return "" if pd.isna(x) else str(x)

# Initialize the metrics evaluator (use the same config as your pipeline)
evaluator = MetricsEvaluator(verbose=False)

# List of columns to recalculate (add/remove as needed)
metric_columns = [
    'semantic_similarity', 'answer_similarity', 'answer_length',
    'flesch_reading_ease', 'flesch_kincaid_grade', 
    'sentiment_polarity', 'sentiment_subjectivity',
    'rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu_score',
    'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
    'entailment_score', 'entailment_label'
]

# Recalculate metrics for all rows
for idx in tqdm(range(len(df))):
    row = df.iloc[idx]
    try:
        metrics = evaluator.evaluate_answer(
            question=safe_str(row['question']),
            model_answer=safe_str(row['model_answer']),
            correct_answer=safe_str(row['correct_answer']),
            system_prompt=safe_str(row.get('system_prompt', None)),
            question_id=safe_str(row.get('question_id', None))
        )
        # Update the DataFrame with new metrics
        for col in metric_columns:
            if col in metrics:
                df.at[idx, col] = metrics[col]
    except Exception as e:
        print(f"Error recalculating metrics for row {idx}: {e}")

# Save the fixed CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"Fixed CSV written to {OUTPUT_CSV}")