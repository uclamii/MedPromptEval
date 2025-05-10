import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
import textwrap


# ---- CONFIG ----
EXPERIMENTS = [
    {
        "name": "HealthCareMagic",
        "file": "final_results/HCM_results.csv",
        "group": ("prompt_type", "prompt_model"),
        "fixed": "answer_model"
    },
    {
        "name": "MedQuAD",
        "file": "final_results/MedQuad_results.csv",
        "group": ("answer_model", "prompt_model"),
        "fixed": "prompt_type"
    },
    {
        "name": "PubMedQA",
        "file": "final_results/PubMed_results.csv",
        "group": ("prompt_type", "prompt_model", "answer_model"),
        "fixed": None
    }
]
METRICS = [
    "semantic_similarity", "answer_similarity", "answer_length", "flesch_reading_ease", "flesch_kincaid_grade",
    "sentiment_polarity", "sentiment_subjectivity", "rouge1_f", "rouge2_f", "rougeL_f", "bleu_score",
    "bertscore_precision", "bertscore_recall", "bertscore_f1", "entailment_score"
]
WEIGHTS = {
    "semantic_similarity": 0.15,
    "answer_similarity": 0.10,
    "answer_length": 0.03,
    "flesch_reading_ease": 0.03,
    "flesch_kincaid_grade": 0.02,
    "sentiment_polarity": 0.02,
    "sentiment_subjectivity": 0.02,
    "rouge1_f": 0.03,
    "rouge2_f": 0.02,
    "rougeL_f": 0.02,
    "bleu_score": 0.02,
    "bertscore_precision": 0.03,
    "bertscore_recall": 0.05,
    "bertscore_f1": 0.15,
    "entailment_score": 0.21
}

# Ensure output directory exists
NORMALIZED_DIR = 'normalized_results'
os.makedirs(NORMALIZED_DIR, exist_ok=True)

# ---- FUNCTIONS ----

def load_and_normalize(file, metrics):
    df = pd.read_csv(file)
    scaler = MinMaxScaler()
    df_metrics = df[metrics].fillna(0)
    df[metrics] = scaler.fit_transform(df_metrics)
    return df

def compute_weighted_score(df, metrics, weights):
    score = np.zeros(len(df))
    for m in metrics:
        score += df[m] * weights.get(m, 1.0/len(metrics))
    df['overall_score'] = score
    return df

def best_config(df, group_cols):
    grouped = df.groupby(list(group_cols))['overall_score'].mean().reset_index()
    best_row = grouped.loc[grouped['overall_score'].idxmax()]
    return best_row, grouped

custom_prompts = [
    'You are a medical chatbot that uses the "tree of thoughts" methodology. When answering a user\'s question, explore multiple analytical branches. Specifically: 1. Consider various clinical interpretations of the data or symptoms. 2. Evaluate different diagnostic possibilities and their implications. 3. Assess multiple treatment or management approaches within the context of the case. At each step, clearly articulate your reasoning paths and how they contribute to your final response.',
    'You are a medical chatbot that uses the "self consistency" methodology. For every question, generate multiple reasoning paths based on different plausible interpretations of the data. Critically compare and contrast these reasoning paths for consistency. Then, converge on the most reliable and well-supported conclusion, making your analytical process transparent to the user.',
    'You are a medical chatbot that uses the "chain of thought" methodology. For each question, break down your reasoning into clear, logical steps. Explain the medical concepts, decision criteria, and inferential steps you use to arrive at a conclusion. Ensure that your explanation improves user understanding and demonstrates the transparency of your medical reasoning.'
]

def plot_experiments(results, bests, out_file="summary_figure.png"):
    sns.set(style="whitegrid", font_scale=1.5)
    # 3 columns, 2 rows: top row = main plots, bottom row = annotation panels
    fig, axes = plt.subplots(
        2, 3, figsize=(28, 12),  # Smaller, more compact figure
        gridspec_kw={'height_ratios': [3, 1.5], 'width_ratios': [1, 1, 1.2]}
    )
    palette = sns.color_palette("viridis", as_cmap=True)

    # --- Main Panels ---
    for i, (df, best, name) in enumerate(zip([r['grouped'] for r in results], bests, [r['name'] for r in results])):
        ax = axes[0, i]
        if i == 0:
            df['label'] = df['prompt_type'].str.title() + '\n' + df['prompt_model'].apply(lambda x: str(x).split('/')[-1])
            sns.barplot(x='overall_score', y='label', data=df, ax=ax, palette="crest")
            ax.set_title(f"{name}: Prompt Type × Prompt Model", fontsize=14)
            ax.set_xlabel("Overall Score", fontsize=11)
            ax.set_ylabel("Prompt Type\nPrompt Model", fontsize=11)
            ax.tick_params(axis='both', labelsize=10)
        elif i == 1:
            df['label'] = df['answer_model'].apply(lambda x: str(x).split('/')[-1]) + '\n' + df['prompt_model'].apply(lambda x: str(x).split('/')[-1])
            sns.barplot(x='overall_score', y='label', data=df, ax=ax, palette="crest")
            ax.set_title(f"{name}: Answer Model × Prompt Model", fontsize=14)
            ax.set_xlabel("Overall Score", fontsize=11)
            ax.set_ylabel("Answer Model\nPrompt Model", fontsize=11)
            ax.tick_params(axis='both', labelsize=10)
        else:
            df['row'] = df['prompt_type'].str.title() + '\n' + df['prompt_model'].apply(lambda x: str(x).split('/')[-1])
            pivot = df.pivot(index='row', columns='answer_model', values='overall_score')
            sns.heatmap(
                pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax,
                cbar_kws={'label': 'Overall Score'}, annot_kws={'size': 10}
            )
            ax.set_title(f"{name}: (Prompt Type × Prompt Model) vs Answer Model", fontsize=14)
            ax.set_xlabel("Answer Model", fontsize=11)
            ax.set_ylabel("Prompt Type\nPrompt Model", fontsize=11)
            best_row_label = f"{best['prompt_type'].title()}\n{str(best['prompt_model']).split('/')[-1]}"
            y, x = list(pivot.index).index(best_row_label), list(pivot.columns).index(best['answer_model'])
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=2))
            # Update x-axis tick labels to only show model name, horizontal and wrapped
            new_xticklabels = [textwrap.fill(str(col).split('/')[-1], width=12) for col in pivot.columns]
            ax.set_xticklabels(new_xticklabels, rotation=0, fontsize=10, ha='center')
            ax.tick_params(axis='both', labelsize=10)

    # --- Annotation Panels ---
    for i, (exp, best) in enumerate(zip(EXPERIMENTS, bests)):
        ax = axes[1, i]
        ax.axis('off')
        df_full = pd.read_csv(exp["file"])
        query = np.ones(len(df_full), dtype=bool)
        for col in exp["group"] if isinstance(exp["group"], tuple) else [exp["group"]]:
            query &= (df_full[col] == best[col])
        best_row = df_full[query].iloc[0]
        config_text = "Best Configuration:\n"
        shown = set()
        for col in exp["group"] if isinstance(exp["group"], tuple) else [exp["group"]]:
            if col not in shown:
                val = best[col]
                if col in ["answer_model", "prompt_model"]:
                    val = str(val).split("/")[-1]
                config_text += f"{col.replace('_', ' ').title()}: {val}\n"
                shown.add(col)
        for col in ["answer_model", "prompt_model", "prompt_type"]:
            if col in best and col not in shown:
                val = best[col]
                if col in ["answer_model", "prompt_model"]:
                    val = str(val).split("/")[-1]
                if col == "prompt_type":
                    val = str(val).title()
                config_text += f"{col.replace('_', ' ').title()}: {val}\n"
                shown.add(col)
        sys_prompt = custom_prompts[i]
        sys_prompt_wrapped = textwrap.fill(sys_prompt, width=80)
        config_text += "\nSystem Prompt:\n" + sys_prompt_wrapped
        ax.text(
            0, 0.95, config_text, fontsize=9, va='top', ha='left', wrap=True, family='monospace', bbox=dict(facecolor='white', alpha=0.95, boxstyle='round,pad=0.5')
        )

    plt.tight_layout(h_pad=3.5)  # Restore some vertical padding between rows
    plt.subplots_adjust(wspace=0.6, hspace=0.15)  # Restore some vertical space between figure and annotation panels
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {out_file}")

# ---- MAIN WORKFLOW ----

results = []
bests = []

for exp in EXPERIMENTS:
    df = load_and_normalize(exp["file"], METRICS)
    df = compute_weighted_score(df, METRICS, WEIGHTS)
    # Save normalized CSV
    base_name = os.path.basename(exp["file"])
    out_path = os.path.join(NORMALIZED_DIR, base_name)
    df.to_csv(out_path, index=False)
    if isinstance(exp["group"], tuple):
        best, grouped = best_config(df, exp["group"])
    else:
        best, grouped = best_config(df, [exp["group"]])
    results.append({"grouped": grouped, "name": exp["name"]})
    bests.append(best)

plot_experiments(results, bests)

# --- Create and save best configuration summary table ---
best_configs_summary = []
experiment_names = {
    'MedQuAD': 'Model Comparison with Standardized Prompts',
    'HealthCareMagic': 'Prompt Type Optimization',
    'PubMedQA': 'Comprehensive Evaluation'
}
for idx, (exp, best) in enumerate(zip(EXPERIMENTS, bests)):
    dataset_name = exp['name']
    row = {
        'Experiment': experiment_names.get(dataset_name, dataset_name),
        'Dataset': dataset_name,
        'Prompt Type': str(best.get('prompt_type', '')).title(),
        'Prompt Model': str(best.get('prompt_model', '')).split('/')[-1],
        'Answer Model': str(best.get('answer_model', '')).split('/')[-1] if 'answer_model' in best else '',
        'Best Overall Score': best['overall_score'],
        'System Prompt': custom_prompts[idx]
    }
    # Use raw (unnormalized) metrics for the best config
    raw_path = exp['file']
    raw_df = pd.read_csv(raw_path)
    query = np.ones(len(raw_df), dtype=bool)
    for col in exp['group'] if isinstance(exp['group'], tuple) else [exp['group']]:
        query &= (raw_df[col] == best[col])
    best_raw_row = raw_df[query].iloc[0]
    for metric in METRICS:
        row[metric] = best_raw_row[metric]
    best_configs_summary.append(row)
# Specify column order
metric_cols = METRICS
col_order = ['Experiment', 'Dataset', 'Prompt Type', 'Prompt Model', 'Answer Model', 'System Prompt'] + metric_cols + ['Best Overall Score']
summary_df = pd.DataFrame(best_configs_summary)[col_order]
summary_df.to_csv('best_configurations_summary.csv', index=False)
