
import pandas as pd
from rouge import Rouge
import numpy as np
import os

# --- Configuration ---
original_csv_path = '/data/ephemeral/home/dev/upstageailab-nlp-summarization-nlp_6/data/train.csv'
augmented_csv_path = '/data/ephemeral/home/dev/upstageailab-nlp-summarization-nlp_6/data/train_augmented_back_translation.csv'

# --- Main Analysis Script ---
def analyze_augmentation():
    print("Starting analysis of the augmented data...")

    # --- Load Data ---
    if not os.path.exists(augmented_csv_path):
        print(f"\nERROR: Augmented file not found at '{augmented_csv_path}'.")
        print("Please run the augmentation script first to generate the file.")
        return

    try:
        df_orig = pd.read_csv(original_csv_path)
        df_combined = pd.read_csv(augmented_csv_path)
        print(f"Successfully loaded original and augmented data.")
    except FileNotFoundError as e:
        print(f"\nERROR: Could not load a file: {e}")
        return

    # Separate the augmented part from the combined file
    # The first half is the original, the second half is the augmented data
    num_original_rows = len(df_orig)
    df_aug_only = df_combined.iloc[num_original_rows:].reset_index(drop=True)

    # Ensure the dataframes are aligned
    if len(df_orig) != len(df_aug_only):
        print("\nERROR: The number of rows in original and augmented data does not match.")
        print(f"Original: {len(df_orig)}, Augmented: {len(df_aug_only)}")
        return

    # --- Analysis ---
    print("\nCalculating differences and similarities...")
    rouge = Rouge()
    char_len_diffs = []
    word_count_diffs = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    # Use a sample for quick analysis or full dataset if preferred
    sample_size = min(len(df_orig), 1000) # Analyze a sample of 1000 to be fast
    df_orig_sample = df_orig.head(sample_size)
    df_aug_sample = df_aug_only.head(sample_size)

    for i in range(len(df_orig_sample)):
        original_text = str(df_orig_sample.loc[i, 'dialogue'])
        augmented_text = str(df_aug_sample.loc[i, 'dialogue'])

        if not original_text or not augmented_text:
            continue

        # Length analysis
        char_len_diffs.append(len(augmented_text) - len(original_text))
        word_count_diffs.append(len(augmented_text.split()) - len(original_text.split()))

        # ROUGE analysis
        try:
            scores = rouge.get_scores([augmented_text], [original_text])
            rouge_1_scores.append(scores[0]['rouge-1']['f'])
            rouge_2_scores.append(scores[0]['rouge-2']['f'])
            rouge_l_scores.append(scores[0]['rouge-l']['f'])
        except Exception:
            continue # Skip if ROUGE calculation fails for any reason

    # --- Print Results ---
    print("\n--- Analysis Results (based on a sample of up to 1000 rows) ---")

    # 1. Length Difference
    print("\n[1] Text Length Analysis:")
    print(f"  - Average Character Difference: {np.mean(char_len_diffs):.2f} characters")
    print(f"  - Average Word Count Difference: {np.mean(word_count_diffs):.2f} words")
    print("    (Positive values mean the augmented text is longer)")

    # 2. ROUGE Score
    print("\n[2] Similarity Analysis (ROUGE F1-Score):")
    print(f"  - Average ROUGE-1 Score: {np.mean(rouge_1_scores):.4f}")
    print(f"  - Average ROUGE-2 Score: {np.mean(rouge_2_scores):.4f}")
    print(f"  - Average ROUGE-L Score: {np.mean(rouge_l_scores):.4f}")
    print("    (Scores closer to 1.0 indicate higher similarity)")

    # 3. Concrete Examples
    print("\n--- Example Comparisons (first 3 rows) ---")
    for i in range(min(3, len(df_orig))):
        print(f"\n----- Example {i+1} -----")
        print(f"[ORIGINAL]:\n{df_orig.loc[i, 'dialogue']}")
        print(f"\n[AUGMENTED]:\n{df_aug_only.loc[i, 'dialogue']}")
        print("----------------------")

if __name__ == "__main__":
    analyze_augmentation()
