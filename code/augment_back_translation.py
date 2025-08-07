import pandas as pd
from google.cloud import translate_v2 as translate
import os
from tqdm import tqdm

# --- Configuration ---
# IMPORTANT: Set your Google Cloud Project ID here
# You can find your project ID in the Google Cloud Console.
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

# File paths
input_csv_path = '/data/ephemeral/home/dev/upstageailab-nlp-summarization-nlp_6/data/train.csv'
output_csv_path = '/data/ephemeral/home/dev/upstageailab-nlp-summarization-nlp_6/data/train_augmented_back_translation.csv'

# --- Helper Functions ---
def back_translate(text, client):
    """Translates text from Korean -> English -> Japanese -> Korean."""
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        # 1. Korean to English
        en_translation = client.translate(text, source_language='ko', target_language='en')
        en_text = en_translation['translatedText']

        # 2. English to Japanese
        ja_translation = client.translate(en_text, source_language='en', target_language='ja')
        ja_text = ja_translation['translatedText']

        # 3. Japanese to Korean
        ko_translation = client.translate(ja_text, source_language='ja', target_language='ko')
        return ko_translation['translatedText']

    except Exception as e:
        # tqdm prints to stderr, so we should too to avoid breaking the progress bar
        import sys
        print(f"An error occurred during translation: {e}", file=sys.stderr)
        return "" # Return empty string on error

# --- Main Script ---
def main():
    # --- Initialization ---
    print("Starting back-translation data augmentation...")

    # Check for Project ID
    if not project_id:
        print("ERROR: GOOGLE_CLOUD_PROJECT environment variable is not set.")
        print("Please set it to your Google Cloud Project ID.")
        return

    # Initialize Translation Client
    try:
        translate_client = translate.Client()
        print("Google Translate client initialized successfully.")
    except Exception as e:
        print(f"ERROR: Could not initialize Google Translate client: {e}")
        print("Please ensure you have authenticated correctly (e.g., `gcloud auth application-default login`).")
        return

    # Load the dataset
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded {input_csv_path} with {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_csv_path}")
        return

    # --- Augmentation ---
    print("Applying back-translation to the 'dialogue' column...")
    # Create a copy for augmentation
    augmented_df = df.copy()

    # Initialize tqdm for pandas
    tqdm.pandas(desc="Translating Dialogues")

    # Apply the back-translation function with a progress bar
    augmented_df['dialogue'] = augmented_df['dialogue'].progress_apply(
        lambda x: back_translate(x, translate_client)
    )

    # Filter out any rows where translation failed
    original_len = len(augmented_df)
    augmented_df = augmented_df[augmented_df['dialogue'] != ""]
    if len(augmented_df) < original_len:
        print(f"Warning: {original_len - len(augmented_df)} rows were removed due to translation errors.")

    print("Back-translation complete.")

    # --- Finalization ---
    # Combine original and augmented data
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    print(f"Original number of rows: {len(df)}")
    print(f"New number of rows after augmentation: {len(combined_df)}")

    # Save the result
    try:
        combined_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"Augmented file successfully saved to: {output_csv_path}")
    except Exception as e:
        print(f"ERROR: Failed to save the output file: {e}")

if __name__ == "__main__":
    main()