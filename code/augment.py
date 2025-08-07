
import pandas as pd
import numpy as np

# File paths
input_csv_path = '/data/ephemeral/home/dev/upstageailab-nlp-summarization-nlp_6/data/train.csv'
output_csv_path = '/data/ephemeral/home/dev/upstageailab-nlp-summarization-nlp_6/data/train_augmented.csv'

# Load the dataset
df = pd.read_csv(input_csv_path)

# Create a copy for augmentation
augmented_df = df.copy()

# Define the augmentation function
def shuffle_dialogue_sentences(dialogue):
  """Shuffles the sentences in a dialogue string."""
  if not isinstance(dialogue, str):
      return ""
  sentences = dialogue.split('\n')
  np.random.shuffle(sentences)
  return '\n'.join(sentences)

# Apply the augmentation
augmented_df['dialogue'] = augmented_df['dialogue'].apply(shuffle_dialogue_sentences)

# Combine original and augmented data
combined_df = pd.concat([df, augmented_df], ignore_index=True)

# Save the result
combined_df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Data augmentation complete.")
print(f"Original number of rows: {len(df)}")
print(f"New number of rows: {len(combined_df)}")
print(f"Augmented file saved to: {output_csv_path}")
