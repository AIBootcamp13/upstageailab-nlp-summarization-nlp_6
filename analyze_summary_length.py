
import pandas as pd
import numpy as np

# Load the training data
try:
    df = pd.read_csv('/data/ephemeral/home/dev/upstageailab-nlp-summarization-nlp_6/data/train.csv')

    # Calculate the length of each summary
    summary_lengths = df['summary'].str.len()

    # Calculate and print statistics
    print("--- Summary Length Analysis (train.csv) ---")
    print(summary_lengths.describe(percentiles=[.25, .50, .75, .90, .95, .99]))

except FileNotFoundError:
    print("Error: train.csv not found.")
except Exception as e:
    print(f"An error occurred: {e}")
