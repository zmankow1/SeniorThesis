import pandas as pd
from collections import Counter
import os
import re

def analyze_world_building():
    """Extracts Locations and Factions using the renamed hybrid files."""
    # RENAMED PATHS
    input_file = "../data/processed_data/manual_labels.csv"
    spacy_file = "../data/processed_data/automated_labels_cleaned.csv"
    output_summary = "processed_data/world_building_summary.txt"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Check your file names!")
        return

    print("Loading data for world-building analysis...")
    df = pd.read_csv(input_file)

    if 'key_entities' not in df.columns and os.path.exists(spacy_file):
        spacy_df = pd.read_csv(spacy_file)
        df = pd.merge(df, spacy_df[['book_id', 'chunk_id', 'key_entities']],
                      on=['book_id', 'chunk_id'], how='left')

    # ... (Rest of hybrid dictionary logic remains same) ...
    print("Report saved.")

if __name__ == "__main__":
    analyze_world_building()