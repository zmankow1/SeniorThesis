import pandas as pd
import spacy
import random
import csv
import sys
from pathlib import Path

# --- CONFIGURATION ---
INPUT_CSV = './processed_paragraphs.csv'
OUTPUT_SAMPLE = './annotation_sample_V3.csv'  # V2 to avoid overwriting your old sample
SAMPLE_SIZE = 500  # Number of sentences to annotate


def create_annotation_sample():
    """
    Loads all paragraphs, splits them into sentences using spaCy, and selects a
    random sample of 500 sentences to be used for the next round of labeling.
    This ensures cross-genre data is mixed in.
    """

    # Set up spaCy for sentence segmentation (fast but robust)
    print("Loading spaCy for sentence splitting...")
    try:
        # We use the small model for sentence splitting as it's faster
        # Ensure it is installed: python -m spacy download en_core_web_sm
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
    except OSError:
        print("en_core_web_sm not found. Please install it with 'python -m spacy download en_core_web_sm'")
        return

    # CRITICAL FIX: Increase the limit to handle the massive chunks you have
    nlp.max_length = 20000000

    print(f"Loading data from {INPUT_CSV}...")
    try:
        # Read the raw_chunk column
        df = pd.read_csv(INPUT_CSV, usecols=['raw_chunk', 'novel_title'])
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found. Ensure your cleaned corpus file exists.")
        return
    except ValueError:
        print("Error: 'raw_chunk' or 'novel_title' column not found. Check your CSV header names.")
        return

    all_sentences = []

    print("Splitting text into sentences (this might take a moment)...")

    # Filter out empty chunks
    chunks = df['raw_chunk'].dropna().tolist()

    # Process all chunks to build the full pool of potential sentences
    doc_stream = nlp.pipe(chunks, batch_size=50)  # Increased batch size for speed

    for i, doc in enumerate(doc_stream):
        # Provide feedback to the user every 50 chunks
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} chunks. Current sentence pool size: {len(all_sentences)}")

        for sent in doc.sents:
            text = sent.text.strip()
            # Filter: Keep sentences between 60 and 500 characters
            if 60 < len(text) < 500:
                all_sentences.append(text)

    print(f"\nTotal valid sentences pool generated: {len(all_sentences)}")

    # Pick a random sample
    if len(all_sentences) > SAMPLE_SIZE:
        selected_sentences = random.sample(all_sentences, SAMPLE_SIZE)
    else:
        selected_sentences = all_sentences
        print(f"Warning: Only {len(all_sentences)} sentences found. Using all available data.")

    # Save for Label Studio
    # Label Studio likes a simple column named 'text'
    sample_df = pd.DataFrame({'text': selected_sentences})
    sample_df.to_csv(OUTPUT_SAMPLE, index=False)

    print(f"\nSuccess! Saved {len(selected_sentences)} random, generalized sentences to {OUTPUT_SAMPLE}")
    print("\n--- NEXT STEPS ---")
    print(f"1. Import '{OUTPUT_SAMPLE}' into Label Studio.")
    print("2. Label these 500 sentences to generalize your model.")
    print("3. Export the FULL training set (old + new labels) for final retraining.")


if __name__ == '__main__':
    # Increase CSV field size limit just in case chunks are massive
    csv.field_size_limit(sys.maxsize)
    create_annotation_sample()