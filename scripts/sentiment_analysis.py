import os
import glob
import pandas as pd
import nltk
import numpy as np
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(script_dir, "..", "data", "corpus_txt")
RESULTS_DIR = os.path.join(script_dir, "..", "data", "results")

# Define major characters to track (Top 3-5 per book/series)
# You can expand this list based on your Neo4j nodes.
CHARACTERS = {
    "Tolkien (Root)": ["aragorn", "frodo", "gandalf", "sam", "boromir", "gollum"],
    "Successors (80s/90s)": ["shea", "flick", "allanon", "rand", "moiraine", "perrin"],
    "Modern (Deconstruction)": ["eddard", "jon", "tyrion", "kaladin", "shallan", "dalinar", "jaime"]
}

# Mapping files to groups (Same as your other scripts)
CORPORA = {
    "Tolkien (Root)": ["FellowshipofTheRing.txt", "TheTwoTowers.txt", "TheReturnofTheKing.txt"],
    "Successors (80s/90s)": ["TheSwordofShannara.txt", "TheEyeofTheWorld.txt"],
    "Modern (Deconstruction)": ["AGameofThrones.txt", "Assassin'sApprentice.txt", "TheWayofKings.txt"]
}


def setup_nltk():
    """Ensures VADER lexicon is downloaded."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("⏳ Downloading VADER lexicon...")
        nltk.download('vader_lexicon')


def load_corpus_map():
    """Inverts the CORPORA dict for file lookup."""
    file_map = {}
    for group, files in CORPORA.items():
        for f in files:
            file_map[f] = group
    return file_map


def extract_character_contexts(text, character_name, window=40):
    """
    Finds all mentions of a character and extracts 'window' words around them.
    Returns a list of text snippets (contexts).
    """
    text_lower = text.lower()
    char_lower = character_name.lower()

    # Simple split (faster than spacy for raw finding)
    words = text.split()

    contexts = []
    # Find indices where character appears (naive partial match but effective for distinct names)
    indices = [i for i, x in enumerate(words) if char_lower in x.lower()]

    for idx in indices:
        # Grab window around the name (e.g., 40 words before, 40 words after)
        start = max(0, idx - window)
        end = min(len(words), idx + window)
        chunk = " ".join(words[start:end])
        contexts.append(chunk)

    return contexts


def analyze_character_morality():
    setup_nltk()
    sia = SentimentIntensityAnalyzer()
    file_map = load_corpus_map()

    # Dictionary to hold ALL scores for a character across ALL books
    # Format: {"Gandalf": {"Group": "Tolkien", "Scores": [...]}}
    aggregated_data = {}

    print("📖 Loading and scanning corpus for character contexts...")

    for filepath in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
        filename = os.path.basename(filepath)
        group = file_map.get(filename)
        if not group: continue

        print(f"   Processing {filename}...")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()

        relevant_chars = CHARACTERS.get(group, [])
        for char in relevant_chars:
            contexts = extract_character_contexts(full_text, char)
            if not contexts: continue

            # Initialize if new
            if char not in aggregated_data:
                aggregated_data[char] = {"Group": group, "Scores": []}

            # Score and extend the master list
            for ctx in contexts:
                res = sia.polarity_scores(ctx)
                aggregated_data[char]["Scores"].append(res['compound'])

    # 4. CALCULATE STATS ON AGGREGATED DATA
    print("\n🧮 Calculating Series-Level Stats...")
    results = []

    for char, data in aggregated_data.items():
        scores = np.array(data["Scores"])

        results.append({
            "Corpus": data["Group"],
            "Character": char.capitalize(),
            "Moral Alignment (Mean)": round(np.mean(scores), 3),
            "Moral Variance (σ)": round(np.std(scores), 3),
            "Mentions": len(scores)
        })

    df = pd.DataFrame(results)
    df = df[df["Mentions"] > 50].sort_values(by=["Corpus", "Moral Alignment (Mean)"], ascending=[True, False])

    return df


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    final_df = analyze_character_morality()

    print("\n" + "=" * 60)
    print("RESULTS: MORAL VARIANCE & CHARACTER COMPLEXITY")
    print("=" * 60)
    print(final_df.to_string(index=False))

    output_path = os.path.join(RESULTS_DIR, "sentiment_variance.csv")
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")