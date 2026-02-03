import os
import glob
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# --- CONFIGURATION ---
# dynamically find the path relative to this script file
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(script_dir, "../data/corpus_txt/")

SPACY_MODEL = "en_core_web_lg"

# Define your thesis corpora structure
CORPORA = {
    "Tolkien (Root)": [
        "FellowshipofTheRing.txt",
        "TheTwoTowers.txt",
        "TheReturnofTheKing.txt"
    ],
    "Successors (80s/90s)": [
        "TheSwordofShannara.txt",
        "TheEyeofTheWorld.txt"
    ],
    "Modern (Deconstruction)": [
        "AGameofThrones.txt",
        "Assassin'sApprentice.txt",
        "TheWayofKings.txt"
    ]
}


def load_and_clean_corpus(nlp):
    """Reads files and returns a dictionary of clean text lists."""
    data = {"Tolkien (Root)": [], "Successors (80s/90s)": [], "Modern (Deconstruction)": []}

    # Invert dictionary for easy lookup
    file_map = {}
    for group, files in CORPORA.items():
        for f in files:
            file_map[f] = group

    print("📖 Loading and cleaning corpus (Advanced Filtering)...")
    for filepath in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
        filename = os.path.basename(filepath)
        group = file_map.get(filename)

        if not group:
            continue

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Increase limit for large books
        nlp.max_length = len(text) + 100000

        # SLICING: We still slice to 1MB for speed, but you can remove [:1000000] for final run
        doc = nlp(text[:1000000])

        # --- THE FIX IS HERE ---
        # 1. POS Tagging: Keep only NOUNS (sword), ADJECTIVES (dark), and PROPN (optional)
        # 2. Filtering: Remove specific names if you want to measure 'Genre' vs 'Character'
        #    (We keep PROPN here to catch 'Gondor', but you might want to exclude them later)
        tokens = []
        for token in doc:
            if token.is_stop or not token.is_alpha:
                continue

            # Keep only descriptive world-building words
            if token.pos_ in ['NOUN', 'ADJ']:
                tokens.append(token.lemma_.lower())

        clean_text = " ".join(tokens)
        data[group].append(clean_text)
        print(f"   ✅ Processed {filename} ({len(tokens)} nouns/adjectives)")

    return data

def calculate_lexical_diffusion(data):
    """
    1. Finds top Tolkien-specific words using TF-IDF on Corpus B.
    2. Measures usage of those specific words in A and C.
    """
    print("\n🔍 Calculating Lexical Diffusion (Vocabulary Inheritance)...")

    # 1. Train on Tolkien ONLY
    tfidf = TfidfVectorizer(max_features=100)
    tfidf.fit(data["Tolkien (Root)"])

    # Extract Tolkien's vocabulary "fingerprint"
    feature_names = tfidf.get_feature_names_out()
    print(f"   Top Tolkien Terms: {', '.join(feature_names[:15])}...")

    results = []

    # 2. Project other corpora onto this vocabulary
    # We use a CountVectorizer fixed to Tolkien's vocabulary to count raw frequencies
    counter = CountVectorizer(vocabulary=feature_names)

    for group, texts in data.items():
        if not texts: continue

        # Transform texts into the "Tolkien Vector Space"
        matrix = counter.transform(texts)
        # Sum counts of Tolkien words per book, normalize by book length
        total_tolkien_words = matrix.sum(axis=1)

        # We average the "Tolkien Density" for the group
        # (This is a simplified density score)
        avg_density = np.mean(total_tolkien_words)
        results.append({"Corpus": group, "Lexical Diffusion Score": avg_density})

    return pd.DataFrame(results)


def calculate_thematic_divergence(data):
    """
    1. Trains LDA Topic Model on Tolkien.
    2. Calculates 'Perplexity' (Confusion) of that model on other books.
    Higher Perplexity = The Topic Model doesn't understand the new book (Divergence).
    """
    print("\n🧠 Calculating Thematic Divergence (LDA Perplexity)...")

    # 1. Train LDA on Tolkien
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm_tolkien = vectorizer.fit_transform(data["Tolkien (Root)"])

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm_tolkien)

    results = []

    for group, texts in data.items():
        if not texts: continue

        dtm_group = vectorizer.transform(texts)
        # Perplexity: Lower is better (good fit). Higher is worse (divergence).
        # We normalize by log length to make it somewhat comparable
        perplexity = lda.perplexity(dtm_group)

        results.append({"Corpus": group, "Thematic Divergence (Perplexity)": perplexity})

    return pd.DataFrame(results)


if __name__ == "__main__":
    nlp = spacy.load(SPACY_MODEL)

    # 1. Load Data
    corpus_data = load_and_clean_corpus(nlp)

    # 2. Run Analyses
    lexical_df = calculate_lexical_diffusion(corpus_data)
    thematic_df = calculate_thematic_divergence(corpus_data)

    # 3. Merge and Display
    final_df = pd.merge(lexical_df, thematic_df, on="Corpus")

    print("\n" + "=" * 60)
    print("RESULTS: QUANTIFYING LITERARY INFLUENCE")
    print("=" * 60)
    print(final_df.to_string(index=False))

    # Save for thesis
    os.makedirs("../data/results", exist_ok=True)
    final_df.to_csv("../data/results/influence_metrics.csv", index=False)