import os
import glob
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from collections import Counter

# --- CONFIGURATION ---
# Dynamically find the path relative to this script file
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(script_dir, "..", "data", "corpus_txt")
RESULTS_DIR = os.path.join(script_dir, "..", "data", "results")

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
    """
    Reads files, CLEANS them, and CHUNKS them into smaller segments
    so LDA has enough data to learn.
    """
    data = {"Tolkien (Root)": [], "Successors (80s/90s)": [], "Modern (Deconstruction)": []}
    file_map = {}
    for group, files in CORPORA.items():
        for f in files: file_map[f] = group

    print("📖 Loading, Cleaning, and Chunking corpus...")

    for filepath in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
        filename = os.path.basename(filepath)
        group = file_map.get(filename)
        if not group: continue

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # 1. Process the whole book (Limit to 2MB to keep it fast but substantial)
        nlp.max_length = len(text) + 100000
        doc = nlp(text[:2000000])

        # 2. Extract meaningful words
        tokens = []
        for token in doc:
            if token.is_stop or not token.is_alpha: continue
            # Keep NOUNS and ADJECTIVES for World Building context
            if token.pos_ in ['NOUN', 'ADJ']:
                tokens.append(token.lemma_.lower())

        # 3. THE FIX: Slice list of tokens into 2,000-word chunks
        CHUNK_SIZE = 2000
        for i in range(0, len(tokens), CHUNK_SIZE):
            chunk = tokens[i:i + CHUNK_SIZE]
            # Only keep chunks that are substantial (ignore tiny end bits)
            if len(chunk) > 500:
                data[group].append(" ".join(chunk))

        print(f"   ✅ Processed {filename} -> {len(tokens) // CHUNK_SIZE} chunks")

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
    counter = CountVectorizer(vocabulary=feature_names)

    for group, texts in data.items():
        if not texts: continue

        # Transform texts into the "Tolkien Vector Space"
        matrix = counter.transform(texts)
        total_tolkien_words = matrix.sum(axis=1)

        # Average the "Tolkien Density" for the group
        avg_density = np.mean(total_tolkien_words)
        results.append({"Corpus": group, "Lexical Diffusion Score": avg_density})

    return pd.DataFrame(results)


def calculate_thematic_divergence(lda_model, vectorizer, data):
    """
    Calculates 'Perplexity' (Confusion) of the Tolkien model on other books.
    Higher Perplexity = The Topic Model doesn't understand the new book (Divergence).
    """
    print("\n🧠 Calculating Thematic Divergence (LDA Perplexity)...")
    results = []

    for group, texts in data.items():
        if not texts: continue

        dtm_group = vectorizer.transform(texts)
        perplexity = lda_model.perplexity(dtm_group)

        results.append({"Corpus": group, "Thematic Divergence (Perplexity)": perplexity})

    return pd.DataFrame(results)


def analyze_topic_distribution(lda_model, vectorizer, data):
    """
    Shows WHICH Tolkien topics survived in the other books.
    """
    print("\n📊 Analyzing Topic Fingerprints...")

    # 1. Print what the Topics actually ARE
    feature_names = vectorizer.get_feature_names_out()
    print("   Tolkien's Thematic Pillars:")
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        print(f"   Topic {topic_idx + 1}: {', '.join(top_words)}")

    results = []

    # 2. Project each corpus onto these topics
    for group, texts in data.items():
        if not texts: continue

        dtm = vectorizer.transform(texts)
        topic_dist = lda_model.transform(dtm)
        avg_dist = np.mean(topic_dist, axis=0)

        row = {"Corpus": group}
        for i, val in enumerate(avg_dist):
            row[f"Topic {i + 1} Share"] = val
        results.append(row)

    return pd.DataFrame(results)


def analyze_archetype_context(nlp, data, target_word="sword"):
    """
    Checks what adjectives describe a specific archetype in each corpus.
    """
    print(f"\n👑 Analyzing Semantic Context for '{target_word}'...")
    results = []

    for group, texts in data.items():
        adjectives = []
        for text in texts:
            # We process a snippet to save time.
            # Ideally, use the original raw text, but here we scan the lemma list roughly.
            # For accurate context, we re-process a chunk of raw text:
            doc = nlp(text[:500000])

            for token in doc:
                if token.lemma_.lower() == target_word:
                    for child in token.children:
                        if child.pos_ == "ADJ":
                            adjectives.append(child.lemma_.lower())

        common = Counter(adjectives).most_common(3)
        desc_str = ", ".join([f"{w} ({c})" for w, c in common])
        results.append({"Corpus": group, f"'{target_word}' Descriptors": desc_str})

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Ensure results folder exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("⏳ Loading spaCy model...")
    nlp = spacy.load(SPACY_MODEL)

    # 1. Load Data
    corpus_data = load_and_clean_corpus(nlp)

    # 2. Prepare Models (Train on Tolkien)
    # min_df=0.2 means a word must appear in 20% of documents to be a "theme"
    # This filters out "Jester" or one-off scene words.
    vectorizer = CountVectorizer(max_df=0.95, min_df=0.2, stop_words='english')
    dtm_tolkien = vectorizer.fit_transform(corpus_data["Tolkien (Root)"])

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm_tolkien)

    # 3. Run All Analyses
    lexical_df = calculate_lexical_diffusion(corpus_data)
    thematic_df = calculate_thematic_divergence(lda, vectorizer, corpus_data)
    topic_dist_df = analyze_topic_distribution(lda, vectorizer, corpus_data)

    # 4. Run Semantic Shift on Multiple Archetypes
    target_words = ["sword", "king", "magic", "shadow", "stone"]

    all_shifts = []
    for word in target_words:
        print(f"   ...analyzing '{word}'")
        df = analyze_archetype_context(nlp, corpus_data, target_word=word)
        # Add a column to track which word this is
        df.insert(0, "Archetype", word)
        all_shifts.append(df)

    # Combine into one big table
    master_shift_df = pd.concat(all_shifts)

    # 5. Merge and Save
    final_df = pd.merge(lexical_df, thematic_df, on="Corpus")

    print("\n" + "=" * 60)
    print("RESULTS 1: QUANTIFYING INFLUENCE")
    print("=" * 60)
    print(final_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("RESULTS 2: TOPIC DISTRIBUTION (Thematic Architecture)")
    print("=" * 60)
    print(topic_dist_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("RESULTS 3: SEMANTIC SHIFT (The Deconstruction)")
    print("=" * 60)
    # Group by Archetype for cleaner reading
    print(master_shift_df.sort_values(by=["Archetype", "Corpus"]).to_string(index=False))

    # Save to CSV
    final_df.to_csv(os.path.join(RESULTS_DIR, "influence_metrics.csv"), index=False)
    topic_dist_df.to_csv(os.path.join(RESULTS_DIR, "topic_distribution.csv"), index=False)
    master_shift_df.to_csv(os.path.join(RESULTS_DIR, "semantic_shift_master.csv"), index=False)

    print(f"\n✅ All reports saved to: {RESULTS_DIR}")