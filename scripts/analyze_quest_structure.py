import os
import glob
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from hmmlearn import hmm

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(script_dir, "..", "data", "corpus_txt")
RESULTS_DIR = os.path.join(script_dir, "..", "data", "results")

# We need consistent chunks to model the "flow" of the story
CHUNK_SIZE = 2000  # Words per chunk

CORPORA = {
    "Tolkien (Root)": ["FellowshipofTheRing.txt", "TheTwoTowers.txt", "TheReturnofTheKing.txt"],
    "Successors (80s/90s)": ["TheSwordofShannara.txt", "TheEyeofTheWorld.txt"],
    "Modern (Deconstruction)": ["AGameofThrones.txt", "Assassin'sApprentice.txt", "TheWayofKings.txt"]
}


def load_corpus_map():
    file_map = {}
    for group, files in CORPORA.items():
        for f in files: file_map[f] = group
    return file_map


def load_and_chunk_sequences(nlp):
    """
    Reads files and breaks them into ordered sequences of chunks.
    Returns: {filename: [chunk1, chunk2, ...]}
    """
    print("📖 Loading and Chunking Sequences...")
    file_map = load_corpus_map()
    sequences = {}

    # We load ALL files to ensure we can project them later
    for filepath in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
        filename = os.path.basename(filepath)
        if filename not in file_map: continue

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Process in spaCy (limit length for speed)
        nlp.max_length = len(text) + 100000
        doc = nlp(text[:2000000])  # Limit to 2MB per book for memory safety

        tokens = [t.lemma_.lower() for t in doc if t.pos_ in ['NOUN', 'ADJ'] and t.is_alpha]

        # Create Chunks
        book_chunks = []
        for i in range(0, len(tokens), CHUNK_SIZE):
            chunk = tokens[i:i + CHUNK_SIZE]
            if len(chunk) > 500:  # Ignore tiny end-of-book fragments
                book_chunks.append(" ".join(chunk))

        sequences[filename] = book_chunks
        print(f"   ✅ {filename}: {len(book_chunks)} chunks")

    return sequences


def train_lda_model(sequences, corpus_map):
    """
    Trains LDA on TOLKIEN ONLY to define the 'Standard Fantasy Topics'.
    """
    print("\n🧠 Training Topic Model (LDA) on Tolkien...")

    # 1. Gather all Tolkien chunks
    tolkien_text = []
    for filename, chunks in sequences.items():
        if corpus_map.get(filename) == "Tolkien (Root)":
            tolkien_text.extend(chunks)

    # 2. Vectorize
    # Aggressive filtering to find clear themes
    vectorizer = CountVectorizer(max_df=0.7, min_df=5, stop_words='english')
    dtm = vectorizer.fit_transform(tolkien_text)

    # 3. LDA
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)

    return lda, vectorizer


def get_topic_sequences(lda, vectorizer, sequences):
    """
    Converts text chunks into sequences of Topic Probabilities.
    Returns: {filename: array of shape (n_chunks, n_topics)}
    """
    topic_seqs = {}
    for filename, chunks in sequences.items():
        if not chunks: continue
        dtm = vectorizer.transform(chunks)
        topic_dist = lda.transform(dtm)
        topic_seqs[filename] = topic_dist
    return topic_seqs


def train_hmm_and_score(topic_sequences, corpus_map):
    """
    1. Trains HMM on Tolkien's topic sequences.
    2. Scores every other book against that model.
    """
    print("\n🔮 Training Hidden Markov Model (Narrative Structure)...")

    # 1. Prepare Training Data (Tolkien)
    tolkien_seqs = []
    tolkien_lengths = []

    for filename, seq in topic_sequences.items():
        if corpus_map.get(filename) == "Tolkien (Root)":
            tolkien_seqs.append(seq)
            tolkien_lengths.append(len(seq))

    if not tolkien_seqs:
        print("❌ No Tolkien data found! Check paths.")
        return pd.DataFrame()

    X_train = np.concatenate(tolkien_seqs)

    # 2. Train Gaussian HMM
    # We assume 3 Hidden Narrative States (e.g., Setup, Journey/Conflict, Resolution)
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(X_train, tolkien_lengths)

    print(f"   HMM Converged: {model.monitor_.converged}")

    # 3. Score All Books
    results = []
    print("\n📏 Calculating Quest Adherence Scores...")

    for filename, seq in topic_sequences.items():
        group = corpus_map.get(filename)

        # Log Likelihood of the sequence given the model
        score = model.score(seq)

        # Normalize by length (longer books have lower likelihoods naturally)
        normalized_score = score / len(seq)

        results.append({
            "Corpus": group,
            "Book": filename,
            "Raw Log-Likelihood": round(score, 2),
            "Quest Adherence Score": round(normalized_score, 2)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    nlp = spacy.load("en_core_web_lg")
    file_map = load_corpus_map()

    # 1. Load Data
    raw_sequences = load_and_chunk_sequences(nlp)

    # 2. Train LDA (The "Words" of the story)
    lda_model, vectorizer = train_lda_model(raw_sequences, file_map)

    # 3. Convert Text -> Topic Flow
    topic_sequences = get_topic_sequences(lda_model, vectorizer, raw_sequences)

    # 4. Train HMM (The "Grammar" of the story) & Score
    df = train_hmm_and_score(topic_sequences, file_map)

    # 5. Aggregate by Group
    print("\n" + "=" * 60)
    print("RESULTS: QUEST ADHERENCE (HMM SCORES)")
    print("=" * 60)

    # Higher Score (closer to 0) = Better Fit to Tolkien's Structure
    # Lower Score (more negative) = Divergent Structure
    summary = df.groupby("Corpus")["Quest Adherence Score"].mean().sort_values(ascending=False)
    print(summary)

    print("\nDetailed Breakdown:")
    print(df.sort_values(by="Quest Adherence Score", ascending=False).to_string(index=False))

    df.to_csv(os.path.join(RESULTS_DIR, "quest_structure_scores.csv"), index=False)
    print(f"\n✅ Saved to {RESULTS_DIR}")