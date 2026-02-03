import spacy
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re

# --- CONFIGURATION ---
MODEL_PATH = "../custom_ner_model/fantasy_ner"
INPUT_CSV = "processed_paragraphs.csv"
OUTPUT_NODES = "neo4j_nodes.csv"
OUTPUT_RELS = "neo4j_relationships.csv"

# Window settings: roughly a page of text
CHUNK_SIZE = 2000  # Characters per chunk
OVERLAP = 200  # Overlap to ensure we don't cut an entity in half


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Splits a massive string into smaller overlapping windows.
    Returns a list of text chunks.
    """
    if not isinstance(text, str):
        return []

    # If text is small enough, just return it
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        # Try to find a period or space near the end to break cleanly
        if end < len(text):
            # Look for the last period in the slice to avoid cutting sentences
            last_period = text.rfind('.', start, end)
            if last_period != -1 and last_period > start + (size // 2):
                end = last_period + 1

        chunk = text[start:end]
        if len(chunk.strip()) > 10:  # Skip empty/tiny chunks
            chunks.append(chunk)

        # Move forward, keeping the overlap
        start = end - overlap

        # Safety break for infinite loops if overlap is messed up
        if start >= end:
            start = end

    return chunks


def extract_entities_and_relationships():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        nlp = spacy.load(MODEL_PATH)
        # CRITICAL FIX: Increase limit slightly just in case, though chunking solves the main issue
        nlp.max_length = 3000000
    except OSError:
        print("Error: Could not load model. Did you run ner_trainer.py?")
        return

    print(f"Loading corpus from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("Error: Input CSV not found.")
        return

    # Prepare data: Explode massive rows into manageable windows
    print("Chunking massive texts into analyzing windows...")

    processing_queue = []

    # Iterate over the original rows
    for index, row in df.iterrows():
        raw_text = row['raw_chunk']
        book_title = row['novel_title']

        # Break the text into windows
        windows = chunk_text(raw_text)

        for window in windows:
            processing_queue.append((window, book_title))

    print(f"Original rows: {len(df)}. Analyzing windows: {len(processing_queue)}")

    extracted_nodes = []
    extracted_relationships = []

    # Unzip the queue for batch processing
    chunks_to_process, titles_to_process = zip(*processing_queue)

    # Process with nlp.pipe
    # Batch size of 50 is safe now that chunks are small (~2000 chars)
    doc_stream = nlp.pipe(chunks_to_process, batch_size=50)

    for doc, book_title in tqdm(zip(doc_stream, titles_to_process), total=len(chunks_to_process)):

        # A. Extract Nodes
        entities_in_window = []

        for ent in doc.ents:
            clean_name = ent.text.strip().replace("\n", " ")

            # Heuristic cleanup
            if clean_name.lower().startswith("the ") and len(clean_name) > 4:
                clean_name = clean_name[4:]
            if clean_name.endswith("'s"):
                clean_name = clean_name[:-2]

            # Simple length filter to remove noise like "A" or "Is"
            if len(clean_name) < 2:
                continue

            extracted_nodes.append({
                "name": clean_name,
                "label": ent.label_,
                "source_book": book_title
            })

            entities_in_window.append((clean_name, ent.label_))

        # B. Extract Relationships (Co-occurrence in this Window)
        unique_ents = list(set(entities_in_window))

        for i in range(len(unique_ents)):
            for j in range(i + 1, len(unique_ents)):
                source_ent, source_label = unique_ents[i]
                target_ent, target_label = unique_ents[j]

                # Sort alphabetically to ensure undirected edges are consistent
                # (e.g. always "Gandalf-Frodo", never "Frodo-Gandalf")
                # This helps aggregation later.
                if source_ent > target_ent:
                    source_ent, target_ent = target_ent, source_ent
                    source_label, target_label = target_label, source_label

                extracted_relationships.append({
                    "source": source_ent,
                    "source_label": source_label,
                    "target": target_ent,
                    "target_label": target_label,
                    "type": "MENTIONED_WITH",
                    "book": book_title
                })

    print("\nSaving Data for Neo4j...")

    # Save Nodes
    nodes_df = pd.DataFrame(extracted_nodes)
    # Deduplicate nodes so we have a clean list
    nodes_df = nodes_df.drop_duplicates(subset=['name', 'label'])
    nodes_df.to_csv(OUTPUT_NODES, index=False)

    # Save Relationships
    # Note: We keep duplicates here because frequency = weight.
    # If Rand and Mat appear together 50 times, we want 50 rows (or we aggregate).
    # For now, let's save raw interactions; we can aggregate in Neo4j or pandas later.
    rels_df = pd.DataFrame(extracted_relationships)
    rels_df.to_csv(OUTPUT_RELS, index=False)

    print(f"Success! Extracted {len(nodes_df)} unique nodes and {len(rels_df)} interaction events.")
    print(f"Files created: {OUTPUT_NODES}, {OUTPUT_RELS}")


if __name__ == '__main__':
    extract_entities_and_relationships()