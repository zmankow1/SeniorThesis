import os
import re
import pandas as pd


def find_books(directory):
    """Finds all .txt files in the specified directory."""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith('.txt')]


def smart_split(text):
    """
    Detects the format of the text and splits it into logical chunks.
    Ensures consistent granularity across different file formats.
    """
    # 1. Standardize line endings and remove carriage returns
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 2. Pre-processing: If it's hard-wrapped or a wall of text, un-wrap it.
    # We define 'Hard Wrapped' as having many newlines but very few double newlines.
    if text.count('\n') > 100 and text.count('\n\n') < (text.count('\n') / 4):
        # Join single lines into a flow, but preserve double newlines as paragraph markers
        # We use a temporary placeholder to protect true paragraphs
        text = re.sub(r'\n\n+', '||PARAGRAPH||', text)
        text = text.replace('\n', ' ')
        text = text.replace('||PARAGRAPH||', '\n\n')

    # 3. Decision Point: How to chunk?
    # To ensure consistent granularity (e.g., ~1000 rows per book),
    # we will use a sentence-based approach for ALL books,
    # but we will use paragraph markers as preferred break points.

    # First, flatten the text to handle all variations the same way
    clean_text = re.sub(r'\s+', ' ', text).strip()

    # Split by sentences using regex (Lookbehind for .!? and Lookahead for Space + Capital)
    print("   [Mode] Applying Universal Sentence-Aware Chunking...")
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean_text)

    # Group sentences into chunks.
    # A standard fantasy paragraph is ~5-8 sentences.
    # Let's use a chunk size of 7 to get high granularity (target ~1500-2000 rows for large books).
    chunk_size = 7
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

    return chunks


def clean_and_process(file_path, book_name):
    """Reads, cleans, and chunks the book."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
    except Exception as e:
        print(f"   [Error] Could not read file: {e}")
        return []

    # Apply the unified splitting logic
    chunks = smart_split(raw_text)

    # Final data structure
    data = []
    for i, chunk in enumerate(chunks):
        # Final cleanup of the chunk
        clean_chunk = re.sub(r'\s+', ' ', chunk).strip()
        if len(clean_chunk) > 30:  # Ignore fragments shorter than a short sentence
            data.append({
                "book_id": book_name.replace(".txt", ""),
                "chunk_id": i,
                "text": clean_chunk
            })

    return data


if __name__ == "__main__":
    input_dir = "../data/corpus_txt"
    output_dir = "../data/processed_data"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_data = []
    book_files = find_books(input_dir)
    # Ensure consistent order for processing
    book_files.sort()

    print(f"--- Found {len(book_files)} novels. ---")

    for i, book in enumerate(book_files, 1):
        print(f"[{i}/{len(book_files)}] Processing: {book}...")
        file_path = os.path.join(input_dir, book)

        book_data = clean_and_process(file_path, book)
        if book_data:
            all_data.extend(book_data)
            print(f"   Found {len(book_data)} rows.")
        else:
            print(f"   [Warning] No valid data extracted from {book}.")

    if all_data:
        # Save to a single CSV for your NLP work
        df = pd.DataFrame(all_data)
        df.to_csv(os.path.join(output_dir, "master_corpus.csv"), index=False)
        print(f"\n--- SUCCESS! Total Rows: {len(df)} ---")

        # Summary by book
        print("\nRow Counts by Book:")
        print(df['book_id'].value_counts())
    else:
        print("\n--- FAILED: No data processed. Check your input directory. ---")