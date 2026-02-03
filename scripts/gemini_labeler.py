import pandas as pd
import os
import json
import time
import requests
import re

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(script_dir, "..", "data", "processed_data", "master_corpus.csv")
OUTPUT_JSON = os.path.join(script_dir, "..", "data", "annotations", "gold_standard_training.json")

SAMPLE_SIZE = 2000
# scripts/gemini_labeler.py
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ Error: GEMINI_API_KEY not found in environment variables.")

def is_metadata(text):
    """Skips Table of Contents and Chapter lists which confuse the AI."""
    metadata_indicators = [r"Chapter \w+", r"Page \d+", r"Contents", r"Appendix", r"Prologue", r"Map of"]
    text_snippet = text[:200]
    if sum(1 for pat in metadata_indicators if re.search(pat, text_snippet, re.I)) > 2:
        return True
    return False


def get_gemini_labels(text):
    if not API_KEY: return {"entities": []}

    prompt = f"""
    Act as a high-level Computational Linguist. Extract entities from this fantasy text.

    CATEGORIES:
    - CHARACTER: Specific people, unique creatures (e.g., 'Ghost', 'Tyrion').
    - LOCATION: Geographic sites, cities, specific buildings (e.g., 'Winterfell', 'The Eyrie').
    - FACTION: Military groups, houses, races (e.g., 'House Stark', 'Aes Sedai', 'Andals').
    - ARTIFACT: Unique named items (e.g., 'Ice', 'Shardblade', 'The One Ring').

    STRICT NEGATIVE CONSTRAINTS:
    1. DO NOT label Titles as Locations. 'King's Hand', 'Maester', and 'Princess' are NOT locations.
    2. STRIP all possessives. 'Illyrio's' becomes 'Illyrio'. 'Winterfell's' becomes 'Winterfell'.
    3. NO METADATA. Ignore chapter names, page numbers, or book titles.
    4. NO DIALOGUE SNIPPETS. 'me?Bran' is forbidden. It must be 'Bran'.
    5. NO ADJECTIVES. 'Alethi' (Faction) is okay, but 'Alethi Soulcaster' is an ARTIFACT.

    Return ONLY JSON: {{"entities": [ {{"text": "Clean Name", "label": "LABEL"}}, ... ]}}

    TEXT: {text}
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"responseMimeType": "application/json"}}

    for delay in [1, 2, 4, 8]:
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])
            time.sleep(delay)
        except:
            time.sleep(delay)
    return {"entities": []}


def run_power_labeler():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    # Filter out chunks that look like Table of Contents
    df['is_meta'] = df['text'].apply(is_metadata)
    clean_df = df[df['is_meta'] == False]

    sample_df = clean_df.sample(n=min(SAMPLE_SIZE, len(clean_df))).copy()
    gold_data = []

    print(f"Starting High-Fidelity Labeling for {SAMPLE_SIZE} chunks...")
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        print(f"   [{i + 1}/{SAMPLE_SIZE}] {row['book_id']} (Chunk {row['chunk_id']})...", end="\r")
        result = get_gemini_labels(row['text'])
        gold_data.append({"text": row['text'], "entities": result.get('entities', [])})
        time.sleep(0.4)  # Faster but safe rate

    with open(OUTPUT_JSON, "w", encoding='utf-8') as f:
        json.dump(gold_data, f, indent=4)

    print(f"\n✅ SUCCESS! Generated {len(gold_data)} high-quality training examples.")


if __name__ == "__main__":
    run_power_labeler()