import pandas as pd
import re
import os


def run_fixer():
    """
    Reads the existing master_corpus.csv and applies a targeted dictionary
    to find characters that spaCy typically misses (like Perrin, Sansa, or The Fool).
    """
    input_file = "../data/processed_data/master_corpus.csv"
    output_file = "../data/processed_data/manual_labels.csv"

    if not os.path.exists(input_file):
        print("Error: master_corpus.csv not found.")
        return

    print("Loading master corpus for targeted fixing...")
    df = pd.read_csv(input_file)

    # The 'Manual Knowledge Base' - names spaCy is bad at catching
    # Key: Book ID, Value: List of names to force-search for
    target_map = {
        "TheEyeofTheWorld": ["Perrin", "Min", "Ba'alzamon", "Elyas", "Loial", "Logain"],
        "AGameofThrones": ["Sansa", "Bran", "Cersei", "Jaime", "Theon", "Varys", "Sandor"],
        "TheWayofKings": ["Teft", "Rock", "Lopen", "Wit", "Hoid", "Renarin", "Moash"],
        "Assassin'sApprentice": ["The Fool", "Patience", "Nighteyes", "Shrewd"],
        "TheSwordofShannara": ["Flick", "Hendel", "Stenmin"],
        "FellowshipofTheRing": ["Theoden", "Galadriel", "Celebrorn"],
        "TheTwoTowers": ["Theoden", "Eowyn", "Faramir", "Uglúk"],
        "TheReturnofTheKing": ["Theoden", "Denethor", "Ioreth"]
    }

    print("Scanning for missing legends...")

    fixed_entities = []

    for _, row in df.iterrows():
        text = str(row['text'])
        book_id = row['book_id']

        # Check if we have specific targets for this book
        targets = target_map.get(book_id, [])
        found_in_chunk = []

        for name in targets:
            # Case-insensitive search but requiring word boundaries
            if re.search(rf'\b{name}\b', text, re.IGNORECASE):
                # We append the proper-case version from our list
                found_in_chunk.append(name)

        # For this script, we'll just store the "Found Manuals"
        # You can later merge this with your spaCy labels
        fixed_entities.append(", ".join(found_in_chunk))

    df['manual_entities'] = fixed_entities

    # Calculate a combined count (Manuals + spaCy if you want, but let's just see these for now)
    df['manual_count'] = df['manual_entities'].apply(lambda x: len(x.split(',')) if x else 0)

    print(f"Saving updated labels to {output_file}...")
    df.to_csv(output_file, index=False)

    # Quick Report
    print("\n--- TARGETED FIX REPORT ---")
    for book in target_map.keys():
        count = df[df['book_id'] == book]['manual_count'].sum()
        print(f"{book}: Found missing characters {count} times.")


if __name__ == "__main__":
    run_fixer()