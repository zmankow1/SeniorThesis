import pandas as pd
import os
import re


def clean_entity_string(entity_str):
    """
    Cleans a comma-separated string of entities.
    Removes punctuation artifacts, generic noise, and lowercase misidentifications.
    """
    if pd.isna(entity_str) or entity_str == "":
        return ""

    # Split into individual entities
    ents = [e.strip() for e in str(entity_str).split(',')]
    cleaned_ents = []

    # 1. Noise words to discard
    noise_words = {
        "maester", "wolf", "bush", "ironwood", "albino", "squires", "king",
        "queen", "m’lord", "m'lord", "lord", "lady", "sir", "a king", "the king",
        "sword", "shield", "plate", "blade", "sun", "moon", "day", "night",
        "gate", "inn", "stable", "horse", "dragon", "cell", "dungeon", "way"
    }

    for ent in ents:
        # A. Strip leading/trailing punctuation (catches ”Bran, etc.)
        ent = re.sub(r"^[^\w]+|[^\w]+$", "", ent).strip()

        # B. Skip if too short or in noise list
        if len(ent) < 3 or ent.lower() in noise_words:
            continue

        # C. Validation: Fantasy proper nouns MUST be capitalized
        # (Catches cases where common words at start of sentences are mislabeled)
        if not ent[0].isupper():
            continue

        # D. Filter generic "The [Noun]" (e.g., 'The Room', 'The King')
        if ent.lower().startswith("the ") and len(ent.split()) == 2:
            continue

        cleaned_ents.append(ent)

    # Re-join into a clean string
    return ", ".join(list(set(cleaned_ents)))


def refine_automated_data():
    """
    Reads the automated_labels.csv and applies the cleaning logic.
    Saves the cleaned version back to the processed_data folder.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(script_dir, "..", "data", "processed_data"))

    input_file = os.path.join(base_path, "automated_labels.csv")
    output_file = os.path.join(base_path, "automated_labels_cleaned.csv")

    if not os.path.exists(input_file):
        print(f"❌ Error: Could not find {input_file}")
        return

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    print("Refining AI-generated labels (scrubbing noise)...")
    df['key_entities'] = df['key_entities'].apply(clean_entity_string)

    # Update count column to reflect cleaned data
    df['entity_count'] = df['key_entities'].apply(lambda x: len(x.split(',')) if x else 0)

    df.to_csv(output_file, index=False)
    print(f"✅ SUCCESS! Cleaned labels saved to: {output_file}")
    print("Now update your analysis scripts to use 'automated_labels_cleaned.csv' as the spacy_file.")


if __name__ == "__main__":
    refine_automated_data()