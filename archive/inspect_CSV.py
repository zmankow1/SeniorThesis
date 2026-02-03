import pandas as pd
import re
import os

INPUT_CSV = "processed_paragraphs.csv"
OUTPUT_CSV = "targeted_labeling_names.csv"


def inspect_and_sample():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ File not found: {INPUT_CSV}")
        return

    # Load the full data
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # 1. DATA VOLUME DIAGNOSTIC
    print(f"\n--- Data Volume Report ---")
    print(f"Total rows in CSV: {len(df)}")

    if 'novel_title' in df.columns:
        print("\nParagraphs per book found in CSV:")
        print(df['novel_title'].value_counts())
    else:
        print("⚠️ 'novel_title' column missing.")

    # 2. COLUMN DETECTION
    text_col = 'raw_chunk' if 'raw_chunk' in df.columns else df.columns[0]
    print(f"\n📖 Searching in column: '{text_col}'")

    # 3. BROADENED SEARCH STRATEGY
    # Added more variants and common names to verify data presence
    names_to_find = [
        'Rand', 'Perrin', 'Mat ', 'Matth', 'Egwen', 'Moiraine', 'Nynaeve',  # WoT
        'Dalinar', 'Kaladin', 'Shallan', 'Szeth', 'Adolin',  # Stormlight
        'Shea', 'Flick', 'Allanon', 'Wil ', 'Amberle'  # Shannara
    ]
    pattern = '|'.join(names_to_find)

    print(f"⚡ Searching for major protagonists...")

    # Use a case-insensitive search
    mask = df[text_col].str.contains(pattern, flags=re.IGNORECASE, na=False)
    targeted_df = df[mask].copy()

    if targeted_df.empty:
        print("❌ Still no matches found for any major characters.")
        print("\nFirst 2 lines of text data for inspection:")
        print(df[text_col].head(2).values)
        return

    print(f"✅ Success! Found {len(targeted_df)} matching rows.")

    # 4. EXPORT
    # We take a larger head (200) since we want plenty of examples to fix the model
    final_df = targeted_df.head(200).rename(columns={text_col: 'text'})
    final_df[['text']].to_csv(OUTPUT_CSV, index=False)

    print(f"💾 Saved {len(final_df)} sentences to {OUTPUT_CSV}")
    print("\n--- Next Steps ---")
    if len(targeted_df) < 20:
        print(
            "⚠️ Data count is very low. You may need to re-run 'CleanText.py' and check if the './corpus_txt/' folder contains all your files.")
    else:
        print("Upload the new 'targeted_labeling_names.csv' to Label Studio to fix the Rand/Perrin labels.")


if __name__ == "__main__":
    inspect_and_sample()