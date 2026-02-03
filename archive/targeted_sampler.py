import pandas as pd
import re

# This script pulls sentences specifically containing your missing characters
# so you can label them and fix the model's bias.

INPUT_CSV = "processed_paragraphs.csv"
OUTPUT_CSV = "targeted_labeling_names.csv"


def get_targeted_sentences():
    try:
        df = pd.read_csv(INPUT_CSV)

        # Search for "Rand" or "Perrin" as whole words
        pattern = r'\bRand\b|\bPerrin\b'
        mask = df['raw_chunk'].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)

        # Take up to 60 targeted sentences
        targeted_df = df[mask].head(60)

        if targeted_df.empty:
            print("❌ No sentences found with those names. Check your input CSV!")
            return

        # Prepare for Label Studio
        targeted_df = targeted_df.rename(columns={'raw_chunk': 'text'})
        targeted_df[['text']].to_csv(OUTPUT_CSV, index=False)

        print(f"✅ Created {OUTPUT_CSV} with {len(targeted_df)} targeted sentences.")
        print("Now upload this CSV to Label Studio and label the names as CHARACTER.")

    except FileNotFoundError:
        print(f"❌ Could not find {INPUT_CSV}")


if __name__ == "__main__":
    get_targeted_sentences()