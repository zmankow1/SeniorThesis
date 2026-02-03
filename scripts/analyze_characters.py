import pandas as pd
from collections import Counter
import os


def run_final_analysis():
    """
    Merges automated and manual labels, cleans the data,
    and produces the final character frequency report.

    Structure assumed: SeniorThesis/scripts/ (this script)
    """
    # PATH SETUP for SeniorThesis structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(script_dir, "..", "data", "processed_data"))

    # Renamed Input Files
    spacy_file = os.path.join(base_path, "automated_labels_cleaned.csv")
    manual_file = os.path.join(base_path, "manual_labels.csv")

    # Output Files
    output_summary = os.path.join(base_path, "final_character_summary.txt")
    output_csv = os.path.join(base_path, "final_character_frequencies.csv")

    if not os.path.exists(manual_file):
        print(f"Error: {manual_file} not found. Ensure you moved your renamed files to the data/processed_data folder!")
        return

    print("Loading merged corpus data...")
    df = pd.read_csv(manual_file)

    # Merge logic: Recover automated labels if not present
    if 'key_entities' not in df.columns:
        if os.path.exists(spacy_file):
            print("   [Info] 'key_entities' column missing. Merging with automated labels...")
            spacy_df = pd.read_csv(spacy_file)
            df = pd.merge(df, spacy_df[['book_id', 'chunk_id', 'key_entities']],
                          on=['book_id', 'chunk_id'], how='left')
        else:
            print(f"   [Warning] '{spacy_file}' not found. Using manual labels only.")

    print("Synthesizing labels...")
    results = []
    summary_output = "FINAL CHARACTER ANALYSIS (MERGED & CLEANED)\n" + "=" * 50 + "\n"

    # Noise filter & Alias mapping
    noise_filter = {"North", "South", "East", "West", "The World", "City", "High", "Low", "maester", "lady", "lord"}
    alias_map = {
        "Kal": "Kaladin", "Strider": "Aragorn", "Baggins": "Frodo",
        "Mithrandir": "Gandalf", "Sméagol": "Gollum", "gollum": "Gollum",
        "Littlefinger": "Petyr Baelish", "Samwise": "Sam", "Peregrin": "Pippin"
    }

    for book in df['book_id'].unique():
        book_df = df[df['book_id'] == book]
        all_final_entities = []

        for _, row in book_df.iterrows():
            spacy_ents = str(row.get('key_entities', '')).split(',') if pd.notnull(row.get('key_entities')) else []
            manual_ents = str(row.get('manual_entities', '')).split(',') if pd.notnull(
                row.get('manual_entities')) else []

            combined = set([e.strip() for e in (spacy_ents + manual_ents)])

            for n in combined:
                if not n or n in noise_filter or len(n) <= 2:
                    continue
                final_name = alias_map.get(n, n)
                all_final_entities.append(final_name)

        counts = Counter(all_final_entities)
        top_25 = counts.most_common(25)

        summary_output += f"\nBOOK: {book}\n" + "-" * len(book) + "\n"
        for name, count in top_25:
            summary_output += f"  - {name:<25} | Count: {count}\n"
            results.append({"book": book, "character": name, "frequency": count})

    # Save to the data/processed_data folder
    with open(output_summary, "w", encoding='utf-8') as f:
        f.write(summary_output)

    pd.DataFrame(results).to_csv(output_csv, index=False)

    print(f"✅ SUCCESS! Final report saved to: {output_summary}")
    print(f"✅ CSV frequencies saved to: {output_csv}")


if __name__ == "__main__":
    run_final_analysis()