import spacy
import pandas as pd
import os


def run_inference():
    """Scales the trained model across the entire 14k chunk corpus."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CORPUS_PATH = os.path.join(script_dir, "..", "data", "processed_data", "master_corpus.csv")
    MODEL_PATH = os.path.join(script_dir, "..", "custom_ner_model", "fantasy_ner_v2")
    OUTPUT_PATH = os.path.join(script_dir, "..", "data", "processed_data", "ai_gold_labels.csv")

    if not os.path.exists(MODEL_PATH):
        print("❌ Error: fantasy_ner_v2 model not found. Train it first!")
        return

    print("Loading Fantasy Brain...")
    nlp = spacy.load(MODEL_PATH)
    df = pd.read_csv(CORPUS_PATH)

    results = []
    print(f"Processing {len(df)} chunks. This may take a few minutes...")

    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 1000 == 0 and i > 0:
            print(f"   Progress: {i}/{len(df)} chunks...")

        doc = nlp(str(row['text']))

        # Format entities for easy reading
        raw_entities = [ent.text for ent in doc.ents]
        labeled_entities = [f"{ent.text}|{ent.label_}" for ent in doc.ents]

        results.append({
            "book_id": row['book_id'],
            "chunk_id": row['chunk_id'],
            "key_entities": ",".join(raw_entities),
            "labeled_entities": ",".join(labeled_entities),
            "entity_count": len(doc.ents)
        })

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ COMPLETE! AI labels generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_inference()