import json
import spacy
from spacy.training import Example
import random
import os
import re


def clean_and_normalize(text):
    """Strips possessives and punctuation so the AI learns the base concept."""
    text = re.sub(r"['’]s$", "", text)
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    return text.strip()


def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(script_dir, "..", "data", "annotations", "gold_standard_training.json")
    OUTPUT_MODEL = os.path.join(script_dir, "..", "custom_ner_model", "fantasy_ner_v4")

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    labels = set()
    for entry in gold_data:
        for ent in entry.get('entities', []):
            labels.add(ent['label'])
    for label in labels: ner.add_label(label)

    train_examples = []
    for entry in gold_data:
        text = entry['text']
        doc = nlp.make_doc(text)
        entities = []

        for ent in entry.get('entities', []):
            cleaned_name = clean_and_normalize(ent['text'])
            start = text.find(cleaned_name)
            if start != -1:
                end = start + len(cleaned_name)
                span = doc.char_span(start, end, label=ent['label'], alignment_mode="contract")
                if span: entities.append((span.start_char, span.end_char, span.label_))

        entities = sorted(list(set(entities)))
        cleaned = []
        last_end = -1
        for s, e, l in entities:
            if s >= last_end:
                cleaned.append((s, e, l))
                last_end = e

        train_examples.append(Example.from_dict(doc, {"entities": cleaned}))

    print(f"Training v4 Model on {len(train_examples)} examples...")
    optimizer = nlp.begin_training()
    for i in range(50):  # 50 iterations for maximum fidelity
        random.shuffle(train_examples)
        losses = {}
        batches = spacy.util.minibatch(train_examples, size=spacy.util.compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, drop=0.4, losses=losses)
        if i % 5 == 0:
            print(f"   Iteration {i:02d} | Loss: {losses['ner']:.4f}")

    nlp.to_disk(OUTPUT_MODEL)
    print(f"\n✅ v4 'Final Brain' saved to: {OUTPUT_MODEL}")


if __name__ == "__main__":
    train_model()