import json
from pathlib import Path
import random
import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding, filter_spans
from spacy.training import Example  # <--- CRITICAL IMPORT

# --- CONFIGURATION ---
TRAINING_DATA_FILE = "fantasy_annotations2.json"
OUTPUT_DIR = Path("../custom_ner_model")
MODEL_NAME = "fantasy_ner"

TARGET_LABELS = ["CHARACTER", "LOCATION", "FACTION", "ARTIFACT"]

LABEL_MAPPING = {
    "PER": "CHARACTER",
    "LOC": "LOCATION",
    "FAC": "FACTION",
    "ORG": "FACTION",
    "ART": "ARTIFACT",
    "ITEM": "ARTIFACT",
    "ARTIFACT": "ARTIFACT",
    "CHARACTER": "CHARACTER",
    "LOCATION": "LOCATION",
    "FACTION": "FACTION"
}


# --- 1. DATA PREPARATION ---

def load_label_studio_data(json_path):
    training_data = []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []

    print(f"Successfully loaded {len(data)} items from JSON.")

    nlp = spacy.blank("en")

    for item in data:
        text = item.get('data', {}).get('text')
        annotations = item.get('annotations', [{}])[0].get('result', [])

        if not text or not annotations:
            continue

        doc = nlp.make_doc(text)
        spans = []

        for ann in annotations:
            val = ann.get('value', {})
            start = val.get('start')
            end = val.get('end')
            label_list = val.get('labels')

            if start is None: start = ann.get('start')
            if end is None: end = ann.get('end')

            if label_list and start is not None and end is not None:
                original_label = label_list[0]

                if original_label in LABEL_MAPPING:
                    final_label = LABEL_MAPPING[original_label]
                    if start < end:
                        span = doc.char_span(start, end, label=final_label)
                        if span is not None:
                            spans.append(span)

        filtered_spans = filter_spans(spans)

        # Convert to dictionary format expected by Example.from_dict
        final_entities = [(span.start_char, span.end_char, span.label_) for span in filtered_spans]

        if final_entities:
            training_data.append((text, {"entities": final_entities}))

    print(f"Successfully converted {len(training_data)} items for training (overlaps resolved).")
    return training_data


def convert_to_docbin(data, output_path):
    nlp = spacy.blank("en")
    doc_bin = DocBin()

    for text, annot in data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annot)  # Create Example object to validate
        doc_bin.add(example.reference)  # Add the reference doc (with correct entities)

    doc_bin.to_disk(output_path)


# --- 2. MODEL TRAINING ---

def train_ner_model(training_data, model_output_path):
    print("\nStarting model training...")
    nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for label in TARGET_LABELS:
        ner.add_label(label)

    # Initialize weights
    nlp.begin_training()

    optimizer = nlp.create_optimizer()

    n_iter = 30
    print(f"Training for {n_iter} iterations...")

    for itn in range(n_iter):
        random.shuffle(training_data)
        losses = {}

        # Batching
        batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))

        for batch in batches:
            examples = []
            for text, annot in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annot)
                examples.append(example)

            # 🌟 FIX: Pass 'examples' list to nlp.update
            nlp.update(examples, drop=0.5, losses=losses, sgd=optimizer)

        if (itn + 1) % 5 == 0:
            print(f"Iteration {itn + 1}/{n_iter} - Loss: {losses.get('ner', 'N/A'):.2f}")

    if model_output_path is not None:
        model_output_path.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(model_output_path)
        print(f"\nModel successfully saved to {model_output_path}")
        return nlp


# --- 3. MAIN EXECUTION ---

if __name__ == '__main__':
    raw_data = load_label_studio_data(TRAINING_DATA_FILE)

    if not raw_data:
        print("\nFATAL ERROR: No valid data found.")
    else:
        random.shuffle(raw_data)

        # Simple split
        split = int(len(raw_data) * 0.8)
        train_data = raw_data[:split]

        # We don't necessarily need DocBin for this simple loop script,
        # but good to have if we switch to CLI training later.
        convert_to_docbin(train_data, Path("train.spacy"))

        trained_nlp = train_ner_model(train_data, OUTPUT_DIR / MODEL_NAME)

        # Test
        print("\n--- Testing Baseline Model ---")
        if trained_nlp:
            test_text = "Eventine must bring his Elven armies across the Plains of Streleheim to reinforce Tyrsis."
            doc = trained_nlp(test_text)
            print(f"Test Sentence: {test_text}")
            for ent in doc.ents:
                print(f"  - '{ent.text}' ({ent.label_})")