import spacy
from spacy import displacy


# You might need to run: python -m spacy download en_core_web_lg
# 'en_core_web_lg' is a 500MB model trained on millions of news and web documents.

def pretrained_model(text_to_test):
    try:
        # Load the "Large" English model
        print("Loading industry-standard model (en_core_web_lg)...")
        nlp = spacy.load("en_core_web_lg")
        print("✅ Model loaded.")
    except OSError:
        print("❌ Model 'en_core_web_lg' not found.")
        print("Please run: python -m spacy download en_core_web_lg")
        return

    doc = nlp(text_to_test)

    print("\n" + "=" * 50)
    print("STANDARD NER RESULTS")
    print("=" * 50)

    # Standard models use labels like PERSON, GPE (Location), and ORG (Organization)
    for ent in doc.ents:
        # Mapping standard labels to your fantasy labels for comparison
        label_map = {
            "PERSON": "CHARACTER",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "ORG": "FACTION/ORG",
            "PRODUCT": "ARTIFACT?"
        }
        mapped_label = label_map.get(ent.label_, ent.label_)
        print(f"[{ent.text}] -> Standard Label: {ent.label_} (Mapped: {mapped_label})")

    if not doc.ents:
        print("No entities found.")
    print("=" * 50)


# Test with your problematic sentences
test_text = """
Rand al'Thor and Perrin Aybara walked through the streets of Caemlyn. 
The Dragon Reborn looked at the White Tower. 
Moiraine Damodred guided Egwene al'Vere.
"""

if __name__ == "__main__":
    pretrained_model(test_text)