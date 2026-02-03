import spacy

# Load your custom model
model_path = "../custom_ner_model/fantasy_ner"
try:
    nlp = spacy.load(model_path)
    print(f"✅ Model loaded from {model_path}")
except:
    print("❌ Could not load model. Check the path!")
    exit()

# Test sentences
test_sentences = [
    "Rand al'Thor gripped his sword and looked at Perrin Aybara.",
    "The Dragon Reborn, Rand, walked through the gates of Caemlyn.",
    "Perrin felt the wolf within him stir as he spoke to Moiraine."
]

# --- 2. ADVANCED DIAGNOSTICS ---
# We use more diverse sentences to see if the model handles titles and apostrophes correctly.
test_cases = [
    {
        "category": "The Big Three (Protagonists)",
        "text": "Rand al'Thor, Perrin Aybara, and Mat Cauthon are from Emond's Field."
    },
    {
        "category": "Titles vs. Names",
        "text": "The Dragon Reborn spoke with the Amyrlin Seat while Moiraine waited."
    },
    {
        "category": "Geography (Locations)",
        "text": "They traveled from Caemlyn to Shienar, crossing the River Arinelle."
    },
    {
        "category": "Apostrophe Sensitivity",
        "text": "Egwene al'Vere and Nynaeve al'Meara joined the Aes Sedai."
    }
]

print("\n" + "=" * 50)
print(f"DIAGNOSTIC REPORT for model: {model_path}")
print("=" * 50)

for case in test_cases:
    print(f"\nCategory: {case['category']}")
    print(f"Input: \"{case['text']}\"")
    doc = nlp(case['text'])

    if not doc.ents:
        print("  ❌ ERROR: No entities detected at all.")
    else:
        for ent in doc.ents:
            # Check for common confusion patterns
            status = "✅"
            if ent.text == "Rand" and ent.label_ == "LOCATION":
                status = "⚠️ CONFUSION (Person labeled as Place)"
            elif ent.text == "Caemlyn" and ent.label_ == "CHARACTER":
                status = "⚠️ CONFUSION (Place labeled as Person)"

            print(f"  {status} [{ent.text}] -> Label: {ent.label_}")

print("\n" + "=" * 50)
print("ANALYSIS & RECOMMENDATION:")
if "fantasy_ner_v2" not in model_path:
    print("- ACTION: You are testing the OLD baseline model. Ensure you run 'ner_trainer_v2.py' first.")
else:
    print("- ACTION: If Rand is still a LOCATION, you need to label ~50 more sentences specifically")
    print("  mentioning Rand and Perrin as CHARACTER to break the 'Short Name = Place' bias.")
print("=" * 50)