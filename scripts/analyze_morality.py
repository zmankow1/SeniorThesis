import os
import glob
import pandas as pd
import re
from collections import defaultdict

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(script_dir, "..", "data", "corpus_txt")
RESULTS_DIR = os.path.join(script_dir, "..", "data", "results")

# 1. DEFINE THE MORAL LEXICON (Mini-MFT)
# We map words to specific moral foundations.
MORAL_LEXICON = {
    "Virtue": {
        "Care": ["safe", "peace", "protect", "help", "kind", "save", "heal", "shelter", "rescue", "life"],
        "Fairness": ["justice", "right", "true", "honest", "equal", "law", "fair", "balance", "proof", "truth"],
        "Loyalty": ["fellowship", "together", "faithful", "honor", "ally", "friend", "promise", "bond", "trust", "kin"],
        "Authority": ["king", "lord", "command", "obey", "lead", "respect", "order", "duty", "father", "rule"],
        "Sanctity": ["pure", "light", "holy", "clean", "sacred", "bless", "spirit", "white", "clear", "divine"]
    },
    "Vice": {
        "Harm": ["kill", "hurt", "war", "blood", "pain", "destroy", "wound", "cruel", "violent", "death"],
        "Cheating": ["lie", "deceit", "traitor", "false", "trick", "steal", "broken", "crooked", "guilt", "fraud"],
        "Betrayal": ["abandon", "alone", "enemy", "betray", "forsake", "rebel", "treason", "deserter", "spy",
                     "turncoat"],
        "Subversion": ["rebel", "defy", "lawless", "chaos", "refuse", "deny", "usurper", "traitor", "mutiny",
                       "disobey"],
        "Degradation": ["corrupt", "dark", "foul", "dirty", "stain", "rot", "disease", "poison", "curse", "filth"]
    }
}

CHARACTERS = {
    "Tolkien (Root)": ["aragorn", "frodo", "gandalf", "sam", "boromir", "gollum", "sauron"],
    "Successors (80s/90s)": ["shea", "flick", "allanon", "rand", "moiraine", "perrin"],
    "Modern (Deconstruction)": ["eddard", "jon", "tyrion", "kaladin", "shallan", "dalinar", "jaime", "cersei"]
}

CORPORA = {
    "Tolkien (Root)": ["FellowshipofTheRing.txt", "TheTwoTowers.txt", "TheReturnofTheKing.txt"],
    "Successors (80s/90s)": ["TheSwordofShannara.txt", "TheEyeofTheWorld.txt"],
    "Modern (Deconstruction)": ["AGameofThrones.txt", "Assassin'sApprentice.txt", "TheWayofKings.txt"]
}


def load_corpus_map():
    file_map = {}
    for group, files in CORPORA.items():
        for f in files: file_map[f] = group
    return file_map


def score_text_morality(text_snippet):
    """Counts moral terms in a snippet."""
    scores = defaultdict(int)
    words = re.findall(r'\w+', text_snippet.lower())

    for word in words:
        # Check Virtues
        for foundation, terms in MORAL_LEXICON["Virtue"].items():
            if word in terms:
                scores[f"{foundation}_Virtue"] += 1

        # Check Vices
        for foundation, terms in MORAL_LEXICON["Vice"].items():
            if word in terms:
                scores[f"{foundation}_Vice"] += 1

    return scores, len(words)


def extract_character_contexts(text, character_name, window=50):
    text_lower = text.lower()
    char_lower = character_name.lower()
    words = text_lower.split()
    contexts = []
    indices = [i for i, x in enumerate(words) if char_lower in x]
    for idx in indices:
        start = max(0, idx - window)
        end = min(len(words), idx + window)
        contexts.append(" ".join(words[start:end]))
    return contexts


def analyze_moral_foundations():
    file_map = load_corpus_map()

    # Data: {"Gandalf": {"Virtue_Care": 10, "Vice_Harm": 2, ... "Total_Words": 5000}}
    char_profiles = {}

    print("⚖️  Scanning Corpus for Moral Signals...")

    for filepath in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
        filename = os.path.basename(filepath)
        group = file_map.get(filename)
        if not group: continue

        print(f"   Reading {filename}...")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()

        relevant_chars = CHARACTERS.get(group, [])

        for char in relevant_chars:
            contexts = extract_character_contexts(full_text, char)
            if not contexts: continue

            if char not in char_profiles:
                char_profiles[char] = {"Group": group, "WordCount": 0, "Scores": defaultdict(int)}

            for ctx in contexts:
                snippet_scores, n_words = score_text_morality(ctx)
                char_profiles[char]["WordCount"] += n_words
                for k, v in snippet_scores.items():
                    char_profiles[char]["Scores"][k] += v

    print("\n🧮 Calculating Moral Profiles...")
    results = []

    for char, data in char_profiles.items():
        total_words = data["WordCount"]
        if total_words < 1000: continue  # Skip minor mentions

        row = {"Group": data["Group"], "Character": char.capitalize()}

        total_virtue_weighted = 0
        total_vice_weighted = 0

        # Explicitly map Virtues to Vices for accurate comparison
        foundation_pairs = {
            "Care": "Harm",
            "Fairness": "Cheating",
            "Loyalty": "Betrayal",
            "Authority": "Subversion",
            "Sanctity": "Degradation"
        }

        for virtue, vice in foundation_pairs.items():
            # Get raw counts
            v_count = data["Scores"][f"{virtue}_Virtue"]
            bad_count = data["Scores"][f"{vice}_Vice"]

            # Normalize per 1,000 words
            row[f"{virtue} (Virtue)"] = round((v_count / total_words) * 1000, 2)
            row[f"{virtue} (Vice)"] = round((bad_count / total_words) * 1000, 2)

            total_virtue_weighted += v_count
            total_vice_weighted += bad_count

        # Overall "Goodness" Ratio
        if total_vice_weighted == 0: total_vice_weighted = 1  # Avoid division by zero
        row["Moral Ratio (Good/Evil)"] = round(total_virtue_weighted / total_vice_weighted, 2)

        results.append(row)

    df = pd.DataFrame(results)
    # Sort by the Ratio to see Heroes at top, Villains at bottom
    return df.sort_values(by="Moral Ratio (Good/Evil)", ascending=False)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = analyze_moral_foundations()

    print("\n" + "=" * 60)
    print("RESULTS: MORAL FOUNDATIONS ANALYSIS")
    print("=" * 60)

    # We print a simplified view for the console
    cols = ["Group", "Character", "Moral Ratio (Good/Evil)", "Care (Virtue)", "Care (Vice)", "Authority (Virtue)"]
    print(df[cols].to_string(index=False))

    output_path = os.path.join(RESULTS_DIR, "moral_foundations.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved detailed report to {output_path}")