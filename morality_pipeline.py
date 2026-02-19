"""
Fantasy Morality Analysis Pipeline
====================================
Analyzes the moral alignment of major characters across 8 fantasy novels,
mapping the evolution of the good/evil axis from Tolkien (1954) to modern fantasy.

Methodology:
  1. Load and clean each book's text
  2. Use seeded character lists + window extraction to associate sentences with characters
  3. Score each sentence on good/evil using a curated moral lexicon
  4. Aggregate per-character alignment scores
  5. Compute book-level moral ambiguity metrics
  6. Produce visualizations: alignment scatter, ambiguity timeline, character heatmaps
"""

import re
import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from collections import defaultdict
from scipy import stats

# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────

BOOKS = {
    "Fellowship of the Ring": {
        "path": "data/corpus_txt/FellowshipofTheRing.txt",
        "year": 1954,
        "author": "Tolkien",
        "characters": {
            # name: (aliases, canonical_alignment)  alignment: -1=evil, 1=good, 0=ambiguous
            "Frodo":    (["Frodo", "Mr. Frodo", "Master Frodo"], 1),
            "Sam":      (["Sam", "Samwise", "Sam Gamgee"], 1),
            "Gandalf":  (["Gandalf", "the wizard", "Mithrandir"], 1),
            "Aragorn":  (["Aragorn", "Strider", "Elessar"], 1),
            "Sauron":   (["Sauron", "the Dark Lord", "the Enemy"], -1),
            "Saruman":  (["Saruman", "the White"], -1),
            "Boromir":  (["Boromir"], 0),
            "Gollum":   (["Gollum", "Sméagol", "Smeagol"], 0),
        }
    },
    "The Two Towers": {
        "path": "data/corpus_txt/TheTwoTowers.txt",
        "year": 1954,
        "author": "Tolkien",
        "characters": {
            "Frodo":    (["Frodo", "Mr. Frodo"], 1),
            "Sam":      (["Sam", "Samwise"], 1),
            "Gandalf":  (["Gandalf", "the White", "Mithrandir"], 1),
            "Aragorn":  (["Aragorn", "Strider"], 1),
            "Gollum":   (["Gollum", "Sméagol", "Smeagol"], 0),
            "Saruman":  (["Saruman"], -1),
            "Sauron":   (["Sauron", "the Dark Lord"], -1),
            "Faramir":  (["Faramir"], 1),
        }
    },
    "The Return of the King": {
        "path": "data/corpus_txt/TheReturnofTheKing.txt",
        "year": 1955,
        "author": "Tolkien",
        "characters": {
            "Frodo":    (["Frodo", "Mr. Frodo"], 1),
            "Sam":      (["Sam", "Samwise"], 1),
            "Gandalf":  (["Gandalf"], 1),
            "Aragorn":  (["Aragorn", "the King", "Elessar"], 1),
            "Gollum":   (["Gollum", "Sméagol", "Smeagol"], 0),
            "Sauron":   (["Sauron", "the Dark Lord", "the Enemy"], -1),
            "Denethor": (["Denethor"], -1),
        }
    },
    "The Sword of Shannara": {
        "path": "data/corpus_txt/TheSwordofShannara.txt",
        "year": 1977,
        "author": "Brooks",
        "characters": {
            "Shea":       (["Shea", "Shea Ohmsford", "Ohmsford"], 1),
            "Allanon":    (["Allanon"], 1),
            "Flick":      (["Flick"], 1),
            "Menion":     (["Menion", "Menion Leah"], 1),
            "Eventine":   (["Eventine"], 1),
            "Brona":      (["Brona", "the Warlock Lord", "the Skull Bearer"], -1),
            "Stenmin":    (["Stenmin"], -1),
            "Balinor":    (["Balinor"], 1),
        }
    },
    "Assassin's Apprentice": {
        "path": "data/corpus_txt/Assassin'sApprentice.txt",
        "year": 1995,
        "author": "Hobb",
        "characters": {
            "Fitz":       (["Fitz", "FitzChivalry", "Boy", "Tom"], 0),
            "Chade":      (["Chade"], 0),
            "Burrich":    (["Burrich"], 0),
            "Shrewd":     (["Shrewd", "King Shrewd"], 0),
            "Regal":      (["Regal", "Prince Regal"], -1),
            "Verity":     (["Verity", "Prince Verity"], 1),
            "Patience":   (["Patience", "Lady Patience"], 1),
            "Galen":      (["Galen"], -1),
        }
    },
    "A Game of Thrones": {
        "path": "data/corpus_txt/AGameofThrones.txt",
        "year": 1996,
        "author": "Martin",
        "characters": {
            "Eddard":     (["Eddard", "Ned", "Lord Stark", "Lord Eddard"], 1),
            "Jon":        (["Jon", "Jon Snow"], 0),
            "Daenerys":   (["Daenerys", "Dany", "Khaleesi"], 0),
            "Tyrion":     (["Tyrion", "the Imp", "Lannister"], 0),
            "Cersei":     (["Cersei", "the Queen"], -1),
            "Joffrey":    (["Joffrey"], -1),
            "Jaime":      (["Jaime", "the Kingslayer"], 0),
            "Littlefinger": (["Littlefinger", "Baelish", "Petyr"], -1),
        }
    },
    "The Eye of the World": {
        "path": "data/corpus_txt/TheEyeofTheWorld.txt",
        "year": 1990,
        "author": "Jordan",
        "characters": {
            "Rand":       (["Rand", "Rand al'Thor"], 0),
            "Mat":        (["Mat", "Matrim"], 0),
            "Perrin":     (["Perrin"], 0),
            "Egwene":     (["Egwene"], 1),
            "Moiraine":   (["Moiraine"], 1),
            "Lan":        (["Lan", "al'Lan"], 1),
            "Ba'alzamon": (["Ba'alzamon", "the Dark One", "Ishamael"], -1),
            "Thom":       (["Thom", "Thom Merrilin"], 0),
        }
    },
    "The Way of Kings": {
        "path": "data/corpus_txt/TheWayofKings.txt",
        "year": 2010,
        "author": "Sanderson",
        "characters": {
            "Kaladin":    (["Kaladin", "Kal"], 0),
            "Dalinar":    (["Dalinar", "Dalinar Kholin"], 1),
            "Szeth":      (["Szeth", "Szeth-son-son-Vallano"], 0),
            "Shallan":    (["Shallan", "Shallan Davar"], 0),
            "Adolin":     (["Adolin"], 1),
            "Sadeas":     (["Sadeas"], -1),
            "Jasnah":     (["Jasnah", "Jasnah Kholin"], 0),
            "Wit":        (["Wit", "Hoid"], 0),
        }
    },
}

# ─────────────────────────────────────────────
# 1. Moral Lexicon  (v2 — precision over recall)
# ─────────────────────────────────────────────
# Rule: only include words that are UNAMBIGUOUSLY moral in nearly all contexts.
# Removed: just, light, guard, free, fair, warm, strength, hope, love, peace,
#          friend, shield, save, help, good — all too common in neutral usage.
# Added: weighted tiers so rarer, stronger words score higher.

# Tier 1 (weight 2.0): unambiguous, strong moral signal
GOOD_T1 = {
    "noble", "righteous", "virtuous", "compassionate", "selfless", "benevolent",
    "merciful", "courageous", "valiant", "valor", "valour", "heroic", "gallant",
    "magnanimous", "incorruptible", "saintly", "blameless", "steadfast",
    "honorable", "honourable",
}
# Tier 2 (weight 1.0): clear moral signal but more common
GOOD_T2 = {
    "brave", "loyal", "faithful", "generous", "honest", "innocent", "gracious",
    "gentle", "wise", "sacrifice", "protect", "defend", "goodness", "purity",
    "mercy", "kindness", "courage", "virtue", "integrity", "devoted",
    "trustworthy", "upright",
}

# Tier 1 (weight 2.0): unambiguous evil
EVIL_T1 = {
    "wicked", "malicious", "treacherous", "villainous", "nefarious", "diabolical",
    "insidious", "sadistic", "monstrous", "abominable", "heinous", "atrocious",
    "despicable", "malevolent", "fiendish", "murderous", "sinister",
}
# Tier 2 (weight 1.0): clear evil signal
EVIL_T2 = {
    "evil", "cruel", "vile", "corrupt", "deceit", "deceive", "betrayal", "betray",
    "treachery", "torture", "tyranny", "tyrant", "ruthless", "merciless",
    "cowardly", "manipulation", "malice", "vicious", "savage", "heartless",
    "hatred", "spite", "greed", "cowardice", "slaughter", "torment",
}

# Atmospheric/environmental words removed from scoring — they cause massive false
# positives in fantasy prose. "darkness fell", "shadow crossed", "under dread of"
# describe settings, not character morality. Same for honor/duty in dialogue.
# These sets are kept for documentation only.
_ATMOSPHERIC_EVIL = {"darkness", "shadow", "doom", "dread", "curse", "ruin"}
_DIALOGUE_GOOD = {"honor", "honour", "sworn", "duty", "truth"}

INTENSIFIERS = {"very", "so", "truly", "deeply", "utterly", "completely", "absolutely", "most", "greatly", "purely"}
NEGATORS = {"not", "never", "no", "without", "nor", "hardly", "scarcely", "barely", "nothing"}

# ─────────────────────────────────────────────
# 2. Text Processing
# ─────────────────────────────────────────────

def load_and_clean(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove chapter headers (all-caps lines)
    text = re.sub(r'\n[A-Z\s]{5,}\n', '\n', text)
    return text

def split_sentences(text):
    """Simple regex sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]

def score_sentence(sentence):
    """
    Score a sentence on the good/evil axis.
    Returns a float: positive = good, negative = evil.

    Improvements over v1:
    - Tiered lexicon weights (T1=2.0, T2=1.0, context=0.5)
    - Window-based negation (±3 tokens)
    - Intensifier boosting (×1.5)
    - Sarcasm/irony guard: possessive constructions like "spoke of X" or 
      "claimed X" preceding a moral word reduce its weight
    """
    words = re.findall(r'\b\w+\b', sentence.lower())
    score = 0.0

    # Sarcasm/distancing verbs — moral words after these get halved
    DISTANCING = {"claimed", "spoke", "pretended", "feigned", "appeared", "seemed",
                  "professed", "supposed", "rumored", "alleged", "called"}

    for i, word in enumerate(words):
        tier = 0
        is_good = False
        if word in GOOD_T1:
            tier = 2.0; is_good = True
        elif word in GOOD_T2:
            tier = 1.0; is_good = True
        elif word in EVIL_T1:
            tier = 2.0; is_good = False
        elif word in EVIL_T2:
            tier = 1.0; is_good = False
        else:
            continue

        weight = tier

        # Intensifier check in window [-2, i)
        for j in range(max(0, i - 2), i):
            if words[j] in INTENSIFIERS:
                weight *= 1.5
                break

        # Negation check in window [-3, i)
        negated = False
        for j in range(max(0, i - 3), i):
            if words[j] in NEGATORS:
                negated = True
                break

        # Distancing verb check in window [-4, i)
        for j in range(max(0, i - 4), i):
            if words[j] in DISTANCING:
                weight *= 0.5
                break

        if is_good:
            score += (-weight if negated else weight)
        else:
            score += (weight if negated else -weight)

    return score

# ─────────────────────────────────────────────
# 3. Character Extraction
# ─────────────────────────────────────────────

def extract_character_sentences(sentences, characters):
    """
    For each character, find sentences that are *about* them specifically.

    v2 improvements over v1:
    - Only score the sentence containing the name mention itself (no window)
    - ALSO score the immediately following sentence IF it starts with a pronoun
      (he/she/they/his/her/their/it) — this catches pronoun continuations
    - Eliminates window contamination where nearby unrelated sentences
      skewed scores heavily

    Returns dict: char_name -> list of sentences
    """
    char_sentences = defaultdict(list)

    # Build alias -> canonical name lookup
    alias_map = {}
    for char, (aliases, _) in characters.items():
        for alias in aliases:
            alias_map[alias.lower()] = char

    PRONOUNS = re.compile(r'^(he|she|they|his|her|their|it|him|the\s+\w+)\b', re.IGNORECASE)

    n = len(sentences)
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        matched_chars = set()
        for alias, char in alias_map.items():
            if re.search(r'\b' + re.escape(alias.lower()) + r'\b', sent_lower):
                matched_chars.add(char)

        if matched_chars:
            for char in matched_chars:
                char_sentences[char].append(sent)
                # Add pronoun-continuation sentence
                if i + 1 < n and PRONOUNS.match(sentences[i + 1]):
                    char_sentences[char].append(sentences[i + 1])

    return char_sentences

# ─────────────────────────────────────────────
# 4. Alignment Scoring
# ─────────────────────────────────────────────

def compute_alignment(char_sentences):
    """
    Returns dict: char -> {
        'mean_score': float,       # core good/evil score
        'std_score': float,        # moral consistency (low std = flat character)
        'n_sentences': int,
        'positive_ratio': float,   # fraction of sentences with positive moral score
    }
    """
    results = {}
    for char, sents in char_sentences.items():
        if len(sents) < 10:
            continue
        scores = [score_sentence(s) for s in sents]
        scores = [s for s in scores if s != 0]  # filter neutral sentences
        if len(scores) < 3:
            scores_all = [score_sentence(s) for s in sents]
            scores = scores_all
        scores = np.array(scores)
        results[char] = {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "n_sentences": len(sents),
            "positive_ratio": float(np.mean(scores > 0)),
            "raw_scores": scores.tolist()
        }
    return results

def normalize_scores(results_by_book):
    """Normalize mean scores to [-1, 1] range across all books."""
    all_means = [r["mean_score"] for book_r in results_by_book.values() 
                 for r in book_r.values()]
    max_abs = max(abs(m) for m in all_means) if all_means else 1.0
    if max_abs == 0:
        max_abs = 1.0
    for book_r in results_by_book.values():
        for r in book_r.values():
            r["normalized_score"] = r["mean_score"] / max_abs
    return results_by_book

# ─────────────────────────────────────────────
# 5. Book-Level Metrics
# ─────────────────────────────────────────────

def compute_book_metrics(book_results, characters):
    """
    Compute aggregate metrics per book:
    - moral_polarization: avg |score| (high = black/white)
    - moral_ambiguity: avg std (high = complex characters)  
    - alignment_accuracy: how well our scores match canonical alignments
    - grey_character_ratio: fraction of chars scoring near 0
    """
    if not book_results:
        return {}
    
    scores = [r["normalized_score"] for r in book_results.values() if "normalized_score" in r]
    stds = [r["std_score"] for r in book_results.values()]
    
    # Canonical alignment comparison
    canonical_matches = 0
    canonical_total = 0
    for char, r in book_results.items():
        if char in characters and "normalized_score" in r:
            _, canon = characters[char]
            pred = np.sign(r["normalized_score"]) if abs(r["normalized_score"]) > 0.05 else 0
            if canon == pred or canon == 0:  # ambiguous chars don't count against
                canonical_matches += 1
            canonical_total += 1
    
    grey_ratio = sum(1 for s in scores if abs(s) < 0.15) / len(scores) if scores else 0
    
    return {
        "moral_polarization": float(np.mean([abs(s) for s in scores])) if scores else 0,
        "moral_ambiguity": float(np.mean(stds)) if stds else 0,
        "alignment_accuracy": canonical_matches / canonical_total if canonical_total else 0,
        "grey_character_ratio": grey_ratio,
        "n_characters_analyzed": len(book_results),
    }

# ─────────────────────────────────────────────
# 6. Run Full Pipeline
# ─────────────────────────────────────────────

print("=" * 60)
print("FANTASY MORALITY ANALYSIS PIPELINE")
print("=" * 60)

all_results = {}    # book_title -> char -> alignment dict
book_metrics = {}   # book_title -> aggregate metrics
all_char_sents = {} # book_title -> char -> raw sentences (for lawful proxy)

for title, meta in BOOKS.items():
    print(f"\n[{meta['year']}] Processing: {title}...")
    
    text = load_and_clean(meta["path"])
    sentences = split_sentences(text)
    print(f"  Sentences extracted: {len(sentences)}")
    
    char_sents = extract_character_sentences(sentences, meta["characters"])
    print(f"  Characters matched: {list(char_sents.keys())}")
    
    char_alignment = compute_alignment(char_sents)
    all_results[title] = char_alignment
    all_char_sents[title] = char_sents
    
    metrics = compute_book_metrics(char_alignment, meta["characters"])
    metrics["year"] = meta["year"]
    metrics["author"] = meta["author"]
    book_metrics[title] = metrics
    
    print(f"  Characters scored: {len(char_alignment)}")
    for char, r in sorted(char_alignment.items(), key=lambda x: x[1]["mean_score"], reverse=True):
        print(f"    {char:20s} mean={r['mean_score']:+.3f}  std={r['std_score']:.3f}  n={r['n_sentences']}")

# Normalize
all_results = normalize_scores(all_results)

# Re-compute metrics with normalized scores
for title, meta in BOOKS.items():
    if title in all_results:
        book_metrics[title].update(
            compute_book_metrics(all_results[title], meta["characters"])
        )
        book_metrics[title]["year"] = meta["year"]
        book_metrics[title]["author"] = meta["author"]

print("\n\nBOOK-LEVEL METRICS:")
print("-" * 60)
for title, m in sorted(book_metrics.items(), key=lambda x: x[1].get("year", 0)):
    print(f"{title} ({m.get('year', '?')})")
    print(f"  Polarization:  {m.get('moral_polarization', 0):.3f}  (↑ = black/white morality)")
    print(f"  Ambiguity:     {m.get('moral_ambiguity', 0):.3f}  (↑ = complex characters)")
    print(f"  Grey chars:    {m.get('grey_character_ratio', 0):.1%}")

# ─────────────────────────────────────────────
# 7. Visualizations
# ─────────────────────────────────────────────

PALETTE = {
    "Tolkien":   "#2E6B3E",
    "Brooks":    "#5B8C3A",
    "Hobb":      "#C47D2E",
    "Martin":    "#8B1A1A",
    "Jordan":    "#4A6FA5",
    "Sanderson": "#7B4F9E",
}

# Aggregate by author for color coding
def get_color(title):
    author = BOOKS[title]["author"]
    return PALETTE.get(author, "#888888")

fig_dir = "/home/claude"

# ──────────────────────────────────────────────────────
# Plot 1: Character Alignment Scatter (all books combined)
# ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.patch.set_facecolor('#1a1a2e')
axes = axes.flatten()

sorted_books = sorted(BOOKS.keys(), key=lambda t: BOOKS[t]["year"])

for ax_i, title in enumerate(sorted_books):
    ax = axes[ax_i]
    ax.set_facecolor('#16213e')
    meta = BOOKS[title]
    results = all_results.get(title, {})
    
    chars_with_data = {c: r for c, r in results.items() if "normalized_score" in r}
    
    if not chars_with_data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', color='white')
        continue
    
    for char, r in chars_with_data.items():
        x = r["normalized_score"]
        y = r["std_score"]
        canon_align = meta["characters"].get(char, ([], 0))[1]
        
        color = "#4CAF50" if canon_align == 1 else ("#F44336" if canon_align == -1 else "#FFC107")
        
        ax.scatter(x, y, s=120, color=color, edgecolors='white', linewidth=0.8, zorder=5)
        ax.annotate(char, (x, y), textcoords="offset points", xytext=(5, 4),
                    fontsize=7.5, color='white', fontweight='bold')
    
    # Dividing line
    ax.axvline(0, color='#aaaaaa', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("← Evil | Good →", color='#cccccc', fontsize=8)
    ax.set_ylabel("Moral Complexity (std)", color='#cccccc', fontsize=8)
    ax.set_title(f"{title}\n({meta['year']}, {meta['author']})", 
                 color='white', fontsize=9, fontweight='bold')
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#4CAF50', edgecolor='white', label='Canonical Good'),
    mpatches.Patch(facecolor='#F44336', edgecolor='white', label='Canonical Evil'),
    mpatches.Patch(facecolor='#FFC107', edgecolor='white', label='Canonical Ambiguous'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
           facecolor='#1a1a2e', edgecolor='#aaaaaa', labelcolor='white',
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle("Character Moral Alignment Across Fantasy Novels\n"
             "X-axis: Moral Sentiment Score  |  Y-axis: Score Variance (moral complexity)",
             color='white', fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"plot1_character_alignment.png", dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print("\n[Saved] plot1_character_alignment.png")

# ──────────────────────────────────────────────────────
# Plot 2: Moral Polarization Over Time (the thesis chart)
# ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#1a1a2e')

for ax in [ax1, ax2]:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#aaaaaa', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

years = []
polarizations = []
ambiguities = []
grey_ratios = []
titles_sorted = []
colors_sorted = []

for title in sorted(book_metrics.keys(), key=lambda t: book_metrics[t].get("year", 0)):
    m = book_metrics[title]
    if "year" not in m:
        continue
    years.append(m["year"])
    polarizations.append(m.get("moral_polarization", 0))
    ambiguities.append(m.get("moral_ambiguity", 0))
    grey_ratios.append(m.get("grey_character_ratio", 0))
    titles_sorted.append(title)
    colors_sorted.append(get_color(title))

# Deduplicate years (Tolkien trilogy)
# For plotting, group Tolkien as single point
year_arr = np.array(years)
pol_arr = np.array(polarizations)
amb_arr = np.array(ambiguities)

# Plot 2a: Polarization over time
ax1.plot(year_arr, pol_arr, color='#888888', linewidth=1, linestyle='--', alpha=0.5, zorder=1)
for i, (yr, pol, title) in enumerate(zip(years, polarizations, titles_sorted)):
    short = BOOKS[title]["author"] + "\n" + str(yr)
    ax1.scatter(yr, pol, s=160, color=colors_sorted[i], edgecolors='white', linewidth=1.2, zorder=5)
    ax1.annotate(short, (yr, pol), textcoords="offset points", xytext=(5, 6),
                 fontsize=8, color='white')

# Trend line
if len(year_arr) > 2:
    z = np.polyfit(year_arr, pol_arr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(year_arr.min(), year_arr.max(), 100)
    ax1.plot(x_line, p(x_line), color='#FF6B6B', linewidth=2, linestyle='-', alpha=0.7, label='Trend')

ax1.set_xlabel("Publication Year", color='#cccccc', fontsize=10)
ax1.set_ylabel("Moral Polarization Score", color='#cccccc', fontsize=10)
ax1.set_title("Moral Polarization Over Time\n(↑ = stronger good/evil binary)", 
              color='white', fontsize=11, fontweight='bold')
ax1.set_facecolor('#16213e')

# Plot 2b: Ambiguity (character complexity) over time
ax2.plot(year_arr, amb_arr, color='#888888', linewidth=1, linestyle='--', alpha=0.5, zorder=1)
for i, (yr, amb, title) in enumerate(zip(years, ambiguities, titles_sorted)):
    short = BOOKS[title]["author"] + "\n" + str(yr)
    ax2.scatter(yr, amb, s=160, color=colors_sorted[i], edgecolors='white', linewidth=1.2, zorder=5)
    ax2.annotate(short, (yr, amb), textcoords="offset points", xytext=(5, 6),
                 fontsize=8, color='white')

if len(year_arr) > 2:
    z2 = np.polyfit(year_arr, amb_arr, 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_line, p2(x_line), color='#4FC3F7', linewidth=2, linestyle='-', alpha=0.7, label='Trend')

ax2.set_xlabel("Publication Year", color='#cccccc', fontsize=10)
ax2.set_ylabel("Avg Character Score Variance", color='#cccccc', fontsize=10)
ax2.set_title("Moral Complexity Over Time\n(↑ = more morally ambiguous characters)", 
              color='white', fontsize=11, fontweight='bold')

fig.suptitle("The Evolution of Morality in Post-Tolkien Fantasy",
             color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"plot2_morality_timeline.png", dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print("[Saved] plot2_morality_timeline.png")

# ──────────────────────────────────────────────────────
# Plot 3: D&D-Style Alignment Grid for selected characters
# ──────────────────────────────────────────────────────

# Proxy "lawful/chaotic" from sentence structure regularity
# (avg sentence length variance near char mentions — ordered prose = lawful)
def compute_lawful_score(char_sents_list):
    """Proxy for lawful/chaotic: chars described in long, complex sentences = more 'lawful'"""
    if not char_sents_list:
        return 0.0
    lengths = [len(s.split()) for s in char_sents_list]
    # Normalize: longer avg = more "deliberate/lawful", higher variance = more "chaotic"
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    # Combine: high mean + low std = lawful; high std = chaotic
    lawful_proxy = (mean_len / 30.0) - (std_len / 40.0)
    return np.clip(lawful_proxy, -1, 1)

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

# Draw alignment grid
for x_div in [-0.33, 0.33]:
    ax.axvline(x_div, color='#444466', linewidth=1.5, alpha=0.8)
for y_div in [-0.33, 0.33]:
    ax.axhline(y_div, color='#444466', linewidth=1.5, alpha=0.8)

# Label grid cells
cell_labels = [
    (-0.66, 0.66, "Lawful Good"), (0, 0.66, "Neutral Good"), (0.66, 0.66, "Chaotic Good"),
    (-0.66, 0, "Lawful Neutral"), (0, 0, "True Neutral"), (0.66, 0, "Chaotic Neutral"),
    (-0.66, -0.66, "Lawful Evil"), (0, -0.66, "Neutral Evil"), (0.66, -0.66, "Chaotic Evil"),
]
for cx, cy, label in cell_labels:
    ax.text(cx, cy, label, ha='center', va='center', color='#555577', fontsize=9,
            fontstyle='italic', alpha=0.7)

# Plot characters — sample key ones across all books
plotted = set()
for title, meta in sorted(BOOKS.items(), key=lambda x: x[1]["year"]):
    results = all_results.get(title, {})
    char_sents_raw = all_char_sents.get(title, {})
    
    for char, r in results.items():
        if "normalized_score" not in r:
            continue
        uid = f"{char}_{title[:6]}"
        if uid in plotted:
            continue
        plotted.add(uid)
        
        good_evil = r["normalized_score"]
        lawful_chaotic = compute_lawful_score(char_sents_raw.get(char, []))
        lawful_chaotic = lawful_chaotic * 1.5  # scale up for visibility
        lawful_chaotic = np.clip(lawful_chaotic, -0.95, 0.95)
        
        canon_align = meta["characters"].get(char, ([], 0))[1]
        color = get_color(title)
        marker = "o" if canon_align == 1 else ("s" if canon_align == -1 else "D")
        
        ax.scatter(lawful_chaotic, good_evil, s=140, color=color, 
                   marker=marker, edgecolors='white', linewidth=0.8, zorder=5)
        
        short_title = meta["author"][:3].upper()
        ax.annotate(f"{char}\n({short_title})", (lawful_chaotic, good_evil),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color='white', alpha=0.9)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axvline(0, color='#888888', linewidth=0.8, alpha=0.4)
ax.axhline(0, color='#888888', linewidth=0.8, alpha=0.4)
ax.set_xlabel("← Lawful | Chaotic →", color='#cccccc', fontsize=11)
ax.set_ylabel("← Evil | Good →", color='#cccccc', fontsize=11)
ax.set_title("D&D Alignment Grid — Characters Across Fantasy Novels\n"
             "Good/Evil axis from moral lexicon scoring; Lawful/Chaotic from prose structure proxy",
             color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='#aaaaaa')
for spine in ax.spines.values():
    spine.set_edgecolor('#444466')

# Author legend
legend_patches = [mpatches.Patch(facecolor=c, edgecolor='white', label=a) 
                  for a, c in PALETTE.items()]
fig.legend(handles=legend_patches, loc='lower right', fontsize=9,
           facecolor='#1a1a2e', edgecolor='#aaaaaa', labelcolor='white',
           title='Author', title_fontsize=9, bbox_to_anchor=(0.99, 0.02))

plt.tight_layout()
plt.savefig(f"plot3_dnd_alignment_grid.png", dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print("[Saved] plot3_dnd_alignment_grid.png")

# ──────────────────────────────────────────────────────
# Plot 4: Heatmap — Moral Scores by Character across Books
# ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

# Build a flat character×book matrix of scores
all_chars_set = set()
for title, results in all_results.items():
    for char in results:
        if "normalized_score" in results[char]:
            all_chars_set.add(char)

# Keep characters that appear in at least 1 book with enough data
col_labels = sorted(BOOKS.keys(), key=lambda t: BOOKS[t]["year"])
row_labels = sorted(all_chars_set)

matrix = np.full((len(row_labels), len(col_labels)), np.nan)
for j, title in enumerate(col_labels):
    for i, char in enumerate(row_labels):
        if char in all_results.get(title, {}) and "normalized_score" in all_results[title][char]:
            matrix[i, j] = all_results[title][char]["normalized_score"]

# Short col labels
short_cols = [f"{BOOKS[t]['author']}\n{BOOKS[t]['year']}" for t in col_labels]

# Custom diverging colormap: red=evil, white=neutral, green=good
cmap = sns.diverging_palette(10, 133, as_cmap=True)

mask = np.isnan(matrix)
sns.heatmap(matrix, ax=ax, xticklabels=short_cols, yticklabels=row_labels,
            cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 7.5, "color": "white"},
            mask=mask, linewidths=0.4, linecolor='#2a2a4a',
            cbar_kws={"label": "Moral Score (← Evil | Good →)", "shrink": 0.7})

ax.set_title("Moral Alignment Heatmap — All Characters × All Books\n(NaN = character not in book)",
             color='white', fontsize=12, fontweight='bold', pad=15)
ax.tick_params(colors='white', labelsize=8)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=15, ha='left')
plt.yticks(rotation=0)

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_color('white')
cbar.ax.tick_params(colors='white')

plt.tight_layout()
plt.savefig(f"plot4_moral_heatmap.png", dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print("[Saved] plot4_moral_heatmap.png")

# ──────────────────────────────────────────────────────
# Plot 5: Grey Character Ratio Bar Chart
# ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

sorted_titles_yr = sorted(book_metrics.keys(), key=lambda t: book_metrics[t].get("year", 0))
grey_vals = [book_metrics[t].get("grey_character_ratio", 0) for t in sorted_titles_yr]
bar_labels = [f"{BOOKS[t]['author']}\n({BOOKS[t]['year']})" for t in sorted_titles_yr]
bar_colors = [get_color(t) for t in sorted_titles_yr]

bars = ax.bar(bar_labels, grey_vals, color=bar_colors, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, grey_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.0%}", ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

ax.set_ylabel("Fraction of Characters Scoring Near Neutral", color='#cccccc', fontsize=10)
ax.set_title("'Grey Character' Ratio by Book\n"
             "(Characters with morally ambiguous scores — higher = more nuanced morality)",
             color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='#aaaaaa', labelsize=9)
ax.set_ylim(0, 1.1)
for spine in ax.spines.values():
    spine.set_edgecolor('#444466')
ax.axhline(0.5, color='#FF6B6B', linestyle='--', alpha=0.5, linewidth=1.2, label='50% threshold')
ax.legend(facecolor='#1a1a2e', edgecolor='#aaaaaa', labelcolor='white')

plt.tight_layout()
plt.savefig(f"plot5_grey_character_ratio.png", dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print("[Saved] plot5_grey_character_ratio.png")

# ──────────────────────────────────────────────────────
# Save results JSON for thesis appendix
# ──────────────────────────────────────────────────────
output_data = {
    "book_metrics": book_metrics,
    "character_scores": {
        title: {
            char: {k: v for k, v in r.items() if k != "raw_scores"}
            for char, r in results.items()
        }
        for title, results in all_results.items()
    }
}

with open(f"morality_results.json", "w") as f:
    json.dump(output_data, f, indent=2)
print("[Saved] morality_results.json")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print("\nKey Thesis Findings:")
for title in sorted(book_metrics.keys(), key=lambda t: book_metrics[t].get("year", 0)):
    m = book_metrics[title]
    print(f"  {title[:35]:35s} ({m.get('year','?')})  "
          f"Polarization={m.get('moral_polarization',0):.3f}  "
          f"GreyChars={m.get('grey_character_ratio',0):.0%}")
