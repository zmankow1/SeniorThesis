"""
morality_pipeline_llm.py
=========================
Full pipeline with LLM-augmented scoring.

This is a wrapper around the original morality_pipeline.py that:
  1. Runs the lexicon pipeline as normal (fast baseline)
  2. Re-scores all flagged sentences using Claude (LLM pass)
  3. Produces a side-by-side comparison of both methods
  4. Generates plots labelled with both methods

Usage:
    # Set your API key first:
    export ANTHROPIC_API_KEY=xx

    # Run full corpus (estimated $3-6, ~20-40 min):
    python morality_pipeline_llm.py

    # Run single book only (estimated $0.50, ~3 min):
    python morality_pipeline_llm.py --book "Fellowship of the Ring"

    # Use Sonnet instead of Haiku for higher accuracy:
    python morality_pipeline_llm.py --model claude-sonnet-4-6
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import everything from the original pipeline
# (make sure morality_pipeline.py is in the same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morality_pipeline import (
    BOOKS, load_and_clean, split_sentences,
    extract_character_sentences, compute_alignment,
    normalize_scores, compute_book_metrics,
    PALETTE, get_color
)
from llm_scorer import LLMScorer, compare_methods

# ── CLI Arguments ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Fantasy Morality Pipeline with LLM Scoring")
parser.add_argument("--book", type=str, default=None,
                    help="Process only this book (exact title). Default: all books.")
parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
                    help="Claude model to use for LLM scoring.")
parser.add_argument("--batch-size", type=int, default=10,
                    help="Sentences per API call.")
parser.add_argument("--api-key", type=str, default=None,
                    help="Anthropic API key (default: ANTHROPIC_API_KEY env var).")
parser.add_argument("--output-dir", type=str, default="./llm_output",
                    help="Directory to save results and plots.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Select books to process ───────────────────────────────────────────────────

if args.book:
    books_to_process = {k: v for k, v in BOOKS.items() if k == args.book}
    if not books_to_process:
        print(f"Book '{args.book}' not found. Available:")
        for t in BOOKS:
            print(f"  {t}")
        sys.exit(1)
else:
    books_to_process = BOOKS

# ── Initialise LLM Scorer ─────────────────────────────────────────────────────

scorer = LLMScorer(
    api_key=args.api_key,
    model=args.model,
    batch_size=args.batch_size,
    verbose=True,
)

# ── Run Pipeline ──────────────────────────────────────────────────────────────

print("=" * 65)
print("FANTASY MORALITY PIPELINE — LLM-AUGMENTED")
print(f"Model: {args.model}")
print("=" * 65)

lexicon_results_all = {}
llm_results_all = {}
book_metrics_lexicon = {}
book_metrics_llm = {}

for title, meta in books_to_process.items():
    print(f"\n[{meta['year']}] {title}")
    print("-" * 50)

    # ── Lexicon pass (same as original pipeline) ──────────────────
    text = load_and_clean(meta["path"])
    sentences = split_sentences(text)
    char_sents = extract_character_sentences(sentences, meta["characters"])
    lexicon_results = compute_alignment(char_sents)
    lexicon_results_all[title] = lexicon_results

    print(f"  Lexicon scores:")
    for char, r in sorted(lexicon_results.items(), key=lambda x: x[1]["mean_score"], reverse=True):
        print(f"    {char:<20} {r['mean_score']:+.3f}")

    # ── LLM pass ──────────────────────────────────────────────────
    print(f"\n  LLM re-scoring:")
    llm_results = scorer.rescore_book(char_sents, title, meta["characters"])
    llm_results_all[title] = llm_results

    print(f"  LLM scores:")
    for char, r in sorted(llm_results.items(), key=lambda x: x[1]["mean_score"], reverse=True):
        print(f"    {char:<20} {r['mean_score']:+.3f}  (n_scored={r['n_scored']})")

    # ── Comparison ────────────────────────────────────────────────
    # Normalize lexicon scores for fair comparison
    temp = {title: lexicon_results}
    temp = normalize_scores(temp)
    compare_methods(temp[title], llm_results, title)

    # ── Book metrics ──────────────────────────────────────────────
    book_metrics_lexicon[title] = compute_book_metrics(lexicon_results, meta["characters"])
    book_metrics_lexicon[title]["year"] = meta["year"]
    book_metrics_lexicon[title]["author"] = meta["author"]

    book_metrics_llm[title] = compute_book_metrics(llm_results, meta["characters"])
    book_metrics_llm[title]["year"] = meta["year"]
    book_metrics_llm[title]["author"] = meta["author"]

scorer.print_stats()

# ── Save JSON results ─────────────────────────────────────────────────────────

output = {
    "model_used": args.model,
    "lexicon_results": {
        t: {c: {k: v for k, v in r.items() if k != "raw_scores"}
            for c, r in res.items()}
        for t, res in lexicon_results_all.items()
    },
    "llm_results": {
        t: {c: r for c, r in res.items()}
        for t, res in llm_results_all.items()
    },
    "book_metrics_lexicon": book_metrics_lexicon,
    "book_metrics_llm": book_metrics_llm,
}

with open(f"{args.output_dir}/llm_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\n[Saved] {args.output_dir}/llm_results.json")

# ── Plot: Lexicon vs LLM Agreement ────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.patch.set_facecolor('#1a1a2e')
axes = axes.flatten()

sorted_books = sorted(books_to_process.keys(), key=lambda t: BOOKS[t]["year"])

for ax_i, title in enumerate(sorted_books[:8]):
    ax = axes[ax_i]
    ax.set_facecolor('#16213e')
    meta = BOOKS[title]

    lex_r = lexicon_results_all.get(title, {})
    llm_r = llm_results_all.get(title, {})

    # Normalize lexicon for comparison
    temp = {title: lex_r}
    temp = normalize_scores(temp)
    lex_norm = temp[title]

    chars = set(lex_norm) & set(llm_r)
    for char in chars:
        lex_score = lex_norm[char].get("normalized_score", 0)
        llm_score = llm_r[char].get("mean_score", 0) / 2.0  # LLM uses [-2,2], normalize to [-1,1]

        canon = meta["characters"].get(char, ([], 0))[1]
        color = "#4CAF50" if canon == 1 else ("#F44336" if canon == -1 else "#FFC107")

        ax.scatter(lex_score, llm_score, s=130, color=color,
                   edgecolors='white', linewidth=0.8, zorder=5)
        ax.annotate(char, (lex_score, llm_score), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5, color='white')

    # Perfect agreement line
    ax.plot([-1, 1], [-1, 1], color='#888888', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(0, color='#555577', linewidth=0.6, alpha=0.4)
    ax.axhline(0, color='#555577', linewidth=0.6, alpha=0.4)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Lexicon Score", color='#cccccc', fontsize=8)
    ax.set_ylabel("LLM Score", color='#cccccc', fontsize=8)
    ax.set_title(f"{title}\n({meta['year']})", color='white', fontsize=9, fontweight='bold')
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

legend_elements = [
    mpatches.Patch(facecolor='#4CAF50', edgecolor='white', label='Canonical Good'),
    mpatches.Patch(facecolor='#F44336', edgecolor='white', label='Canonical Evil'),
    mpatches.Patch(facecolor='#FFC107', edgecolor='white', label='Canonical Ambiguous'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
           facecolor='#1a1a2e', edgecolor='#aaaaaa', labelcolor='white',
           bbox_to_anchor=(0.5, 0.01))
fig.suptitle("Lexicon vs LLM Score Agreement per Character\n"
             "Points on the diagonal = perfect agreement; above = LLM scored higher than lexicon",
             color='white', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{args.output_dir}/plot_lexicon_vs_llm.png", dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f"[Saved] {args.output_dir}/plot_lexicon_vs_llm.png")

# ── Plot: Polarization Timeline — Both Methods ────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#1a1a2e')

for ax in [ax1, ax2]:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#aaaaaa', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

titles_yr = sorted(books_to_process.keys(), key=lambda t: BOOKS[t]["year"])
years = [BOOKS[t]["year"] for t in titles_yr]
pol_lex = [book_metrics_lexicon.get(t, {}).get("moral_polarization", 0) for t in titles_yr]
pol_llm = [book_metrics_llm.get(t, {}).get("moral_polarization", 0) for t in titles_yr]
colors = [get_color(t) for t in titles_yr]

for ax, pols, label, color in [
    (ax1, pol_lex, "Lexicon Method", "#FF6B6B"),
    (ax2, pol_llm, "LLM Method", "#4FC3F7"),
]:
    ax.plot(years, pols, color='#888888', linewidth=1, linestyle='--', alpha=0.4)
    for i, (yr, pol, t) in enumerate(zip(years, pols, titles_yr)):
        ax.scatter(yr, pol, s=160, color=colors[i], edgecolors='white', linewidth=1.2, zorder=5)
        ax.annotate(f"{BOOKS[t]['author']}\n{yr}", (yr, pol),
                    textcoords="offset points", xytext=(5, 6), fontsize=8, color='white')
    if len(years) > 2:
        z = np.polyfit(years, pols, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(years), max(years), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.7)
    ax.set_xlabel("Publication Year", color='#cccccc', fontsize=10)
    ax.set_ylabel("Moral Polarization", color='#cccccc', fontsize=10)
    ax.set_title(f"Polarization Over Time\n({label})", color='white', fontsize=11, fontweight='bold')

fig.suptitle("Moral Polarization Timeline: Lexicon vs LLM Method Comparison",
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{args.output_dir}/plot_timeline_comparison.png", dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f"[Saved] {args.output_dir}/plot_timeline_comparison.png")

print("\n" + "=" * 65)
print("PIPELINE COMPLETE")
print(f"Results saved to: {args.output_dir}/")
print("=" * 65)
