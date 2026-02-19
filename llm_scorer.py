"""
LLM-Augmented Moral Scoring Module
====================================
Drop-in replacement for the lexicon scorer that uses Claude to re-score
sentences that the lexicon already flagged as morally relevant.

Strategy (hybrid pipeline):
  1. Lexicon does a fast first pass over all sentences
  2. Only sentences that triggered at least one lexicon hit are sent to Claude
  3. Claude returns a score in [-2, 2] with reasoning
  4. Claude's score replaces the lexicon score for that sentence
  5. Sentences with zero lexicon hits keep score = 0.0 (neutral)

This keeps API costs manageable:
  - Full corpus: ~120,000 sentences
  - Lexicon-flagged sentences (estimated ~8-15%): ~10,000-18,000 sentences
  - With batching (10 sentences per API call): ~1,000-1,800 calls
  - At ~$0.003 per call (claude-haiku-4-5): ~$3-5 total for the full corpus
  - For a single book (~1,500 flagged sentences): ~$0.50

Usage:
    from llm_scorer import LLMScorer
    scorer = LLMScorer(api_key="your-key-here")  # or set ANTHROPIC_API_KEY env var
    
    # Score a batch of sentences for a specific character in a specific book
    scores = scorer.score_sentences(
        sentences=["Gandalf spoke with great wisdom.", "Sauron laughed at their suffering."],
        character="Gandalf",
        book_title="Fellowship of the Ring"
    )
    # Returns: [1.5, -1.8]  (floats in [-2, 2])

    # Full drop-in: re-score all lexicon-flagged sentences for a book
    updated_char_scores = scorer.rescore_book(
        char_sentences=char_sents,       # dict from extract_character_sentences()
        book_title="Fellowship of the Ring",
        characters=BOOKS["Fellowship of the Ring"]["characters"]
    )
"""

import os
import re
import time
import json
from collections import defaultdict

try:
    import anthropic
except ImportError:
    raise ImportError(
        "anthropic package not found. Run: pip install anthropic"
    )

# ── Lexicon (copied from pipeline so this module is self-contained) ──────────
GOOD_T1 = {
    "noble","righteous","virtuous","compassionate","selfless","benevolent",
    "merciful","courageous","valiant","valor","valour","heroic","gallant",
    "magnanimous","incorruptible","saintly","blameless","steadfast",
    "honorable","honourable",
}
GOOD_T2 = {
    "brave","loyal","faithful","generous","honest","innocent","gracious",
    "gentle","wise","sacrifice","protect","defend","goodness","purity",
    "mercy","kindness","courage","virtue","integrity","devoted",
    "trustworthy","upright",
}
EVIL_T1 = {
    "wicked","malicious","treacherous","villainous","nefarious","diabolical",
    "insidious","sadistic","monstrous","abominable","heinous","atrocious",
    "despicable","malevolent","fiendish","murderous","sinister",
}
EVIL_T2 = {
    "evil","cruel","vile","corrupt","deceit","deceive","betrayal","betray",
    "treachery","torture","tyranny","tyrant","ruthless","merciless",
    "cowardly","manipulation","malice","vicious","savage","heartless",
    "hatred","spite","greed","cowardice","slaughter","torment",
}
ALL_MORAL = GOOD_T1 | GOOD_T2 | EVIL_T1 | EVIL_T2


def has_lexicon_hit(sentence: str) -> bool:
    """Return True if this sentence contains any lexicon word."""
    words = re.findall(r'\b\w+\b', sentence.lower())
    return any(w in ALL_MORAL for w in words)


# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a literary analysis assistant helping score sentences from fantasy novels on a moral alignment axis for a computational linguistics thesis.

You will receive a batch of sentences from a fantasy novel. For each sentence, score it on the good/evil moral axis from the perspective of how it portrays the named character.

SCORING SCALE:
 2.0 = Strongly good (clear heroic act, explicit virtue, self-sacrifice)
 1.0 = Mildly good (positive framing, kind action, honourable behaviour)
 0.0 = Neutral (no moral content, purely descriptive, ambiguous)
-1.0 = Mildly evil (selfish action, deceptive behaviour, cruelty implied)
-2.0 = Strongly evil (explicit atrocity, sadism, irredeemable act)

CRITICAL RULES:
- Detect sarcasm: "He spoke of his great honour" where context implies hypocrisy → score negatively
- Detect irony: narrator describing villain's "noble" deed sarcastically → score negatively  
- Detect reported speech: "He claimed to be virtuous" → reduce score, don't take at face value
- Detect atmospheric use: "darkness fell around them" describes setting, NOT character morality → 0.0
- Focus on what the sentence reveals about the CHARACTER, not the scene
- If the sentence is about a DIFFERENT character than the one named, score 0.0

Respond ONLY with a JSON array of numbers, one per sentence, in the same order.
Example response for 3 sentences: [-1.5, 0.0, 2.0]
No explanation, no markdown, just the JSON array."""


def build_user_prompt(sentences: list[str], character: str, book_title: str) -> str:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    return f"""Book: {book_title}
Character being analyzed: {character}

Sentences to score (score each for how it portrays {character}):
{numbered}

Respond with a JSON array of {len(sentences)} numbers."""


# ── Main Scorer Class ─────────────────────────────────────────────────────────

class LLMScorer:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        api_key     : Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        model       : Model to use. Haiku is recommended for cost efficiency.
                      Use claude-sonnet-4-6 for higher accuracy on ambiguous cases.
        batch_size  : Sentences per API call. 10 is a good balance of cost vs latency.
        max_retries : Retries on API error before giving up and using lexicon score.
        retry_delay : Seconds to wait between retries.
        verbose     : Print progress.
        """
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Pass api_key= or set ANTHROPIC_API_KEY env var."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verbose = verbose

        # Stats tracking
        self.total_api_calls = 0
        self.total_sentences_scored = 0
        self.fallback_count = 0  # times we fell back to lexicon due to API error

    def _call_api(self, sentences: list[str], character: str, book_title: str) -> list[float] | None:
        """Make one API call. Returns list of floats or None on failure."""
        prompt = build_user_prompt(sentences, character, book_title)

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw = response.content[0].text.strip()

                # Parse JSON array from response
                # Strip any accidental markdown fences
                raw = re.sub(r'```json|```', '', raw).strip()
                scores = json.loads(raw)

                if not isinstance(scores, list) or len(scores) != len(sentences):
                    raise ValueError(f"Expected {len(sentences)} scores, got: {raw}")

                # Clamp to [-2, 2]
                scores = [max(-2.0, min(2.0, float(s))) for s in scores]
                self.total_api_calls += 1
                self.total_sentences_scored += len(sentences)
                return scores

            except Exception as e:
                if attempt < self.max_retries - 1:
                    if self.verbose:
                        print(f"    [API retry {attempt+1}/{self.max_retries}] {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    if self.verbose:
                        print(f"    [API FAILED after {self.max_retries} attempts] {e}")
                    return None

    def score_sentences(
        self,
        sentences: list[str],
        character: str,
        book_title: str,
        lexicon_fallback_scores: list[float] | None = None,
    ) -> list[float]:
        """
        Score a list of sentences for a given character.

        Only sends sentences with lexicon hits to the API.
        Sentences without hits return 0.0.
        On API failure, falls back to lexicon_fallback_scores if provided, else 0.0.

        Returns list of floats in [-2, 2], same length as input.
        """
        n = len(sentences)
        final_scores = [0.0] * n

        # Find which sentences have lexicon hits — only these go to the API
        flagged_indices = [i for i, s in enumerate(sentences) if has_lexicon_hit(s)]

        if not flagged_indices:
            return final_scores

        flagged_sentences = [sentences[i] for i in flagged_indices]

        # Batch the flagged sentences
        all_llm_scores = []
        for batch_start in range(0, len(flagged_sentences), self.batch_size):
            batch = flagged_sentences[batch_start:batch_start + self.batch_size]
            llm_scores = self._call_api(batch, character, book_title)

            if llm_scores is None:
                # Fallback: use lexicon scores or 0.0
                self.fallback_count += len(batch)
                if lexicon_fallback_scores:
                    fallback = [lexicon_fallback_scores[flagged_indices[batch_start + j]]
                                for j in range(len(batch))]
                else:
                    fallback = [0.0] * len(batch)
                all_llm_scores.extend(fallback)
            else:
                all_llm_scores.extend(llm_scores)

        # Map scores back to original positions
        for idx, score in zip(flagged_indices, all_llm_scores):
            final_scores[idx] = score

        return final_scores

    def rescore_book(
        self,
        char_sentences: dict[str, list[str]],
        book_title: str,
        characters: dict,
    ) -> dict[str, dict]:
        """
        Re-score all characters in a book using LLM scoring.

        Parameters
        ----------
        char_sentences : Output of extract_character_sentences() — dict of char -> sentences
        book_title     : Used in the prompt for context
        characters     : The character config dict from BOOKS (for canonical labels)

        Returns
        -------
        Dict of char -> {mean_score, std_score, n_sentences, positive_ratio, method}
        """
        import numpy as np
        results = {}

        for char, sentences in char_sentences.items():
            if len(sentences) < 10:
                continue

            if self.verbose:
                flagged = sum(1 for s in sentences if has_lexicon_hit(s))
                print(f"  Scoring {char}: {len(sentences)} sentences, "
                      f"{flagged} flagged ({flagged/len(sentences):.0%}) → API")

            llm_scores = self.score_sentences(sentences, char, book_title)

            # Filter to non-zero scores for aggregation
            # (zero = sentence had no moral content, shouldn't drag mean toward 0)
            nonzero = [s for s in llm_scores if s != 0.0]
            if len(nonzero) < 3:
                nonzero = llm_scores  # fallback if everything scored 0

            arr = np.array(nonzero)
            results[char] = {
                "mean_score": float(np.mean(arr)),
                "std_score": float(np.std(arr)),
                "n_sentences": len(sentences),
                "n_scored": len(nonzero),
                "positive_ratio": float(np.mean(arr > 0)),
                "method": "llm",
            }

        return results

    def print_stats(self):
        print(f"\n[LLM Scorer Stats]")
        print(f"  Total API calls:       {self.total_api_calls}")
        print(f"  Total sentences sent:  {self.total_sentences_scored}")
        print(f"  Fallback to lexicon:   {self.fallback_count} sentences")
        estimated_cost = self.total_api_calls * 0.003
        print(f"  Estimated cost:        ~${estimated_cost:.2f} (haiku rates)")


# ── Comparison Utility ────────────────────────────────────────────────────────

def compare_methods(lexicon_results: dict, llm_results: dict, book_title: str):
    """
    Print a side-by-side comparison of lexicon vs LLM scores.
    Useful for the thesis methodology section.
    """
    print(f"\n{'='*65}")
    print(f"Lexicon vs LLM Score Comparison: {book_title}")
    print(f"{'='*65}")
    print(f"{'Character':<20} {'Lexicon':>10} {'LLM':>10} {'Delta':>10} {'Agrees':>8}")
    print(f"{'-'*65}")

    chars = set(lexicon_results) | set(llm_results)
    for char in sorted(chars):
        lex = lexicon_results.get(char, {}).get("normalized_score", None)
        llm = llm_results.get(char, {}).get("mean_score", None)
        if lex is None or llm is None:
            continue
        delta = llm - lex
        # "Agrees" = both point the same direction (or both near zero)
        agrees = (lex * llm > 0) or (abs(lex) < 0.1 and abs(llm) < 0.1)
        print(f"{char:<20} {lex:>+10.3f} {llm:>+10.3f} {delta:>+10.3f} {'✓' if agrees else '✗':>8}")


# ── Standalone Demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick demo — score a handful of tricky sentences that the lexicon gets wrong.
    Run with: python llm_scorer.py
    Requires ANTHROPIC_API_KEY environment variable.
    """

    DEMO_SENTENCES = [
        # Sarcasm — lexicon would score positive (mentions "honour")
        ("Littlefinger", "A Game of Thrones",
         "Littlefinger smiled and spoke at length about his deep and abiding honour."),

        # Reported evil — lexicon would miss this entirely (no lexicon words)
        ("Joffrey", "A Game of Thrones",
         "He gestured at the man's hands and Ser Ilyn stepped forward."),

        # Atmospheric darkness — lexicon (old version) would score negatively
        ("Sam", "Fellowship of the Ring",
         "Sam peered into the darkness, his heart beating fast."),

        # Genuine virtue — should score positively
        ("Faramir", "The Two Towers",
         "I would not take this thing, if it lay by the highway."),

        # Ambiguous — character doing evil for "good" reasons
        ("Gollum", "The Two Towers",
         "He had promised to help master, and he would keep his promise."),

        # Ironic praise — villain described with heroic language sarcastically
        ("Regal", "Assassin's Apprentice",
         "Prince Regal had, as always, the most noble of intentions toward his family."),
    ]

    print("LLM Scorer Demo — Testing Tricky Cases")
    print("=" * 50)

    scorer = LLMScorer(verbose=True)

    for char, book, sent in DEMO_SENTENCES:
        scores = scorer.score_sentences([sent], char, book)
        print(f"\nCharacter: {char} ({book})")
        print(f"Sentence:  {sent}")
        print(f"Score:     {scores[0]:+.1f}  {'← GOOD' if scores[0] > 0.5 else ('← EVIL' if scores[0] < -0.5 else '← NEUTRAL/AMBIGUOUS')}")

    scorer.print_stats()
