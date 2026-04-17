"""Optional evaluation module for perplexity computation (extra credit)."""

from __future__ import annotations

import math
from typing import List, Optional


class Evaluator:
    """Compute held-out corpus perplexity using model lookup with backoff."""

    def __init__(self, model, normalizer, ngram_order: int) -> None:
        """Store dependencies and settings.

        Args:
            model: Loaded n-gram model with lookup(context).
            normalizer: Shared normalizer instance.
            ngram_order: Maximum n-gram order.

        Returns:
            None.
        """
        self.model = model
        self.normalizer = normalizer
        self.ngram_order = ngram_order

    def score_word(self, word: str, context: List[str]) -> Optional[float]:
        """Return log2 probability for a word given context.

        Args:
            word: Target token.
            context: Context tokens.

        Returns:
            log2 probability or None if unavailable.
        """
        candidates = self.model.lookup(context)
        prob = candidates.get(word)
        if prob is None or prob <= 0:
            return None
        return math.log2(prob)

    def compute_perplexity(self, eval_file: str) -> float:
        """Compute perplexity for an evaluation token file.

        Args:
            eval_file: Path to sentence-per-line tokenized evaluation file.

        Returns:
            Positive perplexity value.
        """
        words: List[str] = []
        with open(eval_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    words.extend(line.split())

        if not words:
            raise ValueError("Evaluation file has no tokens.")

        total_log_prob = 0.0
        evaluated = 0
        skipped = 0

        for idx, word in enumerate(words):
            start = max(0, idx - (self.ngram_order - 1))
            context = words[start:idx]
            safe_context = [w if w in self.model.vocab else "<UNK>" for w in context]
            target = word if word in self.model.vocab else "<UNK>"

            log_prob = self.score_word(target, safe_context)
            if log_prob is None:
                skipped += 1
                continue
            total_log_prob += log_prob
            evaluated += 1

        if evaluated == 0:
            raise ValueError("No evaluable words found for perplexity.")

        if skipped / len(words) > 0.2:
            print("Warning: more than 20% of words were skipped.")

        cross_entropy = -(total_log_prob / evaluated)
        return 2 ** cross_entropy

    def run(self, eval_file: str) -> None:
        """Run evaluation and print summary.

        Args:
            eval_file: Path to eval token file.

        Returns:
            None.
        """
        ppl = self.compute_perplexity(eval_file)
        print(f"Perplexity: {ppl:.2f}")


def main() -> None:
    """Standalone placeholder entry point for evaluation module."""
    print("Evaluator module ready.")


if __name__ == "__main__":
    main()
