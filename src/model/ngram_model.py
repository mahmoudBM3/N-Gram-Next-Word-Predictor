"""N-gram model building, persistence, and backoff lookup."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class NGramModel:
    """Build, store, and query n-gram probability tables across multiple orders."""

    def __init__(self, ngram_order: int, unk_threshold: int) -> None:
        """Initialize model configuration and state.

        Args:
            ngram_order: Maximum n-gram order.
            unk_threshold: Minimum word frequency to keep in vocabulary.

        Returns:
            None.
        """
        self.ngram_order = ngram_order
        self.unk_threshold = unk_threshold
        self.vocab = set()
        self.model: Dict[str, Dict] = {}

    def _read_token_file(self, token_file: str) -> List[List[str]]:
        """Read tokenized sentences from file.

        Args:
            token_file: Path to sentence-per-line token file.

        Returns:
            Nested list where each inner list is a sentence token sequence.
        """
        sentences: List[List[str]] = []
        with open(token_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sentences.append(line.split())
        return sentences

    def build_vocab(self, token_file: str) -> None:
        """Build vocabulary from token file and map low-frequency words to <UNK>.

        Args:
            token_file: Path to normalized token file.

        Returns:
            None.
        """
        sentences = self._read_token_file(token_file)
        counts = Counter(token for sentence in sentences for token in sentence)
        self.vocab = {word for word, count in counts.items() if count >= self.unk_threshold}
        self.vocab.add("<UNK>")

    def build_counts_and_probabilities(self, token_file: str) -> None:
        """Build counts then convert to MLE probabilities for 1..NGRAM_ORDER.

        Args:
            token_file: Path to normalized token file.

        Returns:
            None.
        """
        sentences = self._read_token_file(token_file)
        mapped_sentences: List[List[str]] = []
        for sentence in sentences:
            mapped_sentences.append([
                token if token in self.vocab else "<UNK>" for token in sentence
            ])

        unigram_counts = Counter(token for sentence in mapped_sentences for token in sentence)
        total_tokens = sum(unigram_counts.values())

        model: Dict[str, Dict] = {}
        model["1gram"] = {
            token: count / total_tokens for token, count in unigram_counts.items() if total_tokens
        }

        for order in range(2, self.ngram_order + 1):
            context_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
            for sentence in mapped_sentences:
                if len(sentence) < order:
                    continue
                for idx in range(len(sentence) - order + 1):
                    ngram = sentence[idx : idx + order]
                    context = tuple(ngram[:-1])
                    next_word = ngram[-1]
                    context_counts[context][next_word] += 1

            probs_for_order: Dict[str, Dict[str, float]] = {}
            for context, next_word_counter in context_counts.items():
                denom = sum(next_word_counter.values())
                if denom == 0:
                    continue
                context_key = " ".join(context)
                probs_for_order[context_key] = {
                    word: count / denom for word, count in next_word_counter.items()
                }
            model[f"{order}gram"] = probs_for_order

        self.model = model

    def lookup(self, context: List[str]) -> Dict[str, float]:
        """Backoff lookup from highest order down to unigram.

        Args:
            context: Context word list (typically last NGRAM_ORDER-1 words).

        Returns:
            Dict of candidate next words to probabilities from first matched order.
        """
        safe_context = [word if word in self.vocab else "<UNK>" for word in context]

        for order in range(self.ngram_order, 0, -1):
            key = f"{order}gram"
            if key not in self.model:
                continue

            if order == 1:
                unigram = self.model.get("1gram", {})
                return dict(unigram) if isinstance(unigram, dict) else {}

            needed = order - 1
            if len(safe_context) < needed:
                continue

            context_key = " ".join(safe_context[-needed:])
            candidates = self.model[key].get(context_key, {})
            if candidates:
                return dict(candidates)

        return {}

    def save_model(self, model_path: str) -> None:
        """Save model probability tables to JSON.

        Args:
            model_path: Output JSON path for model probabilities.

        Returns:
            None.
        """
        with open(model_path, "w", encoding="utf-8") as handle:
            json.dump(self.model, handle, ensure_ascii=True, indent=2)

    def save_vocab(self, vocab_path: str) -> None:
        """Save vocabulary list to JSON.

        Args:
            vocab_path: Output JSON path for vocabulary.

        Returns:
            None.
        """
        with open(vocab_path, "w", encoding="utf-8") as handle:
            json.dump(sorted(self.vocab), handle, ensure_ascii=True, indent=2)

    def load(self, model_path: str, vocab_path: str) -> None:
        """Load model and vocabulary from JSON files.

        Args:
            model_path: Input model JSON path.
            vocab_path: Input vocab JSON path.

        Returns:
            None.
        """
        with open(model_path, "r", encoding="utf-8") as handle:
            self.model = json.load(handle)
        with open(vocab_path, "r", encoding="utf-8") as handle:
            self.vocab = set(json.load(handle))


def main() -> None:
    """Run a tiny smoke test for module-local debugging."""
    model = NGramModel(ngram_order=3, unk_threshold=1)
    print(f"Model initialized with order={model.ngram_order}")


if __name__ == "__main__":
    main()
