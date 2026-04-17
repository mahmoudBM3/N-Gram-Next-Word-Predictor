"""Inference module for next-word prediction from a preloaded n-gram model."""

from __future__ import annotations

from typing import List

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel


class Predictor:
    """Normalize input, map OOV words, and return top-k next-word predictions."""

    def __init__(self, model: NGramModel, normalizer: Normalizer, ngram_order: int) -> None:
        """Store injected dependencies and settings.

        Args:
            model: Preloaded n-gram model instance.
            normalizer: Shared normalizer instance.
            ngram_order: Maximum n-gram order used to derive context length.

        Returns:
            None.
        """
        self.model = model
        self.normalizer_obj = normalizer
        self.ngram_order = ngram_order

    def normalize(self, text: str) -> List[str]:
        """Normalize user text and extract the last n-1 words as context.

        Args:
            text: Raw user input string.

        Returns:
            Context words list.
        """
        normalized = self.normalizer_obj.normalize(text)
        tokens = [token for token in normalized.split() if token]
        context_len = max(self.ngram_order - 1, 0)
        return tokens[-context_len:] if context_len > 0 else []

    def map_oov(self, context: List[str]) -> List[str]:
        """Map out-of-vocabulary tokens to <UNK>.

        Args:
            context: Input context tokens.

        Returns:
            Context with OOV terms replaced.
        """
        return [token if token in self.model.vocab else "<UNK>" for token in context]

    def predict_next(self, text: str, k: int) -> List[str]:
        """Predict top-k candidate next words.

        Args:
            text: Raw user input.
            k: Number of predictions to return.

        Returns:
            Top-k words sorted by probability descending.
        """
        if not text or not text.strip():
            return []

        context = self.normalize(text)
        safe_context = self.map_oov(context)
        probs = self.model.lookup(safe_context)
        if not probs:
            return []

        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        predictions = [word for word, _ in ranked[:k]]

        if len(predictions) < k:
            unigram_probs = self.model.model.get("1gram", {})
            unigram_ranked = sorted(
                unigram_probs.items(), key=lambda item: item[1], reverse=True
            )
            for word, _ in unigram_ranked:
                if word not in predictions:
                    predictions.append(word)
                if len(predictions) == k:
                    break

        return predictions[:k]


def main() -> None:
    """Run a tiny smoke test for module-local debugging."""
    print("Predictor module ready.")


if __name__ == "__main__":
    main()
