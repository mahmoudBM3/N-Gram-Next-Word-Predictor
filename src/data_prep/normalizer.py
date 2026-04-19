"""Data preparation utilities for corpus loading, normalization, and tokenization."""

from __future__ import annotations

import os
import re
from typing import List


class Normalizer:
    """Load, clean, tokenize, and save text corpora for model training and inference."""

    def load(self, folder_path: str) -> str:
        """Load all .txt files from a folder and concatenate them into one string.

        Args:
            folder_path: Directory containing raw text files.

        Returns:
            A single concatenated text string from all .txt files.
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Folder not found: {folder_path}. Check TRAIN_RAW_DIR in config/.env."
            )

        txt_files = [
            os.path.join(folder_path, name)
            for name in sorted(os.listdir(folder_path))
            if name.lower().endswith(".txt")
        ]
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in: {folder_path}")

        chunks: List[str] = []
        for path in txt_files:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                chunks.append(handle.read())

        return "\n\n".join(chunks)

    def strip_gutenberg(self, text: str) -> str:
        """Remove Project Gutenberg header and footer sections.

        Args:
            text: Raw text potentially containing Gutenberg markers.

        Returns:
            Text with header/footer removed when markers are present.
        """
        start_match = re.search(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.IGNORECASE | re.DOTALL)
        end_match = re.search(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.IGNORECASE | re.DOTALL)

        start_idx = start_match.end() if start_match else 0
        end_idx = end_match.start() if end_match else len(text)
        return text[start_idx:end_idx]

    def lowercase(self, text: str) -> str:
        """Convert input text to lowercase.

        Args:
            text: Input text.

        Returns:
            Lowercased text.
        """
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation characters while preserving letters, digits, and whitespace.

        Args:
            text: Input text.

        Returns:
            Text with punctuation replaced by spaces.
        """
        return re.sub(r"[^\w\s]", " ", text)

    def remove_numbers(self, text: str) -> str:
        """Remove numeric characters.

        Args:
            text: Input text.

        Returns:
            Text with numbers removed.
        """
        return re.sub(r"\d+", " ", text)

    def remove_whitespace(self, text: str) -> str:
        """Collapse repeated whitespace and blank lines into single spaces.

        Args:
            text: Input text.

        Returns:
            Whitespace-normalized text.
        """
        return " ".join(text.split())

    def normalize(self, text: str) -> str:
        """Apply full normalization pipeline in the required order.

        Args:
            text: Input text.

        Returns:
            Fully normalized text.
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text: str) -> List[str]:
        """Split raw text into sentence-like chunks before final normalization.

        Args:
            text: Raw text.

        Returns:
            List of sentence strings.
        """
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        parts = re.split(r"[.!?]+|\n+", cleaned)
        return [part.strip() for part in parts if part and part.strip()]

    def word_tokenize(self, sentence: str) -> List[str]:
        """Split one normalized sentence into word tokens.

        Args:
            sentence: Sentence text.

        Returns:
            List of non-empty tokens.
        """
        return [token for token in sentence.split(" ") if token]

    def save(self, sentences: List[str], filepath: str) -> None:
        """Write tokenized sentences to a file, one sentence per line.

        Args:
            sentences: Sentences already tokenized and joined by spaces.
            filepath: Output path.

        Returns:
            None.
        """
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as handle:
            for line in sentences:
                handle.write(f"{line}\n")


def main() -> None:
    """Run a tiny smoke test for module-local debugging."""
    sample = "Hello, Holmes! Case #221B started in 1887."
    normalizer = Normalizer()
    print(normalizer.normalize(sample))


if __name__ == "__main__":
    main()
