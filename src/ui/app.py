"""Optional Streamlit UI wrapper for the predictor (extra credit)."""

from __future__ import annotations

import importlib
from typing import List


class PredictorUI:
    """Expose a minimal UI interface around a Predictor instance."""

    def __init__(self, predictor) -> None:
        """Store predictor dependency.

        Args:
            predictor: Predictor-like object with predict_next(text, k).

        Returns:
            None.
        """
        self.predictor = predictor

    def get_predictions(self, text: str, k: int = 3) -> List[str]:
        """Get predictions from predictor while handling empty input.

        Args:
            text: Input phrase.
            k: Number of suggestions.

        Returns:
            List of predicted tokens.
        """
        if not text or not text.strip():
            return []
        return self.predictor.predict_next(text, k)

    def run(self) -> None:
        """Run the UI app.

        Args:
            None.

        Returns:
            None.
        """
        try:
            st = importlib.import_module("streamlit")
        except ImportError as exc:
            raise RuntimeError("streamlit is not installed. Install requirements first.") from exc

        st.set_page_config(page_title="N-Gram Predictor", page_icon="NG")
        st.title("N-Gram Next-Word Predictor")
        text = st.text_input("Enter text")
        k = st.number_input("Top-k", min_value=1, max_value=20, value=3)
        if st.button("Predict"):
            st.write(self.get_predictions(text, int(k)))


def main() -> None:
    """Standalone placeholder entry point for UI module."""
    print("Run Streamlit with: streamlit run src/ui/app.py")


if __name__ == "__main__":
    main()
