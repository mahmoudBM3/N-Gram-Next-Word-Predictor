from pathlib import Path

from src.data_prep.normalizer import Normalizer
from src.evaluation.evaluator import Evaluator
from src.model.ngram_model import NGramModel


def _build_model(tmp_path: Path) -> NGramModel:
    token_file = tmp_path / "tokens.txt"
    token_file.write_text(
        "the game is afoot\n"
        "holmes looked at watson\n"
        "the game is on\n",
        encoding="utf-8",
    )
    model = NGramModel(ngram_order=3, unk_threshold=1)
    model.build_vocab(str(token_file))
    model.build_counts_and_probabilities(str(token_file))
    return model


def test_score_word_seen_returns_negative_float(tmp_path):
    model = _build_model(tmp_path)
    evaluator = Evaluator(model=model, normalizer=Normalizer(), ngram_order=3)
    score = evaluator.score_word("is", ["the", "game"])
    assert isinstance(score, float)
    assert score <= 0


def test_score_word_zero_probability_returns_none(tmp_path):
    model = _build_model(tmp_path)
    evaluator = Evaluator(model=model, normalizer=Normalizer(), ngram_order=3)
    assert evaluator.score_word("nonexistent", ["the", "game"]) is None


def test_compute_perplexity_positive(tmp_path):
    model = _build_model(tmp_path)
    evaluator = Evaluator(model=model, normalizer=Normalizer(), ngram_order=3)

    eval_file = tmp_path / "eval_tokens.txt"
    eval_file.write_text("the game is afoot\n", encoding="utf-8")

    perplexity = evaluator.compute_perplexity(str(eval_file))
    assert perplexity > 1
