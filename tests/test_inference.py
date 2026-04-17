from pathlib import Path

from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel


def _build_model(tmp_path: Path) -> NGramModel:
    token_file = tmp_path / "tokens.txt"
    token_file.write_text(
        "holmes looked at watson\n"
        "holmes looked at the door\n"
        "the game is afoot\n",
        encoding="utf-8",
    )
    model = NGramModel(ngram_order=3, unk_threshold=1)
    model.build_vocab(str(token_file))
    model.build_counts_and_probabilities(str(token_file))
    return model


def test_predict_next_returns_k(tmp_path):
    model = _build_model(tmp_path)
    predictor = Predictor(model=model, normalizer=Normalizer(), ngram_order=3)
    preds = predictor.predict_next("holmes looked", 2)
    assert len(preds) == 2


def test_predict_next_sorted_by_probability(tmp_path):
    model = _build_model(tmp_path)
    predictor = Predictor(model=model, normalizer=Normalizer(), ngram_order=3)
    context = predictor.map_oov(predictor.normalize("holmes looked"))
    probs = model.lookup(context)
    expected = [w for w, _ in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]]
    result = predictor.predict_next("holmes looked", 2)
    assert result[: len(expected)] == expected


def test_predict_next_oov_context_does_not_crash(tmp_path):
    model = _build_model(tmp_path)
    predictor = Predictor(model=model, normalizer=Normalizer(), ngram_order=3)
    preds = predictor.predict_next("zzz qqq", 3)
    assert isinstance(preds, list)


def test_map_oov_replaces_unknown(tmp_path):
    model = _build_model(tmp_path)
    predictor = Predictor(model=model, normalizer=Normalizer(), ngram_order=3)
    mapped = predictor.map_oov(["holmes", "unknown_token"])
    assert mapped[0] == "holmes"
    assert mapped[1] == "<UNK>"
