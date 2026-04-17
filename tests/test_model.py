from pathlib import Path

from src.model.ngram_model import NGramModel


def _write_tokens(path: Path) -> None:
    path.write_text(
        "holmes looked at watson\n"
        "holmes looked at the door\n"
        "watson looked at holmes\n",
        encoding="utf-8",
    )


def test_build_vocab_and_unk(tmp_path):
    token_file = tmp_path / "tokens.txt"
    _write_tokens(token_file)

    model = NGramModel(ngram_order=3, unk_threshold=2)
    model.build_vocab(str(token_file))

    assert "<UNK>" in model.vocab
    assert "door" not in model.vocab


def test_lookup_seen_and_unseen_context(tmp_path):
    token_file = tmp_path / "tokens.txt"
    _write_tokens(token_file)

    model = NGramModel(ngram_order=3, unk_threshold=1)
    model.build_vocab(str(token_file))
    model.build_counts_and_probabilities(str(token_file))

    seen = model.lookup(["holmes", "looked"])
    assert isinstance(seen, dict)
    assert len(seen) > 0

    unseen = model.lookup(["zzz", "qqq"])
    assert isinstance(unseen, dict)
    assert len(unseen) > 0


def test_lookup_empty_when_no_model():
    model = NGramModel(ngram_order=3, unk_threshold=1)
    model.vocab = {"<UNK>"}
    model.model = {}
    assert model.lookup(["x", "y"]) == {}


def test_probabilities_sum_to_one_for_context(tmp_path):
    token_file = tmp_path / "tokens.txt"
    _write_tokens(token_file)

    model = NGramModel(ngram_order=3, unk_threshold=1)
    model.build_vocab(str(token_file))
    model.build_counts_and_probabilities(str(token_file))

    context_probs = model.lookup(["holmes", "looked"])
    total = sum(context_probs.values())
    assert 0.999 <= total <= 1.001
