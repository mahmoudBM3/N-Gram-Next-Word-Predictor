from src.data_prep.normalizer import Normalizer


def test_normalize_pipeline():
    normalizer = Normalizer()
    text = "  HELLO, Holmes 221B!  "
    assert normalizer.normalize(text) == "hello holmes b"


def test_strip_gutenberg_markers():
    normalizer = Normalizer()
    text = (
        "prefix *** START OF THE PROJECT GUTENBERG EBOOK DEMO *** "
        "Body Content" 
        " *** END OF THE PROJECT GUTENBERG EBOOK DEMO *** suffix"
    )
    stripped = normalizer.strip_gutenberg(text)
    assert "Body Content" in stripped
    assert "prefix" not in stripped
    assert "suffix" not in stripped


def test_sentence_and_word_tokenize():
    normalizer = Normalizer()
    sentences = normalizer.sentence_tokenize("One. Two! Three?")
    assert isinstance(sentences, list)
    assert len(sentences) >= 1

    tokens = normalizer.word_tokenize("holmes looked at watson")
    assert tokens == ["holmes", "looked", "at", "watson"]
