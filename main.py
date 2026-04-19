"""Single entry point for data prep, model training, inference, and evaluation."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.evaluation.evaluator import Evaluator
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel


def get_env(name: str) -> str:
    """Read an environment variable or raise an explicit KeyError.

    Args:
        name: Environment variable key.

    Returns:
        Variable value as string.
    """
    value = os.getenv(name)
    if value is None or value == "":
        raise KeyError(f"Missing config variable: {name}. Check config/.env.")
    return value


def setup_logging(level_name: str) -> None:
    """Configure logging based on LOG_LEVEL.

    Args:
        level_name: Logging level string.

    Returns:
        None.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a file path when it is missing.

    Args:
        path: File path whose parent directory should exist.

    Returns:
        None.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_dataprep(normalizer: Normalizer, train_raw_dir: str, train_tokens: str, dev_limit: int) -> None:
    """Run data preprocessing for training corpus.

    Args:
        normalizer: Shared Normalizer instance.
        train_raw_dir: Raw training text folder.
        train_tokens: Output token file path.
        dev_limit: Sentence limit for quick development runs; <=0 means full corpus.

    Returns:
        None.
    """
    if not train_raw_dir or not train_raw_dir.strip():
        raise ValueError("TRAIN_RAW_DIR is empty. Check config/.env")
    if not train_tokens or not train_tokens.strip():
        raise ValueError("TRAIN_TOKENS is empty. Check config/.env")

    logging.info("Data Prep: loading raw corpus from %s", train_raw_dir)
    raw = normalizer.load(train_raw_dir)
    logging.info("Data Prep: loaded %d raw characters", len(raw))
    
    # Skip Gutenberg stripping for concatenated multi-file corpora
    # The files are clean enough as-is
    stripped = raw
    logging.info("Data Prep: skipping Gutenberg strip (multi-file corpus)")
    
    sentences = normalizer.sentence_tokenize(stripped)
    logging.info("Data Prep: tokenized into %d sentences", len(sentences))

    if dev_limit > 0:
        sentences = sentences[:dev_limit]
        logging.info("Data Prep: DEV_SENTENCE_LIMIT applied (%d sentences)", dev_limit)

    token_lines: List[str] = []
    for idx, sentence in enumerate(sentences):
        if not sentence or not sentence.strip():
            continue
        normalized_sentence = normalizer.normalize(sentence)
        tokens = normalizer.word_tokenize(normalized_sentence)
        if tokens:
            token_lines.append(" ".join(tokens))

    normalizer.save(token_lines, train_tokens)
    logging.info("Data Prep: wrote %d lines to %s", len(token_lines), train_tokens)


def run_model(
    model: NGramModel,
    train_tokens: str,
    model_path: str,
    vocab_path: str,
) -> None:
    """Run vocabulary/model construction and persistence.

    Args:
        model: NGramModel instance.
        train_tokens: Input token file.
        model_path: Output model JSON path.
        vocab_path: Output vocab JSON path.

    Returns:
        None.
    """
    if not model_path or not model_path.strip():
        raise ValueError("MODEL path is empty. Check config/.env")
    if not vocab_path or not vocab_path.strip():
        raise ValueError("VOCAB path is empty. Check config/.env")

    logging.info("Model: building vocabulary from %s", train_tokens)
    model.build_vocab(train_tokens)
    logging.info("Model: vocabulary size = %d", len(model.vocab))

    logging.info("Model: building counts and probabilities")
    model.build_counts_and_probabilities(train_tokens)

    ensure_parent_dir(model_path)
    ensure_parent_dir(vocab_path)

    model.save_model(model_path)
    model.save_vocab(vocab_path)
    logging.info("Model: saved model to %s and vocab to %s", model_path, vocab_path)


def run_inference(model: NGramModel, predictor: Predictor, model_path: str, vocab_path: str, top_k: int) -> None:
    """Run interactive next-word prediction CLI loop.

    Args:
        model: NGramModel instance.
        predictor: Predictor instance.
        model_path: Model JSON path.
        vocab_path: Vocab JSON path.
        top_k: Number of predictions to return.

    Returns:
        None.
    """
    logging.info("Inference: loading model artifacts")
    model.load(model_path, vocab_path)

    print("Type text to get next-word predictions. Type 'quit' to exit.")
    while True:
        try:
            text = input("> ").strip()
            if text.lower() == "quit":
                print("Goodbye.")
                break

            predictions = predictor.predict_next(text, top_k)
            if not predictions:
                print("Predictions: []")
            else:
                print(f"Predictions: {predictions}")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break


def run_evaluation(
    evaluator: Evaluator,
    model: NGramModel,
    model_path: str,
    vocab_path: str,
    eval_tokens: str,
) -> None:
    """Run optional perplexity evaluation.

    Args:
        evaluator: Evaluator instance.
        model: NGramModel instance.
        model_path: Model JSON path.
        vocab_path: Vocab JSON path.
        eval_tokens: Evaluation token file path.

    Returns:
        None.
    """
    model.load(model_path, vocab_path)
    evaluator.run(eval_tokens)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        None.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="N-Gram Next-Word Predictor")
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "all", "evaluate"],
        required=True,
        help="Pipeline step to execute",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        raise SystemExit(0)

    return parser.parse_args()


def main() -> None:
    """Load config, wire dependencies, and execute requested pipeline step."""
    load_dotenv(os.path.join("config", ".env"), override=True)

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))

    args = parse_args()

    try:
        train_raw_dir = get_env("TRAIN_RAW_DIR")
        train_tokens = get_env("TRAIN_TOKENS")
        model_path = get_env("MODEL")
        vocab_path = get_env("VOCAB")
        eval_tokens = get_env("EVAL_TOKENS")

        unk_threshold = int(get_env("UNK_THRESHOLD"))
        top_k = int(get_env("TOP_K"))
        ngram_order = int(get_env("NGRAM_ORDER"))
        dev_limit = int(os.getenv("DEV_SENTENCE_LIMIT", "0"))
    except KeyError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    except ValueError as exc:
        print(f"ERROR: Invalid config value: {exc}")
        raise SystemExit(1) from exc

    # Ensure all configured output locations have parent directories.
    ensure_parent_dir(train_tokens)
    ensure_parent_dir(model_path)
    ensure_parent_dir(vocab_path)
    ensure_parent_dir(eval_tokens)

    normalizer = Normalizer()
    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold)
    predictor = Predictor(model=model, normalizer=normalizer, ngram_order=ngram_order)
    evaluator = Evaluator(model=model, normalizer=normalizer, ngram_order=ngram_order)

    if args.step == "dataprep":
        run_dataprep(normalizer, train_raw_dir, train_tokens, dev_limit)
    elif args.step == "model":
        run_model(model, train_tokens, model_path, vocab_path)
    elif args.step == "inference":
        run_inference(model, predictor, model_path, vocab_path, top_k)
    elif args.step == "evaluate":
        run_evaluation(evaluator, model, model_path, vocab_path, eval_tokens)
    elif args.step == "all":
        run_dataprep(normalizer, train_raw_dir, train_tokens, dev_limit)
        run_model(model, train_tokens, model_path, vocab_path)
        run_inference(model, predictor, model_path, vocab_path, top_k)


if __name__ == "__main__":
    main()
