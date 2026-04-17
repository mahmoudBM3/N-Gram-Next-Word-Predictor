# N-Gram Next-Word Predictor

This project builds a statistical next-word prediction system using an n-gram language model trained on Sherlock Holmes novels from Project Gutenberg. The pipeline includes text preprocessing, vocabulary and probability model building, backoff-based inference, and a command-line interface for interactive prediction.

## Requirements

- Python 3.10+
- Anaconda (recommended)
- Install dependencies from `requirements.txt`

## Setup

1. Clone the repository.
2. Create and activate an Anaconda environment:

```bash
conda create -n ngram-predictor python=3.10 -y
conda activate ngram-predictor
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Fill `config/.env` with valid paths and runtime settings.
5. Download the training corpus `.txt` files into `data/raw/train/`:
	- The Adventures of Sherlock Holmes (1661)
	- The Memoirs of Sherlock Holmes (834)
	- The Return of Sherlock Holmes (108)
	- The Hound of the Baskervilles (2852)
6. Optional (extra credit evaluator): put The Valley of Fear (3289) into `data/raw/eval/`.

## Usage

Run each module from `main.py` with `--step`:

```bash
python main.py --step dataprep
python main.py --step model
python main.py --step inference
python main.py --step all
python main.py --step evaluate
```

### Notes

- `--step dataprep`: generates `data/processed/train_tokens.txt`
- `--step model`: generates `data/model/model.json` and `data/model/vocab.json`
- `--step inference`: starts interactive CLI loop
- `--step all`: runs dataprep then model then CLI
- `--step evaluate`: computes perplexity on `EVAL_TOKENS` (extra credit)

### Web GUI (Flask)

Start the web interface:

```bash
python app_web.py
```

Then open your browser to: **http://127.0.0.1:5000**

Type text and click "Predict" to see top-k word suggestions.

## Project Structure

```text
ngram-predictor/
├── config/
│   └── .env
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── eval/
│   ├── processed/
│   │   ├── train_tokens.txt
│   │   └── eval_tokens.txt
│   └── model/
│       ├── model.json
│       └── vocab.json
├── src/
│   ├── data_prep/
│   │   └── normalizer.py
│   ├── model/
│   │   └── ngram_model.py
│   ├── inference/
│   │   └── predictor.py
│   ├── ui/
│   │   └── app.py
│   └── evaluation/
│       └── evaluator.py
├── tests/
├── main.py
├── .gitignore
├── requirements.txt
└── readme.md
```
