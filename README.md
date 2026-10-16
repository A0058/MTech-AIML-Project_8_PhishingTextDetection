# Phishing Text Detection Under Modified Attacks

## Problem
Phishing emails can change their wording to evade text classifiers. This project compares model performance on original test messages and synthetically modified versions.

## Dataset
Put the complete `phishing_email.csv` file in `data/`. The expected columns are:
- `text_combined`: email text
- `label`: binary target

The code assumes `1 = phishing` and `0 = legitimate`. Confirm the label mapping against the dataset's source documentation before interpreting results.

## Installation
```bash
pip install -r requirements.txt
```

## Run
Open `notebooks/experiment.ipynb` in Jupyter and run cells from top to bottom. Or, from the project root:
```bash
python src/evaluation.py
```

The embedding-based model may download `all-MiniLM-L6-v2` the first time and therefore may need internet access.

## Methods
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Sentence Transformer embeddings + Logistic Regression

## Outputs
- `results/results.csv`: original/modified metrics
- `results/comparison_table.csv`: Macro-F1 performance drop
- `results/predictions.csv`: predictions for individual examples
- `figures/`: metric plots and confusion matrices

## Limitations
Modified messages are created with simple lexical substitutions. This is a synthetic baseline, not a validated adversarial benchmark. Inspect modified examples and avoid claiming that every edit preserves the original intent.
