# Spam Email Detection using Machine Learning

A Python-based spam email classification project built with the public SpamAssassin dataset and Scikit-learn.

The project uses a reproducible text-classification pipeline combining **HashingVectorizer**, **TF-IDF weighting**, and a **Decision Tree classifier**. It also evaluates model performance with accuracy, a classification report, and a confusion matrix.

## Tech Stack

- Python
- Scikit-learn
- HashingVectorizer
- TF-IDF
- Decision Tree
- Matplotlib
- Pytest

## Project Workflow

1. Load ham and spam messages from the SpamAssassin dataset.
2. Extract subject/body text from email files.
3. Apply lightweight text cleaning and normalization.
4. Convert text into hashed n-gram features.
5. Apply TF-IDF weighting.
6. Train a Decision Tree classifier.
7. Evaluate predictions using accuracy and a confusion matrix.

## Repository Structure

```text
spam-email-detection-ml/
├── src/
│   └── spam_detection.py
├── tests/
│   └── test_text_utils.py
├── data/
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

This repository is designed to work with the public **SpamAssassin Public Corpus**.

Download/extract the corpus locally and place ham/spam folders under `data/`. A typical layout is:

```text
data/
├── easy_ham/
├── hard_ham/
├── spam/
└── spam_2/
```

The dataset itself is not committed to this repository.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Model

```bash
python src/spam_detection.py --data-dir data
```

The script prints the number of loaded messages, class distribution, accuracy, and classification report. It also saves:

```text
results/confusion_matrix.png
```

## Testing

```bash
pytest
```

## Model Pipeline

```text
Email text
   ↓
HashingVectorizer
   ↓
TF-IDF Transformer
   ↓
DecisionTreeClassifier
   ↓
Ham / Spam prediction
```

## Notes

This repository contains a clean, reproducible implementation of the spam-detection project for portfolio and academic demonstration purposes. Model results depend on the exact SpamAssassin corpus folders and train/test split used locally.
