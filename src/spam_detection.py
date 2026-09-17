"""Spam email detection with Scikit-learn.

This project uses the public SpamAssassin corpus. The pipeline combines
HashingVectorizer, TF-IDF weighting, and a DecisionTreeClassifier.
"""

from __future__ import annotations

import argparse
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier


HAM_LABEL = 0
SPAM_LABEL = 1


def extract_message_text(path: Path) -> str:
    """Extract readable text from an RFC-style email file."""
    try:
        with path.open("rb") as handle:
            message = BytesParser(policy=policy.default).parse(handle)

        subject = str(message.get("subject", ""))
        parts: list[str] = [subject]

        if message.is_multipart():
            for part in message.walk():
                if part.get_content_maintype() == "text" and part.get_content_type() == "text/plain":
                    try:
                        parts.append(part.get_content())
                    except Exception:
                        continue
        else:
            try:
                parts.append(message.get_content())
            except Exception:
                pass

        text = "\n".join(parts)
    except Exception:
        text = path.read_text(encoding="utf-8", errors="ignore")

    return clean_text(text)


def clean_text(text: str) -> str:
    """Apply lightweight normalization while preserving useful email tokens."""
    text = text.replace("\x00", " ")
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " EMAIL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_label(path: Path) -> int | None:
    """Infer ham/spam label from common SpamAssassin folder names."""
    tokens = [part.lower() for part in path.parts]

    if any("spam" in token for token in tokens):
        return SPAM_LABEL
    if any("ham" in token or "easy_ham" in token or "hard_ham" in token for token in tokens):
        return HAM_LABEL
    return None


def load_spamassassin_dataset(data_dir: Path) -> tuple[list[str], list[int]]:
    """Load SpamAssassin messages recursively from a local data directory."""
    texts: list[str] = []
    labels: list[int] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        label = infer_label(path)
        if label is None:
            continue

        text = extract_message_text(path)
        if text:
            texts.append(text)
            labels.append(label)

    if not texts:
        raise ValueError(
            "No labeled emails were found. Place SpamAssassin ham/spam folders "
            "inside the data directory."
        )

    return texts, labels


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create the end-to-end text classification pipeline."""
    return Pipeline(
        steps=[
            (
                "hashing",
                HashingVectorizer(
                    n_features=2**18,
                    alternate_sign=False,
                    norm=None,
                    ngram_range=(1, 2),
                    lowercase=True,
                    stop_words="english",
                ),
            ),
            ("tfidf", TfidfTransformer()),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=random_state,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def evaluate_model(
    pipeline: Pipeline,
    x_test: list[str],
    y_test: list[int],
    output_dir: Path,
) -> None:
    """Print evaluation metrics and save a confusion matrix image."""
    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["ham", "spam"],
            zero_division=0,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["ham", "spam"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title("Spam Email Detection - Confusion Matrix")
    plt.tight_layout()
    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a spam email classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing SpamAssassin ham/spam folders.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset used for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split and classifier.",
    )
    args = parser.parse_args()

    texts, labels = load_spamassassin_dataset(args.data_dir)
    print(f"Loaded {len(texts)} emails")
    print(f"Ham: {labels.count(HAM_LABEL)} | Spam: {labels.count(SPAM_LABEL)}")

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    pipeline = build_pipeline(args.random_state)
    pipeline.fit(x_train, y_train)
    evaluate_model(pipeline, x_test, y_test, Path("results"))


if __name__ == "__main__":
    main()
