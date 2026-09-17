from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from spam_detection import clean_text, infer_label


def test_clean_text_replaces_url_and_email():
    text = "Contact test@example.com and visit https://example.com now."
    cleaned = clean_text(text)

    assert "EMAIL" in cleaned
    assert "URL" in cleaned
    assert "https://example.com" not in cleaned


def test_infer_label_for_spam_path():
    assert infer_label(Path("data/spam/message001")) == 1


def test_infer_label_for_ham_path():
    assert infer_label(Path("data/easy_ham/message001")) == 0
