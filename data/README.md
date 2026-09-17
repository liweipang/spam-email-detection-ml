# Dataset Directory

Place the extracted SpamAssassin Public Corpus folders here.

Example:

```text
data/
├── easy_ham/
├── hard_ham/
├── spam/
└── spam_2/
```

The dataset is intentionally excluded from version control. The training script infers labels from folder names containing `ham` or `spam`.
