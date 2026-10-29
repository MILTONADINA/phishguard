# 02 — Baseline (TF-IDF + logistic regression)

Module: `src/phishguard/baseline.py`

Purpose: a strong, cheap, classical model. The report compares DistilBERT against it (AC-02) to show what fine-tuning a pretrained model actually gained.

## Requirements

### BASE-01 `build_baseline(max_features=50_000, ngram_range=(1, 2), C=1.0, seed=42) -> sklearn.pipeline.Pipeline`
Returns an **unfitted** two-step `Pipeline`:
1. step `"tfidf"`: `TfidfVectorizer(lowercase=True, sublinear_tf=True, max_features=max_features, ngram_range=ngram_range)`
2. step `"clf"`: `LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)`

`train_baseline(texts, labels, **kwargs) -> Pipeline` builds it with `kwargs` and fits it.

### BASE-02 `baseline_predict_proba(model, texts) -> np.ndarray`
Returns a 1-D float array, one `P(phishing)` per text (the class-1 column of `predict_proba`), each value in `[0, 1]`. Empty `texts` → empty array of shape `(0,)`. Trained on clearly separable data, it ranks phishing texts above legitimate ones.

### BASE-03 `save_baseline(model, path)` / `load_baseline(path) -> Pipeline`
Uses `joblib`. Creates parent folders. A loaded model gives predictions identical to the original's.
