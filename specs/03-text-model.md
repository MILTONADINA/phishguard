# 03 — Stage 1: text model (fine-tuned DistilBERT)

Module: `src/phishguard/text_model.py`. Shared types live in `config.py` (`TextTrainConfig`, `TrainResult`).

Pretrained model: `distilbert-base-uncased` (Hugging Face). Sequence classification head with 2 labels; **index 1 = phishing**.

Why DistilBERT (report material): about 40% smaller and 60% faster than BERT-base while keeping most of its accuracy, so it fine-tunes on a laptop GPU (Apple MPS) in hours, not days, and answers fast enough for a live demo.

## Requirements

### TEXT-01 `select_device(prefer=None) -> torch.device`
- `prefer` in `{"cpu", "cuda", "mps", None}`.
- If `prefer` is available, use it. If it isn't, fall back without raising (a warning is fine).
- With `prefer=None`, order is `cuda` → `mps` → `cpu`.
- `select_device("cpu")` is always `torch.device("cpu")`.

### TEXT-02 `PhishTextClassifier`
```python
PhishTextClassifier(model_name_or_path, max_length=256, device=None, batch_size=32)
.predict_proba(texts: Sequence[str]) -> np.ndarray   # shape (len(texts),), P(phishing)
.save(path) -> None                                  # model + tokenizer via save_pretrained
PhishTextClassifier.load(path, **kwargs) -> PhishTextClassifier
attributes: .model, .tokenizer, .device, .max_length
```
- Loads with `AutoTokenizer` and `AutoModelForSequenceClassification` (`num_labels=2`). Works offline from a local directory.
- Inputs are truncated to `max_length` tokens, padded per batch, and processed in batches of `batch_size`.
- `predict_proba` runs in eval mode with no gradients. Output = softmax column 1, values in `[0, 1]`, dtype float.
- Empty input → empty array of shape `(0,)`.
- Deterministic: the same inputs twice give identical outputs.
- `save` then `load` gives identical predictions.

### TEXT-03 `train_text_model(train_texts, train_labels, val_texts, val_labels, config) -> TrainResult`
`config: TextTrainConfig` fields:
- `model_name="distilbert-base-uncased"`, `max_length=256`
- `batch_size=16`, `eval_batch_size=64`, `epochs=2`
- `lr=2e-5`, `weight_decay=0.01`, `warmup_ratio=0.1`
- `patience=1`, `seed=42`
- `output_dir="models/text"`, `device=None`, `max_train_samples=None`

Behaviour:
1. `set_seed(config.seed)`. If `max_train_samples` is set, use a stratified random subset of that size.
2. Hand-written PyTorch loop: AdamW, linear warmup over `warmup_ratio` of the total steps, then linear decay; cross-entropy loss; shuffle each epoch.
3. After each epoch, compute the validation F1 (threshold 0.5) and append to `history`:
   `{"epoch": int, "train_loss": float, "val_f1": float}`
4. When `val_f1` improves on the best so far, save the model with `PhishTextClassifier.save(output_dir)`.
5. Early stopping: stop when `val_f1` hasn't improved for `patience` epochs.
6. Return `TrainResult(history, best_epoch, best_val_f1, output_dir)`.

`output_dir` must then load with `PhishTextClassifier.load`.

### TEXT-04 Training learns
On a tiny, clearly separable dataset with a high learning rate, the last epoch's `train_loss` is lower than the first epoch's.

### TEXT-05 `TrainResult` sanity
`best_epoch` is 1-based and is the epoch with the highest `val_f1` in `history`. `best_val_f1 == max(h["val_f1"] for h in history)`. `len(history) <= epochs`.

## Practical notes (not tested)
- Use `max_train_samples` (e.g. 20 000) for a quick first run. A full run comes after.
- If one epoch on MPS takes more than about 2 h, lower `max_length` to 128 or use Colab.
- Log each epoch to the console. Copy the final history into the report.
