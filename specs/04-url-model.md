# 04 — Stage 2: URL model (character-level CNN)

Module: `src/phishguard/url_model.py`. Shared types in `config.py` (`UrlTrainConfig`, `TrainResult`).

Why character-level (report material): phishing URLs hide in spelling: `paypa1.com`, `secure-login.bank.com.evil.io`, long random paths, IP addresses. A word tokenizer splits these unpredictably. A CNN over characters learns those patterns directly, and it's tiny and fast.

## Requirements

### URL-01 `CharVocab(max_len=200)`
- `PAD = 0`, `UNK = 1`. Printable ASCII characters `chr(32)` … `chr(126)` map to `ord(c) - 30`, giving ids `2 … 96`. `vocab_size == 97`.
- `encode(url) -> list[int]`: strips surrounding whitespace, maps each char (anything outside printable ASCII → `UNK`), keeps the **first** `max_len` characters, pads with `PAD` on the right. The result always has length `max_len`.
- `encode_batch(urls) -> torch.LongTensor` of shape `(len(urls), max_len)`.
- Case is preserved (paths are case-sensitive).

### URL-02 `CharCNN(vocab_size=97, embed_dim=32, num_filters=64, kernel_sizes=(3, 4, 5), dropout=0.3)`
- `nn.Embedding(vocab_size, embed_dim, padding_idx=0)` → one `Conv1d(embed_dim, num_filters, k)` per kernel size → ReLU → global max-pool over time → concatenate → dropout → `Linear(num_filters * len(kernel_sizes), 1)`.
- `forward(x: LongTensor[B, L]) -> FloatTensor[B]`: **logits**, no sigmoid.
- Gradients flow to the embedding weights.

### URL-03 `URLClassifier(model, vocab, device=None)`
- `predict_proba(urls) -> np.ndarray` of shape `(len(urls),)`: sigmoid of the logits, values in `[0, 1]`, eval mode, no gradients.
- Empty list → empty array of shape `(0,)`. Deterministic.

### URL-04 `train_url_model(train_urls, train_labels, val_urls, val_labels, config) -> TrainResult`
`config: UrlTrainConfig` fields:
- `max_len=200`, `embed_dim=32`, `num_filters=64`, `kernel_sizes=(3, 4, 5)`, `dropout=0.3`
- `batch_size=256`, `epochs=5`, `lr=1e-3`, `weight_decay=0.0`
- `patience=2`, `seed=42`
- `output_dir="models/url"`, `device=None`, `max_train_samples=None`

Same loop contract as TEXT-03 (seeded, `BCEWithLogitsLoss`, Adam/AdamW, per-epoch `{"epoch", "train_loss", "val_f1"}`, save best by `val_f1`, early stopping on `patience`, returns `TrainResult`). On a tiny separable set, the last epoch's `train_loss` is lower than the first's.

### URL-05 `URLClassifier.save(path)` / `URLClassifier.load(path, device=None)`
`save` writes `path/url_model.pt` (the `state_dict`) and `path/url_config.json` (`max_len` plus every `CharCNN` constructor argument). `load` rebuilds both and gives identical predictions.

## Known risk to discuss in the report
In PhiUSIIL, many legitimate URLs are bare home pages (`https://www.site.com`) while phishing URLs often have long paths. A model can score highly just by learning **length**. Check this: compare the accuracy on URLs longer than 40 characters against all URLs, and mention the result.
