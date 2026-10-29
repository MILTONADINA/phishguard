"""Command-line interface: `python -m phishguard <command>`. Spec: specs/08-detector-and-cli.md (CLI-01)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from phishguard.config import SEED, TextTrainConfig, UrlTrainConfig

DEVICES = ["cpu", "cuda", "mps"]


def build_parser() -> argparse.ArgumentParser:
    """CLI-01: subcommands prepare-data, train-baseline, train-text, train-url, fit-fusion, evaluate, serve.

    The selected subcommand is stored in args.command. See the spec table for arguments and defaults.
    """
    text_defaults, url_defaults = TextTrainConfig(), UrlTrainConfig()  # one source of truth for defaults
    parser = argparse.ArgumentParser(prog="phishguard", description="Two-stage phishing email detector.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p = commands.add_parser("prepare-data", help="clean, deduplicate and split data/raw into data/processed")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--seed", type=int, default=SEED)

    p = commands.add_parser("train-baseline", help="fit TF-IDF + logistic regression and print val metrics")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out", default="models/baseline.joblib")

    p = commands.add_parser("train-text", help="fine-tune DistilBERT on the email text")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir", default=text_defaults.output_dir)
    p.add_argument("--epochs", type=int, default=text_defaults.epochs)
    p.add_argument("--batch-size", type=int, default=text_defaults.batch_size)
    p.add_argument("--lr", type=float, default=text_defaults.lr)
    p.add_argument("--max-length", type=int, default=text_defaults.max_length)
    p.add_argument("--max-train-samples", type=int, default=None, help="train on a stratified subset (quick run)")
    p.add_argument("--device", choices=DEVICES, default=None, help="default: pick automatically")

    p = commands.add_parser("train-url", help="train the character-level CNN on URLs")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir", default=url_defaults.output_dir)
    p.add_argument("--epochs", type=int, default=url_defaults.epochs)
    p.add_argument("--batch-size", type=int, default=url_defaults.batch_size)
    p.add_argument("--lr", type=float, default=url_defaults.lr)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--device", choices=DEVICES, default=None)

    p = commands.add_parser("fit-fusion", help="choose the fusion weight and threshold on the val split")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--models-dir", default="models")

    p = commands.add_parser("evaluate", help="score every model and write reports/metrics.md")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--reports-dir", default="reports")

    p = commands.add_parser("serve", help="run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--models-dir", default="models")

    return parser


# ---- command handlers ---------------------------------------------------------


def _prepare_data(args: argparse.Namespace) -> None:
    from phishguard.data import prepare_data

    counts = prepare_data(args.raw_dir, args.out_dir, seed=args.seed)
    print(f"Wrote {args.out_dir}/:")
    for name, rows in counts.items():
        print(f"  {name:<14} {rows:>9,} rows")


def _print_metrics(title: str, m: dict) -> None:
    print(f"{title} (n={m['n']:,}, threshold {m['threshold']}):")
    print(
        f"  accuracy {m['accuracy']:.4f}  precision {m['precision']:.4f}  recall {m['recall']:.4f}  "
        f"F1 {m['f1']:.4f}  ROC-AUC {m['roc_auc']:.4f}"
    )
    print(
        f"  false positives {m['fp']:,} (FPR {m['fpr']:.4f})  false negatives {m['fn']:,} (FNR {m['fnr']:.4f})"
    )


def _train_baseline(args: argparse.Namespace) -> None:
    from phishguard.baseline import baseline_predict_proba, save_baseline, train_baseline
    from phishguard.data import read_split
    from phishguard.evaluate import compute_metrics

    data_dir = Path(args.data_dir)
    train, val = read_split(data_dir / "train.csv"), read_split(data_dir / "val.csv")

    start = time.perf_counter()
    model = train_baseline(train["text"].tolist(), train["label"].tolist())
    print(f"Fitted TF-IDF + logistic regression on {len(train):,} emails in {time.perf_counter() - start:.0f} s")

    # Validation only: the test split is reserved for the final evaluation (T13).
    val_probs = baseline_predict_proba(model, val["text"].tolist())
    _print_metrics("Validation", compute_metrics(val["label"].tolist(), val_probs))
    save_baseline(model, args.out)
    print(f"Saved {args.out}")


def _report_training(result, config, start: float) -> None:
    """Print the per-epoch history and keep it next to the model (train_history.json) for the report."""
    minutes = (time.perf_counter() - start) / 60
    print("\nepoch | train_loss | val_f1")
    for h in result.history:
        print(f"{h['epoch']:>5} | {h['train_loss']:>10.4f} | {h['val_f1']:.4f}")
    print(f"Best epoch {result.best_epoch} (val F1 {result.best_val_f1:.4f}), saved to {result.output_dir}")
    print(f"Total time {minutes:.1f} min")

    summary = {"config": vars(config), "history": result.history, "best_epoch": result.best_epoch,
               "best_val_f1": result.best_val_f1, "minutes": round(minutes, 1)}
    (Path(result.output_dir) / "train_history.json").write_text(json.dumps(summary, indent=2))


def _train_text(args: argparse.Namespace) -> None:
    from phishguard.data import read_split
    from phishguard.text_model import train_text_model

    data_dir = Path(args.data_dir)
    train, val = read_split(data_dir / "train.csv"), read_split(data_dir / "val.csv")
    config = TextTrainConfig(
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.out_dir,
        device=args.device,
        max_train_samples=args.max_train_samples,
    )
    print(f"Fine-tuning {config.model_name}: {config}", flush=True)
    start = time.perf_counter()
    result = train_text_model(
        train["text"].tolist(), train["label"].tolist(), val["text"].tolist(), val["label"].tolist(), config
    )
    _report_training(result, config, start)


def _train_url(args: argparse.Namespace) -> None:
    from phishguard.data import read_split
    from phishguard.url_model import train_url_model

    data_dir = Path(args.data_dir)
    train, val = read_split(data_dir / "url_train.csv"), read_split(data_dir / "url_val.csv")
    config = UrlTrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.out_dir,
        device=args.device,
        max_train_samples=args.max_train_samples,
    )
    print(f"Training the character-level CNN: {config}", flush=True)
    start = time.perf_counter()
    result = train_url_model(
        train["url"].tolist(), train["label"].tolist(), val["url"].tolist(), val["label"].tolist(), config
    )
    _report_training(result, config, start)


def _fit_fusion(args: argparse.Namespace) -> None:
    from phishguard.config import MAX_URLS
    from phishguard.data import clean_text, extract_urls, read_split
    from phishguard.detector import PhishDetector
    from phishguard.evaluate import compute_metrics
    from phishguard.fusion import DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS, aggregate_url_scores, fit_fusion, fuse

    models_dir = Path(args.models_dir)
    val = read_split(Path(args.data_dir) / "val.csv")  # validation only: test and held-out stay untouched
    texts, labels = val["text"].tolist(), val["label"].tolist()
    detector = PhishDetector.from_dir(models_dir)  # only its models are used; the fusion is what we fit

    # Same scores PhishDetector.analyze would compute, but in batches.
    print(f"Scoring {len(texts):,} validation emails...", flush=True)
    text_scores = detector.text_model.predict_proba([clean_text(t) for t in texts]).tolist()
    url_scores = [None] * len(texts)
    if detector.url_model is not None:
        links = [extract_urls(t)[:MAX_URLS] for t in texts]
        flat = iter(detector.url_model.predict_proba([u for email_links in links for u in email_links]))
        url_scores = [aggregate_url_scores([next(flat) for _ in email_links]) for email_links in links]
        print(f"  {sum(s is not None for s in url_scores):,} of them contain links", flush=True)

    # How much each weight can reach (best threshold per weight), for the report.
    print("\nweight | best val F1")
    for weight in DEFAULT_WEIGHTS:
        fused = [fuse(t, u, weight) for t, u in zip(text_scores, url_scores)]
        best = max(compute_metrics(labels, fused, threshold)["f1"] for threshold in DEFAULT_THRESHOLDS)
        print(f"{weight:>6} | {best:.4f}")

    params = fit_fusion(text_scores, url_scores, labels)
    params.save(models_dir / "fusion.json")
    chosen = compute_metrics(labels, [fuse(t, u, params.weight) for t, u in zip(text_scores, url_scores)],
                             params.threshold)
    print(f"\nChosen: weight {params.weight} on the text score, threshold {params.threshold} "
          f"(val F1 {chosen['f1']:.4f}). Saved {models_dir / 'fusion.json'}")


def _evaluate(args: argparse.Namespace) -> None:
    from phishguard.evaluate import evaluate_all

    start = time.perf_counter()
    evaluate_all(args.data_dir, args.models_dir, args.reports_dir)
    print((Path(args.reports_dir) / "metrics.md").read_text())
    print(f"Evaluation took {(time.perf_counter() - start) / 60:.1f} min")


def _serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    # phishguard.app builds its `app` (and loads the models) when it is imported, from this
    # variable. Setting it first means the models are loaded once, from --models-dir.
    os.environ["PHISHGUARD_MODELS_DIR"] = args.models_dir
    from phishguard.app import app

    print(f"PhishGuard running at http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


HANDLERS = {
    "prepare-data": _prepare_data,
    "train-baseline": _train_baseline,
    "train-text": _train_text,
    "train-url": _train_url,
    "fit-fusion": _fit_fusion,
    "evaluate": _evaluate,
    "serve": _serve,
}


def main(argv: list[str] | None = None) -> None:
    """Parse args and dispatch to the handler for args.command."""
    args = build_parser().parse_args(argv)
    HANDLERS[args.command](args)
