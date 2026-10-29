"""Command-line interface: `python -m phishguard <command>`. Spec: specs/08-detector-and-cli.md (CLI-01)."""
from __future__ import annotations

import argparse

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


def _not_built_yet(task: str):
    def handler(args: argparse.Namespace) -> None:
        raise NotImplementedError(f"`{args.command}` is built in {task} (see TASKS.md)")

    return handler


HANDLERS = {
    "prepare-data": _prepare_data,
    "train-baseline": _not_built_yet("T6"),
    "train-text": _not_built_yet("T8"),
    "train-url": _not_built_yet("T12"),
    "fit-fusion": _not_built_yet("T13"),
    "evaluate": _not_built_yet("T13"),
    "serve": _not_built_yet("T11"),
}


def main(argv: list[str] | None = None) -> None:
    """Parse args and dispatch to the handler for args.command."""
    args = build_parser().parse_args(argv)
    HANDLERS[args.command](args)
