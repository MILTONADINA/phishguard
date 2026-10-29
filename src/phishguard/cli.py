"""Command-line interface: `python -m phishguard <command>`. Spec: specs/08-detector-and-cli.md (CLI-01)."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """CLI-01: subcommands prepare-data, train-baseline, train-text, train-url, fit-fusion, evaluate, serve.

    The selected subcommand is stored in args.command. See the spec table for arguments and defaults.
    """
    raise NotImplementedError("CLI-01: see specs/08-detector-and-cli.md")


def main(argv: list[str] | None = None) -> None:
    """Parse args and dispatch to the handler for args.command."""
    raise NotImplementedError("CLI-01: see specs/08-detector-and-cli.md")
