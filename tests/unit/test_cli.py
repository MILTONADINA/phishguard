"""specs/08-detector-and-cli.md (CLI-01) — parsing only."""
from __future__ import annotations

import pytest

from phishguard.cli import build_parser

COMMANDS = ["prepare-data", "train-baseline", "train-text", "train-url", "fit-fusion", "evaluate", "serve"]


@pytest.mark.req("CLI-01")
@pytest.mark.parametrize("cmd", COMMANDS)
def test_every_command_parses(cmd):
    args = build_parser().parse_args([cmd])
    assert args.command == cmd


@pytest.mark.req("CLI-01")
def test_defaults_and_types():
    p = build_parser()
    a = p.parse_args(["prepare-data"])
    assert (a.raw_dir, a.out_dir, a.seed) == ("data/raw", "data/processed", 42)
    t = p.parse_args(["train-text", "--epochs", "3", "--lr", "3e-5", "--max-train-samples", "20000", "--device", "mps"])
    assert t.epochs == 3 and t.lr == pytest.approx(3e-5) and t.max_train_samples == 20000 and t.device == "mps"
    t0 = p.parse_args(["train-text"])
    assert (t0.epochs, t0.batch_size, t0.max_length, t0.max_train_samples, t0.out_dir) == (2, 16, 256, None, "models/text")
    u = p.parse_args(["train-url"])
    assert (u.epochs, u.batch_size, u.out_dir) == (5, 256, "models/url")
    s = p.parse_args(["serve", "--port", "9000"])
    assert s.port == 9000 and s.host == "127.0.0.1" and s.models_dir == "models"
    e = p.parse_args(["evaluate"])
    assert (e.data_dir, e.models_dir, e.reports_dir) == ("data/processed", "models", "reports")


@pytest.mark.req("CLI-01")
def test_unknown_command_fails():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["fly-to-the-moon"])
