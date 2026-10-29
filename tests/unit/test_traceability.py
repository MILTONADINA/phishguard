"""Spec ↔ test traceability: every requirement ID defined in specs/ must be referenced by
at least one @pytest.mark.req("ID"), and every referenced ID must exist in a spec."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ID_RE = r"(?:DATA|BASE|TEXT|URL|FUSE|EVAL|API|DET|CLI|UTIL|AC)-\d{2}"


def _spec_ids() -> set[str]:
    ids: set[str] = set()
    for f in (ROOT / "specs").glob("*.md"):
        text = f.read_text(encoding="utf-8")
        ids |= set(re.findall(rf"^###\s+({ID_RE})\b", text, flags=re.M))  # requirement headings
        ids |= set(re.findall(rf"^\|\s*\*\*({ID_RE})\*\*", text, flags=re.M))  # acceptance table rows
    return ids


def _test_ids() -> set[str]:
    ids: set[str] = set()
    for f in (ROOT / "tests").rglob("test_*.py"):
        ids |= set(re.findall(rf'req\(\s*"({ID_RE})"\s*\)', f.read_text(encoding="utf-8")))
    return ids


def test_every_spec_requirement_has_a_test():
    missing = sorted(_spec_ids() - _test_ids())
    assert not missing, f"Requirements without tests: {missing}"


def test_every_test_id_exists_in_specs():
    unknown = sorted(_test_ids() - _spec_ids())
    assert not unknown, f"Tests reference unknown requirement IDs: {unknown}"


def test_specs_define_requirements():
    assert len(_spec_ids()) >= 40
