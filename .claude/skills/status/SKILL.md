---
description: Show project progress — test results per spec area, which TASKS.md items are done, and the next task.
---

1. Run `.venv/bin/python -m pytest -q -rN --no-header -p no:warnings` and summarise pass/fail counts per test file (one file per spec area).
2. Read the progress checklist in `TASKS.md` and list the done and remaining tasks.
3. Show `git log --oneline -15`.
4. If `reports/metrics.md` exists, show it.
5. State the next task, how many days remain before Wed Nov 11, and whether the calendar in `TASKS.md` is on track.

Read-only: don't edit any file.
