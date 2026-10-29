---
description: Implement the next unchecked task from TASKS.md using the spec-driven workflow in CLAUDE.md, then commit and stop.
argument-hint: "[task id, e.g. T3 — optional; defaults to the first unchecked task]"
---

Implement one task from `TASKS.md`. If `$ARGUMENTS` names a task (e.g. `T3`), do that one. Otherwise do the first unchecked task in the progress checklist.

Follow the workflow in `CLAUDE.md` exactly:
1. Read the task row in `TASKS.md` and every spec section it references.
2. Run the task's test command and show that it fails (red).
3. Implement in `src/phishguard/` only. Keep the stub signatures.
4. Re-run the task's tests until they pass, then run `.venv/bin/python -m pytest -q` for regressions.
5. Append to `NOTES.md` (date, task, what happened, why/numbers).
6. Tick the task in the `TASKS.md` checklist.
7. `git add -A && git commit -m "<message from TASKS.md>"`.
8. Report: test counts before and after, the files changed, anything that needs Milton's decision. Then stop.

Never edit `tests/` or `specs/` to get a pass. If something there looks wrong, stop and explain.
For training tasks (T8, T12), run training in the background with a log in `logs/` and report progress. Don't block on a multi-hour run.
