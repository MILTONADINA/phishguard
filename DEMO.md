# Demo script (5 minutes max)

Aim for 4:30 so there's room for a slow page load or a question.

## Before you start
- `make demo` running (text-only mode). Check http://127.0.0.1:8000/health shows `"models_loaded": true`.
- Browser on http://127.0.0.1:8000, zoomed so the back row can read it.
- The three files in `demo_samples/` open in an editor, ready to copy: `legit.txt`, `obvious_phish.txt`, `subtle_phish.txt`.
- `reports/metrics.md` open in a second tab or window.
- Recording: Cmd+Shift+5 → Record Selected Portion (or Entire Screen) → Options: microphone on.

## 0:00–0:30 Problem
- Phishing email is the most common way into a small business, and the person reading it usually has no security team behind them.
- PhishGuard: paste an email and get a verdict, a score, the links it found and plain-English reasons.

## 0:30–1:15 How it works
- Show the architecture diagram in `REPORT.md`, section 1.
- Stage 1: DistilBERT, a pretrained transformer, fine-tuned on 65,000 labelled emails in 19 minutes on this Mac.
- Stage 2: a small character-level CNN that scores each link.
- The two scores are combined with a weight fitted on validation data, then served by FastAPI.
- A TF-IDF baseline for comparison, and a held-out set from a different source that was never trained or tuned on.

## 1:15–3:00 Live demo
1. **Normal email** (`legit.txt`): **legitimate, 0%**. "A team meeting note: nothing to flag."
2. **Obvious phish** (`obvious_phish.txt`): **phishing, 100%**. Point at the reasons and the extracted link `paypa1-secure-login.com`: a 1 in place of the l.
3. **Subtle phish** (`subtle_phish.txt`): **phishing, 78%**. "A polite W-9 request, no threats, a look-alike SharePoint link. It's caught, but it's less sure, and this is the kind real attackers send."
- Point out that links say "not checked". That is the link model's finding, covered in the next section.

## 3:00–4:00 Results (`reports/metrics.md`)
- **On the test split everything looks great:** the baseline 0.989 F1, DistilBERT 0.994.
- **On the held-out emails from a different source:** the baseline flags **290 of 1,000** normal emails. DistilBERT flags **11**. That's what fine-tuning bought.
- **The link model looks perfect on its own test set** (0.998), but it learned a shortcut. Every legitimate URL in that dataset is a bare home page, so it treats any link with a path as phishing, including github.com. When I fitted the fusion weight on real emails, it gave the link model zero weight on its own, so the demo runs text-only.

## 4:00–4:30 What I'd improve
- Retrain the link model on legitimate URLs that have real paths.
- Look at the 94 phishing emails the held-out set still slips past (recall 0.906).
- Highlight the words that drove the score.

## Likely questions
- **Why DistilBERT and not BERT?** About 40% smaller, keeps most of the accuracy, and trains in 19 minutes on a laptop.
- **Why threshold 10%?** The fusion search picked it on validation data, by a margin of about 3 emails. It's slightly worse on held-out, but it was left as fitted: changing it after seeing the held-out results would be tuning on the test.
- **Can I see the link model?** Yes: `make serve` loads both models (screenshot in `REPORT.md`, section 1).
