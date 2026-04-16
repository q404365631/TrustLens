# TrustLens — Positioning & Viral Copy

> Use these across GitHub, social media, HN/Reddit posts, conference talks.

---

## Tagline Options

| # | Tagline | Tone |
|---|---------|------|
| **Current** | *Debug your ML models beyond accuracy.* | Technical |
| **Option 1** | *Accuracy lies. Trust scores don't.* | Bold / provocative |
| **Option 2** | *The last check before your model goes live.* | Practical / safety-focused |
| **Option 3** | *One number tells you if your model is ready for the world.* | Accessible / product-y |
| **Option 4** | *Your model passed the test. Did it pass TrustLens?* | Challenge-based |

**Recommended:** Use **Option 2** in README (less confrontational for enterprise), use **Option 1** for HackerNews/Twitter/Reddit posts where provocation earns clicks.

---

## 3 Alternative Positioning Lines

### 1. "The Deployment Checklist No One Had"

> Every engineer has a mental checklist before deploying:
> - [x] Accuracy ≥ threshold
> - [x] Tests pass
> - [x] Reviewed in PR
>
> **TrustLens adds the checks that actually matter:**
> - [ ] Is the model calibrated?
> - [ ] Does it fail gracefully, or catastrophically and confidently?
> - [ ] Does it perform equally across demographic groups?

---

### 2. "The Gap Between Accuracy and Accountability"

> Healthcare, finance, hiring, fraud detection — in high-stakes domains,
> 90% accuracy isn't good enough if the 10% of errors are **systematically
> wrong** about the most vulnerable users.
>
> TrustLens bridges the gap between model performance and model responsibility.

---

### 3. "Interpretable by Default, Not as an Afterthought"

> Most interpretability tools are bolted on after development.
> TrustLens is designed to be part of every model evaluation loop:
> fast enough to run in CI, rich enough for a research paper,
> simple enough that your first call is `analyze(model, X, y)`.

---

## 3 Viral Social Media Phrases

### 1. The Shocking Comparison (for Twitter/LinkedIn)

> We compared two models on the same dataset.
>
> Model A: 87% accuracy
> Model B: 84% accuracy
>
> We almost deployed Model A.
>
> Then we ran TrustLens.
> Trust Score: A=54/100, B=73/100
>
> Model A was overconfident on its worst mistakes.
> Model B failed far more gracefully.
>
> Accuracy almost got us in trouble.
> Trust saved us.
>
> [GitHub link]

---

### 2. The One-Liner (for HackerNews / Reddit)

> "Show HN: TrustLens — a Python library that gives your ML model a Trust Score (0–100), surfaces high-confidence failures, and flags bias in one `analyze()` call"

---

### 3. The Developer Empathy Hook (for LinkedIn / dev.to)

> Most ML engineering posts talk about model architecture.
> Almost none talk about what happens after you hit 90% accuracy:
>
> • Is the model overconfident on edge cases?
> • Does it fail equally badly across all user groups?
> • If a regulator asks "why," do you have an answer?
>
> We built a library to answer all three. It's called TrustLens.
> One function call. One Trust Score. Full picture.

---

## HackerNews Launch Post Template

```
Title: Show HN: TrustLens – ML model analysis beyond accuracy (trust score, failure showcase, bias detection)

Body:
I built TrustLens to solve a problem I kept hitting: models with great accuracy that had hidden
calibration problems, overconfident failures, and subgroup bias that only appeared in production.

The core idea: analyze(model, X, y) runs a full trust analysis and returns a Trust Score (0–100)
combining calibration quality, failure patterns, bias risk, and embedding geometry.

Key features:
 - Trust Score: single 0–100 number that combines Brier Score, ECE, confidence gap, and fairness metrics
 - summary_plot(): 6-panel dashboard showing everything at a glance
 - show_failures(): surfaces top high-confidence wrong predictions, ranked by "danger"
 - Plugin system: extend with your own metrics in 10 lines of Python
 - 67 passing tests, clean sklearn-compatible API

GitHub: https://github.com/Khanz9664/TrustLens

Would love feedback on the Trust Score formula and what dimensions matter most to others.
```

---

## Tweet Thread Template

```
Tweet 1/5:
We built a tool that gives ML models a "Trust Score" from 0-100.

92% accuracy? Great.
Trust Score 41/100? Ship nothing.

Here's what we found:

Tweet 2/5:
Accuracy measures how often you're right.
Trust Score measures how right you are when you're confident,
how fair you are across groups, and how well your probabilities reflect reality.

One function call:
report = trustlens.analyze(model, X, y)
print(report.trust_score) # e.g. 73/100 [B]

Tweet 3/5:
The failure showcase changed how we debug models.

report.show_failures(top_k=10)

It finds the highest-confidence WRONG predictions — the mistakes
your model is most certain about. These are the dangerous ones.

Tweet 4/5:
We ran two models through it.

Model A: 87% accuracy, Trust Score 54/100 [C] — overconfident, biased
Model B: 84% accuracy, Trust Score 73/100 [B] — calibrated, fair

We almost deployed Model A.

Tweet 5/5:
TrustLens is open-source, MIT licensed, and built for contributors.

Adding a new metric takes 4 steps and ~20 lines of code.
We have 50 GitHub issues ready — from 1-hour beginner tasks to week-long research.

 Star if you've ever shipped a model you weren't 100% sure about.
[link]
```
