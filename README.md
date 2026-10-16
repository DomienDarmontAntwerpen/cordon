# Cordon — AI Cost & Data Gateway

**A self-hosted layer between a company's apps and AI models such as ChatGPT that protects personal data and cuts costs - and proves every saving with a measurement.**

![Python](https://img.shields.io/badge/Python-3.11_%7C_3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o_%7C_4.1--mini-412991?logo=openai&logoColor=white)
![LiteLLM](https://img.shields.io/badge/LiteLLM-gateway-555555)
![scikit-learn](https://img.shields.io/badge/scikit--learn-quality_check-F7931E?logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-ledger-003B57?logo=sqlite&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
[![Tests](https://github.com/DomienDarmontAntwerpen/cordon/actions/workflows/ci.yml/badge.svg)](https://github.com/DomienDarmontAntwerpen/cordon/actions/workflows/ci.yml)

> **In short:** every AI request from a company's apps passes through Cordon first. It masks
> personal data before it leaves the company, blocks manipulation attempts, keeps each team within
> its budget - and saves money: a cache for repeated questions, shorter prompts, smaller images,
> and a cheaper model **only where a benchmark on real models proved it as good**. If Cordon goes
> down, a failsafe sends requests straight to the provider, so the business never stops.

![Cordon dashboard, Overview tab: alerts, cost, savings and savings per technique (mock traffic from demo teams)](docs/images/dashboard.png)

---

## What it does

- **Protects data:** emails, phone numbers, IBANs, national ID and card numbers become placeholders before a request leaves the company, and are put back in the answer.
- **Blocks manipulation:** prompt-injection attempts are flagged or blocked; leaks of the hidden instructions are redacted.
- **Saves money, measured:** 80% less on a simulated working day; a cheaper model is used only where it was proven as good on hundreds of real questions - see [A measured working day](#a-measured-working-day).
- **Controls spend:** a budget and rate limit per team, and a stop for AI agents stuck in a loop.
- **Shows it all:** a dashboard per team, with an *observe* mode that measures what Cordon *would* save before anything changes, and a panel to send a request and see what Cordon did.
- **Keeps working:** a failsafe in front of Cordon sends requests straight to the provider when Cordon is down.

## A measured working day

To see what Cordon saves on a company's bill, we **simulated one working day**: 300 requests built
from real, public data - texts to classify, customer-support questions (popular ones asked again
and again), support chats sent with their history, news articles to summarize. Every request was
sent **live to OpenAI twice**: straight to gpt-4o, as a company pays today, and through Cordon with
its savings on. The costs are **OpenAI's prices for the tokens it really billed**. Classifications are scored by their
known label; the other answers by closeness to the dataset's reference answer (ROUGE-L plus an LLM
judge, gpt-4o itself), on a 0-1 scale of closeness - not a share of correct answers.

| Kind of request | Share of the bill | Saved | Quality 0-1 (straight / Cordon) | How |
|---|---|---|---|---|
| Chats with history | 34% | 86% | 0.37 / 0.45 | gpt-4.1-mini, proven on 367 conversations and in a blind check on 193 fresh ones |
| Support questions | 30% | 78% | 0.30 / 0.29 | repeats come from the cache; the rest from a cheaper model |
| Summaries | 26% | 75% | 0.46 / 0.46 | gpt-4.1-mini, proven as good on 365 articles |
| Classifying texts | 10% | 72% | 0.93 / 0.95 | gpt-4.1-mini, proven as good on 729 questions |
| **Whole day** | | **80%** | no measurable loss over the day | EUR 0.64 straight vs EUR 0.13 through Cordon |

- One composed day, not a real company's traffic: more repeats, summaries or classifications save
  more, long open-ended work less. `tools/audit.py` estimates it from a company's own logs.
- Per kind, 60-100 requests are too few to prove a difference either way. The chats are support
  conversations stitched from a public dataset. gpt-4.1-mini scores *higher* there mostly because
  gpt-4o writes twice as long as the reference; so a **blind check** followed, fixed in advance: on 193
  fresh conversations gpt-4o, as judge, preferred gpt-4.1-mini's reply 56 times, its own 36 times,
  and called 101 a tie - at least as good.
- Arithmetic and code stay on gpt-4o. How each number was measured, and what it does *not* show:
  [docs/REFERENCE.md - Evidence](docs/REFERENCE.md#evidence). Repeat it: `python -m evaluation.company_day`.

## What I decided

Built with an AI coding assistant as the tool; the design calls were mine:

- **Every claim measured, or not made** - savings only from live runs, with the plan and the
  decision rule committed before the answers exist, and the result accepted either way.
- **The business never stops because of Cordon** - hence the failsafe that bypasses it when it is down.
- **Data protection before savings** - a broken security step stops the request; a broken saving never does.
- **No per-employee view** - monitoring staff needs a stated purpose, informed employees and, in
  Belgium, the works council (CAO 81); cost stays visible per team.
- **Semantic cache off by default** - measured: 1-5% of its hits answer a different question.
- **When two options score alike, the simpler one ships.**

## How it works

```mermaid
flowchart LR
    A[Check access] --> B[Protect input] --> C[Stop loops] --> D[Shrink] --> E[Cache]
    E -->|seen before| I[Protect output]
    E -->|new| F[Pick model] --> G[Ask the AI] --> H[Check quality]
    H -->|doubtful: ask the strong model once| G
    H --> I --> J[Answer]
```

1. **Protect:** personal data is masked and manipulation attempts are caught before anything leaves.
2. **Save:** repeated questions come from an encrypted cache; duplicated text and oversized images are trimmed.
3. **Route:** a cheaper model answers only where it was proven as good; a small ML model (the quality check) sends any answer it doubts to the strong model.
4. **Account:** every request is booked - cost, what it would have cost, what was blocked - never its text.

If a money-saving step fails, the request goes through without that saving; if a security step fails, the request is stopped.

![Try a request, live on OpenAI: the next turn of a support conversation, email masked, gpt-4.1-mini chosen, a real answer, 84% saved](docs/images/dashboard_try.png)

<details>
<summary>Teams, each in observe or enforce mode</summary>

![Teams tab: one row per team with Observe / Enforce buttons, features, requests, cost and savings](docs/images/dashboard_teams.png)
</details>

## Tech stack

| Layer | Technology |
|---|---|
| Gateway | FastAPI, LiteLLM (OpenAI-compatible API) |
| Models | gpt-4o (strong), gpt-4.1-mini (proven route); Mistral supported |
| Quality check (ML) | scikit-learn classifier on answer features and the model's own confidence (logprobs) |
| Data protection | patterns plus checks (Luhn for cards, real digits for phones), AES-GCM encrypted cache |
| Storage | SQLite (ledger, teams, cache) |
| Dashboard | plain HTML/CSS/JS, Chart.js |
| Failsafe | nginx |
| Deployment | Docker Compose |
| Testing | pytest: 300+ offline tests, 5 live checks, a 10,000-request fuzz run |

## Getting started

No API key needed - in mock mode Cordon fakes the AI's answers.

```bash
python3.11 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m tools.setup_models

.venv/bin/python -m tools.demo             # a real server, 10 scenarios: savings, cache, routing, blocking, privacy
.venv/bin/python -m tools.dashboard_demo   # the dashboard with demo teams; prints a key for "Try a request"
```

Apps use the normal OpenAI library with Cordon's address:

```python
client = OpenAI(api_key="<Cordon team key>", base_url="http://127.0.0.1:8787/v1")
```

Going live, team admin and deployment: [docs/REFERENCE.md](docs/REFERENCE.md#running-and-managing-cordon).

## Running the tests

```bash
.venv/bin/python -m pytest tests/              # offline
.venv/bin/python -m pytest -m live             # 5 live checks, needs OPENAI_API_KEY, < 1 cent
```

## Project structure

```
src/           the gateway: request pipeline, protection, savings, configuration, the trained check
tools/         what an operator runs: team admin, setup, the audit, the demos
evaluation/    the proof: benchmark, route proof, training, load and failsafe checks
notebooks/     the ML evaluations, with their results
tests/         the test suite
docs/          REFERENCE.md: every file, error code, design choice, failure mode, the evidence and limits
```

Also: [SECURITY.md](SECURITY.md) (threat model, data handling) and [reports/model_card.md](reports/model_card.md) (the quality check, measured).

## Status & next steps

A working, tested prototype; every saving and quality number above was measured live on OpenAI (about EUR 10 in total; the dashboard overview shows mock traffic) and can be repeated (`evaluation/company_day.py`). How each was measured, and what it does *not* show: [docs/REFERENCE.md - Evidence](docs/REFERENCE.md#evidence). In progress, each measured before it is switched on: a route for arithmetic (asked plainly, gpt-4o-mini comes close: 95% vs 97%), shorter chat histories, a batch mode for work that can wait - and a pilot on a company's own traffic, the real test of any route.

Reviewed in rounds by AI review agents (software and security, ML evaluation, a recruiter and a client's CTO); every confirmed bug has a regression test that fails without the fix.

## Author
**Domien Darmont** — Operational leader building applied AI systems, Antwerp, Belgium  
GitHub: [@DomienDarmontAntwerpen](https://github.com/DomienDarmontAntwerpen)  
LinkedIn: [Domien Darmont](https://www.linkedin.com/in/domiendarmont/)

A live demo, the project presentation or answers to any questions are available on request.

## License
© 2026 Domien Darmont. All rights reserved.

The source code of this project is published for viewing and evaluation only. No permission is granted to copy, modify, distribute or use it, in whole or in part, without prior written consent from the author.

Third-party evaluation data in `data/` (questions from public datasets) remains under its own licence - see [docs/REFERENCE.md - Data sources](docs/REFERENCE.md#data-sources).
