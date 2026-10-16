# Cordon - technical reference

The README explains what Cordon is and how to try it. This document holds the details, each in
one place: every file, every error code, the design choices, what happens when something fails,
how to deploy it, and the evidence behind every number - with what it does *not* show.

## Contents

1. [The request pipeline](#the-request-pipeline)
2. [Running and managing Cordon](#running-and-managing-cordon)
3. [Repository layout, file by file](#repository-layout-file-by-file)
4. [Error codes](#error-codes)
5. [Design choices](#design-choices)
6. [What happens when something fails](#what-happens-when-something-fails)
7. [Deploying](#deploying)
8. [Evidence](#evidence)
9. [Security and privacy](#security-and-privacy)
10. [Topics covered](#topics-covered)
11. [Why not LiteLLM Proxy, Helicone or Portkey?](#why-not-litellm-proxy-helicone-or-portkey)
12. [Considered and left out](#considered-and-left-out)
13. [Limitations and next steps](#limitations-and-next-steps)

---

## The request pipeline

Every request goes through nine steps, one file each in `src/server/pipeline/`; `run.py` runs them
in order.

| Step | File | In plain words |
|---|---|---|
| 1. Check access | `step1_admission.py` | A valid team key, an allowed model, a sane size, within the rate limit and budget? |
| 2. Protect input | `step2_input_protection.py` | Replace personal data with placeholders (`<EMAIL_1>`); flag or block manipulation attempts. |
| 3. Stop loops | `step3_loop_detection.py` | Stop an AI agent that keeps sending the same message. |
| 4. Shrink | `step4_optimize.py` | Remove duplicated text, shorten long histories and tool output, downscale images. |
| 5. Cache | `step5_cache.py` | Answered before? Return that answer for free. |
| 6. Pick model | `step6_routing.py` | A cheaper model only where it was proven - together with its check - as good; otherwise the strong model. The answer names the kind of request (`X-Cordon-Kind`). |
| 7. Ask the AI | `step7_generate.py` | Call the model; if a provider fails, try the next one. |
| 8. Check quality | `step8_quality.py` | A cheaper model's answer is scored; below the route's proven threshold, the strong model is asked once. `X-Cordon-Check` says what decided (`rules_only`, `classifier`, or the rule that fired). |
| 9. Protect output | `step9_output.py` | Redact leaked personal data or hidden instructions, put the real values back, book the cost. |

**Golden rule:** if a *money-saving* step breaks, the request goes through without that saving; if
a *security* step breaks, the request is stopped - Cordon never sends data out unprotected.

## Running and managing Cordon

Run the server in mock mode and open `http://127.0.0.1:8787/dashboard`:

```bash
export CORDON_ADMIN_KEY=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")
MOCK_MODE=true .venv/bin/python -c "import uvicorn; from src.server.gateway import create_app; uvicorn.run(create_app(), host='127.0.0.1', port=8787)"
```

Manage teams (changes apply at once, even while the server runs; the dashboard has the same switches):

```bash
.venv/bin/python -m tools.admin new-team "Support team"       # starts in observe mode, nothing on
.venv/bin/python -m tools.admin key "Support team"            # the API key for the app
.venv/bin/python -m tools.admin mode "Support team" enforce   # observe | enforce
.venv/bin/python -m tools.admin on "Support team" trim cache routing
.venv/bin/python -m tools.admin set "Support team" budget_daily_eur=50
.venv/bin/python -m tools.admin teams                         # overview
```

**Going live:** copy `.env.example` to `.env`, set `MOCK_MODE=false`, add `OPENAI_API_KEY` and the
three secrets (the file shows how to generate each).

## Repository layout, file by file

```
cordon/
├── src/                  The server - everything that runs in production
│   ├── server/           The app itself: the request pipeline (pipeline/), settings, database, dashboard
│   ├── protection/       Security, personal-data guardrails, budgets and rate limits
│   ├── savings/          Shrinking, caching, model routing, quality scoring, cost accounting
│   ├── config/           YAML configuration (models, prices, limits, detection patterns)
│   ├── models/           Trained model files
│   └── paths.py          The location of every file, in one place
├── tools/                What an operator runs: team admin, model setup, audit, demos, CI
├── evaluation/           Proof and measurement: benchmark, route proof, training, load and failsafe checks
├── notebooks/            Evaluations, with their results (the quality check, the semantic cache threshold)
├── tests/                Automated tests - all offline
├── data/                 Benchmark questions and training data
├── reports/              Generated reports (not in git, except the model card)
├── deploy/               The failsafe in front of Cordon (nginx)
└── docs/                 This reference and the dashboard screenshots
```

### The server: `src/server/`

| File | What it does |
|---|---|
| `gateway.py` | Builds the app and connects everything. No request logic: start reading at `pipeline/run.py`. |
| `pipeline/` | **The heart.** One file per step a request goes through ([the pipeline](#the-request-pipeline)). |
| `routes_public.py` | Health check, model list, metrics for the dashboard, the dashboard itself. |
| `routes_admin.py` | The admin API: teams, keys, CSV export. |
| `services.py` | The long-lived objects every request shares (database, cache, limiters). |
| `schemas.py` | The shape of every JSON body the API accepts or returns. |
| `http_utils.py` | Small helpers: error responses, reading the request body, API key parsing. |
| `ledger.py` | Writes one record per request (costs, savings, what was blocked - never any text). |
| `settings.py` | Loads configuration and refuses to start if something is unsafe. |
| `store.py` | The database: teams, keys, ledger, cache. |
| `telemetry.py` | Optional tracing to Langfuse (with personal data removed). |
| `static/` | The dashboard (HTML, CSS, JavaScript, Chart.js). |

### Protection: `src/protection/`

| File | What it does |
|---|---|
| `security.py` | Keys, admin access, blocking too many wrong-key attempts, encryption helpers. |
| `guardrails.py` | Finds and hides personal data; detects manipulation attempts and leaks. |
| `limits.py` | Budgets, rate limits, and the circuit breaker that stops runaway agent loops. |

### Savings: `src/savings/`

| File | What it does |
|---|---|
| `optimize.py` | Shrinks prompts, chat histories, tool output and images. |
| `cache.py` | Remembers previous answers (encrypted, separated per team). |
| `routing.py` | Takes the proven route for each type of question (a cheaper model plus its check), else the strong model. |
| `quality.py` | The quality check: rules plus a small model decide whether a cheaper model's answer goes to the strong model. |
| `costing.py` | Calculates what each request cost, and what it *would* have cost without Cordon. |

### Configuration and models: `src/config/`, `src/models/`

| File | What it does |
|---|---|
| `config/config.yaml` | All settings: which models, prices, default limits per team. |
| `config/patterns.yaml` | What personal data and manipulation attempts look like (EN/NL/FR). |
| `config/routing_table.yaml` | The proven routes per task type, each with its evidence - written by `evaluation/route_proof.py` (run via `evaluation.benchmark --routing-table-from`). |
| `models/model.pkl` | The small quality-scoring model (made by `evaluation/train.py`). |

### Tools you run by hand: `tools/`

| File | What it does |
|---|---|
| `admin.py` | Manage teams from the command line: create them, give out keys, switch mode and features. |
| `setup_models.py` | Downloads the helper models once, after installing. |
| `demo.py` | The full showcase: 10 scenarios against a real running server. |
| `dashboard_demo.py` | Starts Cordon with mock traffic from three teams to explore the dashboard; `--screenshot` regenerates `docs/images/dashboard.png` and `dashboard_teams.png`. |
| `live_try_screenshot.py` | The "Try a request" tab with one real request through a live Cordon (under 0.1 cent): `docs/images/dashboard_try.png`. |
| `audit.py` | Sales tool: analyzes a company's existing AI logs and estimates what Cordon would save - without connecting to anything. |
| `ci.sh` | The automatic checks; `.github/workflows/ci.yml` runs it on every change. |

### Proof and measurement: `evaluation/`

| File | What it does |
|---|---|
| `benchmark.py` | Compares answer quality and cost with and without Cordon; collects labelled answers from one model (`--collect`). |
| `company_day.py` | A realistic working day of requests, sent live straight to gpt-4o and through Cordon: the real cost and quality of both; also writes the chat, summary and confirmation rounds. |
| `route_proof.py` | Proves a route: a cheaper model *with its quality check* against the strong model, per kind of request; writes the routing table. |
| `pairwise_judge.py` | A blind check of one route: a judge sees two replies (in both orders, models hidden) and picks the more helpful one. |
| `train.py` | Trains the quality check on real labelled answers (never on route-evidence questions) and writes the model card. |
| `gate_eval.py` | The check's evaluation: metrics, candidates (incl. an MLP, a fine-tuned transformer, an LLM judge). |
| `fetch_eval_sets.py` | Public questions with known answers (GSM8K, SST-2, MBPP, AG News): the evaluation set, the training set and extra rounds, never overlapping. |
| `semantic_threshold.py` | The calculations behind the semantic cache threshold (used by its notebook). |
| `measure_overhead.py` | How much time Cordon itself adds per request. |
| `load_test.py` | Many requests at once against a real server: throughput and exact bookkeeping. |
| `failsafe_check.py` | The failsafe end to end in Docker (Cordon up, a refusal, Cordon down, restart, Cordon back). |

### Everything else

| Path | What it is |
|---|---|
| `tests/` | Automated tests - all run offline; `tests/test_live.py` runs only with an OpenAI key. |
| `data/golden_set.jsonl` | The benchmark's 60 exam questions with their correct answers, 10 per task type. |
| `data/labelled_answers/` | Every live answer with its label (right or wrong): the routes' evidence and the quality check's training data. |
| `data/gate_eval/` | The LLM judge's stored scores, so the notebook runs offline. |
| `reports/` | Generated reports (only `model_card.md` is kept in git). |
| `notebooks/semantic_cache_threshold.ipynb` | How similar two questions must be before Cordon reuses an answer - measured on 40,430 labelled question pairs. |
| `notebooks/quality_gate_real_answers.ipynb` | The quality check on real answers: baselines, over/underfitting, learning curve, neural networks, an LLM judge, the threshold trade-off. |
| `requirements-eval.txt` | Packages for the proof and measurement tools in `evaluation/` (installed by CI; not needed to run Cordon). |
| `requirements-dev.txt` | Extra packages for the notebooks and the screenshot tools (includes `requirements-eval.txt`). |
| `deploy/failsafe.conf.template` | The failsafe (nginx) in front of Cordon: requests go straight to the provider when Cordon cannot be reached. |
| `Dockerfile`, `docker-compose.yml` | For running Cordon on a server. |
| `SECURITY.md` | Full security and privacy write-up. |

---

## Error codes

Cordon answers in the same error format as OpenAI, so existing apps handle
these errors without changes. The `code` field says exactly what happened.

| HTTP | `code` | What happened | What to do |
|---|---|---|---|
| 400 | `param_not_allowed` | The request contains a field Cordon does not allow (for example `api_base`). | Remove the field. |
| 400 | `invalid_param_value` | An allowed field has an invalid value (e.g. `max_tokens` not a positive integer, `temperature` outside 0-2, `n` not 1). | Fix the value. |
| 400 | `model_not_allowed` | The model name is not on Cordon's list. | Use one of the models from `GET /v1/models`. |
| 400 | `model_not_priced` | Cordon does not know this model's price, so it cannot enforce budgets. | Add a price in `config.yaml`, or use another model. |
| 400 | `provider_not_allowed` | This team may not use that provider (for example DeepSeek). | Ask the admin to allow it, or pick another model. |
| 400 | `provider_unavailable` | No API key is configured for that provider. | Add the provider key to `.env`. |
| 400 | `invalid_tag` | A tag is malformed, or there are more than 5. | Fix the tags. |
| 400 | `pii_blocked` | The question contains personal data and the team is set to block it. | Remove the personal data. |
| 400 | `injection_blocked` | The question looks like a manipulation attempt and the team is set to block it. | Rephrase; ask the admin if this is a false alarm. |
| 400 | *(none)* | The body is not valid JSON, or `messages` is empty. | Fix the request. |
| 401 | *(none)* | The API key is wrong, expired or revoked. | Check the key. |
| 402 | `daily_cap` | The team's daily budget is used up. | Wait until midnight (UTC), or raise the budget. |
| 402 | `monthly_budget` | The team's monthly budget is used up. | Wait until next month, or raise the budget. |
| 413 | `body_too_large` | The request is larger than 20 MB. | Send less data. |
| 413 | `too_many_messages` | More than 200 messages in one request. | Shorten the conversation. |
| 413 | `input_too_large` | The question is longer than the team's input-token limit. | Shorten the question. |
| 413 | `too_many_images` | More images than the team allows. | Send fewer images. |
| 413 | `image_too_large` | An image is over 40 megapixels, or its size cannot be read. | Send a smaller image. |
| 400 | `provider_rejected` | The AI provider refused the request itself as invalid (for example a parameter value it does not accept) - Cordon does not try other models then, they would refuse it too - or every allowed model refused it for its own reason (for example a parameter that model does not support). | Fix the request or ask for another model; the message carries the reason. |
| 422 | - | Admin API only: a setting has an invalid value (a negative limit, text where a number belongs, holdout above 1, an infinite budget). The message names the field. | Fix the value. |
| 422 | `output_blocked` | The answer contained someone else's personal data or the hidden instructions, and the team is set to block that. | Ask again; check the system prompt and tools. |
| 429 | `rate_limit_exceeded` | Too many requests or tokens per minute. Comes with a `Retry-After` header. | Wait the number of seconds in `Retry-After`. |
| 429 | `auth_throttled` | Too many wrong keys from this IP address (10 in 5 minutes). | Wait 5 minutes and fix the key. |
| 429 | `agent_loop_detected` | An agent sent the same message over and over. | Fix the agent's loop. |
| 502 | `provider_error` | Every model on the ladder failed at the provider - or a routed answer did not pass the quality check and the strong model could not be reached (the doubted answer is never served). | Check the provider's status page; retry later. |
| 503 | *(none)*, header `X-Cordon-Error: security` | A security step crashed, so Cordon stopped the request rather than send it unprotected. | Check the server logs; this is a bug to fix. |
| 503 | *(none)*, header `X-Cordon-Error: internal` | An unexpected crash after the model call; the cost so far is booked and the rest of the budget reservation released. | Check the server logs; this is a bug to fix. |
| 503 | `server_busy` | More requests in flight than `server.max_in_flight` (config.yaml). Comes with `Retry-After: 1`. | Retry after a second; raise the limit or add capacity if it persists. |

---

## Design choices

- **No word-level editing of prompts.** Shrinking only removes *exact*
  duplicate sentences/paragraphs, never "unimportant" words - meaning is
  never touched. The system prompt is never shortened, because providers
  cache it on their side.
- **Images follow OpenAI's own token formula** (fit 2048×2048, shortest side
  768, 85 + 170 tokens per tile). Pixel size is checked before decoding;
  photos default to `detail: low`. Re-encoding strips EXIF/GPS data.
- **"Net savings" = what it would have cost minus what it did cost.**
  Routing savings are always marked *estimated*, because the expensive
  model was never actually called, so its cost can't be measured.
- **Fast check inline, heavy proof offline.** Only a light classifier runs
  during a request (well under 40 ms). The proof is offline: `evaluation/benchmark.py`
  and `evaluation/route_proof.py` on real answers, with statistical tests.
- **Savings fail open, security fails closed** (see the golden rule above).
- **Strict list of allowed request parameters** - stops a caller from, for
  example, redirecting Cordon's own provider keys to another server.
- **The cache never stores readable text.** Entries are AES-GCM encrypted,
  tied to the team (another team can't decrypt them), and looked up by a
  keyed hash.
- **Routing is based on evidence, and a route includes its check.** A cheaper model is used
  only for a request type where `evaluation/route_proof.py` proved the cascade - the cheaper
  model, with the check sending doubtful answers to the strong model - as good as the strong
  model (the rule is under [Evidence](#evidence)). Without that evidence the strong model is
  kept. Streams and requests with tools never take a route: they cannot be checked (a stream
  cannot be taken back, a tool call has no text). A routed request reserves budget for both
  models, and after the route's model and the strong model nothing else is tried. Each team
  can choose its own models.
- **A default `max_tokens` limit is never counted as a saving.**
- **One server process.** Rate limits, budgets and the circuit breaker are
  kept in memory; running several copies would need Redis for that state.

## What happens when something fails

| Failure | What Cordon does |
|---|---|
| A money-saving step crashes (shrink, images, cache, routing) | Request continues without that saving; logged. |
| A security step crashes (input/output protection) | Request is stopped with a 503 (default `security_fail_mode: closed`). |
| Any other unexpected crash, after access was checked | `closed` (default): 503. `open`: the request is sent straight to the requested model, unprotected, and marked `X-Cordon-Error: internal`. |
| The AI provider fails | Next model in line is tried; if all fail, 502. A routed request tries only the route's model and the strong model. |
| The quality check crashes, or a routed answer has no confidence to check | The strong model is asked (a route was only proven with its check); if it cannot answer, 502 - the doubted answer is never served or cached. |
| Rate limiter / budget / circuit breaker crashes | Request is allowed and logged (these protect cost, not data). |
| Cordon itself is down | The failsafe in front of Cordon sends each request Cordon does not answer straight to the provider, so the apps keep working - without Cordon's protection for those requests. They carry `X-Cordon-Failsafe: direct`. A refusal by Cordon is never sent direct. |

## Deploying

**A pilot, concretely:**

| Phase | What | Done when |
|---|---|---|
| Audit (weeks 1-2) | `tools/audit.py` reads an export of the company's AI logs offline - nothing installed | The report shows the savings potential and how much personal data leaves today |
| Observe (weeks 3-6) | Cordon in front of 1-2 teams, changing nothing, measuring what it *would* do | Real "would save" figures. Observe masks nothing, so start with teams without sensitive data |
| Enforce (week 7+) | Features on one by one, per team | Savings measured; no quality loss on the client's own questions (150+ per kind before a route) |

Before the pilot, on the client's side: a data-protection impact assessment (DPIA), OpenAI's EU
data processing addendum, and an agreement on who runs Cordon.

- **One Cordon per client**, in the client's own (preferably EU) infrastructure, with **the
  client's own AI keys** - the provider bills them directly. Each team or app gets its own key,
  settings, limits and data export.
- **Docker:** `docker compose up -d` (needs `OPENAI_API_KEY` in `.env`: the failsafe's direct path
  uses it). The container runs as user 10001 on a read-only file system; only `./data` (the
  database) is writable, so first run `mkdir -p data && sudo chown 10001 data`. Image build,
  start and failsafe re-verified locally on 15 October 2026 (Docker 29), and run in CI.
- **The failsafe (nginx) keeps the apps working when Cordon is down.** When Cordon gives no answer
  at all (down, unreachable, the connection drops), the failsafe sends that request straight to
  OpenAI with the client's own key. Any answer from Cordon - including a refusal - goes back
  unchanged. Requests sent direct are not protected and are marked `X-Cordon-Failsafe: direct`.
  Verified with `evaluation/failsafe_check.py` (13 checks) and the full `docker compose` stack
  against OpenAI. To agree with the client's IT: the direct path serves OpenAI models only; a
  Cordon that *hangs* is bypassed only after 5 minutes, and if it had already called the provider,
  that request is paid twice.
- **Behind HTTPS** (for example a Caddy reverse proxy).
- **Sizing:** one small server (2 vCPU, 2-4 GB RAM - Cordon uses about 750 MB after start, most of
  it the small helper models); a 3 GB image; about 90 requests per second on one process with raised
  capacity, measured with a simulated provider. Several copies need shared limits (Redis, not built).
- **Upkeep:** rotate keys, watch the dashboard's alerts, and re-run the benchmark when a provider
  changes its models.

## Evidence

All live runs used OpenAI (13-16 October 2026), about **EUR 10** together (the sum of each run's own
cost count). About 3,000 questions with known or reference answers, from public datasets (see
[Data sources](#data-sources)), drawn by `evaluation/fetch_eval_sets.py` and
`evaluation/company_day.py` in never-overlapping samples. Every answer is stored with its score in
`data/labelled_answers/`, so everything below is rebuilt without API calls. Rounds that decided a
route were committed (plan and decision rule) before any of their answers existed. The 5 live checks
(`pytest -m live`) pass, including billed cost within 5% of OpenAI's own usage.

### A measured working day

`evaluation/company_day.py`: 300 realistic requests - 100 texts to classify, 80 support questions
where popular ones come back (Zipf), 60 support chats sent with their history, 60 news articles to
summarize - each sent live straight to gpt-4o and through Cordon (enforce; trim, cache, routing,
answer-length cap, loop stop, masking). Real cost of both; quality of both scored the same way
(exact labels; ROUGE-L plus an LLM judge against the dataset's reference answer).

| Kind | Requests | EUR straight | EUR Cordon | Saved | Quality straight / Cordon |
|---|---|---|---|---|---|
| Classification | 100 | 0.0626 | 0.0173 | 72% | 0.93 / 0.95 |
| Support questions | 80 | 0.1932 | 0.0424 | 78% | 0.30 / 0.29 |
| Chats with history | 60 | 0.2153 | 0.0297 | 86% | 0.37 / 0.45 |
| Summaries | 60 | 0.1668 | 0.0415 | 75% | 0.46 / 0.46 |
| **Whole day** | 300 | 0.6380 | 0.1308 | **80%** | **0.56 / 0.58** |

Over the whole day Cordon's quality is no lower (95% interval of the difference +0.004 to +0.036).
Per kind the intervals are wide at 60-100 requests ("not proven" at this size), and the figures
move a few points between runs (support was 0.28 / 0.35 in the previous run); the routes
themselves are proven on larger rounds below. The same day saved 29% before the chat and summary
routes, and 54% before the conversation route. The day was measured just before a fix to the
"cut off" rule (below); the fix only lowers how often an answer is sent on.

### Answer quality through Cordon

Through Cordon (trimming and protection on) versus straight to gpt-4o: no measurable loss. The
first live run found two real problems, both fixed and re-measured: invoice dates were masked as
phone numbers, and the benchmark's parser misread answers that put "FINAL:" mid-line.

### Routing: where a cheaper model is good enough

What runs live is a *cascade*: the cheaper model answers, and a quality check sends every answer
it doubts to gpt-4o. `evaluation/route_proof.py` measures exactly that, per kind of request, for
every cheaper model at four check thresholds (0 = rules only, 0.5, 0.7, 0.9). **The rule:** at
least 30 questions; the average quality within 3 points of gpt-4o; and an exact (Clopper-Pearson)
bound that at most 3% of questions get a wrong or failed answer - at a confidence raised for every
model and threshold tried on the same questions (Bonferroni). On right/wrong questions any lower
score is a wrong answer. Graded answers (summaries, support replies, scored against a reference)
differ a little on almost every question even between two runs of gpt-4o, so they count towards
the average, and as a loss only when they fail outright. The routing table
(`src/config/routing_table.yaml`) is written by the proof and carries its evidence.

| Kind of request | Questions | Route | Quality vs gpt-4o | Cost vs gpt-4o |
|---|---|---|---|---|
| Classification | 729 | gpt-4.1-mini + check at 0.5 (17% sent on) | 91.1% vs 90.8% | 33% |
| Summaries (long texts) | 365 | gpt-4.1-mini | 0.465 vs 0.460 | 16% |
| Conversations (sent with earlier turns) | 375 | gpt-4.1-mini, rules only (1% sent on) | 0.457 vs 0.347 | 17% |
| Chat / general questions | 536 | gpt-4.1-mini + check at 0.5 (44% sent on) | 75.4% vs 75.0% | 60% |
| Arithmetic / reasoning | 429 | none proven | gpt-4o-mini 95% vs 97% (asked plainly) | - |
| Code | 189 | none proven | | - |

What these numbers do **not** show:

- **The rule for graded answers was changed after seeing data** (owner decision, commit message of
  1a72479): of 379 summaries and support chats, 9 scored clearly below gpt-4o - read one by one,
  none was wrong, all were different or less complete. Both routes were then **confirmed on fresh
  questions fixed in advance** (cefee7f): 200 new summaries (0.471 vs 0.468) and 200 general
  questions - open questions, advice, brainstorming, writing (0.401 vs 0.405, within the margin).
- **Conversations became their own kind after seeing data:** the chat route's 200 support chats
  showed gpt-4.1-mini doing well on requests sent with earlier turns, mixed in with general
  questions where it does not. The split (`routing.classify_request`: a chat with an earlier
  assistant reply is a conversation) was committed with its decision rule before a **fresh round of
  200 conversations** was answered (4922b26); on the 197 both models answered: 0.459 vs 0.339, a
  failed answer on at most 1.5%, 17% of the cost - passed. The conversations are support questions
  from one public dataset stitched into chats, scored by closeness to that dataset's reference
  answers; on a company's own chats the gap may be smaller.
- **Personal data changes what a model says:** with a masked email (`<EMAIL_1>`) in a support
  question, gpt-4.1-mini more often answers "I can't access your account"; the rules see a refusal
  and ask gpt-4o, so such a request costs *more* than going straight (both models are paid), not
  less quality. In six tries of one such prompt, three were sent on; on the measured day, 2 of 60
  conversations were. A pilot on real traffic shows the real share.
- **The "cut off" rule was fixed** (16 October): a complete answer ending in a short word and a
  question or exclamation mark ("Would you like that?") counted as cut off - 39 of 9,255 stored
  answers. Retraining left the ML model byte-identical; the routes stayed the same, and the share
  sent on dropped (conversations 1.9% to 0.8%, summaries 0.5% to 0%).
- **The quality check adds little on classification and summaries.** On gpt-4.1-mini's
  classification answers it separates right from wrong barely better than chance (AUC 0.58); the
  summary route runs on the rules only. Those routes hold because gpt-4.1-mini itself is about as
  good as gpt-4o there. On chats the check sends most answers on, so it saves little.
- **Cost** is priced from list prices; checked once against the real tokens of stored answers
  (27% real vs 28% list for an earlier classification setting).
- **An escalated answer counts as gpt-4o's stored answer** to that question; live it is a fresh call.
- **Arithmetic:** the first rounds ended with "Answer with just the number"; gpt-4o ignored it,
  cheaper models obeyed (gpt-4o-mini 61%). Asked plainly, gpt-4o-mini reaches 95% vs 97% on 276
  questions - close, not proven.
- **The judge is gpt-4o itself**, which if anything favours gpt-4o's own answers.
- **The classification questions were reused across rounds:** several models, thresholds and a
  retrained check were tried on the same 729 questions. Bonferroni corrects within one round, not
  across rounds; on the 400 fresh questions alone the routed setting (check at 0.5) did not pass,
  the stricter 0.7 did - the routed setting is proven only on the pooled 729.

### The quality check (the ML model)

Two layers: rules for unmistakable failures (empty, cut off, "N/A", refusal, error, invalid JSON),
then a classifier on the answer's shape, the kind of question, its format and the model's own
confidence (logprobs). Trained on 1,197 real answers (to 997 questions) of the cheaper models, never on a question that
proves a route (the model file records which it saw). Selection: the simplest model within 0.01
AUC of the best (Logistic Regression, then Random Forest, then Gradient Boosting). Full figures,
per kind of question: [`reports/model_card.md`](../reports/model_card.md); the comparison with an
MLP, a fine-tuned MiniLM transformer and gpt-4o-mini as a judge (none beat the classic models):
[`notebooks/quality_gate_real_answers.ipynb`](../notebooks/quality_gate_real_answers.ipynb).
Out of fold it separates right from wrong best on arithmetic (AUC 0.87) and worst on code (0.68);
on gpt-4.1-mini's classification answers it is near chance (0.58).

### Measured offline

| What | Result | How |
|---|---|---|
| Shrinking one long request | €0.01489 → €0.00897 (40% cheaper), a constructed example | `tools/demo.py`, trim scenario |
| One large invoice photo | 765 → 85 tokens (89% cheaper), location data removed | OpenAI's image-token formula |
| Time Cordon adds per request | 24 ms typical, 31 ms for the slowest 5% (mock provider, so Cordon's own time) | `evaluation/measure_overhead.py`, all features on, a MacBook |
| Under load | 1,000 requests at once, 1 s per provider call: all answered and booked exactly, about 90 per second on one process (`--worker-threads 256 --max-in-flight 1024`). At the default capacity 96 are served at once and the rest get a clean "busy, retry" (503) | `evaluation/load_test.py` |
| Hostile input | 10,000 random and malformed requests (5,000 per mode): no crash, every error in OpenAI's format | `CORDON_FUZZ_REQUESTS=5000 pytest tests/test_fuzz.py` (300 per mode normally) |
| Cordon itself down | The apps keep working through the failsafe | `evaluation/failsafe_check.py` |

**Semantic cache** - measured on 40,430 human-labelled Quora question pairs, with a tuning half
and a held-out half ([notebook](../notebooks/semantic_cache_threshold.ipynb)):

| At threshold 0.97 (held-out half) | Before | After adding the name guard |
|---|---|---|
| Hits that answer a different question (dataset labels) | 9.4% | 7.6% |
| Corrected for label errors (30 pairs re-read by an AI assistant, one reader, not blind; 95% range) | - | roughly 1-5% |
| Share of true duplicates caught | 11.7% | 9.8% |

No threshold reached the 1% target; 0.97 was kept, not selected. The remaining wrong hits are
reversed roles ("guys/girls", "vendors/customers") that similarity cannot see, so the semantic
cache is off by default and meant for FAQ-style traffic. The exact cache has no such risk.

### Data sources

The question text in `data/labelled_answers/` and `data/gate_eval/` comes from these public datasets,
downloaded on demand by `evaluation/fetch_eval_sets.py` and `evaluation/company_day.py` (the datasets themselves are not in the repo).
Used here for non-commercial evaluation only; each stays under its own terms.

| Dataset | Used for | Licence / terms |
|---|---|---|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (OpenAI) | arithmetic | MIT |
| [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) (Stanford, GLUE) | sentiment | research use, see the dataset card |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) (Google) | Python code | CC BY 4.0 |
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | news topics | non-commercial research use, see the dataset card |
| [Quora Question Pairs](https://huggingface.co/datasets/nyu-mll/glue) (GLUE QQP) | semantic cache threshold | Quora's terms; not stored in the repo |
| [Bitext customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | support questions and chats | CDLA-Sharing 1.0 |
| [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) | summaries | Apache 2.0 |
| [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (Databricks) | general questions | CC BY-SA 3.0 |

### Reproduce

```bash
.venv/bin/python -m pytest -m live -v                                  # 5 live checks, < 1 cent
.venv/bin/python -m evaluation.fetch_eval_sets                         # the evaluation set
.venv/bin/python -m evaluation.benchmark --live --golden-set data/external/extended_golden_set.jsonl
.venv/bin/python -m evaluation.fetch_eval_sets --extra agnews 200      # an extra round, disjoint from all others
.venv/bin/python -m evaluation.benchmark --collect gpt-4.1-mini --golden-set data/external/extra_agnews_200.jsonl
.venv/bin/python -m evaluation.train                                   # the quality check
.venv/bin/python -m evaluation.company_day --max-eur 6                 # the measured working day (live, about EUR 1.50)
.venv/bin/python -m evaluation.benchmark --routing-table-from data/labelled_answers/*.jsonl \
    --questions data/external/extended_golden_set.jsonl data/external/extra_*.jsonl   # no API calls
```

Without a key, `tools.demo` and the mock benchmark check the wiring only; their scores are
simulated and say nothing about quality.

## Security and privacy

Full details in **[SECURITY.md](../SECURITY.md)**. In short: keys are stored
hashed, the cache is encrypted and separated per team, personal data is
masked in enforce mode, manipulation attempts are detected (and optionally
blocked), and every new team starts with nothing switched on. Note: a new
team starts in observe mode, which only measures - personal data is *not*
masked until the team is switched to enforce.

**EU AI Act (not legal advice):** Cordon itself is not a high-risk AI
system - it is infrastructure. Its per-request records (what was sent,
blocked, masked) *support* a client's own record-keeping and human-oversight
duties, but do not fulfil them on their own.

## Topics covered

| Topic | Where |
|---|---|
| Token & cost optimization, context engineering | `optimize.py`, `cache.py`, `costing.py` |
| Model routing, fallbacks & reliability | `routing.py`, `pipeline/step7_generate.py`, `deploy/failsafe.conf.template` |
| Latency optimization | caching, streaming, overhead measurement |
| Guardrails, prompt-injection defense | `guardrails.py` |
| Structured outputs | `quality.py` (JSON schema) |
| Eval sets, LLM-as-judge, human-in-the-loop | `benchmark.py` |
| Observability & monitoring | `telemetry.py`, dashboard |
| Security | `security.py`, `SECURITY.md` |
| Data governance, GDPR / EU AI Act, data residency | `store.py`, `guardrails.py`, provider allowlist, `SECURITY.md` |
| CI/CD for AI | `tools/ci.sh` |
| AI infrastructure | `Dockerfile`, `docker-compose.yml` |

**Out of scope** (belongs in the client's own app): RAG, chunking,
reranking, vector databases, multi-agent orchestration, MCP, fine-tuning the
client's own LLMs.

## Why not LiteLLM Proxy, Helicone or Portkey?

Cordon uses LiteLLM internally, so it builds *on top of* it rather than
replacing it.

- **LiteLLM Proxy** - routing, fallbacks, spend tracking. No prompt
  shrinking, image-aware downscaling, personal-data masking, injection
  detection, benchmark-based routing or offline audit. More general;
  supports more providers.
- **Helicone** - mainly logging and dashboards (much richer than Cordon's).
  Doesn't optimize, mask or guard.
- **Portkey** - closest in ambition, as a hosted commercial platform.
  Cordon is small, self-hosted, one instance per client, and carries its
  own harness to measure, on the client's own questions, whether quality is preserved.

Cordon's niche: a **consultant's tool** - one small, inspectable instance
per client, able to measure (not just promise) that the savings don't
hurt answer quality.

## Considered and left out

| Idea | What it would have done | Why it was left out |
|---|---|---|
| Custom prompt-injection classifier | Fine-tune a small transformer to replace the phrase list. | Good off-the-shelf models exist (Meta Prompt Guard, deepset's DeBERTa models); a custom one only makes sense benchmarked against those, and it adds a transformer to every request. |
| Custom name and address detection (NER) | Detect names and addresses as personal data. | Microsoft Presidio already does this well; plugging it in is the right move. |
| Local model fine-tuned with QLoRA | A small open model for one narrow task, as a free pool model. | QLoRA's 4-bit training relies on an NVIDIA GPU; a local model adds a component to maintain alone. |
| Spend forecasting | Predict the monthly bill from the ledger. | Needs months of real traffic; on demo data it would only fit noise. |
| Power BI as the dashboard | A richer dashboard. | No Mac version and a licence per viewer; `/admin/export` CSV feeds it for clients who want it. |
| A per-employee view | Cost per person, for training needs. | Monitoring staff needs a stated purpose, informed employees and, in Belgium, the works council (CAO 81). |

## Limitations and next steps

- **Live evidence covers one provider** (OpenAI) and public question sets; Mistral is supported
  but not live-verified. A client's own traffic - 150+ questions per kind of request - is the
  real test. The limits of the routing evidence are listed under [Evidence](#evidence).
- **Personal data:** structured identifiers only (email, phone, IBAN, national ID, card numbers),
  not names or addresses. The app's own system message is only masked when a team turns on
  `mask_instructions` (for RAG apps that paste documents there).
- **Manipulation detection** is phrase-based (EN/NL/FR); new phrasings can slip through.
- **No harmful-content filtering** on answers - pair with a moderation API if needed.
- **The quality check** judges the signs around an answer, not the fact, and needs the model's
  confidence (OpenAI logprobs).
- **One process:** rate limits, budgets and the loop detector live in memory (budgets are restored
  from the ledger at restart); several copies need Redis.
- **Semantic cache:** roughly 1-5% wrong answers on a hard benchmark; off by default. A
  cross-encoder re-check is the next step if it must become safer.
- `requirements.txt` pins versions but not hashes.
- **Next, each measurable with the same tools:** a small pairwise judge model (such as TypeSafe's JEV) instead of gpt-4o as the evaluation judge, if it holds up on our own labelled answers; an arithmetic round without "just the number";
  a batch mode for work that can wait (half price; `audit.py` already spots candidates); an
  optional "answer concisely" instruction, kept only if quality holds; a periodic benchmark
  re-run to catch a provider's quality drifting.
