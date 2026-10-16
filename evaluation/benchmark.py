"""
The quality proof: do Cordon's optimizations and routing keep answers as good?

- Four conditions per golden-set item: A strong model, original prompt (baseline);
  B strong model through Cordon; C every pool model; D the full pipeline.
- Category scores (field F1, exact FINAL answer, pass@1, ROUGE-L + judge) and
  a paired bootstrap per category -> pass / warn / fail (pass needs >= 30 items).
- `--write-routing-table`: routes proven together with their quality check (evaluation/route_proof.py).
- Mock mode checks the wiring only; real conclusions need `--live`.

    .venv/bin/python -m evaluation.benchmark            # mock
    .venv/bin/python -m evaluation.benchmark --live --limit 3 --max-eur 5
    .venv/bin/python -m evaluation.benchmark --collect gpt-4o-mini --golden-set data/external/extended_golden_set.jsonl
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import litellm
import numpy as np
from sklearn.metrics import cohen_kappa_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cordon.benchmark")

SEED = 42
from src.paths import GOLDEN_SET_PATH, REPO_ROOT
HUMAN_REVIEW_PATH = REPO_ROOT / "reports" / "human_review.csv"
REPORT_MD_PATH = REPO_ROOT / "reports" / "benchmark_report.md"
REPORT_HTML_PATH = REPO_ROOT / "reports" / "benchmark_report.html"
LIVE_RESULTS_PATH = REPO_ROOT / "reports" / "live_results.json"
# Every live answer with its label: the training data for the quality gate (evaluation/train.py).
# Only benchmark questions (no client data), so it is kept in the repo for reproducible training.
LABELLED_ANSWERS_DIR = REPO_ROOT / "data" / "labelled_answers"

CATEGORIES = ("factual", "extraction", "summarization", "classification", "code", "reasoning")
CONDITIONS = ("A", "B", "C", "D")

# --- Section: mock mode - deterministic answers with a known error rate per condition ---
MOCK_ERROR_RATE_A = 0.05
MOCK_ERROR_RATE_B = 0.07
MOCK_ERROR_RATE_D = 0.10
MOCK_ERROR_RATE_C_BY_ALIAS = {"mistral-small": 0.25, "gpt-4o-mini": 0.15, "mistral-large": 0.10, "gpt-4o": 0.05}

BOOTSTRAP_RESAMPLES = 2000
VERDICT_MARGIN = 0.03
MIN_ITEMS_FOR_PASS = 30  # fewer paired items can show a large loss, but never prove "no loss"


# --- Section: golden set loading ---
def load_golden_set(path: Path, limit: int | None = None) -> list[dict]:
    """
    Load the golden set from JSONL.

    Parameters: path - path to a JSONL file (one JSON object per line).
                 limit - if set, the first `limit` items of every category, so
                 even a small smoke run covers each category.
    Returns: a list of item dicts (id, category, messages, expected, ...).
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    for item in items:
        if item.get("category") in FINAL_CATEGORIES:
            item["messages"] = _with_final_instruction(item["messages"])
    if limit is not None:
        seen: dict[str, int] = {}
        kept = []
        for item in items:
            seen[item["category"]] = seen.get(item["category"], 0) + 1
            if seen[item["category"]] <= limit:
                kept.append(item)
        items = kept
    return items


def _with_final_instruction(messages: list[dict]) -> list[dict]:
    """A copy of the messages with FINAL_INSTRUCTION added to the last user message."""
    copied = [dict(m) for m in messages]
    for message in reversed(copied):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            message["content"] = f"{message['content']}\n\n{FINAL_INSTRUCTION}"
            break
    return copied


# --- Section: deterministic mock-mode simulation ---
def _stable_draw(item_id: str, tag: str, seed: int = SEED) -> float:
    """A stable [0, 1) draw for one (item, tag), via a hash - never Python's random module, so repeated runs with the same seed always agree."""
    digest = hashlib.sha256(f"{seed}:{item_id}:{tag}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _simulate_success(item_id: str, condition: str, alias: str, seed: int = SEED) -> bool:
    """
    Deterministically decide success/failure for one (item, condition, model) in mock mode.

    Parameters: item_id, condition ("A"|"B"|"C"|"D"), alias, seed.
    Returns: bool.
    Why B and D are correlated with A rather than independent: B and D are
    "A, plus Cordon's processing" - modeling their failures as fully
    independent draws lets D get randomly *luckier* than A on a small
    sample purely by chance, which is not just unrealistic but would make a
    mock-mode CI run **deterministically** red forever at a fixed seed (not
    flaky - permanently broken), since the same seed always redraws the
    same unlucky outcome. Gating B/D's extra failure chance on A already
    having succeeded keeps D "at least as good as A, minus a small
    Cordon-specific risk" - which is what these conditions are actually
    meant to represent. C (a genuinely different, independent model) is not
    gated this way; comparing it to A's independent draw is specifically
    the point of condition C.
    """
    if condition in ("A", "C"):
        error_rate = MOCK_ERROR_RATE_A if condition == "A" else MOCK_ERROR_RATE_C_BY_ALIAS.get(alias, 0.15)
        return _stable_draw(item_id, f"{condition}:{alias}", seed) >= error_rate

    baseline_success = _stable_draw(item_id, f"A:{alias}", seed) >= MOCK_ERROR_RATE_A
    if not baseline_success:
        return False
    extra_error_rate = max(0.0, (MOCK_ERROR_RATE_B if condition == "B" else MOCK_ERROR_RATE_D) - MOCK_ERROR_RATE_A)
    return _stable_draw(item_id, f"{condition}:{alias}", seed) >= extra_error_rate


def _mock_answer_text(item: dict, success: bool) -> str:
    """Build a deterministic mock answer for one item: the correct-shaped answer on success, a clearly wrong/empty one on failure."""
    category = item["category"]
    if category in FINAL_CATEGORIES:
        return f"FINAL: {item['expected'] if success else 'N/A'}"
    if not success:
        return "N/A" if category != "code" else "```python\ndef placeholder():\n    pass\n```"
    if category == "extraction":
        return json.dumps(item["expected"])
    if category == "code":
        return f"```python\n{item['solution']}\n```"
    return str(item["expected"])


# --- Section: per-category scoring ---
def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two word lists (used by rouge_l)."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(reference: str, candidate: str) -> float:
    """Word-level ROUGE-L F1 between a reference and a candidate text. Returns 0.0 if either is empty."""
    ref_words, cand_words = reference.lower().split(), candidate.lower().split()
    if not ref_words or not cand_words:
        return 0.0
    lcs = _lcs_length(ref_words, cand_words)
    precision, recall = lcs / len(cand_words), lcs / len(ref_words)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_extraction(expected: dict, answer_text: str) -> float:
    """Per-field F1 with numeric tolerance (1% relative, min 0.01 absolute) against a JSON object parsed from the answer."""
    parsed = None
    try:
        parsed = json.loads(answer_text)
    except ValueError:
        match = re.search(r"\{.*\}", answer_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except ValueError:
                parsed = None
    if not isinstance(parsed, dict):
        return 0.0
    correct = 0
    for field_name, expected_value in expected.items():
        got = parsed.get(field_name)
        if isinstance(expected_value, (int, float)) and isinstance(got, (int, float)):
            if abs(got - expected_value) <= max(0.01, abs(expected_value) * 0.01):
                correct += 1
        elif str(got).strip().lower() == str(expected_value).strip().lower():
            correct += 1
    return correct / len(expected)


_CODE_BLOCK_REGEX = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


CODE_TIMEOUT_SECONDS = 10
_TESTS_PASSED = "__cordon_tests_passed__"


def score_code(expected_test: str, answer_text: str, timeout: float = CODE_TIMEOUT_SECONDS) -> float:
    """
    pass@1: run the answer's code block, then the expected assertions, in a separate Python process.

    Parameters: expected_test - Python assert statements.
                 answer_text - the model's raw answer (expected to contain a ```python block).
                 timeout - seconds before the run counts as a failure (an endless loop).
    Returns: 1.0 if every assertion passes, 0.0 on any error or timeout.
    Why a separate process: model-written code may loop forever, exit, or
    crash the interpreter; none of that can reach the benchmark itself. It is
    still not a sandbox (the code can use the file system and network) - an
    operator-run harness on benchmark questions, as SECURITY.md notes.
    """
    match = _CODE_BLOCK_REGEX.search(answer_text)
    # The marker is printed only after every assertion ran: code that exits early never prints it.
    program = (match.group(1) if match else answer_text) + "\n\n" + expected_test + f"\nprint({_TESTS_PASSED!r})\n"
    with tempfile.TemporaryDirectory() as workdir:
        try:
            done = subprocess.run([sys.executable, "-I", "-c", program], cwd=workdir, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 0.0
    return 1.0 if done.returncode == 0 and _TESTS_PASSED in done.stdout else 0.0


# --- Section: the FINAL-line answer protocol (classification, factual, reasoning) ---
# Free-text answers ("not negative; it is positive", "4 euros (10 - 6)") cannot be
# scored reliably by rules - every rule invites the next edge case. So these items
# ask the model to end with one fixed line, and only that line is scored.
FINAL_CATEGORIES = ("classification", "factual", "reasoning")
FINAL_INSTRUCTION = "End your reply with one line in exactly this form: FINAL: <your answer>"
# "FINAL:" anywhere on a line - models often end the sentence with it ("... Paris. FINAL: Paris")
# instead of starting a new line - possibly in markdown bold; the answer runs to the end of that line.
_FINAL_REGEX = re.compile(r"\bFINAL\s*\**\s*:\s*\**\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUMBER_REGEX = re.compile(r"-?\d+(?:\.\d+)?")
_NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                 "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def extract_final(answer_text: str) -> str | None:
    """The answer after the last `FINAL:`, up to the end of its line, without markdown or trailing punctuation; None if absent."""
    matches = _FINAL_REGEX.findall(answer_text or "")
    return matches[-1].strip().strip("*`\"'").rstrip(".!").strip() if matches else None


def _normalized(text: str) -> str:
    """Lowercase, words joined by single underscores ("Not spam" -> "not_spam")."""
    return "_".join(re.findall(r"[a-z0-9]+", text.lower()))


def _as_number(text: str) -> float | None:
    """The first number in a text, number words included ("seven" -> 7); None if there is none."""
    match = _NUMBER_REGEX.search(text.replace(",", ""))
    if match:
        return float(match.group())
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _NUMBER_WORDS:
            return float(_NUMBER_WORDS[word])
    return None


def score_final(expected: str, answer_text: str, category: str) -> float:
    """
    Score an answer by its FINAL line only.

    classification - the label must equal the expected one ("Not spam" = "not_spam");
    a numeric expected answer - the number on the FINAL line must equal it;
    any other answer - the expected words must appear as whole words on the FINAL line.
    No FINAL line scores 0 (the report counts these as format misses).
    """
    final = extract_final(answer_text)
    if final is None:
        return 0.0
    if category == "classification":
        return 1.0 if _normalized(final) == _normalized(expected) else 0.0
    expected_number = _as_number(expected) if _NUMBER_REGEX.fullmatch(expected.strip()) or expected.lower() in _NUMBER_WORDS else None
    if expected_number is not None:
        return 1.0 if _as_number(final) == expected_number else 0.0
    return 1.0 if re.search(rf"(?<!\w){re.escape(expected.strip().lower())}(?!\w)", final.lower()) else 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, 0.0 on a zero vector (avoids importing quality.py just for this one helper in non-embedding callers)."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def heuristic_judge(reference: str, answer_text: str) -> float:
    """Deterministic mock-mode judge: word-overlap-based 1-5 rubric, rescaled to 0-1."""
    reference_words = set(reference.lower().split())
    if not reference_words:
        return 0.0
    overlap = len(reference_words & set(answer_text.lower().split())) / len(reference_words)
    rubric = 1 + round(min(overlap, 1.0) * 4)
    return (rubric - 1) / 4.0


class JudgeFailed(Exception):
    """The LLM judge gave no usable 1-5 rating: the answer is scored as a failed call (0), never as a neutral 3."""


def llm_judge(strong_litellm_name: str, reference: str, answer_text: str, config: dict, on_cost=None) -> float:
    """
    Real LLM-as-judge: a 1-5 faithfulness rubric call to the strong model, rescaled to 0-1.

    Parameters: config - config.yaml (for the provider timeout and pricing).
    on_cost - optional callback receiving the judge call's cost in EUR, so
    the run's cost guard counts judge calls too.
    Raises: JudgeFailed when the call fails or the reply holds no 1-5 rating. A
    neutral default would hand both sides of a comparison the same score and
    push the verdict toward "pass".
    """
    from src.savings import costing

    prompt = (
        "Rate how faithful this summary is to the reference on a scale of 1 to 5 "
        "(5 = fully faithful, 1 = not faithful at all). Respond with only the number.\n\n"
        f"Reference: {reference}\nSummary: {answer_text}"
    )
    try:
        response = litellm.completion(
            model=strong_litellm_name, messages=[{"role": "user", "content": prompt}],
            timeout=config["provider"]["timeout_seconds"],
        )
        if on_cost is not None:
            on_cost(costing.actual_cost_eur(
                strong_litellm_name, response.usage.prompt_tokens, response.usage.completion_tokens, config
            ))
    except Exception as exc:
        raise JudgeFailed(f"judge call failed: {exc}") from exc
    match = re.search(r"[1-5]", response.choices[0].message.content or "")
    if not match:
        raise JudgeFailed("judge reply holds no 1-5 rating")
    return (int(match.group()) - 1) / 4.0


def score_summarization(reference: str, answer_text: str, judge_fn, embed_fn) -> float:
    """Mean of ROUGE-L, embedding cosine similarity, and judge faithfulness."""
    rouge = rouge_l(reference, answer_text)
    try:
        vectors = embed_fn([reference, answer_text])
        cosine = max(0.0, _cosine(vectors[0], vectors[1]))
    except Exception:
        cosine = 0.0
    faithfulness = judge_fn(reference, answer_text)
    return (rouge + cosine + faithfulness) / 3.0


def score_item(item: dict, answer_text: str, judge_fn, embed_fn) -> float:
    """Dispatch to the category-specific scorer. Parameters: item (golden-set entry), answer_text, judge_fn, embed_fn. Returns: a 0-1 score."""
    category = item["category"]
    if category == "extraction":
        return score_extraction(item["expected"], answer_text)
    if category == "code":
        return score_code(item["expected_test"], answer_text)
    if category == "summarization":
        return score_summarization(item["expected"], answer_text, judge_fn, embed_fn)
    return score_final(str(item["expected"]), answer_text, category)


# --- Section: one run's outcome ---
@dataclass
class RunResult:
    """One (item, condition[, model]) call's outcome."""

    answer_text: str = ""
    cost_eur: float = 0.0
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    served_model: str | None = None
    error: str | None = None
    score: float = 0.0
    valid_json: bool | None = None
    refused: bool = False
    confidence: dict | None = None  # quality.answer_confidence(): the model's own certainty (live OpenAI calls)


_REFUSAL_REGEX = re.compile(r"\b(i cannot|i can['’]t|i['’]m not able|i am unable)\b", re.IGNORECASE)


def _check_valid_json(item: dict, answer_text: str) -> bool | None:
    """True/False if item requested a response_format and the answer is/isn't valid JSON; None if no schema was requested."""
    if not item.get("response_format"):
        return None
    try:
        json.loads(answer_text)
        return True
    except ValueError:
        match = re.search(r"\{.*\}", answer_text, re.DOTALL)
        if match:
            try:
                json.loads(match.group(0))
                return True
            except ValueError:
                pass
        return False


def _finalize_result(item: dict, result: RunResult, condition: str, alias: str, mock_mode: bool, judge_fn, embed_fn) -> RunResult:
    """Fill in score/valid_json/refused on a RunResult that already has answer_text/cost/latency/tokens set."""
    if result.error:
        return result
    result.valid_json = _check_valid_json(item, result.answer_text)
    result.refused = bool(_REFUSAL_REGEX.search(result.answer_text))
    if mock_mode:
        result.score = 1.0 if _simulate_success(item["id"], condition, alias) else 0.0
        return result
    try:
        result.score = score_item(item, result.answer_text, judge_fn, embed_fn)
    except JudgeFailed as exc:
        logger.warning("%s on %s (%s): scored as a failed call", exc, item["id"], condition)
        result.error, result.score = "judge_failed", 0.0
    return result


# --- Section: running the four conditions ---
def call_direct(litellm_name: str, item: dict, condition: str, alias: str, mock_mode: bool, config: dict) -> RunResult:
    """
    Call a model directly through LiteLLM, bypassing Cordon entirely -
    conditions A (strong, original prompt) and C (each pool model, original prompt).

    Parameters: litellm_name - the LiteLLM model name. item - the golden-set
    entry. condition - "A" or "C" (used only for the mock-mode error-rate
    lookup and the deterministic mock answer). alias - the model's config.yaml
    alias (same reason). mock_mode, config.
    Returns: a RunResult with answer_text/cost_eur/latency_ms/tokens set
    (score/valid_json/refused are filled in by the caller via _finalize_result).
    """
    from src.savings import costing

    kwargs: dict = {"model": litellm_name, "messages": item["messages"], "timeout": config["provider"]["timeout_seconds"]}
    if item.get("response_format"):
        kwargs["response_format"] = item["response_format"]
    if mock_mode:
        success = _simulate_success(item["id"], condition, alias)
        kwargs["mock_response"] = _mock_answer_text(item, success)
    elif litellm_name.startswith("openai/"):
        kwargs["logprobs"] = True  # the model's certainty per token: training data for the quality gate

    start = time.perf_counter()
    try:
        response = litellm.completion(**kwargs)
    except Exception as exc:
        return RunResult(error=str(exc), latency_ms=(time.perf_counter() - start) * 1000)
    latency_ms = (time.perf_counter() - start) * 1000

    answer_text = response.choices[0].message.content or ""
    usage = response.usage
    if mock_mode:
        prompt_tokens = litellm.token_counter(model=litellm_name, messages=item["messages"])
        completion_tokens = litellm.token_counter(model=litellm_name, text=answer_text) if answer_text else 0
        usage.prompt_tokens, usage.completion_tokens = prompt_tokens, completion_tokens
    cost_eur = costing.actual_cost_eur(litellm_name, usage.prompt_tokens, usage.completion_tokens, config)
    from src.savings import quality

    return RunResult(
        answer_text=answer_text, cost_eur=cost_eur, latency_ms=latency_ms,
        tokens_in=usage.prompt_tokens, tokens_out=usage.completion_tokens, served_model=alias,
        confidence=None if mock_mode else quality.answer_confidence(response),
    )


def call_through_cordon(client, headers: dict, item: dict, strong_alias: str, condition: str, mock_mode: bool) -> RunResult:
    """
    Call the model through the real (in-process) Cordon pipeline - conditions
    B (optimizations + guardrails, no routing) and D (the full pipeline).

    Parameters: client - a TestClient wrapping create_app()'s ASGI app.
                 headers - the benchmark tenant's Authorization header.
                 item - the golden-set entry.
                 strong_alias - the alias requested (Cordon resolves routing/
                 downgrade itself for condition D; B's tenant has routing off).
                 condition - "B" or "D" (mock-mode error-rate lookup only).
                 mock_mode.
    Returns: a RunResult.
    """
    body: dict = {"model": strong_alias, "messages": item["messages"]}
    if item.get("response_format"):
        body["response_format"] = item["response_format"]

    request_headers = dict(headers)
    if mock_mode and not _simulate_success(item["id"], condition, strong_alias):
        request_headers["X-Mock-Bad-Response"] = "true"

    start = time.perf_counter()
    response = client.post("/v1/chat/completions", headers=request_headers, json=body)
    latency_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        return RunResult(error=f"http {response.status_code}: {response.text[:200]}", latency_ms=latency_ms)

    data = response.json()
    answer_text = data["choices"][0]["message"]["content"] or ""
    cost_eur = float(response.headers.get("X-Cordon-Cost", 0.0))
    served_model = response.headers.get("X-Cordon-Route") or strong_alias
    usage = data.get("usage") or {}
    return RunResult(
        answer_text=answer_text, cost_eur=cost_eur, latency_ms=latency_ms,
        tokens_in=usage.get("prompt_tokens", 0), tokens_out=usage.get("completion_tokens", 0), served_model=served_model,
    )


# --- Section: statistics — paired bootstrap and the pass/warn/fail verdict ---
def paired_bootstrap_ci(diffs: list[float], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED,
                        confidence: float = 0.95) -> tuple[float, float]:
    """
    Bootstrap confidence interval (95% by default) for the mean of paired per-item differences.

    Parameters: diffs - per-item (condition - A) score differences.
                 resamples, seed. confidence - the interval's two-sided level.
    Returns: (lower, upper) bound of the interval. (0.0, 0.0) if diffs is empty.
    """
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    tail = (1.0 - confidence) / 2
    lower = means[int(tail * resamples)]
    upper = means[min(int((1.0 - tail) * resamples), resamples - 1)]
    return lower, upper


def verdict_from_ci(lower: float, upper: float, n: int | None = None, margin: float = VERDICT_MARGIN) -> str:
    """
    Turn a confidence interval for (condition - A) into pass / fail / warn.

    fail: even the upper bound is below -margin (clearly worse).
    pass: the lower bound clears -margin AND there are at least
          MIN_ITEMS_FOR_PASS paired items. With fewer, identical scores give a
          bootstrap interval of (0, 0) that looks like proof but is not: on 10
          items that all agree, the real loss could still be about 26%.
    warn: anything else - inconclusive.
    n=None skips the sample-size rule (for callers that checked n themselves).
    """
    if upper < -margin:
        return "fail"
    if lower > -margin and (n is None or n >= MIN_ITEMS_FOR_PASS):
        return "pass"
    return "warn"


def loss_rate_bound(diffs: list[float], confidence: float = 0.95) -> float:
    """
    Exact (Clopper-Pearson) one-sided upper bound on the share of items where the condition scored lower.

    Why next to the bootstrap: with no or very few losses the bootstrap interval collapses to
    the observed values (0 losses in 70 gives [0, 0+]) and looks like proof; this bound says how
    many losses could still hide in a sample that size (0 in 70: up to 4.2%).
    """
    from scipy.stats import beta

    losses, n = sum(1 for d in diffs if d < 0), len(diffs)
    return 1.0 if n == 0 else float(beta.ppf(confidence, losses + 1, n - losses)) if losses < n else 1.0


@dataclass
class PairedVerdict:
    """A condition against A on paired items: n, mean difference, 95% bootstrap CI, the loss-rate bound, verdict."""

    n: int
    mean_diff: float
    lower: float
    upper: float
    loss_bound: float
    verdict: str


def paired_verdict(diffs: list[float], margin: float = VERDICT_MARGIN, confidence: float = 0.95,
                   loss_diffs: list[float] | None = None) -> PairedVerdict:
    """
    The one verdict rule for every comparison in the benchmark.

    pass needs all three: at least MIN_ITEMS_FOR_PASS items, the bootstrap lower bound above
    -margin, and the exact loss-rate bound within the margin (in practice >= ~100 items when
    the cheaper model never loses). fail: the bootstrap upper bound is below -margin.
    confidence - raised when several settings are tried on the same answers (route_proof.py).
    """
    lower, upper = paired_bootstrap_ci(diffs, confidence=confidence)
    # By default every lower score is a loss; route_proof passes graded differences as 0 (see loss_diff).
    bound = loss_rate_bound(diffs if loss_diffs is None else loss_diffs, confidence)
    verdict = verdict_from_ci(lower, upper, len(diffs), margin)
    if verdict == "pass" and bound > margin:
        verdict = "warn"
    return PairedVerdict(len(diffs), sum(diffs) / len(diffs), lower, upper, bound, verdict)


VERDICT_SYMBOLS = {"pass": "✅", "fail": "❌", "warn": "⚠️"}


# --- Section: human-in-the-loop review ---
def export_human_review(judged_items: list[dict], path: Path = HUMAN_REVIEW_PATH, n: int = 20, seed: int = SEED) -> None:
    """
    Export a deterministic sample of summarization items for human review.

    Parameters: judged_items - [{"id", "reference", "answer", "judge_score"}, ...]
    (summarization items only - faithfulness rating is only defined there).
                 path - output CSV path.
                 n - sample size.
                 seed - sampling seed.
    Returns: None. Writes columns: id, reference, answer, judge_score,
    human_score (blank, for the reviewer to fill in 1-5).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sample = judged_items if len(judged_items) <= n else rng.sample(judged_items, n)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "reference", "answer", "judge_score_1_5", "human_score_1_5"])
        for row in sample:
            judge_1_5 = round(row["judge_score"] * 4 + 1)
            writer.writerow([row["id"], row["reference"], row["answer"], judge_1_5, ""])
    logger.info("wrote %d items for human review: %s", len(sample), path)


def compute_kappa(human_review_path: Path) -> float | None:
    """
    Read a completed human_review.csv and compute quadratic-weighted Cohen's kappa.

    Parameters: human_review_path - path to a human_review.csv with human_score_1_5 filled in.
    Returns: the kappa value, or None if no rows have a human score filled in.
    """
    judge_scores, human_scores = [], []
    with open(human_review_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human_raw = (row.get("human_score_1_5") or "").strip()
            if not human_raw:
                continue
            judge_scores.append(int(row["judge_score_1_5"]))
            human_scores.append(int(human_raw))
    if not judge_scores:
        return None
    return float(cohen_kappa_score(judge_scores, human_scores, weights="quadratic"))


# --- Section: report writing ---
def _render_markdown(summary: dict) -> str:
    """Render the benchmark report as Markdown. Parameters: summary - the full results dict built by run_benchmark(). Returns: the report text."""
    lines = [
        "# Cordon benchmark report",
        "",
        f"Mode: **{'live' if summary['live'] else 'mock'}**. "
        + ("Real conclusions come only from `--live` runs." if not summary["live"] else "These are real model results."),
        "",
        "## Quality verdicts per category (paired bootstrap, 2,000 resamples, seed 42)",
        "",
        f"A ✅ needs at least {MIN_ITEMS_FOR_PASS} paired items, so with fewer every verdict is ⚠️ "
        "(inconclusive) or ❌ by rule - a client's own golden set should use 50-100 items per "
        "category for a conclusive result.",
        "",
        "| Category | n | D − A verdict | D − A mean | B − A verdict | B − A mean |",
        "|---|---|---|---|---|---|",
    ]
    for category in CATEGORIES:
        row = summary["category_verdicts"].get(category)
        if not row:
            continue
        lines.append(
            f"| {category} | {row['n']} | {VERDICT_SYMBOLS[row['d_verdict']]} | {row['d_mean']:+.3f} | "
            f"{VERDICT_SYMBOLS[row['b_verdict']]} | {row['b_mean']:+.3f} |"
        )

    lines += [
        "",
        "## Model x category quality and cost matrix (condition C: each pool model, original prompt)",
        "",
        "| Model | " + " | ".join(CATEGORIES) + " | mean cost/item (EUR) |",
        "|---|" + "---|" * (len(CATEGORIES) + 1),
    ]
    for alias in summary["pool_aliases"]:
        cells = []
        for category in CATEGORIES:
            cell = summary["model_category_matrix"].get(alias, {}).get(category)
            cells.append(f"{cell:.2f}" if cell is not None else "-")
        mean_cost = summary["model_mean_cost"].get(alias, 0.0)
        lines.append(f"| {alias} | " + " | ".join(cells) + f" | {mean_cost:.6f} |")

    lines += [
        "",
        "## Quality gate",
        "",
        "Trained and evaluated on these labelled answers by `evaluation/train.py` - see `reports/model_card.md`.",
        "",
        "## Other metrics",
        "",
        f"- Valid JSON rate (extraction items): {summary['valid_json_rate']:.1%}",
        f"- Refusal rate: {summary['refusal_rate']:.1%}",
        f"- Cost per correct answer (condition D, total cost / correct answers): EUR {summary['cost_per_correct_d']:.6f}",
        "- Failed calls (scored 0): " + ", ".join(f"{c}: {n}" for c, n in summary.get("error_counts", {}).items()),
        "- Answers without the required FINAL line (scored 0): " + ", ".join(f"{c}: {n}" for c, n in summary.get("format_misses", {}).items()),
        f"- Mean latency (condition D): {summary['mean_latency_d_ms']:.1f} ms",
    ]
    if summary.get("kappa") is not None:
        lines.append(f"- Human-vs-judge quadratic-weighted Cohen's kappa: {summary['kappa']:.3f}")
    return "\n".join(lines) + "\n"


def _render_html(summary: dict, markdown_body: str) -> str:
    """Wrap the markdown report in a minimal, dependency-free HTML page (no templating library needed for this simple a layout)."""
    import html

    escaped = html.escape(markdown_body)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Cordon benchmark report</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 920px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
pre {{ white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.9rem; line-height: 1.5; }}
</style></head>
<body><pre>{escaped}</pre></body></html>
"""


def write_reports(summary: dict) -> None:
    """Write reports/benchmark_report.md and .html. Parameters: summary. Returns: None."""
    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    markdown_body = _render_markdown(summary)
    REPORT_MD_PATH.write_text(markdown_body, encoding="utf-8")
    REPORT_HTML_PATH.write_text(_render_html(summary, markdown_body), encoding="utf-8")
    logger.info("wrote %s and %s", REPORT_MD_PATH, REPORT_HTML_PATH)


# --- Section: main orchestration ---
class _BudgetGuard:
    """Tracks live-mode spend and stops starting new calls before the next one would pass --max-eur."""

    def __init__(self, max_eur: float | None, safety_share: float = 0.10) -> None:
        """
        Parameters: max_eur - the run's cost ceiling (None = no limit, mock mode).
        safety_share - stop this share of the ceiling early (at most EUR 0.50), since the next call's
        exact cost is not known in advance. A fixed margin made small ceilings (EUR 0.50) spend nothing.
        """
        self.max_eur = max_eur
        self.safety_margin_eur = min(0.50, safety_share * max_eur) if max_eur else 0.0
        self.spent_eur = 0.0

    def record(self, cost_eur: float) -> None:
        """Add a completed call's cost to the running total."""
        self.spent_eur += cost_eur

    def should_stop(self) -> bool:
        """True if the next call should be skipped because spend is within the safety margin of the ceiling."""
        return self.max_eur is not None and self.spent_eur >= (self.max_eur - self.safety_margin_eur)


def _ensure_throwaway_secrets() -> None:
    """
    Generate Cordon's three secrets for this process if they are not set.

    Why this is safe: the benchmark runs its own in-process Cordon on a
    temporary database that is deleted afterwards, so nothing encrypted or
    hashed with these values outlives the run. It means a live run needs only
    the provider key (OPENAI_API_KEY) - not a full production .env.
    Values already set (environment or .env) are kept.
    """
    import base64
    import secrets

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    generated = {
        "CORDON_ADMIN_KEY": lambda: secrets.token_urlsafe(32),
        "CORDON_DATA_KEY": lambda: base64.b64encode(os.urandom(32)).decode(),
        "CORDON_HASH_SALT": lambda: secrets.token_urlsafe(16),
    }
    for name, make in generated.items():
        if not os.environ.get(name):
            os.environ[name] = make()


# --- Section: one benchmark run, step by step ---
@dataclass
class _Run:
    """Everything one benchmark run uses and collects, shared by the steps below."""

    live: bool
    config: dict
    pool_aliases: list[str]
    strong_alias: str
    guard: "_BudgetGuard"
    client: object = None
    headers_b: dict = field(default_factory=dict)
    headers_d: dict = field(default_factory=dict)
    # Scores keyed by item id, so A, B and D are always compared on the same question.
    category_scores: dict = field(default_factory=lambda: {c: {"A": {}, "B": {}, "D": {}} for c in CATEGORIES})
    error_counts: dict = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "D": 0})
    format_misses: dict = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "D": 0})  # answers without a FINAL line
    model_category_scores: dict = field(default_factory=dict)
    model_costs: dict = field(default_factory=dict)
    valid_json: list = field(default_factory=list)
    refusals: list = field(default_factory=list)
    d_cost_total: float = 0.0
    d_correct: int = 0
    latency_d: list = field(default_factory=list)
    human_review_rows: list = field(default_factory=list)
    answers: list = field(default_factory=list)  # one labelled record per scored answer (see _answer_record)
    stopped_early: bool = False

    @property
    def strong_litellm_name(self) -> str:
        """LiteLLM name of the strong (baseline) model."""
        return self.config["models"]["model_aliases"][self.strong_alias]

    def judge(self, reference: str, answer: str) -> float:
        """Summary faithfulness: the LLM judge live (its cost counted), a heuristic in mock mode."""
        if not self.live:
            return heuristic_judge(reference, answer)
        return llm_judge(self.strong_litellm_name, reference, answer, self.config, self.guard.record)


def run_benchmark(
    golden_set_path: Path, limit: int | None, live: bool, max_eur: float | None, write_table: bool, export_review: bool
) -> dict:
    """
    Run the full four-condition benchmark and return the results summary.

    Parameters: golden_set_path - JSONL golden set. limit - cap the number of
    items (None = all). live - False runs entirely offline/mock. max_eur -
    live-mode cost ceiling (ignored in mock mode). write_table - write
    src/config/routing_table.yaml. export_review - write reports/human_review.csv.
    Returns: a summary dict consumed by write_reports() and the console printout.
    """
    os.environ["MOCK_MODE"] = "false" if live else "true"
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    _ensure_throwaway_secrets()

    from fastapi.testclient import TestClient

    from src.server import settings as settings_module
    from src.server.gateway import create_app

    settings_obj = settings_module.load_settings()
    run = _Run(live=live, config=settings_obj.config, pool_aliases=_benchmark_pool(settings_obj, live),
               strong_alias=settings_obj.config["models"]["default_strong"], guard=_BudgetGuard(max_eur if live else None))
    run.model_category_scores = {a: {c: [] for c in CATEGORIES} for a in run.pool_aliases}
    run.model_costs = {a: [] for a in run.pool_aliases}

    with tempfile.TemporaryDirectory() as tmp_dir:
        app = create_app(settings_obj=settings_obj, db_path=str(Path(tmp_dir) / "benchmark.sqlite"))
        with TestClient(app) as client:
            run.client = client
            admin = {"Authorization": f"Bearer {settings_obj.admin_key}"}
            run.headers_b = _benchmark_tenant(run, admin, "benchmark-b", ["trim", "vision"])
            run.headers_d = _benchmark_tenant(run, admin, "benchmark-d", ["trim", "vision", "routing"])
            for item in load_golden_set(golden_set_path, limit=limit):
                if run.guard.should_stop():
                    logger.warning("live cost guard reached; stopping before item %s", item["id"])
                    run.stopped_early = True
                    break
                _run_item(run, item)

    if export_review and run.human_review_rows:
        export_human_review(run.human_review_rows)
    if write_table:
        from evaluation import route_proof

        items = load_golden_set(golden_set_path, limit=limit)
        route_proof.write_routing_table(route_proof.prove_routes(run.answers, items, run.config),
                                        source=f"a {'live' if live else 'mock'} run on {Path(golden_set_path).name}")
    return _summarize(run)


def collect_answers(golden_set_path: Path, alias: str, max_eur: float, repeat: bool = False) -> Path:
    """
    Live: one model's answers to every golden-set item, scored and labelled, with its confidence.

    The cheapest way to grow the quality gate's training data: no Cordon conditions and no
    comparison, just answers from the model the gate will judge (the judge only scores summaries).
    Returns: the labelled-answers file written to data/labelled_answers/.
    """
    os.environ["MOCK_MODE"] = "false"
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    _ensure_throwaway_secrets()
    from src.savings import quality
    from src.server import settings as settings_module

    config = settings_module.load_settings().config
    strong_name = config["models"]["model_aliases"][config["models"]["default_strong"]]
    guard = _BudgetGuard(max_eur)
    judge = lambda reference, answer: llm_judge(strong_name, reference, answer, config, guard.record)  # noqa: E731
    answers = []
    for item in load_golden_set(golden_set_path):
        if guard.should_stop():
            logger.warning("cost guard reached; stopping before item %s", item["id"])
            break
        # The strong model's answers are the reference (condition A) routing evidence is measured against.
        # The strong model's answers are condition A (or R: a repeat run, kept as data).
        condition = ("R" if repeat else "A") if alias == config["models"]["default_strong"] else "C"
        result = call_direct(config["models"]["model_aliases"][alias], item, condition, alias, False, config)
        result = _finalize_result(item, result, condition, alias, False, judge, quality.embed_texts)
        guard.record(result.cost_eur)
        answers.append(_answer_record(item, result, condition, alias))
    path = write_labelled_answers(answers)
    print(f"collected {len(answers)} answers from {alias}, spent EUR {guard.spent_eur:.4f}: {path}")
    return path


def _benchmark_pool(settings_obj, live: bool) -> list[str]:
    """The pool models to compare: all of them in mock mode, only those with a provider key live."""
    aliases = settings_obj.config["models"]["model_aliases"]
    pool = list(settings_obj.config["models"]["default_pool"])
    return pool if not live else [a for a in pool if settings_obj.provider_has_key(aliases[a].split("/", 1)[0])]


def _benchmark_tenant(run: _Run, admin_headers: dict, name: str, features: list[str]) -> dict:
    """Create a Cordon tenant for one condition (B or D) with generous limits; return its auth header."""
    aliases = run.config["models"]["model_aliases"]
    providers = sorted({aliases[a].split("/", 1)[0] for a in [*run.pool_aliases, run.strong_alias]})
    tenant = run.client.post("/admin/tenants", headers=admin_headers, json={
        "name": name, "mode": "enforce", "allowed_providers": providers, "features": features,
        "strong": run.strong_alias, "pool": run.pool_aliases, "holdout_pct": 0.0,
        "rpm": 100000, "tpm": 100000000, "budget_daily_eur": 100000.0, "budget_monthly_eur": 100000.0,
    }).json()
    key = run.client.post(f"/admin/tenants/{tenant['id']}/keys", headers=admin_headers, json={}).json()["key"]
    return {"Authorization": f"Bearer {key}"}


def _run_item(run: _Run, item: dict) -> None:
    """Run conditions A, B and D, then C per pool model, for one golden-set item, and record everything."""
    mock = not run.live
    result_a = _scored(run, item, call_direct(run.strong_litellm_name, item, "A", run.strong_alias, mock, run.config), "A", run.strong_alias)
    result_b = _scored(run, item, call_through_cordon(run.client, run.headers_b, item, run.strong_alias, "B", mock), "B", run.strong_alias)
    result_d = _scored(run, item, call_through_cordon(run.client, run.headers_d, item, run.strong_alias, "D", mock), "D", run.strong_alias)

    # A failed call is a wrong answer (score 0), never a skipped one.
    for condition, result in (("A", result_a), ("B", result_b), ("D", result_d)):
        run.category_scores[item["category"]][condition][item["id"]] = 0.0 if result.error else result.score
        run.error_counts[condition] += 1 if result.error else 0

    if result_a.valid_json is not None:
        run.valid_json.append(bool(result_a.valid_json))
    run.refusals.append(result_a.refused)
    run.d_cost_total += result_d.cost_eur
    if not result_d.error:
        run.latency_d.append(result_d.latency_ms)
        run.d_correct += 1 if result_d.score >= 0.5 else 0
    if not result_a.error:
        if item["category"] == "summarization":
            try:
                judge_score = run.judge(item["expected"], result_a.answer_text)
            except JudgeFailed:
                judge_score = None  # no judgement to compare with a human's: the row is left out
            if judge_score is not None:
                run.human_review_rows.append({"id": item["id"], "reference": item["expected"],
                                              "answer": result_a.answer_text, "judge_score": judge_score})
    _run_condition_c(run, item)


def _run_condition_c(run: _Run, item: dict) -> None:
    """Condition C: every pool model directly (with A, the routing evidence - see route_proof.py)."""
    for alias in run.pool_aliases:
        if run.guard.should_stop():
            run.stopped_early = True
            return
        litellm_name = run.config["models"]["model_aliases"][alias]
        result_c = _scored(run, item, call_direct(litellm_name, item, "C", alias, not run.live, run.config), "C", alias)
        score = 0.0 if result_c.error else result_c.score
        run.error_counts["C"] += 1 if result_c.error else 0
        run.model_category_scores[alias][item["category"]].append(score)
        run.model_costs[alias].append(result_c.cost_eur)


def _scored(run: _Run, item: dict, result: "RunResult", condition: str, alias: str) -> "RunResult":
    """Score one call's answer and add its cost to the run's cost guard."""
    from src.savings import quality

    result = _finalize_result(item, result, condition, alias, not run.live, run.judge, quality.embed_texts)
    run.guard.record(result.cost_eur)
    run.answers.append(_answer_record(item, result, condition, alias))
    # Live only: in mock mode the scores are simulated and Cordon's mock answer is an echo, so a missing FINAL line means nothing.
    if run.live and item["category"] in FINAL_CATEGORIES and not result.error and extract_final(result.answer_text) is None:
        run.format_misses[condition] += 1
    return result


def _answer_record(item: dict, result: "RunResult", condition: str, alias: str) -> dict:
    """
    One answer with its label, for training the quality gate on real model output.

    label - 1 if the answer scored >= 0.5 against the item's known correct
    answer, else 0; None for a failed call (no answer to learn from).
    label_source - "ground_truth" (exact scoring against the expected answer or
    the unit test), or "judge" for summaries, where an LLM judge's faithfulness
    rating is part of the score because no single correct summary exists.
    """
    question = item["messages"][-1]["content"].replace(f"\n\n{FINAL_INSTRUCTION}", "")
    return {
        "item_id": item["id"], "category": item["category"], "condition": condition, "model": alias,
        "question": question, "answer": result.answer_text, "score": round(result.score, 4),
        "label": None if result.error else int(result.score >= 0.5),
        "label_source": "judge" if item["category"] == "summarization" else "ground_truth",
        "error": result.error,
        "confidence": result.confidence,
    }


def write_labelled_answers(answers: list[dict], directory: Path = LABELLED_ANSWERS_DIR) -> Path:
    """Write one live run's labelled answers to <directory>/answers_<UTC time>.jsonl and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"answers_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
    path.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers), encoding="utf-8")
    return path


def _category_verdicts(category_scores: dict) -> dict:
    """Per golden-set category: paired B-A and D-A differences, bootstrap intervals and verdicts."""
    verdicts = {}
    for category in CATEGORIES:
        a, b, d = (category_scores[category][c] for c in ("A", "B", "D"))
        paired = [item_id for item_id in a if item_id in b and item_id in d]
        if not paired:
            continue
        n = len(paired)
        b_diffs = [b[i] - a[i] for i in paired]
        d_diffs = [d[i] - a[i] for i in paired]
        verdicts[category] = {
            "n": n,
            "b_mean": sum(b_diffs) / n, "b_verdict": paired_verdict(b_diffs).verdict,
            "d_mean": sum(d_diffs) / n, "d_verdict": paired_verdict(d_diffs).verdict,
        }
    return verdicts


def _summarize(run: _Run) -> dict:
    """The run's results, in the shape write_reports() and the console printout expect."""
    def mean(values: list) -> float:
        """Average, 0.0 for an empty list."""
        return sum(values) / len(values) if values else 0.0

    return {
        "live": run.live,
        "answers": run.answers,
        "stopped_early": run.stopped_early,
        "error_counts": run.error_counts,
        "format_misses": run.format_misses,
        "spent_eur": run.guard.spent_eur,
        "category_verdicts": _category_verdicts(run.category_scores),
        "pool_aliases": run.pool_aliases,
        "model_category_matrix": {
            alias: {c: (mean(v) if v else None) for c, v in cats.items()} for alias, cats in run.model_category_scores.items()
        },
        "model_mean_cost": {alias: mean(v) for alias, v in run.model_costs.items()},
        "valid_json_rate": mean([float(v) for v in run.valid_json]),
        "refusal_rate": mean([float(v) for v in run.refusals]),
        # Total spent on condition D divided by its correct answers (failed and wrong answers cost money too).
        "cost_per_correct_d": run.d_cost_total / run.d_correct if run.d_correct else 0.0,
        "mean_latency_d_ms": mean(run.latency_d),
        "kappa": None,
    }


def write_live_results(summary: dict, limit: int | None, max_eur: float, path: Path = LIVE_RESULTS_PATH) -> None:
    """
    Write reports/live_results.json after a live run: date, models, real spend and verdicts.

    Parameters: summary - run_benchmark()'s result. limit, max_eur - the run's
    CLI settings, recorded so the numbers can be reproduced. path - output file.
    """
    record = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "litellm_version": _installed_version("litellm"),
        "limit": limit,
        "max_eur": max_eur,
        "spent_eur": round(summary["spent_eur"], 6),
        "stopped_early": summary["stopped_early"],
        "models": summary["pool_aliases"],
        "model_mean_cost_eur": summary["model_mean_cost"],
        "category_verdicts": summary["category_verdicts"],
        "valid_json_rate": summary["valid_json_rate"],
        "refusal_rate": summary["refusal_rate"],
        "mean_latency_d_ms": summary["mean_latency_d_ms"],
        "cost_per_correct_d_eur": summary["cost_per_correct_d"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def _installed_version(package: str) -> str | None:
    """Return an installed package's version, or None."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Parameters: argv - for testing; None reads sys.argv. Returns: process exit code (1 if any category verdict is a clear fail, per tools/ci.sh's gate)."""
    parser = argparse.ArgumentParser(description="Cordon benchmark: proof that optimizations and guardrails preserve answer quality.")
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-eur", type=float, default=5.0)
    parser.add_argument("--write-routing-table", action="store_true")
    parser.add_argument("--export-human-review", action="store_true")
    parser.add_argument("--human-review", type=Path, default=None)
    parser.add_argument("--collect", default=None, metavar="MODEL",
                        help="live: collect one model's labelled answers (with confidence) as quality-gate training data")
    parser.add_argument("--repeat", action="store_true",
                        help="with --collect of the strong model: a second run (condition R), to measure its own noise")
    parser.add_argument("--routing-table-from", type=Path, nargs="+", default=None, metavar="ANSWERS",
                        help="rebuild the routing table from stored live runs' labelled answers (no API calls)")
    parser.add_argument("--questions", type=Path, nargs="+", default=None, metavar="JSONL",
                        help="with --routing-table-from: the question files those runs used (default: --golden-set)")
    args = parser.parse_args(argv)

    if args.routing_table_from is not None:
        from evaluation import route_proof

        return route_proof.rebuild_routing_table(args.routing_table_from, args.questions or [args.golden_set])
    if args.collect is not None:
        collect_answers(args.golden_set, args.collect, args.max_eur, args.repeat)
        return 0

    if args.human_review is not None:
        kappa = compute_kappa(args.human_review)
        if kappa is None:
            print("No human-reviewed rows found (fill in human_score_1_5 first).")
            return 1
        print(f"Quadratic-weighted Cohen's kappa (judge vs human): {kappa:.3f}")
        return 0

    summary = run_benchmark(
        args.golden_set, args.limit, args.live, args.max_eur, args.write_routing_table, args.export_human_review
    )
    write_reports(summary)
    if args.live:
        write_live_results(summary, args.limit, args.max_eur)
        print(f"  labelled answers: {write_labelled_answers(summary['answers'])}")

    print(f"\nCordon benchmark ({'live' if args.live else 'mock'} mode)")
    if summary["stopped_early"]:
        print(f"  stopped early: live cost guard reached (spent EUR {summary['spent_eur']:.2f})")
    any_fail = False
    for category, row in summary["category_verdicts"].items():
        b_symbol, d_symbol = VERDICT_SYMBOLS[row["b_verdict"]], VERDICT_SYMBOLS[row["d_verdict"]]
        print(f"  {category:15s} n={row['n']:2d}  B-A {b_symbol} ({row['b_mean']:+.3f})  D-A {d_symbol} ({row['d_mean']:+.3f})")
        if row["b_verdict"] == "fail" or row["d_verdict"] == "fail":
            any_fail = True
    print(f"  failed calls (scored 0): {summary['error_counts']}  answers without FINAL line: {summary['format_misses']}")
    print(f"  valid JSON rate: {summary['valid_json_rate']:.1%}  refusal rate: {summary['refusal_rate']:.1%}")
    print(f"  mean latency (D): {summary['mean_latency_d_ms']:.1f} ms  cost/correct (D): EUR {summary['cost_per_correct_d']:.6f}")
    if not args.live:
        print("  NOTE: mock mode - these numbers prove the pipeline wiring, not real quality. Run --live for real conclusions.")
    print(f"  reports: {REPORT_MD_PATH}, {REPORT_HTML_PATH}" + (f", {LIVE_RESULTS_PATH}" if args.live else ""))

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
