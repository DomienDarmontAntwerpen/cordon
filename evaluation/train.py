"""
Trains the quality gate's classifier on real, labelled model answers.

- Data: the cheaper models' answers in data/labelled_answers/ (written by the live
  benchmark and `evaluation.benchmark --collect`), each labelled against a known correct answer.
- Features: quality.extract_features() - the answer's shape, the kind of question,
  format and size, and the model's own confidence (logprobs) when the provider gives it.
- Candidates: Logistic Regression, Random Forest, Gradient Boosting, compared with 5-fold
  cross-validation grouped by question (a question is never in both training and test);
  the simplest one within 0.01 AUC of the best wins and is refit on all answers.
- Never trained on the questions that prove routes (those the strong model also answered):
  evaluation/route_proof.py checks the gate on exactly those, so it must not have seen them.
- The thresholds that run belong to the routes (proven per kind of request by the benchmark);
  the model card reports, for comparison, what a threshold keeping 90% of good answers catches.
- Writes src/models/model.pkl ({"classifier", "trained_on"}), its SHA-256 into config.yaml,
  and reports/model_card.md.

    .venv/bin/python -m evaluation.train
"""

import logging
from collections import Counter
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.model_selection import GroupKFold

from src.paths import CONFIG_PATH, DATA_DIR, MODEL_PATH, REPORTS_DIR
from src.protection import security
from src.savings import quality
from evaluation import gate_eval

logger = logging.getLogger("cordon.train")
MODEL_CARD_PATH = REPORTS_DIR / "model_card.md"
ANSWERS_DIR = DATA_DIR / "labelled_answers"
GOOD_KEPT_TARGET = 0.90  # the gate may send at most 10% of good answers to the strong model needlessly
MIN_GOOD_FOR_OWN_THRESHOLD = 50  # fewer good answers of a kind: use the shared default threshold
SOURCE_NAMES = {"gsm8k": "GSM8K", "sst2": "SST-2", "mbpp": "MBPP", "agnews": "AG News"}  # id prefix -> public set


# --- Section: data ---
def training_rows(config: dict, directory: Path = ANSWERS_DIR) -> list[dict]:
    """
    The cheaper models' labelled answers that carry the model's confidence (exactly what the
    classifier judges live), minus every question the strong model also answered: those are
    route evidence, and the gate is checked on them (evaluation/route_proof.py).
    """
    strong = config["models"]["default_strong"]
    answers = gate_eval.load_answers(directory)
    evidence = {r["item_id"] for r in answers if r["model"] == strong}
    return gate_eval.with_confidence([r for r in answers if r["model"] != strong and r["item_id"] not in evidence])


# --- Section: selection, threshold, fit ---
def out_of_fold(name: str, features: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Out-of-fold P(good) of one candidate, folds grouped by question."""
    return gate_eval.grouped_cv_probabilities(gate_eval.CANDIDATES[name], features, rows)


def choose_threshold(labels: np.ndarray, good_probability: np.ndarray, good_kept: float = GOOD_KEPT_TARGET) -> float:
    """The highest threshold that still leaves `good_kept` of the good answers alone (out-of-fold scores)."""
    return float(np.quantile(good_probability[labels == 1], 1.0 - good_kept))


def type_only_baseline(rows: list[dict], features: np.ndarray) -> float:
    """AUC of knowing only the kind of question (its good-rate on the training folds): what the model must beat."""
    from sklearn.metrics import roc_auc_score

    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["item_id"] for r in rows])
    first = quality.FEATURE_NAMES.index("type_chat")
    kind = features[:, first:first + 6].argmax(axis=1)
    probability = np.zeros(len(rows))
    for train_idx, test_idx in GroupKFold(n_splits=5).split(rows, labels, groups):
        rate = {k: labels[train_idx][kind[train_idx] == k].mean() for k in set(kind[train_idx])}
        probability[test_idx] = [rate.get(k, labels[train_idx].mean()) for k in kind[test_idx]]
    return round(float(roc_auc_score(labels, probability)), 3)


def kinds_of(features: np.ndarray) -> np.ndarray:
    """The kind of question of every row, read from its one-hot type features (as quality.question_kind)."""
    first = quality.FEATURE_NAMES.index("type_chat")
    names = [n.removeprefix("type_") for n in quality.FEATURE_NAMES[first:first + 6]]
    return np.array(names)[features[:, first:first + 6].argmax(axis=1)]


def choose_thresholds(labels: np.ndarray, probability: np.ndarray, kinds: np.ndarray) -> tuple[dict, float]:
    """
    A threshold per kind of question with enough good answers, and a shared default for the rest.

    Why per kind: the model scores some kinds high by nature (classification); one shared threshold
    then almost never fires there - measured: 4% of classification's wrong answers caught, 35% with its own.
    """
    default = round(choose_threshold(labels, probability), 3)
    own = {str(kind): round(choose_threshold(labels[kinds == kind], probability[kinds == kind]), 3)
           for kind in sorted(set(kinds)) if (labels[kinds == kind] == 1).sum() >= MIN_GOOD_FOR_OWN_THRESHOLD}
    return own, default


def train(rows: list[dict]) -> tuple[dict, dict]:
    """Compare the candidates on held-out questions, pick the simplest best, report per kind, refit on all rows."""
    features = gate_eval.full_feature_matrix(rows)
    labels = np.array([r["label"] for r in rows])
    results = gate_eval.compare_candidates(rows, features, threshold=0.5)
    selected = gate_eval.pick_simplest(results)
    probability = out_of_fold(selected, features, rows)
    kinds = kinds_of(features)
    thresholds, threshold = choose_thresholds(labels, probability, kinds)
    per_row_threshold = np.array([thresholds.get(k, threshold) for k in kinds])
    classifier = gate_eval.CANDIDATES[selected]().fit(features, labels)
    model = {"classifier": classifier, "trained_on": sorted({r["item_id"] for r in rows})}
    metrics = {
        "rows": len(rows), "bad": int((labels == 0).sum()), "questions": len({r["item_id"] for r in rows}),
        "sources": sorted({SOURCE_NAMES.get(r["item_id"].split("_")[0], "golden set") for r in rows}),
        "answers_per_model": dict(sorted(Counter(r["model"] for r in rows).items())),
        "candidates": results, "selected": selected, "threshold": threshold,
        "thresholds": thresholds,
        "at_threshold": gate_eval.gate_metrics(labels, probability - per_row_threshold + 0.5, 0.5),
        "shape_only": gate_eval.gate_metrics(labels, out_of_fold(selected, _shape_only(features), rows), 0.5)["auc"],
        "type_only": type_only_baseline(rows, features),
        "weights": _weights(classifier),
        "per_kind": _per_kind(kinds, labels, probability, thresholds, threshold),
    }
    return model, metrics


def _per_kind(kinds: np.ndarray, labels: np.ndarray, probability: np.ndarray, thresholds: dict, default: float) -> dict:
    """Out-of-fold results per kind of question, each at its own threshold (or the default)."""
    table = {}
    for kind in sorted(set(kinds)):
        mask = kinds == kind
        threshold = thresholds.get(kind, default)
        table[str(kind)] = {**gate_eval.gate_metrics(labels[mask], probability[mask], threshold), "threshold": threshold}
    return table


def repeat_disagreement(rows: list[dict]) -> tuple[int, int]:
    """Label noise: of the questions one model answered in several runs, how many got different labels."""
    labels: dict[str, list[int]] = {}
    for row in rows:
        labels.setdefault(row["item_id"], []).append(row["label"])
    repeated = [v for v in labels.values() if len(v) > 1]
    return sum(1 for v in repeated if len(set(v)) > 1), len(repeated)


def _shape_only(features: np.ndarray) -> np.ndarray:
    """The six answer-shape features only, other columns zeroed: what the first gate saw."""
    masked = np.zeros_like(features)
    masked[:, :6] = features[:, :6]
    return masked


def _weights(model) -> list[tuple[str, float]]:
    """Logistic Regression's weights per (standardized) feature, largest effect first; empty for tree models."""
    classifier = model[-1] if hasattr(model, "steps") else model
    if not hasattr(classifier, "coef_"):
        return []
    pairs = zip(quality.FEATURE_NAMES, classifier.coef_[0])
    return sorted(((name, round(float(w), 3)) for name, w in pairs), key=lambda p: -abs(p[1]))


# --- Section: outputs ---
def update_config(config_path: Path, sha256_hex: str) -> None:
    """Write the model file's SHA-256 into config.yaml's model_sha256 line (comments kept)."""
    text = config_path.read_text(encoding="utf-8")
    text, count = re.subn(r"^(\s*model_sha256:).*$", rf'\g<1> "{sha256_hex}"', text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError("could not find exactly one model_sha256 line in config.yaml")
    config_path.write_text(text, encoding="utf-8")


def write_model_card(metrics: dict, sha256_hex: str, path: Path = MODEL_CARD_PATH) -> None:
    """Write reports/model_card.md from this run's numbers only."""
    candidates = "\n".join(
        f"| {name} | {r['auc']:.3f} | {r['average_precision_bad']:.3f} |" for name, r in metrics["candidates"].items())
    at = metrics["at_threshold"]
    weights = ("## Largest weights (standardized features; positive = more likely a good answer)\n\n| Feature | Weight |\n"
               "|---|---|\n" + "\n".join(f"| {name} | {w:+.3f} |" for name, w in metrics["weights"][:10]) + "\n\n"
               if metrics["weights"] else "")  # a tree model has no weights to show
    per_kind = "\n".join(
        f"| {kind} | {m['n']} | {m['n_bad']} | {m['auc'] if m['auc'] is not None else '-'} | {m['threshold']} | "
        f"{'-' if m['bad_caught'] is None else format(m['bad_caught'], '.0%')} | {m['good_kept']:.0%} |"
        for kind, m in metrics["per_kind"].items())
    repeat = metrics["repeat_disagreement"]
    path.write_text(f"""# Quality gate model card

## What it is
The classifier behind each route's quality check: when routing sent a request to a cheaper
model, an answer scoring below the route's threshold is asked again of the strong model, once.
The thresholds belong to the routes - the benchmark proves each route *with* its check
(`evaluation/route_proof.py`). It catches the *signs* of a wrong answer; it cannot check a fact or a
calculation itself.

## Data
{metrics['rows']} real answers of the cheaper models ({', '.join(f"{m}: {n}" for m, n in metrics['answers_per_model'].items())}) to {metrics['questions']} questions with known
correct answers ({', '.join(metrics['sources'])}), of which {metrics['bad']} are wrong.
Questions the strong model also answered are left out: they prove the routes, and the check is
measured on them, so it never trains on them (the model file stores the ids it trained on).

Every answer carries the model's own confidence over the answer text
(OpenAI logprobs, measured before the benchmark's FINAL line, as a live answer has none).
Labels come from exact scoring against the known answer (summaries: an LLM judge's rating).
Written by the live benchmark and `evaluation.benchmark --collect` to `data/labelled_answers/`.

## Features
`quality.extract_features()`, {len(quality.FEATURE_NAMES)} in all: the answer's shape (similarity to the question,
length, refusal, error, repetition, cut off), the kind of question, format and size (code block,
valid Python, share of digits), and the model's confidence. Without confidence (a provider
that returns no logprobs) the classifier cannot judge: such an answer goes to the strong model
(a route at threshold 0 uses the rules only and needs no confidence). Answers are featurized as an app
would receive them: the benchmark's own "FINAL:" line is removed first, so the gate never
learns a quirk of the test setup that real traffic does not have.

## Evaluation - 5-fold cross-validation grouped by question (never scored on a seen question)

| Candidate | AUC | Average precision (wrong answers) |
|---|---|---|
{candidates}

Selected: **{metrics['selected']}** (the simplest within 0.01 AUC of the best), refit on all answers.

What it has to beat:
- knowing only the kind of question: AUC {metrics['type_only']}
- the six answer-shape features only (the first gate): AUC {metrics['shape_only']}

For comparison only (the thresholds that run are the routes'): a threshold per kind of question
that keeps 90% of its good answers (kinds with fewer than {MIN_GOOD_FOR_OWN_THRESHOLD} good answers share {metrics['threshold']})
catches, out-of-fold, **{at['bad_caught']:.0%} of wrong answers** while {at['good_kept']:.0%} of good answers are left alone.

Neural networks (an MLP on embeddings, a fine-tuned MiniLM transformer) and an LLM judge
were measured against this model in `notebooks/quality_gate_real_answers.ipynb`.

### Per kind of question (out-of-fold, each at its own threshold)

The check only ever acts where routing sends a cheaper model. One shared threshold would leave
some kinds nearly unguarded; that is why each route has its own, proven threshold.

| Kind | Answers | Wrong | AUC | Threshold | Wrong answers caught | Good left alone |
|---|---|---|---|---|---|---|
{per_kind}

### Noise and coverage
- {repeat[0]} of the {repeat[1]} questions gpt-4o-mini answered in more than one run got different labels:
  the same model answers differently from run to run, which limits how precise any figure here can be.
- The rule layer (empty, cut off, non-answer, refusal, error, invalid JSON) fired on none of these
  real answers - real models rarely produce such failures. It is tested on constructed cases
  (`tests/test_quality.py`), and is there for the day a provider does.

{weights}## Limits
- It reads signs, not truth: a wrong number in a confident, well-formed answer passes.
- Trained on the cheaper models' answers listed under Data, to the kinds of questions in the
  table above. On another model or kind of traffic it can be much weaker - that is measured, not
  assumed: `evaluation/route_proof.py` proves each route with the check on that route's own
  model. On gpt-4.1-mini's classification answers it stays barely better than chance (AUC about
  0.6, also after training on 400 of that model's own answers): a wrong label is usually a
  confident one. That route passes because gpt-4.1-mini itself is nearly as good as gpt-4o.
- It is never used on its own: a route is proven together with its check, and an answer
  without the model's confidence goes to the strong model.

## File integrity
SHA-256 of `src/models/model.pkl`: `{sha256_hex}` (recorded in `src/config/config.yaml`,
checked before every load).
""", encoding="utf-8")


def main() -> int:
    """Train on the labelled answers, save the model, update config.yaml, write the model card."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = training_rows(config)
    if not rows:
        logger.error("no labelled answers in %s - run the live benchmark or `evaluation.benchmark --collect` first", ANSWERS_DIR)
        return 1
    model, metrics = train(rows)
    strong = config["models"]["default_strong"]
    metrics["repeat_disagreement"] = repeat_disagreement(
        [r for r in gate_eval.load_answers(ANSWERS_DIR) if r["model"] != strong and r["condition"] == "C"])
    logger.info("selected %s, thresholds %s (default %s), overall %s", metrics["selected"], metrics["thresholds"],
                metrics["threshold"], metrics["at_threshold"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    sha256_hex = security.file_sha256(str(MODEL_PATH))
    update_config(CONFIG_PATH, sha256_hex)
    write_model_card(metrics, sha256_hex)
    logger.info("wrote %s (sha256 %s) and %s", MODEL_PATH, sha256_hex, MODEL_CARD_PATH)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
