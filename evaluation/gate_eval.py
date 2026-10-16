"""
Evaluation of the quality gate on real, labelled model answers.

The gate decides whether a cheaper model's answer is bad enough to ask the
strong model again. So the question is: how many wrong answers does it catch,
and how many good answers does it send back needlessly?

- `load_answers()`: the labelled answers the live benchmark wrote (data/labelled_answers/).
- `feature_matrix()`: quality.extract_features() for every answer.
- `compare_candidates()`: Logistic Regression, Random Forest and Gradient Boosting,
  trained on real answers with cross-validation grouped by question, so a question
  is never in both the training and the test fold.
- `gate_metrics()`: AUC, average precision, and at a threshold: share of wrong
  answers caught, share of good answers kept.
- `full_feature_matrix()`: everything the shipped gate sees (quality.extract_features:
  shape, question type, format, length, the model's own confidence).
- Neural candidates for comparison: an MLP on embeddings, and a
  fine-tuned MiniLM transformer that reads question and answer as text and
  learns its own features (`finetuned_transformer_cv()`).
Used by notebooks/quality_gate_real_answers.ipynb.
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.savings import quality

SEED = 42
CANDIDATES = {
    "LogisticRegression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=SEED),
    "GradientBoosting": lambda: HistGradientBoostingClassifier(max_depth=3, class_weight="balanced", random_state=SEED),
}
SIMPLEST_FIRST = ["LogisticRegression", "RandomForest", "GradientBoosting"]


# --- Section: data ---
def load_answers(directory: Path) -> list[dict]:
    """Every labelled answer (label 0 or 1) from the answers_*.jsonl files; failed calls (label None) are left out."""
    rows = []
    for path in sorted(directory.glob("answers_*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("label") is not None]


def feature_matrix(rows: list[dict]) -> np.ndarray:
    """Only the six answer-shape features (the first gate's view), for comparison."""
    return full_feature_matrix(rows)[:, :6]


# --- Section: metrics ---
def gate_metrics(labels: np.ndarray, good_probability: np.ndarray, threshold: float) -> dict:
    """
    How well a "probability the answer is good" separates good from bad answers.

    auc - 0.5 is guessing, 1.0 perfect. average_precision_bad - for finding the
    bad answers (the gate's job). At `threshold` (below it the gate escalates):
    bad_caught - share of wrong answers escalated; good_kept - share of good
    answers left alone (every good one escalated costs a strong-model call).
    """
    bad = labels == 0
    escalated = good_probability < threshold
    has_both = 0 < bad.sum() < len(labels)
    return {
        "n": int(len(labels)), "n_bad": int(bad.sum()),
        "auc": round(float(roc_auc_score(labels, good_probability)), 3) if has_both else None,
        "average_precision_bad": round(float(average_precision_score(bad, 1 - good_probability)), 3) if has_both else None,
        "bad_caught": round(float(escalated[bad].mean()), 3) if bad.any() else None,
        "good_kept": round(float((~escalated[~bad]).mean()), 3) if (~bad).any() else None,
    }


def compare_candidates(rows: list[dict], features: np.ndarray, threshold: float, folds: int = 5) -> dict:
    """Out-of-fold metrics per candidate; folds are grouped by question (item_id)."""
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["item_id"] for r in rows])
    results = {}
    for name in SIMPLEST_FIRST:
        probability = cross_val_predict(CANDIDATES[name](), features, labels, groups=groups,
                                        cv=GroupKFold(n_splits=folds), method="predict_proba")[:, 1]
        results[name] = gate_metrics(labels, probability, threshold)
    return results


def pick_simplest(results: dict, tolerance: float = 0.01) -> str:
    """The simplest candidate whose AUC is within `tolerance` of the best (the agreed tie-break)."""
    best = max(r["auc"] or 0 for r in results.values())
    return next(name for name in SIMPLEST_FIRST if (results[name]["auc"] or 0) >= best - tolerance)


# --- Section: more information for the model ---
# The benchmark asks models to end with "FINAL: <answer>" so it can score them; real apps never do.
_FINAL_LINE = re.compile(r"\s*\**\s*FINAL\s*\**\s*:\s*\**\s*(.*)$", re.IGNORECASE | re.DOTALL)


def as_served(answer: str) -> str:
    """
    The answer as a real app would get it: the benchmark's "FINAL:" marker removed, its value kept
    as the closing line unless the text already ends with it ("... the cost is:\nFINAL: 145" ->
    "... the cost is:\n145"; "Paris. FINAL: Paris" -> "Paris."; "FINAL: 4" -> "4").
    """
    match = _FINAL_LINE.search(answer)
    if not match:
        return answer
    before, value = answer[:match.start()].rstrip(), match.group(1).strip().strip("*")
    if not before:
        return value
    return before if before.rstrip(".!").endswith(value) else f"{before}\n{value}"


def with_confidence(rows: list[dict]) -> list[dict]:
    """The answers that carry the model's confidence over the answer text - the only ones the classifier learns from."""
    return [r for r in rows if (r.get("confidence") or {}).get("body_prob") is not None]


def full_feature_matrix(rows: list[dict]) -> np.ndarray:
    """quality.extract_features() on the answer as served, with its confidence: what the shipped gate sees."""
    return np.stack([quality.extract_features(r["question"], as_served(r["answer"]), r.get("confidence")) for r in rows])


def embedding_matrix(rows: list[dict]) -> np.ndarray:
    """The embeddings of question and answer side by side (384 + 384 numbers): raw material a neural network can learn from."""
    vectors = quality.embed_texts([r["question"] for r in rows] + [as_served(r["answer"]) or " " for r in rows])
    return np.hstack([vectors[:len(rows)], vectors[len(rows):]])


def mlp() -> object:
    """A small neural network (one hidden layer of 64), with weight decay and early stopping against overfitting."""
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, early_stopping=True,
                                                        max_iter=500, random_state=SEED))


def grouped_cv_probabilities(model_factory, features: np.ndarray, rows: list[dict], folds: int = 5) -> np.ndarray:
    """Out-of-fold P(good) for every answer; folds grouped by question."""
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["item_id"] for r in rows])
    return cross_val_predict(model_factory(), features, labels, groups=groups, cv=GroupKFold(n_splits=folds),
                             method="predict_proba")[:, 1]


# --- Section: a fine-tuned transformer that learns its own features ---
BASE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"


def finetuned_transformer_cv(rows: list[dict], folds: int = 5, epochs: int = 3, max_length: int = 256) -> np.ndarray:
    """
    Out-of-fold P(good) from MiniLM fine-tuned per fold on (question, answer) as text.

    A pre-trained transformer plus one new output layer, trained further on the
    training folds (transfer learning); it finds its own features in the words.
    Folds are grouped by question, so it is always scored on unseen questions.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_TRANSFORMER)
    labels = np.array([r["label"] for r in rows], dtype=np.float32)
    groups = np.array([r["item_id"] for r in rows])
    probabilities = np.zeros(len(rows))
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(rows, labels, groups):
        encoder = AutoModel.from_pretrained(BASE_TRANSFORMER).to(device)
        head = torch.nn.Linear(encoder.config.hidden_size, 1).to(device)
        optimizer = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], lr=2e-5, weight_decay=0.01)
        bad_weight = torch.tensor(float((labels[train_idx] == 1).sum() / max((labels[train_idx] == 0).sum(), 1)), dtype=torch.float32, device=device)
        for _ in range(epochs):
            order = np.random.default_rng(SEED).permutation(train_idx)
            for start in range(0, len(order), 16):
                batch = order[start:start + 16]
                logits = _forward(encoder, head, tokenizer, [rows[i] for i in batch], device, max_length)
                target = torch.tensor(labels[batch], device=device)
                # Bad answers are rare: weigh them up so the model cannot win by calling everything good.
                weights = torch.where(target == 0, bad_weight, torch.ones_like(target))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, weight=weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            for start in range(0, len(test_idx), 32):
                batch = test_idx[start:start + 32]
                probabilities[batch] = torch.sigmoid(
                    _forward(encoder, head, tokenizer, [rows[i] for i in batch], device, max_length)).cpu().numpy()
    return probabilities


def _forward(encoder, head, tokenizer, batch_rows: list[dict], device: str, max_length: int):
    """One batch through MiniLM (question and answer as a text pair), mean-pooled, then the output layer."""
    encoded = tokenizer([r["question"] for r in batch_rows], [as_served(r["answer"]) or " " for r in batch_rows],
                        truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
    hidden = encoder(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return head(pooled).squeeze(-1)



# --- Section: an existing model as judge, for comparison ---
JUDGE_PROMPT = ("Is the answer below a correct and complete answer to the question? Reply with only yes or no.\n\n"
                "Question: {question}\n\nAnswer: {answer}")


def judge_probabilities(rows: list[dict], litellm_name: str, config: dict, workers: int = 8) -> tuple[np.ndarray, float]:
    """
    Live: an LLM judges every answer as served (as_served - the same text the classifier sees);
    P(good) is its probability of replying "yes" (from logprobs).

    Returns: (probabilities, total cost in EUR). The comparison point for the gate: an existing,
    much larger model reading every answer - at the price of one extra call per answer.
    """
    from concurrent.futures import ThreadPoolExecutor

    results = list(ThreadPoolExecutor(max_workers=workers).map(lambda r: _judge_one(r, litellm_name, config), rows))
    return np.array([p for p, _ in results]), sum(cost for _, cost in results)


def _judge_one(row: dict, litellm_name: str, config: dict) -> tuple[float, float]:
    """One judge call: (P(yes), cost in EUR)."""
    import litellm

    from src.savings import costing

    response = litellm.completion(
        model=litellm_name, max_tokens=1, logprobs=True, top_logprobs=5, timeout=config["provider"]["timeout_seconds"],
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=row["question"], answer=as_served(row["answer"]))}])
    cost = costing.actual_cost_eur(litellm_name, response.usage.prompt_tokens, response.usage.completion_tokens, config)
    logprobs = response.choices[0].logprobs
    top = logprobs["content"][0]["top_logprobs"] if isinstance(logprobs, dict) else \
        [{"token": t.token, "logprob": t.logprob} for t in logprobs.content[0].top_logprobs]
    yes = sum(float(np.exp(t["logprob"])) for t in top if t["token"].strip().lower() == "yes")
    no = sum(float(np.exp(t["logprob"])) for t in top if t["token"].strip().lower() == "no")
    return (yes / (yes + no) if yes + no > 0 else 0.5), cost
