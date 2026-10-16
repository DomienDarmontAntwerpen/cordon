"""
Route proof: is a cheaper model *together with its quality check* as good as the strong model?

What runs live is a cascade: the cheaper model answers, and the check (quality.gate_verdict
at the route's threshold - the same function step 8 uses) sends every answer it doubts to the
strong model. This module measures exactly that, on stored answers to questions both models
answered, per kind of request (routing.classify_request):

- for each threshold in THRESHOLDS, an answer the check sends on counts as the strong model's
  own answer (difference 0) and a kept one as the cheaper model's;
- the benchmark's verdict rule (benchmark.paired_verdict), made stricter because several
  settings are tried on the same answers: 95% becomes 1 - 0.05 / (models x thresholds tried
  in that bucket) (Bonferroni);
- the cheapest passing setting becomes the route - if the cascade (the cheaper model always,
  plus the strong model for the share sent on) costs less than the strong model alone.

An answer the check sends on counts as the strong model's *stored* answer to that question
(difference 0). Live it is a fresh call, with gpt-4o's own run-to-run variation (on these
classification questions: worse on 1 of 130 when asked twice), so the proof slightly
understates the variation of an escalated answer.

The check must never have been trained on the questions that prove it: those are left out
(quality.trained_questions).

    .venv/bin/python -m evaluation.benchmark --routing-table-from data/labelled_answers/*.jsonl \\
        --questions data/external/extended_golden_set.jsonl data/external/extra_*.jsonl
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.paths import ROUTING_TABLE_PATH
from evaluation.benchmark import PairedVerdict, load_golden_set, paired_verdict
from evaluation.gate_eval import as_served

logger = logging.getLogger("cordon.route_proof")

THRESHOLDS = (0.0, 0.5, 0.7, 0.9)  # 0 = only the rules check the cheaper model's answers

CONFIDENCE = 1 - 0.05 / len(THRESHOLDS)  # one model's thresholds; prove_routes divides by the models tried too


def confidence_for(settings_tried: int) -> float:
    """Bonferroni: every extra setting tried on the same answers must clear a stricter bar."""
    return 1 - 0.05 / settings_tried


@dataclass(frozen=True)
class Pair:
    """One question answered by both models: what the check reads, and both scores."""

    question: str
    answer: str  # the cheaper model's answer as an app receives it
    confidence: dict | None
    response_format: dict | None
    cheap_score: float
    strong_score: float
    graded: bool = False  # scored against a reference (summaries, support replies), not right/wrong


@dataclass(frozen=True)
class RouteEvidence:
    """One cheaper model with its check at one threshold, against the strong model."""

    model: str
    threshold: float
    escalated: float  # share of answers the check sent to the strong model
    cost_share: float  # the cascade's cost as a share of the strong model alone
    cascade_mean: float
    strong_mean: float
    verdict: PairedVerdict
    confidence: float

    @property
    def proven(self) -> bool:
        """As good as the strong model by the verdict rule, and cheaper."""
        return self.verdict.verdict == "pass" and self.cost_share < 1.0


def cascade_evidence(pairs: list[Pair], model: str, threshold: float, price_share: float, config: dict,
                     confidence: float = CONFIDENCE) -> RouteEvidence:
    """The cascade at one threshold: which answers the check sends on, the quality and the cost."""
    from src.savings import quality

    sent_on = [quality.gate_verdict(p.question, p.answer, config, threshold, p.response_format, p.confidence).escalate
               for p in pairs]
    diffs = [0.0 if on else p.cheap_score - p.strong_score for p, on in zip(pairs, sent_on)]
    escalated = sum(sent_on) / len(pairs)
    return RouteEvidence(
        model=model, threshold=threshold, escalated=escalated, cost_share=price_share + escalated,
        cascade_mean=sum(p.strong_score if on else p.cheap_score for p, on in zip(pairs, sent_on)) / len(pairs),
        strong_mean=sum(p.strong_score for p in pairs) / len(pairs),
        verdict=paired_verdict(diffs, confidence=confidence,
                               loss_diffs=[loss_diff(d, p) for d, p in zip(diffs, pairs)]),
        confidence=confidence,
    )


def loss_diff(diff: float, pair: Pair) -> float:
    """
    The difference as it counts towards the rule's loss bound: on right/wrong questions any loss is a
    wrong answer; a graded answer (scored against a reference) differs a little on almost every
    question even between two runs of the strong model, so it counts as a loss only when the cheaper
    model failed outright (score 0 where the strong model scored). Graded answers move the average.
    """
    failed = pair.cheap_score == 0.0 and pair.strong_score > 0.0
    return diff if (not pair.graded or failed) else 0.0


def best_route(evidence: list[RouteEvidence]) -> RouteEvidence:
    """The cheapest proven setting; without one, the closest to proof (highest lower bound), for the report."""
    proven = [e for e in evidence if e.proven]
    if proven:
        return min(proven, key=lambda e: e.cost_share)
    return max(evidence, key=lambda e: (e.verdict.lower, -e.cost_share))


def pairs_by_bucket(answers: list[dict], items: list[dict], config: dict, excluded: frozenset) -> dict:
    """
    {(category, difficulty): {cheaper model: [Pair]}} from labelled answers: every question with
    the strong model's answer (condition A) and a cheaper model's (condition C), minus `excluded`.
    Per question and model the first answer counts - for a cheaper model the first one carrying its confidence.
    """
    from src.savings import routing

    strong = config["models"]["default_strong"]
    first: dict = {}
    for a in answers:
        if (a["condition"] == "A" and a["model"] == strong) or (a["condition"] == "C" and a["model"] != strong):
            key = (a["item_id"], a["model"])
            # A cheaper model's answer the check can read (with its confidence) wins over one it cannot.
            if key not in first or (_checkable(a) and not _checkable(first[key])):
                first[key] = a
    pairs: dict = {}
    for item in items:
        a = first.get((item["id"], strong))
        if a is None or item["id"] in excluded:
            continue
        bucket = routing.classify_request(item["messages"], item.get("response_format"))
        for (item_id, model), c in first.items():
            if item_id == item["id"] and model != strong:
                pairs.setdefault(bucket, {}).setdefault(model, []).append(Pair(
                    question=c["question"], answer=as_served(c.get("answer") or ""), confidence=c.get("confidence"),
                    response_format=item.get("response_format"),
                    cheap_score=0.0 if c.get("error") else c["score"], strong_score=0.0 if a.get("error") else a["score"],
                    graded=item["category"] == "summarization"))
    return pairs


def _checkable(answer: dict) -> bool:
    """True when the answer carries the model's confidence over its text, which the check reads."""
    return (answer.get("confidence") or {}).get("body_prob") is not None


def prove_routes(answers: list[dict], items: list[dict], config: dict) -> dict:
    """{bucket: [best RouteEvidence per cheaper model]} - every setting tried, the best one kept per model."""
    from src.savings import costing, quality

    aliases = config["models"]["model_aliases"]
    strong_price = costing.cost_eur(aliases[config["models"]["default_strong"]], 1000, 1000, config)
    excluded = quality.trained_questions(config)
    proofs = {}
    for bucket, by_model in sorted(pairs_by_bucket(answers, items, config, excluded).items()):
        proofs[bucket] = []
        confidence = confidence_for(len(by_model) * len(THRESHOLDS))
        for model, pairs in sorted(by_model.items()):
            share = costing.cost_eur(aliases[model], 1000, 1000, config) / strong_price
            proofs[bucket].append(best_route([cascade_evidence(pairs, model, t, share, config, confidence)
                                              for t in THRESHOLDS]))
    return proofs


# --- Section: the routing table ---
def write_routing_table(proofs: dict, path: Path = ROUTING_TABLE_PATH, source: str = "") -> None:
    """
    Write src/config/routing_table.yaml: per bucket the cheapest proven route, with its evidence
    as a comment, so the file explains itself and is never edited by hand. A bucket without a
    proven route is left out: its requests stay on the strong model.
    """
    lines = [
        "# Routing table - written by `evaluation.benchmark --write-routing-table` (or --routing-table-from); do not edit by hand.",
        f"# Written {time.strftime('%Y-%m-%d', time.gmtime())}" + (f" from {source}" if source else "") + ".",
        "# Per (category, difficulty): the cheaper model that, together with its quality check at gate_threshold,",
        "# proved as good as the strong model (>= 30 questions; interval and exact loss-rate bound within the margin,",
        "# at 1 - 0.05 / (models x thresholds tried in the bucket), shown per route) and cheaper. A missing bucket",
        "# routes to the strong model.",
    ]
    entries = [(bucket, min((e for e in evidence if e.proven), key=lambda e: e.cost_share))
               for bucket, evidence in sorted(proofs.items()) if any(e.proven for e in evidence)]
    lines.append("categories:" if entries else "categories: {}")
    current = None
    for (category, difficulty), e in entries:
        if category != current:
            lines.append(f"  {category}:")
            current = category
        lines.append(f"    # {describe(e)}")
        lines.append(f"    {difficulty}: {{model: {e.model}, gate_threshold: {e.threshold}, escalated: {round(e.escalated, 3)}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote routing table: %s", path)


def describe(e: RouteEvidence) -> str:
    """One line of evidence for a route."""
    v = e.verdict
    return (f"{e.model} + check at {e.threshold}: {e.cascade_mean:.1%} vs the strong model's {e.strong_mean:.1%} on {v.n} "
            f"questions ({e.confidence:.2%}: {v.lower:+.3f} to {v.upper:+.3f}; a wrong or failed answer on at most "
            f"{v.loss_bound:.1%}); "
            f"{e.escalated:.0%} sent to the strong model; costs {e.cost_share:.0%} of it - {v.verdict}")


def rebuild_routing_table(answers_paths: list[Path], question_paths: list[Path]) -> int:
    """Write the routing table from stored live runs (labelled answers + the question files they ran on); no API calls."""
    import yaml

    from src.paths import CONFIG_PATH

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    answers = [json.loads(line) for path in answers_paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [item for path in question_paths for item in load_golden_set(path)]
    proofs = prove_routes(answers, items, config)
    write_routing_table(proofs, source=f"{', '.join(p.name for p in answers_paths)} ({', '.join(p.name for p in question_paths)})")
    for (category, difficulty), evidence in proofs.items():
        for e in evidence:
            print(f"  {category}/{difficulty}: {describe(e)}")
    return 0
