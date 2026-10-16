"""
How similar must two questions be before the semantic cache reuses an answer?
Functions used by notebooks/semantic_cache_threshold.ipynb: apply Cordon's real
matching rule to labelled question pairs, sweep thresholds for hit rate and
wrong-hit rate, and bootstrap a confidence interval.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.savings.cache import semantic_guard_tokens

SEED = 42


def load_pairs(path: Path, n: int, seed: int = SEED) -> pd.DataFrame:
    """
    Load a reproducible random sample of labelled question pairs.

    Parameters: path - a parquet file with question1, question2, label columns
    (GLUE's QQP layout). n - how many pairs to sample. seed - for reproducibility.
    Returns: a DataFrame with columns question1, question2, label.
    """
    frame = pd.read_parquet(path)[["question1", "question2", "label"]].dropna()
    return frame.sample(n=min(n, len(frame)), random_state=seed).reset_index(drop=True)


def pair_scores(pairs: pd.DataFrame, embed_fn, guard_fn=semantic_guard_tokens) -> pd.DataFrame:
    """
    Score each pair with Cordon's semantic-cache rule.

    Parameters: pairs - from load_pairs(). embed_fn - texts -> list of vectors
    (quality.embed_texts in practice). guard_fn - text -> guard tokens; the
    default is the rule Cordon uses, other values let a notebook compare rules.
    Returns: pairs plus `cosine` (embedding similarity) and `tokens_equal`
    (both questions have the same negation words, numbers and names in the
    same order - cache.semantic_guard_tokens(); without that, cache.py never
    serves a hit, whatever the similarity).
    """
    texts_a = pairs["question1"].tolist()
    texts_b = pairs["question2"].tolist()
    vectors_a = np.asarray(embed_fn(texts_a), dtype=float)
    vectors_b = np.asarray(embed_fn(texts_b), dtype=float)
    norms = np.linalg.norm(vectors_a, axis=1) * np.linalg.norm(vectors_b, axis=1)
    cosine = np.sum(vectors_a * vectors_b, axis=1) / np.where(norms == 0, 1.0, norms)
    tokens_equal = [
        guard_fn(a) == guard_fn(b) for a, b in zip(texts_a, texts_b)
    ]
    scored = pairs.copy()
    scored["cosine"] = cosine
    scored["tokens_equal"] = tokens_equal
    return scored


def is_hit(scored: pd.DataFrame, threshold: float) -> np.ndarray:
    """True where Cordon would serve a cache hit at this threshold."""
    return (scored["cosine"].to_numpy() >= threshold) & scored["tokens_equal"].to_numpy()


def sweep(scored: pd.DataFrame, thresholds) -> pd.DataFrame:
    """
    Evaluate every threshold.

    Returns: one row per threshold with
    hits - pairs that would be served from the cache;
    wrong_hits - hits on pairs that are really different questions;
    wrong_hit_rate - wrong_hits / hits (share of cache answers that are wrong);
    hit_rate - correct hits / all true duplicates (share of savings captured).
    """
    labels = scored["label"].to_numpy() == 1
    positives = int(labels.sum())
    rows = []
    for threshold in thresholds:
        hit = is_hit(scored, threshold)
        hits = int(hit.sum())
        wrong = int((hit & ~labels).sum())
        rows.append({
            "threshold": round(float(threshold), 4),
            "hits": hits,
            "wrong_hits": wrong,
            "wrong_hit_rate": wrong / hits if hits else 0.0,
            "hit_rate": (hits - wrong) / positives if positives else 0.0,
        })
    return pd.DataFrame(rows)


def choose_threshold(sweep_table: pd.DataFrame, max_wrong_hit_rate: float, min_hits: int = 30) -> float | None:
    """
    The lowest threshold whose wrong-hit rate is within the limit, with enough hits to be meaningful.

    Why the lowest: a lower threshold means more cache hits, so more savings.
    Why min_hits: a wrong-hit rate measured on a handful of hits says nothing.
    Returns: the threshold, or None if no threshold meets the limit.
    """
    eligible = sweep_table[(sweep_table["wrong_hit_rate"] <= max_wrong_hit_rate) & (sweep_table["hits"] >= min_hits)]
    return None if eligible.empty else float(eligible["threshold"].min())


def bootstrap_wrong_hit_ci(scored: pd.DataFrame, threshold: float, resamples: int = 2000, seed: int = SEED) -> tuple[float, float]:
    """
    95% bootstrap confidence interval for the wrong-hit rate at a threshold.

    Resamples the pairs with replacement; returns (lower, upper).
    """
    rng = np.random.default_rng(seed)
    hit = is_hit(scored, threshold)
    wrong = hit & (scored["label"].to_numpy() != 1)
    n = len(scored)
    rates = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        hits = hit[idx].sum()
        rates.append(wrong[idx].sum() / hits if hits else 0.0)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))
