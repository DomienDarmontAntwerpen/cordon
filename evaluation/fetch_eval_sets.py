"""
Builds a larger benchmark set: our own 60 golden-set questions plus public questions with known answers.

Why: a routing "pass" needs at least 30 paired questions per kind of request,
and the golden set has 10 per category. Three public sets add questions whose
correct answer is fixed, so they are scored exactly, with no judge:
- GSM8K (MIT licence): arithmetic word problems -> "reasoning", the number is the answer;
- SST-2 (GLUE): movie-review sentiment -> "classification", positive or negative;
- MBPP sanitized (CC BY 4.0): Python tasks with unit tests -> "code", pass@1;
- AG News: news articles by topic -> "classification", so classification evidence is not
  only sentiment (a different task, four labels).
A fixed-seed sample of each is converted to the golden-set format and written,
with the golden set, to data/external/extended_golden_set.jsonl (downloads stay
in data/external/, which is not committed).

    .venv/bin/python -m evaluation.fetch_eval_sets                    # the evaluation set (60 per public set)
    .venv/bin/python -m evaluation.fetch_eval_sets --gate-set 200     # quality-gate training questions, disjoint
    .venv/bin/python -m evaluation.benchmark --live --golden-set data/external/extended_golden_set.jsonl
"""

import argparse
import io
import json
import random
from pathlib import Path

import httpx
import pandas as pd

from src.paths import DATA_DIR, GOLDEN_SET_PATH

EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_PATH = EXTERNAL_DIR / "extended_golden_set.jsonl"
GATE_SET_PATH = EXTERNAL_DIR / "gate_training_set.jsonl"
SEED = 42
SOURCES = {
    "gsm8k": "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet",
    "sst2": "https://huggingface.co/datasets/stanfordnlp/sst2/resolve/main/data/validation-00000-of-00001.parquet",
    "mbpp": "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/sanitized/test-00000-of-00001.parquet",
    "agnews": "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/test-00000-of-00001.parquet",
}
# Sources only used for extra rounds: the evaluation and gate sets never drew from them.
EXTRA_ONLY_SOURCES = {
    "mbpp_train": "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/sanitized/train-00000-of-00001.parquet",
}
AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


# --- Section: download ---
def load_source(name: str) -> pd.DataFrame:
    """The source's parquet file as a DataFrame, downloaded once into data/external/."""
    path = _download(f"{name}.parquet", {**SOURCES, **EXTRA_ONLY_SOURCES}[name])
    return pd.read_parquet(io.BytesIO(path.read_bytes()))


def _download(filename: str, url: str) -> Path:
    """A public dataset file, downloaded once into data/external/ (gitignored); returns its path."""
    path = EXTERNAL_DIR / filename
    if not path.exists():
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        response = httpx.get(url, follow_redirects=True, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


# --- Section: conversion to the golden-set format ---
def gsm8k_item(index: int, row) -> dict:
    """A GSM8K problem: the answer is the number after '####'."""
    answer = row["answer"].split("####")[-1].strip().replace(",", "")
    return {"id": f"gsm8k_{index}", "category": "reasoning", "source": "gsm8k",
            "messages": [{"role": "user", "content": f"{row['question']} Answer with just the number."}],
            "expected": answer}


def sst2_item(index: int, row) -> dict:
    """An SST-2 review: label 1 is positive, 0 negative."""
    review = row["sentence"].strip()
    return {"id": f"sst2_{index}", "category": "classification", "source": "sst2",
            "messages": [{"role": "user", "content": f'Classify the sentiment of this movie review as positive or negative: "{review}"'}],
            "expected": "positive" if int(row["label"]) == 1 else "negative"}


def mbpp_item(index: int, row) -> dict:
    """An MBPP task: the first test is shown (it fixes the function name), all tests are run."""
    tests = list(row["test_list"])
    setup = list(row["test_imports"]) if row["test_imports"] is not None else []
    prompt = (f"{row['prompt']}\nYour code should pass this test:\n{tests[0]}\n"
              "Return only the function definition in a python code block.")
    return {"id": f"mbpp_{index}", "category": "code", "source": "mbpp",
            "messages": [{"role": "user", "content": prompt}],
            "expected_test": "\n".join(setup + tests)}


def agnews_item(index: int, row) -> dict:
    """An AG News article: the topic is one of four labels."""
    article = row["text"].strip()
    return {"id": f"agnews_{index}", "category": "classification", "source": "agnews",
            "messages": [{"role": "user", "content":
                          f'Classify the topic of this news article as World, Sports, Business or Sci/Tech: "{article}"'}],
            "expected": AGNEWS_LABELS[int(row["label"])]}


def gsm8k_plain_item(index: int, row) -> dict:
    """A GSM8K problem asked as a user would, without "Answer with just the number" (which cheaper models obey and gpt-4o ignores)."""
    return {**gsm8k_item(index, row), "id": f"gsm8kplain_{index}", "source": "gsm8k_plain",
            "messages": [{"role": "user", "content": row["question"]}]}


def mbpp_train_item(index: int, row) -> dict:
    """An MBPP task from its training split (more code questions once the test split is used up)."""
    return {**mbpp_item(index, row), "id": f"mbpp_train_{index}", "source": "mbpp_train"}


CONVERTERS = {"gsm8k": gsm8k_item, "sst2": sst2_item, "mbpp": mbpp_item, "agnews": agnews_item,
              "mbpp_train": mbpp_train_item, "gsm8k_plain": gsm8k_plain_item}
# A variant asks the same dataset differently: it shares the dataset's rows, so it excludes the same ones.
VARIANT_OF = {"gsm8k_plain": "gsm8k"}


EVAL_PER_SET = 60  # questions per public set in the evaluation set (routing evidence)
GATE_PER_SET = 200  # questions per public set in the gate-training set


def split_indices(n_rows: int, eval_per_set: int, gate_per_set: int) -> tuple[list[int], list[int]]:
    """
    The evaluation sample (seed 42, unchanged since the first live run, so the routing
    evidence stays reproducible) and a gate-training sample (its own seed) drawn only from
    the rows left over - the quality gate never trains on an evaluation question.
    """
    eval_rows = sorted(random.Random(SEED).sample(range(n_rows), eval_per_set))
    taken = set(eval_rows)
    remaining = [i for i in range(n_rows) if i not in taken]
    gate_rows = sorted(random.Random(SEED + 1).sample(remaining, min(gate_per_set, len(remaining))))
    return eval_rows, gate_rows


def build(per_set: int = EVAL_PER_SET, output: Path = OUTPUT_PATH) -> dict[str, int]:
    """Write the golden set plus per_set evaluation items of every public source; return the count per source."""
    lines = [line for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = {"golden_set": len(lines)}
    for name in SOURCES:
        frame = load_source(name)
        eval_rows, _ = split_indices(len(frame), per_set, 0)
        counts[name] = len(eval_rows)
        lines += [json.dumps(CONVERTERS[name](i, frame.iloc[i]), ensure_ascii=False) for i in eval_rows]
    _write(output, lines)
    return counts


def build_gate_set(per_set: int, output: Path = GATE_SET_PATH) -> dict[str, int]:
    """Write per_set gate-training items of every public source, none of them in the evaluation set."""
    lines, counts = [], {}
    for name in SOURCES:
        frame = load_source(name)
        _, gate_rows = split_indices(len(frame), EVAL_PER_SET, per_set)
        counts[name] = len(gate_rows)
        lines += [json.dumps(CONVERTERS[name](i, frame.iloc[i]), ensure_ascii=False) for i in gate_rows]
    _write(output, lines)
    return counts


def extra_indices(n_rows: int, count: int, extra_only: bool = False, earlier: frozenset = frozenset()) -> list[int]:
    """
    `count` rows (seed 44) in neither the evaluation nor the gate-training sample (`extra_only`: a
    source those never drew from) nor an earlier extra round (`earlier`).
    """
    eval_rows, gate_rows = ([], []) if extra_only else split_indices(n_rows, EVAL_PER_SET, GATE_PER_SET)
    taken = set(eval_rows) | set(gate_rows) | set(earlier)
    remaining = [i for i in range(n_rows) if i not in taken]
    return sorted(random.Random(SEED + 2).sample(remaining, min(count, len(remaining))))


def build_extra(source: str, count: int) -> Path:
    """
    Extra questions of one source, disjoint from the evaluation and gate-training samples and from
    every earlier extra round of that source: more evidence for one kind of request, with the
    sample size fixed before anything is run. Round k is written to extra_<source>_<n>[_k].jsonl.
    """
    base = VARIANT_OF.get(source, source)
    frame = load_source(base)
    earlier_files = sorted({*EXTERNAL_DIR.glob(f"extra_{base}_*.jsonl"), *EXTERNAL_DIR.glob(f"extra_{source}_*.jsonl")})
    earlier = frozenset(int(json.loads(line)["id"].rsplit("_", 1)[1])
                        for path in earlier_files for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    rows = extra_indices(len(frame), count, extra_only=base in EXTRA_ONLY_SOURCES, earlier=earlier)
    suffix = f"_{len(earlier_files) + 1}" if earlier_files else ""
    path = EXTERNAL_DIR / f"extra_{source}_{len(rows)}{suffix}.jsonl"
    _write(path, [json.dumps(CONVERTERS[source](i, frame.iloc[i]), ensure_ascii=False) for i in rows])
    return path


def _write(path: Path, lines: list[str]) -> None:
    """Write JSONL lines, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Build the evaluation set (default) or, with --gate-set N, the gate-training set."""
    parser = argparse.ArgumentParser(description="Build the extended evaluation set or the gate-training set.")
    parser.add_argument("--per-set", type=int, default=EVAL_PER_SET, help="evaluation questions per public set")
    parser.add_argument("--gate-set", type=int, default=None, metavar="N",
                        help="instead: N gate-training questions per public set, disjoint from the evaluation set")
    parser.add_argument("--extra", nargs=2, metavar=("SOURCE", "N"), default=None,
                        help="instead: N extra questions of one source, disjoint from both other sets")
    args = parser.parse_args(argv)
    if args.extra:
        print(f"wrote {build_extra(args.extra[0], int(args.extra[1]))}")
    elif args.gate_set:
        print(f"wrote {GATE_SET_PATH}: {build_gate_set(args.gate_set)}")
    else:
        counts = build(args.per_set)
        print(f"wrote {OUTPUT_PATH}: {counts} ({sum(counts.values())} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
