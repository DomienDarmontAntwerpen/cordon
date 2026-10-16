"""
A company's working day, measured live: the same requests straight to gpt-4o, and through Cordon.

Questions are fine for a benchmark but tell a buyer little: what matters is the bill for real
traffic. This builds one day of realistic requests from public data, with realistic sizes and
repetition, and sends every request twice - directly to gpt-4o (what the company pays today) and
through an in-process Cordon with its savings on - measuring the real cost and the answer quality
of both.

- classification: short texts to label (news topic, review sentiment) - known correct answers
- support: customer-support questions, popular ones asked again and again (Bitext)
- chat: support questions sent with the conversation so far, as chat apps do (Bitext)
- summaries: news articles to summarize (CNN/DailyMail)

Answers are scored as in the benchmark: exact labels for classification; for the rest, ROUGE-L plus
an LLM judge against the dataset's reference answer. Data is downloaded on demand to data/external/.

    .venv/bin/python -m evaluation.company_day --max-eur 6
"""

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

from evaluation import benchmark
from evaluation.fetch_eval_sets import EXTERNAL_DIR, _download
from src.paths import REPORTS_DIR

SEED = 42
BITEXT_URL = ("https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/"
              "resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
CNN_URL = "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/1.0.0/test-00000-of-00001.parquet"
CLASSIFICATION_FILES = ("extra_sst2_200.jsonl", "extra_agnews_200_2.jsonl")
DAY = {"classification": 100, "support": 80, "chat": 60, "summaries": 60}
SUPPORT_UNIQUE = 30  # the support questions asked; popular ones come back (Zipf), as in a real help desk
REPORT_PATH = REPORTS_DIR / "company_day.json"


# --- Section: the day's requests ---
def build_day(rng: random.Random) -> list[dict]:
    """The day's requests, each a benchmark item with a `kind`, in a shuffled order."""
    support = pd.read_csv(_download("bitext.csv", BITEXT_URL)).sample(frac=1, random_state=SEED)
    items = _classification(rng) + _support(support, rng) + _chats(support) + _summaries()
    rng.shuffle(items)
    return items


def _classification(rng: random.Random) -> list[dict]:
    """Short texts to label, with known answers."""
    pool = [i for name in CLASSIFICATION_FILES for i in benchmark.load_golden_set(EXTERNAL_DIR / name)]
    return [{**item, "kind": "classification"} for item in rng.sample(pool, DAY["classification"])]


def _support(frame: pd.DataFrame, rng: random.Random) -> list[dict]:
    """Support questions; the popular ones are asked again and again (weights 1/rank)."""
    unique = frame.head(SUPPORT_UNIQUE)
    picks = rng.choices(range(SUPPORT_UNIQUE), weights=[1 / (rank + 1) for rank in range(SUPPORT_UNIQUE)], k=DAY["support"])
    return [_reference_item(f"support_{i}", "support", [{"role": "user", "content": unique.iloc[i]["instruction"]}],
                            unique.iloc[i]["response"]) for i in picks]


def _chats(frame: pd.DataFrame, count: int = DAY["chat"]) -> list[dict]:
    """A support question sent with the four earlier turns of the conversation, as chat apps send it."""
    rows = frame.iloc[SUPPORT_UNIQUE:SUPPORT_UNIQUE + count * 5]
    items = []
    for n in range(count):
        turns = rows.iloc[n * 5:(n + 1) * 5]
        history = [m for _, t in turns.iloc[:4].iterrows()
                   for m in ({"role": "user", "content": t["instruction"]}, {"role": "assistant", "content": t["response"]})]
        last = turns.iloc[4]
        items.append(_reference_item(f"chat_{n}", "chat", history + [{"role": "user", "content": last["instruction"]}],
                                     last["response"]))
    return items


def _summaries() -> list[dict]:
    """News articles to summarize, with the article's own highlights as the reference."""
    return _summary_items(pd.read_parquet(_download("cnn_dailymail_test.parquet", CNN_URL)).sample(DAY["summaries"], random_state=SEED))


def _summary_items(frame: pd.DataFrame) -> list[dict]:
    """A summary request per article."""
    return [_reference_item(f"summary_{n}", "summaries",
                            [{"role": "user", "content": f"Summarize this article in three sentences:\n\n{row['article']}"}],
                            row["highlights"]) for n, (_, row) in enumerate(frame.iterrows())]


def _reference_item(item_id: str, kind: str, messages: list[dict], reference: str) -> dict:
    """A request scored against a reference answer (ROUGE-L plus the LLM judge, as summaries are)."""
    return {"id": item_id, "kind": kind, "category": "summarization", "messages": messages, "expected": reference}


# --- Section: route evidence for the long requests ---
def build_route_rounds(count: int = 200) -> list[Path]:
    """
    Chats and summaries never used by the day (later Bitext rows, other CNN articles), written as
    question files for `benchmark --collect`: evidence on whether a cheaper model can take them.
    """
    support = pd.read_csv(_download("bitext.csv", BITEXT_URL)).sample(frac=1, random_state=SEED)
    start = SUPPORT_UNIQUE + DAY["chat"] * 5  # past every row the day uses
    chats = [{**item, "id": f"chatround_{n}"} for n, item in enumerate(_chats(support.iloc[start - SUPPORT_UNIQUE:], count))]
    articles = pd.read_parquet(_download("cnn_dailymail_test.parquet", CNN_URL))
    used = set(articles.sample(DAY["summaries"], random_state=SEED).index)
    fresh = articles.drop(index=list(used)).sample(count, random_state=SEED + 1)
    summaries = [{**item, "id": f"summaryround_{n}"} for n, item in enumerate(_summary_items(fresh))]
    paths = []
    for name, items in (("extra_chats", chats), ("extra_summaries", summaries)):
        path = EXTERNAL_DIR / f"{name}_{len(items)}.jsonl"
        path.write_text("".join(json.dumps({k: v for k, v in i.items() if k != "kind"}) + "\n" for i in items), encoding="utf-8")
        paths.append(path)
    return paths


# --- Section: confirmation rounds for the long-request routes ---
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
GENERAL_KINDS = ("open_qa", "general_qa", "brainstorming", "creative_writing")  # questions without a given text


def build_conversation_confirmation(count: int = 200) -> Path:
    """
    Fresh support conversations (Bitext rows past every earlier round: the day's 330 and the route
    round's 1,000), to confirm the conversation route on data it was not chosen on.
    """
    support = pd.read_csv(_download("bitext.csv", BITEXT_URL)).sample(frac=1, random_state=SEED)
    start = SUPPORT_UNIQUE + DAY["chat"] * 5 + 200 * 5  # past the day and the 200-chat route round
    chats = [{**item, "id": f"chatconfirm_{n}"} for n, item in enumerate(_chats(support.iloc[start - SUPPORT_UNIQUE:], count))]
    path = EXTERNAL_DIR / f"extra_chats_confirm_{len(chats)}.jsonl"
    path.write_text("".join(json.dumps({k: v for k, v in i.items() if k != "kind"}) + "\n" for i in chats), encoding="utf-8")
    return path


def build_confirmation_rounds(count: int = 200) -> list[Path]:
    """
    Fresh evidence for the chat and summary routes: summaries of CNN articles no earlier round used,
    and general questions (Dolly: open questions, advice, brainstorming, writing) with a reference
    answer - the everyday requests that land in the chat bucket.
    """
    articles = pd.read_parquet(_download("cnn_dailymail_test.parquet", CNN_URL))
    used = set(articles.sample(DAY["summaries"], random_state=SEED).index)
    used |= set(articles.drop(index=list(used)).sample(count, random_state=SEED + 1).index)
    fresh = articles.drop(index=list(used)).sample(count, random_state=SEED + 2)
    summaries = [{**item, "id": f"summaryconfirm_{n}"} for n, item in enumerate(_summary_items(fresh))]
    dolly = pd.read_json(_download("dolly.jsonl", DOLLY_URL), lines=True)
    general = dolly[dolly["category"].isin(GENERAL_KINDS)].sample(count, random_state=SEED)
    questions = [_reference_item(f"generalconfirm_{n}", "general", [{"role": "user", "content": row["instruction"]}],
                                 row["response"]) for n, (_, row) in enumerate(general.iterrows())]
    paths = []
    for name, items in (("extra_summaries_confirm", summaries), ("extra_general_confirm", questions)):
        path = EXTERNAL_DIR / f"{name}_{len(items)}.jsonl"
        path.write_text("".join(json.dumps({k: v for k, v in i.items() if k != "kind"}) + "\n" for i in items), encoding="utf-8")
        paths.append(path)
    return paths


# --- Section: the run ---
def run_day(items: list[dict], max_eur: float) -> list[dict]:
    """Send every request straight to gpt-4o and through Cordon; one scored record per request and route."""
    os.environ.update(MOCK_MODE="false", LITELLM_LOCAL_MODEL_COST_MAP="True")
    benchmark._ensure_throwaway_secrets()
    import tempfile

    from fastapi.testclient import TestClient

    from src.savings import quality
    from src.server import settings as settings_module
    from src.server.gateway import create_app

    settings_obj = settings_module.load_settings()
    config, strong = settings_obj.config, settings_obj.config["models"]["default_strong"]
    strong_name = config["models"]["model_aliases"][strong]
    guard = benchmark._BudgetGuard(max_eur)
    judge = lambda reference, answer: benchmark.llm_judge(strong_name, reference, answer, config, guard.record)  # noqa: E731
    records = []
    with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(settings_obj=settings_obj, db_path=f"{tmp}/day.sqlite")) as client:
        headers = _cordon_team(client, settings_obj.admin_key)
        for item in items:
            if guard.should_stop():
                break
            direct = benchmark.call_direct(strong_name, item, "A", strong, False, config)
            through = benchmark.call_through_cordon(client, headers, item, strong, "D", False)
            for route, result in (("direct", direct), ("cordon", through)):
                result = benchmark._finalize_result(item, result, route, strong, False, judge, quality.embed_texts)
                guard.record(result.cost_eur)
                records.append({"id": item["id"], "kind": item["kind"], "route": route, "cost_eur": result.cost_eur,
                                "score": 0.0 if result.error else result.score, "model": result.served_model or strong})
    return records


def _cordon_team(client, admin_key: str) -> dict:
    """A team with Cordon's savings and protections on, as a company would run it."""
    admin = {"Authorization": f"Bearer {admin_key}"}
    team = client.post("/admin/tenants", headers=admin, json={
        "name": "company day", "mode": "enforce", "holdout_pct": 0.0, "allowed_providers": ["openai"],
        "features": ["trim", "cache", "routing", "max_tokens", "circuit_breaker"], "pii_action": "mask",
        "rpm": 100000, "tpm": 100000000, "budget_daily_eur": 1000.0, "budget_monthly_eur": 1000.0}).json()
    return {"Authorization": "Bearer " + client.post(f"/admin/tenants/{team['id']}/keys", headers=admin, json={}).json()["key"]}


# --- Section: the result ---
def summarize(records: list[dict]) -> dict:
    """Per kind of request and for the whole day: cost straight vs through Cordon, and answer quality of both."""
    rows = {}
    for kind in [*DAY, "whole day"]:
        mine = [r for r in records if kind == "whole day" or r["kind"] == kind]
        # Records come in pairs (direct, then Cordon, for the same request), so the two lists line up.
        direct, cordon = [r for r in mine if r["route"] == "direct"], [r for r in mine if r["route"] == "cordon"]
        if not direct:
            continue
        cost_d, cost_c = sum(r["cost_eur"] for r in direct), sum(r["cost_eur"] for r in cordon)
        diffs = [c["score"] - d["score"] for d, c in zip(direct, cordon)]
        # The route proof's rule: classification is right/wrong; the other kinds are graded answers,
        # a loss only when Cordon's answer failed outright.
        losses = [diff if d["kind"] == "classification" or (c["score"] == 0.0 < d["score"]) else 0.0
                  for diff, d, c in zip(diffs, direct, cordon)]
        verdict = benchmark.paired_verdict(diffs, loss_diffs=losses)
        rows[kind] = {"requests": len(direct), "eur_direct": round(cost_d, 4), "eur_cordon": round(cost_c, 4),
                      "saved_pct": round(100 * (1 - cost_c / cost_d), 1) if cost_d else 0.0,
                      "quality_direct": round(sum(r["score"] for r in direct) / len(direct), 3),
                      "quality_cordon": round(sum(r["score"] for r in cordon) / len(cordon), 3),
                      "quality_verdict": verdict.verdict, "quality_interval": [round(verdict.lower, 3), round(verdict.upper, 3)]}
    return rows


def main(argv: list[str] | None = None) -> int:
    """Build the day, run it live, print and save the result."""
    parser = argparse.ArgumentParser(description="One realistic working day, live: straight to gpt-4o vs through Cordon.")
    parser.add_argument("--max-eur", type=float, default=6.0)
    parser.add_argument("--route-rounds", action="store_true", help="instead: write the chat and summary evidence rounds")
    parser.add_argument("--confirm-rounds", action="store_true", help="instead: write the fresh confirmation rounds")
    args = parser.parse_args(argv)
    if args.confirm_rounds:
        print("\n".join(f"wrote {p}" for p in [*build_confirmation_rounds(), build_conversation_confirmation()]))
        return 0
    if args.route_rounds:
        print("\n".join(f"wrote {p}" for p in build_route_rounds()))
        return 0
    records = run_day(build_day(random.Random(SEED)), args.max_eur)
    result = summarize(records)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({"summary": result, "records": records}, indent=1), encoding="utf-8")
    print(f"{'kind':15s} {'requests':>8s} {'EUR direct':>11s} {'EUR Cordon':>11s} {'saved':>7s} {'quality direct/Cordon':>22s}  verdict")
    for kind, r in result.items():
        print(f"{kind:15s} {r['requests']:8d} {r['eur_direct']:11.4f} {r['eur_cordon']:11.4f} {r['saved_pct']:6.1f}% "
              f"{r['quality_direct']:>10.3f} / {r['quality_cordon']:.3f}  {r['quality_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
