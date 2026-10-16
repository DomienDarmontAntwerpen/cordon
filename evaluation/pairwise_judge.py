"""
A blind pairwise check of one route: for each question, a judge sees the conversation and two
replies (which model wrote which is hidden) and picks the one that helps the customer more.

Why: the reference-based score (ROUGE-L plus a 1-5 judge against the dataset's answer) favours
replies close to the reference's length; this asks the question a company cares about instead.
Each pair is judged twice, in both orders, so a judge's preference for the first or second reply
cancels out: a model wins a question only when it is preferred both times, otherwise it is a tie.

Decision rule (fixed before any judgement was made): the cheaper model's share - wins plus half
the ties - must have a 95% bootstrap interval whose lower end is at least 0.45 (0.5 = as good).
The judge is gpt-4o itself, which if anything favours its own replies.

    .venv/bin/python -m evaluation.pairwise_judge --questions data/external/extra_chats_confirm_200.jsonl \
        --cheap data/labelled_answers/answers_20261016T085811Z.jsonl \
        --strong data/labelled_answers/answers_20261016T091025Z.jsonl --max-eur 2
"""
import argparse
import json
import re
from pathlib import Path

import litellm
import yaml

from evaluation.benchmark import load_golden_set, paired_bootstrap_ci
from src.paths import CONFIG_PATH, REPORTS_DIR
from src.savings import costing, routing

JUDGE_MODEL = "openai/gpt-4o"
MIN_SHARE = 0.45
REPORT_PATH = REPORTS_DIR / "pairwise_conversations.json"
PROMPT = (
    "You review replies from a customer-support assistant. Read the conversation, then the two "
    "replies to the customer's last message. Which reply helps the customer more: correct, relevant "
    "to what was asked, clear, and complete enough? Length is not a merit in itself.\n"
    "Answer with only A, B or TIE.\n\n{conversation}\n\nReply A:\n{a}\n\nReply B:\n{b}"
)


def judge_once(conversation: str, a: str, b: str, config: dict, on_cost) -> str:
    """One judgement: "A", "B" or "TIE" ("TIE" too when the reply holds no clear choice)."""
    response = litellm.completion(model=JUDGE_MODEL, timeout=config["provider"]["timeout_seconds"],
                                  messages=[{"role": "user", "content": PROMPT.format(conversation=conversation, a=a, b=b)}])
    on_cost(costing.actual_cost_eur(JUDGE_MODEL, response.usage.prompt_tokens, response.usage.completion_tokens, config))
    match = re.search(r"\b(TIE|A|B)\b", (response.choices[0].message.content or "").upper())
    return match.group(1) if match else "TIE"


def outcome(first: str, second: str) -> float:
    """
    The cheaper model's score on one question from two judgements: first with the cheaper reply as
    A, second with it as B. 1 = preferred both times, 0 = the other preferred both times, else 0.5.
    """
    if first == "A" and second == "B":
        return 1.0
    if first == "B" and second == "A":
        return 0.0
    return 0.5


def conversation_text(messages: list[dict]) -> str:
    """The conversation as the judge reads it."""
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def run(questions: list[dict], cheap: dict, strong: dict, config: dict, max_eur: float, judge=judge_once) -> dict:
    """Judge every conversation both models answered; stop before the spend could pass max_eur."""
    spent = [0.0]
    records = []
    for item in questions:
        if routing.classify_request(item["messages"], None)[0] != "conversation":
            continue
        if item["id"] not in cheap or item["id"] not in strong or spent[0] > max_eur * 0.95:
            continue
        text, on_cost = conversation_text(item["messages"]), (lambda eur: spent.__setitem__(0, spent[0] + eur))
        first = judge(text, cheap[item["id"]], strong[item["id"]], config, on_cost)
        second = judge(text, strong[item["id"]], cheap[item["id"]], config, on_cost)
        records.append({"id": item["id"], "first": first, "second": second, "cheap_score": outcome(first, second)})
    return summarize(records, spent[0])


def summarize(records: list[dict], spent_eur: float) -> dict:
    """The cheaper model's share with its 95% interval, and the verdict by the fixed rule."""
    scores = [r["cheap_score"] for r in records]
    share = sum(scores) / len(scores) if scores else 0.0
    lower, upper = paired_bootstrap_ci(scores)
    return {
        "questions": len(scores), "cheap_wins": scores.count(1.0), "strong_wins": scores.count(0.0),
        "ties": scores.count(0.5), "cheap_share": round(share, 3), "interval": [round(lower, 3), round(upper, 3)],
        "verdict": "pass" if scores and lower >= MIN_SHARE else "fail", "spent_eur": round(spent_eur, 4),
        "records": records,
    }


def _answers(path: Path) -> dict:
    """item_id -> answer text, for answers without an error."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["item_id"]: r["answer"] for r in rows if not r.get("error") and r.get("answer")}


def main(argv: list[str] | None = None) -> int:
    """Run the pairwise check and write reports/pairwise_conversations.json."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--cheap", type=Path, required=True)
    parser.add_argument("--strong", type=Path, required=True)
    parser.add_argument("--max-eur", type=float, default=2.0)
    args = parser.parse_args(argv)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    result = run(load_golden_set(args.questions), _answers(args.cheap), _answers(args.strong), config, args.max_eur)
    REPORT_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print({k: v for k, v in result.items() if k != "records"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
