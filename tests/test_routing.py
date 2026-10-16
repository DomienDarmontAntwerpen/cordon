"""
src/savings/routing.py: request classification, candidate models, the ladder, escalation.

Run: .venv/bin/python -m pytest tests/test_routing.py -v
"""
import pytest



from helpers import CONFIG


def test_classify_request_categories():
    """classify request categories."""
    from src.savings import routing

    assert routing.classify_request([{"role": "user", "content": "Please extract the following fields: name, date"}], None)[0] == "extraction"
    assert routing.classify_request([{"role": "user", "content": "Summarize this article for me please"}], None)[0] == "summarization"
    assert routing.classify_request([{"role": "user", "content": "```python\nprint(1)\n```"}], None)[0] == "code"
    assert routing.classify_request([{"role": "user", "content": "What's the weather like?"}], None)[0] == "chat"


def test_classify_request_difficulty_hard_reasoning():
    """classify request difficulty hard reasoning."""
    from src.savings import routing

    category, difficulty = routing.classify_request(
        [{"role": "user", "content": "Solve this step by step: first, find x. then, find y. finally, compute x+y."}], None
    )
    assert category == "reasoning"
    assert difficulty == "hard"


def test_classify_request_response_format_hints_extraction():
    """classify request response format hints extraction."""
    from src.savings import routing

    category, _difficulty = routing.classify_request(
        [{"role": "user", "content": "give me data"}], {"type": "json_schema", "json_schema": {}}
    )
    assert category == "extraction"


def test_classify_request_recognises_code_requests_without_a_code_block():
    """"Write a Python function..." is a code request (the golden set's code items were classified as chat)."""
    from src.savings import routing

    category, _ = routing.classify_request([{"role": "user", "content": "Write a Python function add(a, b) that returns the sum."}], None)
    assert category == "code"


def test_classify_request_uses_only_the_users_question():
    """A system prompt that says 'summarize' or 'you must' must not decide the category or difficulty."""
    from src.savings import routing

    messages = [
        {"role": "system", "content": "You must summarize carefully. You should ensure accuracy and need to be brief."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    assert routing.classify_request(messages, None) == ("chat", "easy")
    assert routing.classify_request([{"role": "user", "content": "I need to build a class schedule for next week."}], None)[0] != "code"
    assert routing.classify_request([{"role": "user", "content": "How do we implement a new travel policy?"}], None)[0] != "code"


def test_candidate_models_always_includes_strong():
    """candidate models always includes strong."""
    from src.savings import routing

    config = {
        "money": CONFIG["money"],
        "models": {
            "model_aliases": {"mistral-small": "mistral/mistral-small-latest", "gpt-4o": "openai/gpt-4o"},
            "default_pool": ["mistral-small"], "default_strong": "gpt-4o", "capability_overrides": {},
        }
    }
    tenant_settings = {"allowed_providers": ["mistral", "openai"]}
    candidates, strong = routing.candidate_models(tenant_settings, config, set(), {}, mock_mode=True)
    assert strong == "gpt-4o"
    assert "gpt-4o" in candidates
    assert "mistral-small" in candidates


def test_candidate_models_excludes_disallowed_provider():
    """candidate models excludes disallowed provider."""
    from src.savings import routing

    config = {
        "money": CONFIG["money"],
        "models": {
            "model_aliases": {"mistral-small": "mistral/mistral-small-latest", "gpt-4o": "openai/gpt-4o"},
            "default_pool": ["mistral-small"], "default_strong": "gpt-4o", "capability_overrides": {},
        }
    }
    tenant_settings = {"allowed_providers": ["openai"]}
    candidates, _strong = routing.candidate_models(tenant_settings, config, set(), {}, mock_mode=True)
    assert "mistral-small" not in candidates


CANDIDATES = ["mistral-small", "gpt-4o-mini", "gpt-4o"]


def test_choose_route_starts_with_strong_without_table_entry(monkeypatch):
    """Without benchmark evidence for a bucket, the ladder starts with the strong model (cheaper ones only as failover)."""
    from src.savings import routing

    monkeypatch.setattr(routing, "ROUTING_TABLE", {})
    assert routing.choose_route("chat", "easy", CANDIDATES, "gpt-4o", frozenset()) == (["gpt-4o", "mistral-small", "gpt-4o-mini"], None)


def test_a_route_puts_its_model_first_and_the_strong_model_second(monkeypatch):
    """The proven model answers first, then the strong model - never an unproven, unchecked model after them."""
    from src.savings import routing

    route = routing.Route("gpt-4o-mini", 0.7, 0.4)
    monkeypatch.setattr(routing, "ROUTING_TABLE", {"reasoning": {"easy": route}})
    ladder, taken = routing.choose_route("reasoning", "easy", CANDIDATES, "gpt-4o", frozenset({"gpt-4o-mini"}))
    assert (ladder, taken) == (["gpt-4o-mini", "gpt-4o"], route)


def test_a_route_is_not_taken_where_its_check_cannot_run(monkeypatch):
    """A route proven with the classifier needs the model's confidence; a stream or a tool call cannot be checked at all."""
    from src.savings import routing

    monkeypatch.setattr(routing, "ROUTING_TABLE", {"reasoning": {"easy": routing.Route("gpt-4o-mini", 0.7, 0.4)}})
    assert routing.choose_route("reasoning", "easy", CANDIDATES, "gpt-4o", frozenset())[1] is None
    assert routing.choose_route("reasoning", "easy", CANDIDATES, "gpt-4o", frozenset({"gpt-4o-mini"}), answer_uncheckable=True)[1] is None
    assert routing.choose_route("reasoning", "easy", ["gpt-4o"], "gpt-4o", frozenset({"gpt-4o-mini"}))[1] is None  # not allowed


def test_the_routing_table_file_loads_as_routes(tmp_path, monkeypatch):
    """What route_proof.write_routing_table writes is what routing.py reads."""
    from src.savings import routing
    from evaluation import route_proof

    path = tmp_path / "routing_table.yaml"
    route_proof.write_routing_table({}, path)
    monkeypatch.setattr(routing, "_ROUTING_TABLE_PATH", path)
    assert routing._load_routing_table() == {}
    path.write_text("categories:\n  code:\n    easy: {model: gpt-4o-mini, gate_threshold: 0.7, escalated: 0.46}\n")
    assert routing._load_routing_table() == {"code": {"easy": routing.Route("gpt-4o-mini", 0.7, 0.46)}}


def test_a_chat_turn_with_earlier_replies_is_a_conversation():
    """Conversations are measured and routed apart from one-off questions; other kinds keep their category."""
    from src.savings import routing

    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello! How can I help?"}]
    assert routing.classify_request([{"role": "user", "content": "Where is my order?"}], None)[0] == "chat"
    assert routing.classify_request(history + [{"role": "user", "content": "Where is my order?"}], None)[0] == "conversation"
    assert routing.classify_request(history + [{"role": "user", "content": "Classify this as spam or not: win now"}], None)[0] == "classification"


_HISTORY = [{"role": "user", "content": "I ordered a jacket last week."}, {"role": "assistant", "content": "It ships today."}]


@pytest.mark.parametrize("messages", [
    # earlier arithmetic: the follow-up is arithmetic too, which no cheaper model is proven on
    [{"role": "user", "content": "Tom has 12 apples and gives 5 away."}, {"role": "assistant", "content": "7 are left."},
     {"role": "user", "content": "And if he then buys 3 bags of 4 apples each, what is his total now?"}],
    [{"role": "user", "content": "Tom has 12 apples and gives 5 away. How many are left?"},
     {"role": "assistant", "content": "7."}, {"role": "user", "content": "And if he then buys 3 bags of 4 apples each?"}],
    # instructions (a coding assistant), tools, attachments, code and long contexts were never in the evidence
    [{"role": "user", "content": "Can you help with my script?"}, {"role": "assistant", "content": "```python\nprint(1)\n```"},
     {"role": "user", "content": "Now make it faster."}],
    [{"role": "system", "content": "You are a Python coding assistant."}, *_HISTORY, {"role": "user", "content": "Can you change it?"}],
    [{"role": "user", "content": "What is the weather in Ghent?"},
     {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}]},
     {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 14}"}],
    [{"role": "user", "content": [{"type": "text", "text": "What is in this picture?"},
                                  {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]},
     {"role": "assistant", "content": "A dog."}, {"role": "user", "content": "What breed is it?"}],
    [{"role": "user", "content": "Here is our contract. " * 300}, {"role": "assistant", "content": "Noted."},
     {"role": "user", "content": "Can I cancel it?"}],
])
def test_only_the_proven_shape_of_conversation_takes_its_route(messages):
    """A conversation is plain user/assistant text within the evidence's size, with chat questions only."""
    from src.savings import routing

    assert routing.classify_request(messages, None)[0] != "conversation"
    assert routing.classify_request(_HISTORY + [{"role": "user", "content": "Can I change the address?"}], None)[0] == "conversation"
