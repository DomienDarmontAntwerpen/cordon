"""
src/savings/cache.py: eligibility rules, tenant isolation, semantic matching and its guard tokens, no text stored.

Run: .venv/bin/python -m pytest tests/test_cache.py -v
"""



from helpers import make_cache


def test_cache_eligibility_never_written_rules():
    """cache eligibility never written rules."""
    from src.savings import cache

    assert not cache.check_eligibility(True, False, False, False, 0.5, 1.0, "hi", False).eligible
    assert not cache.check_eligibility(False, True, False, False, 0.5, 1.0, "hi", False).eligible
    assert not cache.check_eligibility(False, False, True, False, 0.5, 1.0, "hi", False).eligible
    assert not cache.check_eligibility(False, False, False, True, 0.5, 1.0, "hi", False).eligible
    assert not cache.check_eligibility(False, False, False, False, 1.5, 1.0, "hi", False).eligible
    assert not cache.check_eligibility(False, False, False, False, 0.5, 1.0, "what is today's date", False).eligible
    assert not cache.check_eligibility(False, False, False, False, 0.5, 1.0, "hi", True).eligible
    assert cache.check_eligibility(False, False, False, False, 0.5, 1.0, "hi", False).eligible


def test_cache_eligibility_missing_temperature_counts_as_one():
    """cache eligibility missing temperature counts as one."""
    from src.savings import cache

    # cache_max_temperature below 1.0: a missing temperature (None) must be treated as 1.0 and therefore ineligible.
    assert not cache.check_eligibility(False, False, False, False, None, 0.5, "hi", False).eligible


def test_exact_cache_isolated_per_tenant(tmp_path):
    """exact cache isolated per tenant."""
    cache, _store = make_cache(tmp_path)
    messages = [{"role": "user", "content": "hello"}]
    key_a = cache.exact_key("tenant-a", "enforce", "openai/gpt-4o", messages, 0.0, None, None)
    key_b = cache.exact_key("tenant-b", "enforce", "openai/gpt-4o", messages, 0.0, None, None)
    assert key_a != key_b
    cache.put_exact("tenant-a", key_a, "answer for a")
    assert cache.get_exact("tenant-a", key_a) == "answer for a"
    assert cache.get_exact("tenant-b", key_a) is None


def test_semantic_cache_hits_rephrasing_but_not_negation_flip(tmp_path):
    """semantic cache hits rephrasing but not negation flip."""
    cache, _store = make_cache(tmp_path)
    messages = [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "What is the capital of France?"}]
    context_hash = cache.context_hash("t1", "enforce", "openai/gpt-4o", messages, 0.0, None, None)
    cache.put_semantic("t1", context_hash, "What is the capital of France?", "Paris.")

    hit = cache.get_semantic("t1", context_hash, "What is the capital city of France?")
    assert hit == "Paris."

    miss = cache.get_semantic("t1", context_hash, "What is not the capital of France?")
    assert miss is None


def test_cache_rows_never_contain_message_or_answer_text(tmp_path):
    """cache rows never contain message or answer text."""
    cache, store = make_cache(tmp_path)
    secret_question = "zzz_unique_question_marker_zzz"
    secret_answer = "zzz_unique_answer_marker_zzz"
    messages = [{"role": "user", "content": secret_question}]
    key = cache.exact_key("t1", "enforce", "openai/gpt-4o", messages, 0.0, None, None)
    cache.put_exact("t1", key, secret_answer)

    context_hash = cache.context_hash("t1", "enforce", "openai/gpt-4o", messages, 0.0, None, None)
    cache.put_semantic("t1", context_hash, secret_question, secret_answer)

    raw = open(store._db_path, "rb").read()
    assert secret_question.encode() not in raw
    assert secret_answer.encode() not in raw


def test_extract_negation_and_number_tokens():
    """extract negation and number tokens."""
    from src.savings import cache

    tokens = cache.extract_negation_and_number_tokens("The cost is not 500 euros, never 600.")
    assert "not" in tokens
    assert "never" in tokens
    assert "500" in tokens
    assert "600" in tokens


def test_semantic_guard_requires_same_names_in_same_order():
    """Questions that differ only in a name, or in the order of two names, must not share a semantic cache entry."""
    from src.savings import cache

    assert cache.extract_name_tokens("Which rehab center in Bartow County GA is best?") == ["bartow", "county", "ga"]
    assert cache.extract_name_tokens("What is cheaper in India than in the UK? Ask Paris.") == ["india", "uk", "paris"]
    assert cache.extract_name_tokens("I think I should go") == []  # sentence starts and "I" are not names
    same = cache.semantic_guard_tokens
    assert same("Best rehab center in Bartow County?") != same("Best rehab center in Towns County?")
    assert same("Is food cheaper in India than in the UK?") != same("Is food cheaper in the UK than in India?")
    assert same("How do I learn Python?") == same("How can I learn Python?")


def test_semantic_cache_never_stores_names_or_numbers_in_clear(tmp_path):
    """The semantic cache row stores its guard tokens (names, numbers) only as keyed hashes - never as readable text."""
    import os

    from src.savings.cache import Cache
    from src.server.store import Store

    store = Store(str(tmp_path / "cache.sqlite"))
    cache_obj = Cache(store, os.urandom(32), "salt-salt-salt-salt", semantic_threshold=0.97, cache_ttl_seconds=3600)
    cache_obj.put_semantic("tenant-a", "ctx", "Where does Jan Peeters live in Antwerpen, number 4711?", "an answer")
    raw = (tmp_path / "cache.sqlite").read_bytes() + b"".join(
        p.read_bytes() for p in tmp_path.glob("cache.sqlite-wal")
    )
    for fragment in (b"jan", b"peeters", b"antwerpen", b"Jan", b"Peeters", b"4711"):
        assert fragment not in raw
    assert cache_obj.get_semantic("tenant-a", "ctx", "Where does Jan Peeters live in Antwerpen, number 4711?") == "an answer"
    assert cache_obj.get_semantic("tenant-a", "ctx", "Where does Jan Janssens live in Antwerpen, number 4711?") is None
    assert cache_obj.get_semantic("tenant-a", "ctx", "Where does Jan Peeters live in Antwerpen, number 4712?") is None
