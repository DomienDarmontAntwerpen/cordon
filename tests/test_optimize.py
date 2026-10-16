"""
src/savings/optimize.py: meaning-preserving trim, history and tool-output limits, image tokens and pixel limits.

Run: .venv/bin/python -m pytest tests/test_optimize.py -v
"""

import pytest


from helpers import png_data_url


def test_trim_text_deduplicates_sentences_and_paragraphs():
    """trim_text deduplicates sentences and paragraphs."""
    from src.savings import optimize

    text = "Hello there. Hello there. This is new.\n\nHello there. Hello there. This is new.\n\nParagraph two."
    result = optimize.trim_text(text)
    assert result.count("Hello there.") == 1
    assert "This is new." in result
    assert "Paragraph two." in result


def test_trim_text_never_touches_code_blocks():
    """trim_text never touches code blocks."""
    from src.savings import optimize

    text = "Look:\n\n```python\nprint(1)\nprint(1)\n```\n\nThanks. Thanks."
    result = optimize.trim_text(text)
    assert "```python\nprint(1)\nprint(1)\n```" in result
    assert result.count("Thanks.") == 1


def test_trim_text_never_removes_negations():
    """trim_text never removes negations."""
    from src.savings import optimize

    text = "The server is not running. The server is not running."
    result = optimize.trim_text(text)
    assert "not running" in result
    assert result.count("not running") == 1


def test_trim_messages_leaves_system_and_tool_messages_untouched():
    """trim_messages leaves system and tool messages untouched."""
    from src.savings import optimize

    messages = [
        {"role": "system", "content": "Repeat. Repeat."},
        {"role": "user", "content": "Repeat. Repeat."},
        {"role": "tool", "tool_call_id": "1", "content": "Repeat. Repeat."},
    ]
    trimmed = optimize.trim_messages(messages)
    assert trimmed[0]["content"] == "Repeat. Repeat."
    assert trimmed[1]["content"].count("Repeat.") == 1
    assert trimmed[2]["content"] == "Repeat. Repeat."


def test_trim_history_keeps_system_and_last_user_message():
    """trim_history keeps system and last user message."""
    from src.savings import optimize

    messages = [{"role": "system", "content": "sys"}]
    for _ in range(20):
        messages.append({"role": "user", "content": "word " * 50})
        messages.append({"role": "assistant", "content": "reply " * 50})
    trimmed = optimize.trim_history(messages, history_max_tokens=200, litellm_name="gpt-4o")
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1] == messages[-1]
    assert len(trimmed) < len(messages)


def test_trim_history_noop_when_already_under_budget():
    """trim_history noop when already under budget."""
    from src.savings import optimize

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert optimize.trim_history(messages, history_max_tokens=6000, litellm_name="gpt-4o") == messages


def test_truncate_tool_output_adds_marker_with_excess_count():
    """truncate_tool_output adds marker with excess count."""
    from src.savings import optimize

    message = {"role": "tool", "tool_call_id": "1", "content": "word " * 2000}
    result = optimize.truncate_tool_output([message], tool_output_max_tokens=50, litellm_name="gpt-4o")
    assert "truncated" in result[0]["content"]
    assert "by Cordon" in result[0]["content"]
    assert len(result[0]["content"]) < len(message["content"])


def test_truncate_tool_output_noop_under_budget():
    """truncate_tool_output noop under budget."""
    from src.savings import optimize

    message = {"role": "tool", "tool_call_id": "1", "content": "short result"}
    result = optimize.truncate_tool_output([message], tool_output_max_tokens=50, litellm_name="gpt-4o")
    assert result[0]["content"] == "short result"


@pytest.mark.parametrize("model, width, height, detail, billed", [
    # what OpenAI billed live (16 October 2026): prompt tokens with the image minus without it
    ("gpt-4o", 2000, 1500, "high", 765), ("gpt-4o", 2000, 1500, "low", 85), ("gpt-4o", 512, 384, "high", 255),
    ("gpt-4o-mini", 2000, 1500, "high", 25501), ("gpt-4o-mini", 2000, 1500, "low", 2833), ("gpt-4o-mini", 512, 384, "high", 8500),
    ("gpt-4.1-mini", 2000, 1500, "low", 4796), ("gpt-4.1-mini", 512, 384, "high", 311), ("gpt-4.1-mini", 4000, 3000, "high", 4976),
    ("gpt-4.1-mini", 1000, 1000, "high", 1658), ("gpt-4.1-mini", 700, 300, "high", 356),
    ("gpt-4.1-nano", 2000, 1500, "high", 7284), ("gpt-4.1-nano", 512, 384, "high", 472),
])
def test_image_tokens_match_what_openai_billed(model, width, height, detail, billed):
    """Each model is billed by its own rule (gpt-4o-mini ~33x gpt-4o's tokens; the gpt-4.1 minis per 32-px patch)."""
    from src.savings import optimize

    assert optimize.image_tokens(f"openai/{model}", width, height, detail) == billed


def test_image_savings_are_zero_when_the_tile_count_is_unchanged():
    """No saving is counted when shrinking leaves the billed tiles the same."""
    from src.savings import optimize

    part = {"type": "image_url", "image_url": {"url": png_data_url(4000, 3000)}}
    # A generous max_side that still collapses to the same tile count OpenAI
    # would have produced from the original image: savings must be 0.
    result = optimize.optimize_image_part(part, image_max_side=1600, detail_low_for_photos=False, is_document=True, model="gpt-4o")
    assert result.savings_tokens == 0


def test_optimize_image_part_downgrades_detail_and_reports_savings():
    """optimize_image_part downgrades detail and reports savings."""
    from src.savings import optimize

    part = {"type": "image_url", "image_url": {"url": png_data_url(3000, 4000)}}
    result = optimize.optimize_image_part(part, image_max_side=512, detail_low_for_photos=True, is_document=False, model="gpt-4o")
    assert result.detail == "low"
    assert result.savings_tokens > 0
    assert result.exif_stripped


def test_check_image_pixel_limits_rejects_oversized_image():
    """check_image_pixel_limits rejects oversized image."""
    from src.savings import optimize

    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": png_data_url(10000, 10000)}}]}]
    with pytest.raises(optimize.ImageTooLargeError):
        optimize.check_image_pixel_limits(messages, max_image_pixels=40_000_000)


def test_check_image_pixel_limits_allows_image_within_limit():
    """check_image_pixel_limits allows image within limit."""
    from src.savings import optimize

    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": png_data_url(800, 600)}}]}]
    optimize.check_image_pixel_limits(messages, max_image_pixels=40_000_000)  # must not raise


def test_check_image_pixel_limits_ignores_remote_urls():
    """check_image_pixel_limits ignores remote urls."""
    from src.savings import optimize

    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/huge.png"}}]}]
    optimize.check_image_pixel_limits(messages, max_image_pixels=1)  # must not raise - Cordon never fetches it


def test_a_request_is_counted_with_its_images_as_the_model_bills_them():
    """The cost without Cordon (baseline) includes each attached image at the requested model's own rate."""
    from src.savings import costing

    image = {"type": "image_url", "image_url": {"url": png_data_url(2000, 1500), "detail": "high"}}
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}, image]}]
    text_only = costing.count_message_tokens("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}])
    assert costing.count_message_tokens("openai/gpt-4o-mini", messages) - text_only == 25501
    assert costing.count_message_tokens("openai/gpt-4o", messages) - text_only == 765
