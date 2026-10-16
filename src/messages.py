"""
The one module that knows the shape of an OpenAI chat message.

- Content may be a string or a list of parts (text, images, files, audio).
- "system" and "developer" messages are the app's hidden instructions.
- Every other module reads messages through these helpers, so a new message
  shape is handled here once.
"""

from typing import Any, Callable

INSTRUCTION_ROLES = frozenset({"system", "developer"})


def is_instruction(message: dict) -> bool:
    """True for the app's hidden instructions: a "system" or "developer" message."""
    return message.get("role") in INSTRUCTION_ROLES


def message_text(content: Any) -> str:
    """
    The text of a message's content.

    Parameters: content - a message's "content": a string, a list of parts, or None.
    Returns: the string itself, or the "text" parts of a list joined by newlines.
    Image, file and audio parts contribute nothing; anything else gives "".
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        )
    return ""


def text_fields(message: dict) -> list[str]:
    """Every piece of text in a message: its text content and each tool call's arguments."""
    texts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        texts.extend(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        )
    for tool_call in message.get("tool_calls") or []:
        arguments = (tool_call.get("function") or {}).get("arguments")
        if isinstance(arguments, str):
            texts.append(arguments)
    return texts


def map_text_fields(message: dict, fn: Callable[[str], str]) -> dict:
    """
    A copy of `message` with every text field passed through fn.

    Text content (string or text parts) and tool-call arguments are rewritten;
    image, file and audio parts are copied unchanged. The input is not mutated.
    """
    new_message = dict(message)
    content = message.get("content")
    if isinstance(content, str):
        new_message["content"] = fn(content)
    elif isinstance(content, list):
        new_message["content"] = [
            {**part, "text": fn(part["text"])}
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
            else part
            for part in content
        ]
    if message.get("tool_calls"):
        new_calls = []
        for tool_call in message["tool_calls"]:
            new_call = dict(tool_call)
            function = dict(tool_call.get("function") or {})
            if isinstance(function.get("arguments"), str):
                function["arguments"] = fn(function["arguments"])
            new_call["function"] = function
            new_calls.append(new_call)
        new_message["tool_calls"] = new_calls
    return new_message


def last_user_index(messages: list[dict]) -> int | None:
    """Index of the last "user" message, or None if there is none."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return None


def last_user_text(messages: list[dict]) -> str:
    """The text of the last user message (the question being asked), or ""."""
    index = last_user_index(messages)
    return "" if index is None else message_text(messages[index].get("content"))


def instructions_text(messages: list[dict]) -> str | None:
    """The text of every system and developer message, joined; None if there is none."""
    joined = "\n".join(t for t in (message_text(m.get("content")) for m in messages if is_instruction(m)) if t)
    return joined or None


def is_plain_conversation(messages: list[dict], max_messages: int, max_words: int) -> bool:
    """
    True for a plain chat conversation: user and assistant text turns only (no instructions, tools
    or attachments), at least one earlier assistant reply, ending on the user's turn, and within
    max_messages and max_words.
    """
    if not messages or len(messages) > max_messages or messages[-1].get("role") != "user":
        return False
    if not any(m.get("role") == "assistant" for m in messages[:-1]):
        return False
    if any(m.get("role") not in ("user", "assistant") or m.get("tool_calls") or not is_text_only(m) for m in messages):
        return False
    return sum(len(message_text(m.get("content")).split()) for m in messages) <= max_words


def is_text_only(message: dict) -> bool:
    """True if the message carries no image, file, audio or other non-text part."""
    content = message.get("content")
    return not isinstance(content, list) or all(isinstance(part, dict) and part.get("type") == "text" for part in content)
