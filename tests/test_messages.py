"""
src/messages.py: reading every message shape the same way.

Run: .venv/bin/python -m pytest tests/test_messages.py -v
"""




def test_messages_module_reads_every_message_shape():
    """Plain strings, lists of parts and both instruction roles are read the same way everywhere."""
    from src import messages

    request = [
        {"role": "developer", "content": [{"type": "text", "text": "Be brief."}]},
        {"role": "system", "content": "Answer in English."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}]},
        {"role": "user", "content": [{"type": "text", "text": "Summarize"}, {"type": "file", "file": {"file_data": "..."}}]},
    ]
    assert messages.instructions_text(request) == "Be brief.\nAnswer in English."
    assert messages.last_user_index(request) == 4
    assert messages.last_user_text(request) == "Summarize"
    assert not messages.is_text_only(request[4]) and messages.is_text_only(request[2])
    assert messages.text_fields(request[3]) == ["{\"x\": 1}"]
    upper = messages.map_text_fields(request[4], str.upper)
    assert upper["content"][0]["text"] == "SUMMARIZE" and upper["content"][1] == request[4]["content"][1]
