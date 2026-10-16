"""
src/server/telemetry.py: what reaches Langfuse.

Run: .venv/bin/python -m pytest tests/test_telemetry.py -v
"""




def test_telemetry_none_mode_sends_no_content(monkeypatch):
    """Telemetry none mode sends no content."""
    from src.server.telemetry import Telemetry

    captured = {}

    class FakeObservation:
        """FakeObservation."""
        def end(self):
            """End."""
            pass

    class FakeClient:
        """FakeClient."""
        def start_observation(self, **kwargs):
            """Start observation."""
            captured.update(kwargs)
            return FakeObservation()

    telemetry = Telemetry("pk-fake", "sk-fake", "https://cloud.langfuse.com")
    telemetry._client = FakeClient()
    future = telemetry.record_request(
        "t1", "none", {"served_model": "gpt-4o"}, input_text="the real prompt text", output_text="the real answer text"
    )
    future.result(timeout=5)
    assert captured["input"] is None
    assert captured["output"] is None


def test_telemetry_redacted_mode_redacts_pii(monkeypatch):
    """Telemetry redacted mode redacts pii."""
    from src.server.telemetry import Telemetry

    captured = {}

    class FakeObservation:
        """FakeObservation."""
        def end(self):
            """End."""
            pass

    class FakeClient:
        """FakeClient."""
        def start_observation(self, **kwargs):
            """Start observation."""
            captured.update(kwargs)
            return FakeObservation()

    telemetry = Telemetry("pk-fake", "sk-fake", "https://cloud.langfuse.com")
    telemetry._client = FakeClient()
    future = telemetry.record_request(
        "t1", "redacted", {"served_model": "gpt-4o"}, input_text="contact me at person@example.com", output_text="ok"
    )
    future.result(timeout=5)
    assert "person@example.com" not in captured["input"]
    assert "[REDACTED_EMAIL]" in captured["input"]
