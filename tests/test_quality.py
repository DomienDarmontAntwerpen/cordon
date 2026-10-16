"""
src/savings/quality.py: answer features, JSON validation, the quality classifier.

Run: .venv/bin/python -m pytest tests/test_quality.py -v
"""

from helpers import CONFIG

CONF = {"mean_prob": 0.9, "min_prob": 0.5, "final_prob": 0.9, "body_prob": 0.9}  # an answer's confidence


def test_extract_features_good_vs_refusal():
    """extract features good vs refusal."""
    from src.savings import quality

    good = quality.extract_features("What is the capital of France?", "The capital of France is Paris.", CONF)
    refusal = quality.extract_features("What is the capital of France?", "I cannot help with that request.", CONF)
    assert good[2] == 0.0  # refusal_flag
    assert refusal[2] == 1.0
    assert good[0] > refusal[0]  # higher question-answer cosine for the on-topic answer


def test_extract_features_empty_answer_flagged():
    """extract features empty answer flagged."""
    from src.savings import quality

    features = quality.extract_features("What is the capital of France?", "", CONF)
    assert features[5] == 1.0  # empty_or_truncated_flag


def test_extract_features_error_flag():
    """extract features error flag."""
    from src.savings import quality

    features = quality.extract_features("How do I fix this?", "Traceback (most recent call last):\nTypeError: bad", CONF)
    assert features[3] == 1.0  # error_flag


def test_validate_structured_output_no_schema_is_valid():
    """validate structured output no schema is valid."""
    from src.savings import quality

    assert quality.validate_structured_output("anything at all", None)


def test_validate_structured_output_valid_and_invalid_json():
    """validate structured output valid and invalid json."""
    from src.savings import quality

    schema = {"type": "json_schema", "json_schema": {"schema": {"type": "object", "required": ["a"]}}}
    assert quality.validate_structured_output('{"a": 1}', schema)
    assert not quality.validate_structured_output("not json", schema)
    assert not quality.validate_structured_output("{}", schema)


def test_score_answer_good_vs_bad(cordon_settings):
    """score_answer good vs bad."""
    from src.savings import quality

    config = cordon_settings.config
    good_score = quality.score_answer("What is the capital of France?", "The capital of France is Paris.", config, confidence=CONF)
    bad_score = quality.score_answer("What is the capital of France?", "I cannot help with that request.", config)
    assert good_score > bad_score


def test_score_answer_invalid_json_schema_scores_zero(cordon_settings):
    """score_answer invalid json schema scores zero."""
    from src.savings import quality

    config = cordon_settings.config
    schema = {"type": "json_schema", "json_schema": {"schema": {"type": "object", "required": ["a"]}}}
    score = quality.score_answer("give me json", "not json at all", config, response_format=schema)
    assert score == 0.0


def test_quality_gate_accepts_short_correct_answers_and_flags_real_truncation():
    """Short correct answers ("150", "Paris") pass the gate; answers cut off mid-sentence are flagged."""
    from src.savings import quality

    assert not quality._is_empty_or_truncated("150")
    assert not quality._is_empty_or_truncated("Paris")
    assert quality._is_empty_or_truncated("")
    assert quality._is_empty_or_truncated("The capital of France is")
    assert quality._is_empty_or_truncated("Water boils at 100 degrees and")
    assert quality.obvious_failure("150") is None and quality.obvious_failure("Paris") is None


def test_quality_error_flag_only_for_answers_that_are_errors():
    """A correct answer that explains an error is not itself an error answer."""
    from src.savings import quality

    assert quality._ERROR_REGEX.search("Traceback (most recent call last):\n  File ...")
    assert quality._ERROR_REGEX.search("TypeError: unsupported operand type(s)")
    assert quality._ERROR_REGEX.search("Error: connection refused")
    assert not quality._ERROR_REGEX.search("The TypeError means you are adding incompatible types.")
    assert not quality._ERROR_REGEX.search("An exception is raised because the file is missing.")


# --- The two-layer gate: rules for unmistakable failures, the trained model for the rest ---
def test_obvious_failures_are_rules_not_learned():
    """Empty, cut-off, non-answers, refusals and errors score 0 whatever the model says; normal answers pass the rules."""
    from src.savings import quality

    for bad in ("", "N/A", "Unknown", "I cannot help with that request.", "Traceback (most recent call last):\nTypeError: x",
                "The total is,"):
        assert quality.obvious_failure(bad), bad
        assert quality.score_answer("Extract the fields.", bad, CONFIG) == 0.0, bad
    for fine in ("Paris", "150", "The answer is none of the above.", "{\"a\": 1}"):
        assert quality.obvious_failure(fine) is None, fine


def test_answer_confidence_feeds_the_features():
    """The model's certainty over the answer text is the last feature; without it the classifier is not consulted."""
    from src.savings import quality

    features = quality.extract_features("What is 2+2?", "4", {"mean_prob": 0.9, "body_prob": 0.7})
    assert len(features) == len(quality.FEATURE_NAMES)
    assert abs(float(features[-1]) - 0.7) < 1e-6
    assert quality.score_answer("What is 2+2?", "4", CONFIG) == 1.0     # no confidence: rules only
    assert quality.score_answer("What is 2+2?", "N/A", CONFIG) == 0.0   # the rules still apply


def test_the_check_runs_at_the_routes_threshold():
    """Threshold 0: only the rules. Above 0: the classifier must reach it, and an answer without confidence goes on."""
    from src.savings import quality

    question = "Classify the sentiment of this review as positive or negative: \"great\""
    assert quality.gate_verdict(question, "positive", CONFIG, 0.0) == quality.GateVerdict(False, 1.0, "rules_only")
    assert quality.gate_verdict(question, "N/A", CONFIG, 0.0).escalate
    assert quality.gate_verdict(question, "positive", CONFIG, 0.5) == quality.GateVerdict(True, 0.0, "no_confidence")
    score = quality.good_probability(question, "positive", CONFIG, CONF)
    assert quality.gate_verdict(question, "positive", CONFIG, score + 0.01, confidence=CONF).escalate
    assert not quality.gate_verdict(question, "positive", CONFIG, max(score - 0.01, 0.001), confidence=CONF).escalate


def test_the_model_file_names_the_questions_it_was_trained_on():
    """route_proof.py must be able to leave those questions out of the evidence."""
    from src.savings import quality

    trained = quality.trained_questions(CONFIG)
    assert trained and all(isinstance(item_id, str) for item_id in trained)


def quality_model_path():
    """The shipped gate model's path."""
    from src.paths import MODEL_PATH

    return MODEL_PATH

