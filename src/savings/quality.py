"""
Embeddings and the answer-quality classifier.

- One shared embedding model (all-MiniLM-L6-v2), loaded once per process.
- `extract_features()` / `score_answer()`: a small classifier (src/models/model.pkl,
  hash-checked before loading) gives the probability that an answer is good.
- `validate_structured_output()`: an answer that breaks its JSON schema scores 0.
The classifier is triage, not proof: the benchmark is the proof.
"""

import ast
import json
import logging
import re
import threading
from dataclasses import dataclass

import jsonschema
import numpy as np

logger = logging.getLogger("cordon.quality")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_model_lock = threading.Lock()

_classifier = None
_classifier_lock = threading.Lock()


def get_embedding_model():
    """
    Load (once) and return the shared sentence-transformers embedding model.

    Parameters: none.
    Returns: a sentence_transformers.SentenceTransformer instance, the same
    one on every call within this process.
    Why double-checked locking: the model is loaded lazily on first use
    (not at import time, so importing this module in a test or CLI tool
    that never needs embeddings stays cheap), but multiple request threads
    could race to load it concurrently without the lock.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("loading embedding model: %s", EMBEDDING_MODEL_NAME)
                _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of texts in one call.

    Parameters: texts - a list of distinct strings (callers should
    deduplicate before calling: each text is embedded at most once per
    request - this function itself does not deduplicate).
    Returns: an (n, dim) numpy array, row i is the embedding of texts[i].
    Why only masked text: callers must only ever pass already-masked text
    (guardrails.mask_messages()'s output) - embedding raw PII would leak it
    into the vector, which is stored in the cache/circuit-breaker state.
    """
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.

    Parameters: a, b - embedding vectors of the same dimension.
    Returns: a float in [-1, 1] (in practice [0, 1] for these embeddings);
    0.0 if either vector is all-zero (avoids a division by zero).
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# --- Section: answer-quality features — shared by train.py, score_answer(), and benchmark.py ---
_REFUSAL_REGEX = re.compile(
    r"\b(i cannot|i can['’]t|i['’]m not able|i am not able|i['’]m unable|i am unable|"
    r"as an ai|i don['’]t have access|i do not have access)\b",
    re.IGNORECASE,
)
# An answer that IS an error: it starts like one, or contains a Python traceback.
# A correct answer that explains an error ("The TypeError means...") must not match.
_ERROR_REGEX = re.compile(
    r"^\s*(traceback|exception\b|error:|\w+(error|exception):|stack trace|segmentation fault|"
    r"undefined is not a function)|traceback \(most recent call last\)",
    re.IGNORECASE,
)

_CATEGORIES = ("chat", "classification", "code", "extraction", "reasoning", "summarization")
FEATURE_NAMES: tuple[str, ...] = (
    # the answer's shape
    "qa_cosine", "length_ratio", "refusal_flag", "error_flag", "repetition_ratio", "empty_or_truncated_flag",
    # the kind of question (as step 4 classifies it) - some kinds go wrong more often
    *(f"type_{c}" for c in _CATEGORIES), "hard",
    # format and size
    "has_code", "code_parses", "digit_share", "answer_words_log", "question_words_log",
    # the model's own certainty over the answer text (OpenAI logprobs, answer_confidence()["body_prob"])
    "confidence",
)
_CODE_BLOCK_REGEX = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _repetition_ratio(text: str) -> float:
    """Ratio of distinct words to total words (1.0 = no repeated words, lower = more repetitive). 1.0 for empty text (no repetition to flag; emptiness is its own feature)."""
    words = text.split()
    if not words:
        return 1.0
    return len(set(words)) / len(words)


_TRAILING_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "by", "for", "with", "from", "as",
    "is", "are", "was", "were", "be", "that", "which", "because", "if", "then", "using", "like", "such",
})


def _is_empty_or_truncated(text: str) -> bool:
    """
    True if the answer is empty, or visibly cut off mid-sentence.

    "Cut off" means it ends on a comma/colon/dash, or a longer answer (4+
    words) ends on a word that cannot end a sentence ("... and", "... the").
    A short answer without a full stop ("150", "Paris") is NOT truncated:
    treating every missing full stop as truncation made the classifier reject
    correct one-word answers (measured on the golden set; see the model card).
    """
    stripped = text.strip()
    if not stripped:
        return True
    if stripped[-1] in ",:;-(":
        return True
    words = stripped.split()
    last_word = re.sub(r"[^\w]", "", words[-1].lower())
    return len(words) >= 4 and last_word in _TRAILING_FUNCTION_WORDS


def extract_features(question: str, answer: str, confidence: dict) -> np.ndarray:
    """
    The quality gate's feature vector for one (question, answer) pair, in FEATURE_NAMES order.

    Parameters: question, answer - masked text only. confidence - answer_confidence() of the
    response (the model is only trained and used on answers that carry it).
    The same function serves training (evaluation/train.py) and the live gate (score_answer), so
    the two can never compute features differently.
    """
    from src.savings import routing

    if answer.strip():
        question_vector, answer_vector = embed_texts([question, answer])
        cosine = cosine_similarity(question_vector, answer_vector)
    else:
        cosine = 0.0
    question_words, answer_words = len(question.split()), len(answer.split())
    shape = [cosine, min(answer_words / (question_words or 1), 10.0), float(bool(_REFUSAL_REGEX.search(answer))),
             float(bool(_ERROR_REGEX.search(answer))), _repetition_ratio(answer), float(_is_empty_or_truncated(answer))]

    category, difficulty = routing.classify_request([{"role": "user", "content": question}], None)
    kind = [*(float(category == c) for c in _CATEGORIES), float(difficulty == "hard")]  # the routing taxonomy

    block = _CODE_BLOCK_REGEX.search(answer)
    form = [float(block is not None), float(block is not None and _parses(block.group(1))),
            sum(ch.isdigit() for ch in answer) / max(len(answer), 1),
            float(np.log1p(answer_words)), float(np.log1p(question_words))]

    certainty = [confidence.get("body_prob", confidence["mean_prob"])]
    return np.array(shape + kind + form + certainty, dtype=np.float32)


def _parses(code: str) -> bool:
    """True if the code is valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


# --- Section: the model's own confidence (OpenAI logprobs) ---
def answer_confidence(response) -> dict | None:
    """
    How sure the model was of its own answer, from a completion's logprobs.

    Returns: {"mean_prob", "min_prob", "final_prob", "body_prob"} - the average and lowest
    probability of all the answer's tokens; the average over the tokens after the last
    "FINAL:" (the benchmark's answer line); and the average over the tokens *before* it -
    the answer as a real app gets it, so the one the gate is trained and served on
    (body_prob equals mean_prob when there is no FINAL line).
    None when the provider returned no logprobs.
    Why: a model that hesitates over its answer is wrong more often - a signal
    the answer's shape alone cannot give.
    """
    try:
        return _confidence(response)
    except Exception:  # odd provider data must never break a request that is already paid for
        logger.warning("could not read the answer's logprobs; scoring without confidence")
        return None


def _confidence(response) -> dict | None:
    """answer_confidence() without the safety net."""
    logprobs = getattr(response.choices[0], "logprobs", None)  # absent when they were not asked for: normal
    if logprobs is None:
        return None
    tokens = logprobs["content"] if isinstance(logprobs, dict) else logprobs.content
    pairs = [((t["token"], t["logprob"]) if isinstance(t, dict) else (t.token, t.logprob)) for t in tokens or []]
    pairs = [(tok, lp) for tok, lp in pairs if isinstance(tok, str) and isinstance(lp, (int, float))]
    if not pairs:
        return None
    probs = [float(np.exp(lp)) for _, lp in pairs]
    text, final_start = "", None
    for i, (token, _) in enumerate(pairs):
        text += token
        if text.rstrip().upper().endswith("FINAL:"):
            final_start = i + 1
    final = probs[final_start:] if final_start is not None and final_start < len(probs) else probs
    marker = _final_marker_start(pairs) if final_start is not None else None
    body = probs[:marker] if marker else probs
    return {"mean_prob": round(sum(probs) / len(probs), 4), "min_prob": round(min(probs), 4),
            "final_prob": round(sum(final) / len(final), 4), "body_prob": round(sum(body) / len(body), 4)}


def _final_marker_start(pairs: list[tuple[str, float]]) -> int | None:
    """Index of the token where the last "FINAL" marker begins (None if there is none)."""
    text, starts = "", []
    for i, (token, _) in enumerate(pairs):
        if "FINAL" in (text[-6:] + token).upper() and "FINAL" not in text[-6:].upper():
            starts.append(i)
        text += token
    return starts[-1] if starts else None


# --- Section: structured-output validation ---
def validate_structured_output(answer_text: str, response_format: dict | None) -> bool:
    """
    Validate an answer against a requested JSON-schema response_format.

    Parameters: answer_text - the model's raw answer text.
                 response_format - the request's `response_format`, or None.
    Returns: True if there is nothing to validate (no response_format, or
    not a json_schema one), or the answer parses as JSON and satisfies the
    schema; False otherwise (invalid JSON, or schema violation).
    """
    if not response_format or response_format.get("type") != "json_schema":
        return True
    schema = (response_format.get("json_schema") or {}).get("schema")
    if not schema:
        return True
    try:
        parsed = json.loads(answer_text)
    except ValueError:
        return False
    try:
        jsonschema.validate(parsed, schema)
    except jsonschema.ValidationError:
        return False
    return True


# --- Section: the trained classifier — hash-checked load, once per process ---
def load_classifier(model_path: str, expected_sha256: str):
    """
    Load and return the trained quality classifier, verifying its hash first.

    Parameters: model_path - path to src/models/model.pkl.
                 expected_sha256 - config.yaml's retention.model_sha256.
    Returns: the unpickled gate - {"classifier", "trained_on" (question ids)} - cached
    after the first call.
    Raises: ValueError if the file's hash does not match expected_sha256.
    Why the hash check comes first: joblib.load() unpickles arbitrary
    objects, which can execute code - this must never run on a file that
    does not match the hash recorded at training time (security.py's model
    integrity control).
    """
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                from src.protection.security import verify_model_integrity

                if not verify_model_integrity(model_path, expected_sha256):
                    raise ValueError(f"model file does not match the configured hash: {model_path}")
                import joblib

                _classifier = joblib.load(model_path)
    return _classifier


# An answer that is nothing but a placeholder: no answer at all, however it is spelled.
_NON_ANSWER_REGEX = re.compile(r"^\W*(n/?a|none|null|unknown|i don'?t know|no answer|-+|\?+)\W*$", re.IGNORECASE)


def obvious_failure(answer: str, response_format: dict | None = None) -> str | None:
    """
    The first layer of the gate: an unmistakably bad answer, named - or None.

    Empty or cut off, a non-answer ("N/A", "unknown"), a refusal, an error or stack trace, or
    JSON that breaks the requested schema. These are rules, not learned: real models rarely produce them, so a model
    trained on real answers never sees enough of them to learn them.
    """
    if not validate_structured_output(answer, response_format):
        return "invalid_json"
    if _is_empty_or_truncated(answer):
        return "empty_or_truncated"
    if _NON_ANSWER_REGEX.match(answer.strip()):
        return "non_answer"
    if _REFUSAL_REGEX.search(answer):
        return "refusal"
    if _ERROR_REGEX.search(answer):
        return "error"
    return None


@dataclass
class GateVerdict:
    """The gate's decision on one answer: escalate or not, the score, and why."""

    escalate: bool
    score: float  # the classifier's probability that the answer is good; 0 for an obvious failure
    reason: str  # an obvious_failure() name, "classifier", "rules_only" or "no_confidence"


def gate_verdict(question: str, answer: str, config: dict, threshold: float, response_format: dict | None = None,
                 confidence: dict | None = None) -> GateVerdict:
    """
    Decide whether a cheaper model's answer goes to the strong model after all.

    The same function proves a route (evaluation/route_proof.py) and serves it (step 8), so the
    benchmark measures exactly what runs. Rules first (obvious_failure); with a threshold of 0
    only the rules apply. Otherwise the classifier's probability that the answer is good must
    reach the route's threshold; an answer without the model's confidence cannot be checked,
    so it goes to the strong model - the route was only proven with the check.
    """
    failure = obvious_failure(answer, response_format)
    if failure:
        return GateVerdict(True, 0.0, failure)
    if threshold <= 0:
        return GateVerdict(False, 1.0, "rules_only")
    if not confidence:
        return GateVerdict(True, 0.0, "no_confidence")
    score = good_probability(question, answer, config, confidence)
    return GateVerdict(score < threshold, score, "classifier")


def good_probability(question: str, answer: str, config: dict, confidence: dict) -> float:
    """The trained classifier's probability that the answer is good (model.pkl, hash-checked)."""
    classifier = _gate_model(config)["classifier"]
    return float(classifier.predict_proba(extract_features(question, answer, confidence).reshape(1, -1))[0][1])


def trained_questions(config: dict) -> frozenset:
    """The question ids the gate was trained on: they can never be evidence for a route."""
    return frozenset(_gate_model(config).get("trained_on") or ())


def _gate_model(config: dict) -> dict:
    """The trained gate ({"classifier", "trained_on"}), hash-checked before loading."""
    from src.server.settings import MODEL_PATH

    model_sha256 = (config.get("retention") or {}).get("model_sha256")
    if not model_sha256:
        raise ValueError("no model_sha256 configured")
    return load_classifier(str(MODEL_PATH), model_sha256)


def score_answer(question: str, answer: str, config: dict, response_format: dict | None = None,
                 confidence: dict | None = None) -> float:
    """
    An answer's quality score, 0 to 1 (the X-Cordon-Quality header): 0 for an obvious failure,
    else the classifier's probability that it is good - or 1.0 without the model's confidence,
    when only the rules can judge it.
    """
    if obvious_failure(answer, response_format):
        return 0.0
    return good_probability(question, answer, config, confidence) if confidence else 1.0
