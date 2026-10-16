"""
Guardrails: personal data, prompt injection and leaks of the hidden instructions.

- `apply_input_guardrails()`: masks personal data with placeholders (`<EMAIL_1>`)
  or blocks, and scores prompt injection. Observe mode only counts.
- `apply_output_guardrails()`: redacts a stranger's personal data and leaked
  instructions in the answer; the caller's own data is left alone.
- `restore_message()`: puts the real values back into the answer.
- `StreamingOutputGuardrail`: the same output checks for a streamed answer.
- `redact()`: scrubs log lines and telemetry; not part of the request path.
"""

import logging
import datetime
import re
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger("cordon.guardrails")

# --- Section: pattern loading — PII regexes, injection phrases, time words, negations ---
from src import messages as messages_module
from src.paths import PATTERNS_PATH as _PATTERNS_PATH


def _load_patterns() -> dict:
    """
    Load patterns.yaml once at import time.

    Parameters: none.
    Returns: the parsed YAML document as a dict.
    Why: all detection data lives in one file (patterns.yaml) so non-English
    samples never need to be hard-coded in Python or in tests.
    """
    with open(_PATTERNS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_PATTERNS = _load_patterns()

# Compiled once; every regex here is linear-time (no nested quantifiers), see
# the header comment in patterns.yaml - this runs on every request.
_PII_REGEXES: dict[str, re.Pattern] = {
    name: re.compile(spec["regex"]) for name, spec in _PATTERNS["pii"].items()
}

# Secret-looking substrings that must never reach a log line, independent of
# the deployment-specific secret values themselves (which security.py's log
# filter also scrubs literally).
_SECRET_REGEXES: list[re.Pattern] = [
    re.compile(r"sk-[A-Za-z0-9_-]{10,}"),
    re.compile(r"cdn_[A-Za-z0-9_-]{10,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{10,}"),
]

# Belgian/international phone numbers have at least this many digits once
# separators are stripped; a shorter digit run (an IPv4 address, a version
# string) is a false positive from the regex alone.
_MIN_PHONE_DIGITS = 8

INJECTION_PHRASES: dict[str, list[str]] = _PATTERNS["injection_phrases"]
BASE64_SUSPICIOUS_REGEX = re.compile(_PATTERNS["base64_suspicious"]["regex"])
TIME_WORDS: dict[str, list[str]] = _PATTERNS["time_words"]
NEGATIONS: dict[str, list[str]] = _PATTERNS["negations"]


# --- Section: PII detection — a regex finds candidates, a confirmer per type accepts them ---
def _digit_count(value: str) -> int:
    """Count ASCII digit characters in a string."""
    return sum(1 for ch in value if ch.isdigit())


# Real dates the phone regex also matches: 2026-02-03, 2026/02/03, 03-02-2026, 03/02/2026, 03.02.2026.
_DATE_REGEX = re.compile(r"(?<!\d)(?:(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})|(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}))(?!\d)")


def _is_calendar_date(match: re.Match) -> bool:
    """True if a _DATE_REGEX match is a real calendar date with a plausible year (1900-2100)."""
    year, month, day = (match.group(1), match.group(2), match.group(3)) if match.group(1) else (
        match.group(6), match.group(5), match.group(4))
    try:
        return 1900 <= int(year) <= 2100 and bool(datetime.date(int(year), int(month), int(day)))
    except ValueError:
        return False


def _is_phone(text: str, start: int, end: int) -> bool:
    """
    Confirm a phone candidate text[start:end]: at least 8 digits, not all of them part of real dates.

    The regex alone also matches an IPv4 address (127.0.0.1, 7 digits) and dates (2026-02-03,
    or a range "2026-01-01 - 2026-12-31") - masking a date made the model lose it, which the
    first live benchmark showed on invoice extraction. Judged on the whole text, so a range the
    regex cuts short is still seen as dates; a phone number with a date-shaped piece is still a phone.
    """
    if _digit_count(text[start:end]) < _MIN_PHONE_DIGITS:
        return False
    in_dates = set()
    for match in _DATE_REGEX.finditer(text):
        if _is_calendar_date(match):
            in_dates.update(range(*match.span()))
    return any(text[i].isdigit() and i not in in_dates for i in range(start, end))


def _is_card(text: str, start: int, end: int) -> bool:
    """Confirm a card candidate text[start:end] with the Luhn checksum (ordinary 13-19 digit numbers fail it)."""
    return _luhn_check(re.sub(r"[ \-]", "", text[start:end]))


def _luhn_check(digits: str) -> bool:
    """
    Validate a digit string with the Luhn checksum algorithm.

    Parameters: digits - a string of ASCII digits only (no separators).
    Returns: True if the Luhn checksum is valid, False otherwise.
    Why: the credit_card regex only finds candidate digit runs; without a Luhn
    check, ordinary 13-19 digit numbers (invoice numbers, phone numbers already
    caught by another pattern) would be flagged as credit cards.
    """
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Types whose regex only finds candidates; each must pass its confirmer, which sees the whole text.
_CONFIRMERS = {"phone": _is_phone, "credit_card": _is_card}


def find_pii(text: str) -> list[tuple[str, tuple[int, int], str]]:
    """
    Find all PII matches in a text.

    Parameters: text - the text to scan.
    Returns: a list of (pii_type, (start, end), matched_value) tuples, sorted
    by start offset. Phone and card candidates must pass their confirmer
    (_CONFIRMERS): no dates or IP addresses as phones, Luhn for cards.
    Why: shared by redact() and the input and output guardrails, so
    detection logic lives in one place.
    """
    if not text:
        return []
    matches: list[tuple[str, tuple[int, int], str]] = []
    for name, rx in _PII_REGEXES.items():
        for m in rx.finditer(text):
            value = m.group()
            confirm = _CONFIRMERS.get(name)
            if confirm is None or confirm(text, *m.span()):
                matches.append((name, m.span(), value))
    matches.sort(key=lambda t: t[1][0])
    return matches


# --- Section: redact — conservative text scrubber for logs and telemetry only ---
def redact(text: str) -> str:
    """
    Replace PII and secret-looking substrings with type-labelled placeholders.

    Parameters: text - arbitrary text (a log message, an error string, a
    telemetry payload field).
    Returns: the text with every PII and secret match replaced by
    "[REDACTED_<TYPE>]"; never the original value.
    Why: this is the one function the log filter (security.py) and telemetry
    content mode "redacted" (telemetry.py) run text through. It intentionally
    over-redacts rather than under-redacts, since its only job is to keep
    secrets and personal data out of places that are not the model call itself.
    """
    if not text:
        return text
    scrubbed = text
    for rx in _SECRET_REGEXES:
        scrubbed = rx.sub("[REDACTED_SECRET]", scrubbed)
    # Apply PII replacements back-to-front so earlier offsets stay valid.
    for name, (start, end), _value in sorted(find_pii(scrubbed), key=lambda t: t[1][0], reverse=True):
        scrubbed = scrubbed[:start] + f"[REDACTED_{name.upper()}]" + scrubbed[end:]
    return scrubbed


# --- Section: own data — PII that appeared in this request's user messages ---
def _normalize_pii_value(pii_type: str, value: str) -> str:
    """
    Normalize a PII value for own-data comparison.

    Parameters: pii_type - the PII type name (e.g. "email", "phone").
                 value - the matched text.
    Returns: a normalized form so "0471 23 45 67" and "0471234567" compare
    equal; emails are lowercased, everything else is reduced to alphanumerics.
    """
    if pii_type == "email":
        return value.strip().lower()
    return re.sub(r"[^A-Za-z0-9]", "", value).lower()


def collect_own_data(messages: list[dict]) -> set[str]:
    """
    Collect the set of PII values that appear in this request's user messages.

    Parameters: messages - the full request's messages.
    Returns: a set of normalized PII values (see _normalize_pii_value) found
    in `role: user` messages only - the definition of "own
    data": PII the caller stated themselves, not PII from a system or tool
    message, which must not get the own-data exception.
    """
    own: set[str] = set()
    for message in messages:
        if message.get("role") != "user":
            continue
        for text in messages_module.text_fields(message):
            for name, _span, value in find_pii(text):
                own.add((name, _normalize_pii_value(name, value)))
    return own


# --- Section: masking — reversible placeholders, system prompt untouched ---
_PLACEHOLDER_TYPE_LABELS: dict[str, str] = {
    "email": "EMAIL",
    "phone": "PHONE",
    "iban": "IBAN",
    "be_national_number": "NATID",
    "credit_card": "CARD",
}


def mask_messages(messages: list[dict], include_instructions: bool = False) -> tuple[list[dict], dict[str, str], dict[str, int]]:
    """
    Replace PII in user/assistant/tool messages with reversible placeholders.

    Parameters: messages - the full request's messages.
                 include_instructions - also mask the system/developer
                 messages (the `mask_instructions` setting, for RAG apps that
                 paste documents there).
    Returns: (masked_messages, mapping, counts).
    masked_messages - a new list; by default the system message is returned
    unchanged, so the provider's prompt caching on that stable prefix keeps working.
    mapping - {"<EMAIL_1>": "person@example.com", ...}, kept in memory only
    for the lifetime of this request.
    counts - number of distinct values masked per PII type (never the values).
    Why one placeholder per distinct value: if the same email appears twice,
    both occurrences become the same "<EMAIL_1>" so the model's answer stays
    coherent, and restore() only needs one mapping entry per value.
    """
    mapping: dict[str, str] = {}
    value_to_placeholder: dict[tuple[str, str], str] = {}
    counts: dict[str, int] = {}
    counters: dict[str, int] = {}

    def get_placeholder(pii_type: str, value: str) -> str:
        """Return the (possibly new) placeholder for one PII value, assigning the next number per type."""
        key = (pii_type, _normalize_pii_value(pii_type, value))
        if key in value_to_placeholder:
            return value_to_placeholder[key]
        counters[pii_type] = counters.get(pii_type, 0) + 1
        placeholder = f"<{_PLACEHOLDER_TYPE_LABELS[pii_type]}_{counters[pii_type]}>"
        value_to_placeholder[key] = placeholder
        mapping[placeholder] = value
        counts[pii_type] = counts.get(pii_type, 0) + 1
        return placeholder

    def mask_text(text: str) -> str:
        """Replace every PII match in one text string with its placeholder."""
        matches = find_pii(text)
        if not matches:
            return text
        pieces = []
        last = 0
        for name, (start, end), value in matches:
            pieces.append(text[last:start])
            pieces.append(get_placeholder(name, value))
            last = end
        pieces.append(text[last:])
        return "".join(pieces)

    masked_messages = [
        messages_module.map_text_fields(message, mask_text) if _scanned(message, include_instructions) else dict(message)
        for message in messages
    ]
    return masked_messages, mapping, counts


def _scanned(message: dict, include_instructions: bool) -> bool:
    """Whether a message is checked for personal data: always, except the app's own instructions unless asked."""
    return include_instructions or not messages_module.is_instruction(message)


# --- Section: restore — tolerant placeholder substitution ---
_BRACKETED_TOKEN_REGEX = re.compile(r"[<\[]([A-Z]+_\d+)[>\]]")
_BARE_TOKEN_REGEX = re.compile(r"\b([A-Z]+_\d+)\b")
PLACEHOLDER_TOKEN_REGEX = re.compile(r"[<\[]?[A-Z]+_\d+[>\]]?")


def restore_placeholders(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """
    Replace placeholder tokens in text with their original values.

    Parameters: text - masked text (an answer, or tool-call arguments).
                 mapping - {"<EMAIL_1>": "person@example.com", ...} from mask_messages().
    Returns: (restored_text, unrestored_count). unrestored_count is the
    number of bracketed placeholder-looking tokens (`<TYPE_N>` or `[TYPE_N]`)
    that did not match any key in mapping - logged as a count, never the token.
    Why tolerant: the model may echo "<EMAIL_1>" back as "[EMAIL_1]" or even
    bare "EMAIL_1"; all three forms are accepted, but only the bracketed
    forms count as a (failed) restoration attempt - a bare "TYPE_N" could
    coincidentally be unrelated text.
    """
    if not mapping or not text:
        return text, 0
    token_to_value = {placeholder[1:-1]: value for placeholder, value in mapping.items()}
    unrestored = 0

    def bracket_sub(match: re.Match) -> str:
        """Resolve one <TYPE_N>/[TYPE_N] match, counting a miss if the token is unknown."""
        nonlocal unrestored
        token = match.group(1)
        if token in token_to_value:
            return token_to_value[token]
        unrestored += 1
        return match.group(0)

    text = _BRACKETED_TOKEN_REGEX.sub(bracket_sub, text)

    def bare_sub(match: re.Match) -> str:
        """Resolve one bare TYPE_N match if known; otherwise leave it untouched (not counted as a miss)."""
        token = match.group(1)
        return token_to_value.get(token, match.group(0))

    text = _BARE_TOKEN_REGEX.sub(bare_sub, text)
    return text, unrestored


def restore_message(message: dict, mapping: dict[str, str]) -> tuple[dict, int]:
    """
    Restore placeholders throughout one message (content and tool-call arguments).

    Parameters: message - the (masked) message to restore.
                 mapping - from mask_messages().
    Returns: (restored_message, unrestored_count).
    """
    total_unrestored = 0

    def restore_text(text: str) -> str:
        """Restore placeholders in one text field, accumulating the unrestored count."""
        nonlocal total_unrestored
        restored, count = restore_placeholders(text, mapping)
        total_unrestored += count
        return restored

    return messages_module.map_text_fields(message, restore_text), total_unrestored


# --- Section: prompt injection — phrase matching plus a suspicious-base64 signal ---
def detect_injection(text: str) -> float:
    """
    Score how likely a text contains a prompt-injection attempt.

    Parameters: text - the text to score (one message's text).
    Returns: a score in [0, 1]. Each distinct injection phrase match (from
    patterns.yaml, across every language) contributes 0.5, capped; a
    suspicious long base64-looking run contributes 0.3. Any score > 0 means
    "detected" for the flag/block decision in apply_input_guardrails().
    Why these weights: a single exact injection phrase match ("ignore
    previous instructions") is already a strong, specific signal and should
    not need corroboration to be flagged; base64 alone is weaker evidence
    (plenty of legitimate content is base64) so it alone keeps the score
    below the "detected" threshold's typical confidence but still visible.
    """
    if not text:
        return 0.0
    lowered = text.lower()
    phrase_score = 0.0
    for phrases in INJECTION_PHRASES.values():
        for phrase in phrases:
            if phrase.lower() in lowered:
                phrase_score = min(1.0, phrase_score + 0.5)
    base64_score = 0.3 if BASE64_SUSPICIOUS_REGEX.search(text) else 0.0
    return min(1.0, phrase_score + base64_score)


# --- Section: system-prompt leakage — verbatim word-run overlap ---
def _find_leak_spans(answer_text: str, system_prompt: str, leak_min_words: int) -> list[tuple[int, int]]:
    """
    Find character spans in answer_text that verbatim-overlap the system prompt.

    Parameters: answer_text - the model's answer.
                 system_prompt - the tenant's/request's system prompt text.
                 leak_min_words - the minimum run length that counts as a leak.
    Returns: a list of (start, end) character spans in answer_text, merged
    where overlapping runs are adjacent, each covering a maximal run of
    consecutive words also found (as a contiguous run) in system_prompt.
    Why word n-grams in a set: this makes the check O(len(system_prompt) +
    len(answer_text)) rather than the naive O(n*m) substring search.
    """
    if not system_prompt or not answer_text:
        return []
    sys_words = system_prompt.lower().split()
    if len(sys_words) < leak_min_words:
        return []
    sys_ngrams = {tuple(sys_words[i : i + leak_min_words]) for i in range(len(sys_words) - leak_min_words + 1)}

    answer_tokens = list(re.finditer(r"\S+", answer_text))
    answer_words_lower = [t.group().lower() for t in answer_tokens]
    n = len(answer_tokens)
    spans: list[tuple[int, int]] = []
    i = 0
    while i <= n - leak_min_words:
        if tuple(answer_words_lower[i : i + leak_min_words]) in sys_ngrams:
            j = i + leak_min_words
            while j < n and tuple(answer_words_lower[j - leak_min_words + 1 : j + 1]) in sys_ngrams:
                j += 1
            spans.append((answer_tokens[i].start(), answer_tokens[j - 1].end()))
            i = j
        else:
            i += 1
    return spans


def detect_system_prompt_leak(answer_text: str, system_prompt: str, leak_min_words: int) -> bool:
    """
    True if the answer contains a verbatim run of >= leak_min_words system-prompt words.

    Parameters: answer_text, system_prompt, leak_min_words - see _find_leak_spans.
    Returns: bool.
    Why it matters: a verbatim system-prompt echo is the typical sign of a
    successful prompt-injection attack asking the model to reveal its
    instructions.
    """
    return bool(_find_leak_spans(answer_text, system_prompt, leak_min_words))


def _redact_leak_spans(answer_text: str, spans: list[tuple[int, int]]) -> str:
    """Replace each (start, end) span in answer_text with a single '[REDACTED_SYSTEM_PROMPT]' marker."""
    if not spans:
        return answer_text
    pieces = []
    last = 0
    for start, end in spans:
        pieces.append(answer_text[last:start])
        pieces.append("[REDACTED_SYSTEM_PROMPT]")
        last = end
    pieces.append(answer_text[last:])
    return "".join(pieces)


# --- Section: output PII — foreign PII only; own data and placeholders stay ---
def _redact_or_block_output_pii(
    text: str, own_data: set[tuple[str, str]], action: str
) -> tuple[str, dict[str, int], bool, str | None]:
    """
    Apply output_pii_action to the non-own-data PII found in text.

    Parameters: text - the (masked) answer text to check.
                 own_data - from collect_own_data(); (type, normalized_value) pairs.
                 action - "redact" or "block" (never called with "off").
    Returns: (processed_text, counts, blocked, blocked_type).
    processed_text equals text unchanged if action == "block" and something
    was blocked (the caller discards it and returns a 422 instead).
    Why placeholders need no special case: a placeholder like "<EMAIL_1>"
    does not match any PII regex, so find_pii() never flags it here - only
    genuine, unmasked PII reaches this function's redact/block decision.
    """
    matches = find_pii(text)
    counts: dict[str, int] = {}
    foreign_matches = [
        (name, span, value)
        for name, span, value in matches
        if (name, _normalize_pii_value(name, value)) not in own_data
    ]
    if not foreign_matches:
        return text, {}, False, None

    if action == "block":
        first_type = foreign_matches[0][0]
        for name, _span, _value in foreign_matches:
            counts[name] = counts.get(name, 0) + 1
        return text, counts, True, first_type

    pieces = []
    last = 0
    for name, (start, end), _value in foreign_matches:
        pieces.append(text[last:start])
        pieces.append(f"[REDACTED_{name.upper()}]")
        counts[name] = counts.get(name, 0) + 1
        last = end
    pieces.append(text[last:])
    return "".join(pieces), counts, False, None


# --- Section: apply_input_guardrails / apply_output_guardrails — the pipeline entry points ---
@dataclass
class InputGuardrailResult:
    """
    Outcome of apply_input_guardrails().

    Fields: masked_messages - the messages to send to the model (unchanged
    unless pii_action == "mask" and not observe). placeholder_mapping - from
    mask_messages(), empty if nothing was masked. own_data - from
    collect_own_data(), always populated (needed by the output guardrails
    regardless of pii_action). pii_counts - count per PII type found in the
    input (not just what was masked). injection_score, injection_detected.
    blocked / blocked_reason ("pii"|"injection") / blocked_pii_type - set
    when this request must be rejected with a 400.
    """

    masked_messages: list[dict]
    placeholder_mapping: dict[str, str]
    own_data: set[tuple[str, str]]
    pii_counts: dict[str, int]
    injection_score: float
    injection_detected: bool
    blocked: bool = False
    blocked_reason: str | None = None
    blocked_pii_type: str | None = None


def apply_input_guardrails(
    messages: list[dict], pii_action: str, injection_action: str, observe: bool, include_instructions: bool = False
) -> InputGuardrailResult:
    """
    Run PII detection/masking and prompt-injection detection on a request.

    Parameters: messages - the request's messages, system prompt included.
                 pii_action - "off" | "mask" | "block".
                 injection_action - "flag" | "block".
                 include_instructions - also check and mask the system/developer messages.
                 observe - True in observe mode: detect and count only, never
                 mask or block (observe mode never changes
                 or blocks, except the always-on checks in pipeline/step1_admission.py, which
                 this function is not one of).
    Returns: an InputGuardrailResult.
    """
    own_data = collect_own_data(messages)

    pii_counts: dict[str, int] = {}
    any_pii = False
    for message in messages:
        if not _scanned(message, include_instructions):
            continue
        for text in messages_module.text_fields(message):
            for name, _span, _value in find_pii(text):
                pii_counts[name] = pii_counts.get(name, 0) + 1
                any_pii = True

    injection_score = 0.0
    for message in messages:
        for text in messages_module.text_fields(message):
            injection_score = max(injection_score, detect_injection(text))
    injection_detected = injection_score > 0.0

    blocked = False
    blocked_reason = None
    blocked_pii_type = None
    if not observe:
        if pii_action == "block" and any_pii:
            blocked = True
            blocked_reason = "pii"
            blocked_pii_type = next(iter(pii_counts))
        elif injection_action == "block" and injection_detected:
            blocked = True
            blocked_reason = "injection"

    masked_messages = messages
    placeholder_mapping: dict[str, str] = {}
    if not observe and not blocked and pii_action == "mask" and any_pii:
        masked_messages, placeholder_mapping, _mask_counts = mask_messages(messages, include_instructions)

    return InputGuardrailResult(
        masked_messages=masked_messages,
        placeholder_mapping=placeholder_mapping,
        own_data=own_data,
        pii_counts=pii_counts,
        injection_score=injection_score,
        injection_detected=injection_detected,
        blocked=blocked,
        blocked_reason=blocked_reason,
        blocked_pii_type=blocked_pii_type,
    )


@dataclass
class OutputGuardrailResult:
    """
    Outcome of apply_output_guardrails().

    Fields: text - the processed answer text (redactions applied; equals the
    input text unchanged if blocked). blocked / blocked_reason
    ("pii"|"system_prompt_leak") / blocked_pii_type - set when the answer
    must be rejected with a 422. pii_redacted_counts - per type. leak_redacted
    - whether a system-prompt leak span was found and redacted.
    """

    text: str
    blocked: bool = False
    blocked_reason: str | None = None
    blocked_pii_type: str | None = None
    pii_redacted_counts: dict[str, int] = field(default_factory=dict)
    leak_redacted: bool = False


def apply_output_guardrails(
    answer_text: str,
    own_data: set[tuple[str, str]],
    system_prompt: str | None,
    output_pii_action: str,
    output_leak_action: str,
    leak_min_words: int,
    observe: bool,
) -> OutputGuardrailResult:
    """
    Run output PII and system-prompt-leak checks on the model's (masked) answer.

    Parameters: answer_text - the raw model answer, still masked (placeholders
    not yet restored).
                 own_data - from apply_input_guardrails()'s InputGuardrailResult.
                 system_prompt - the request's system message text, or None.
                 output_pii_action - "off" | "redact" | "block".
                 output_leak_action - "off" | "redact" | "block".
                 leak_min_words - minimum verbatim word-run length counted as a leak.
                 observe - True in observe mode: detect and count only.
    Returns: an OutputGuardrailResult.
    """
    text = answer_text
    pii_counts: dict[str, int] = {}
    pii_blocked = False
    pii_blocked_type: str | None = None

    if not observe and output_pii_action != "off":
        text, pii_counts, pii_blocked, pii_blocked_type = _redact_or_block_output_pii(text, own_data, output_pii_action)
    elif observe:
        # Count only; find_pii() already excludes placeholders by construction.
        for name, _span, _value in find_pii(answer_text):
            pii_counts[name] = pii_counts.get(name, 0) + 1

    if not observe and pii_blocked:
        return OutputGuardrailResult(
            text=answer_text, blocked=True, blocked_reason="pii", blocked_pii_type=pii_blocked_type,
            pii_redacted_counts={}, leak_redacted=False,
        )

    leak_spans = _find_leak_spans(answer_text, system_prompt or "", leak_min_words) if leak_min_words > 0 else []
    leak_redacted = False
    if leak_spans and not observe and output_leak_action != "off":
        if output_leak_action == "block":
            return OutputGuardrailResult(
                text=answer_text, blocked=True, blocked_reason="system_prompt_leak", blocked_pii_type=None,
                pii_redacted_counts=pii_counts, leak_redacted=False,
            )
        text = _redact_leak_spans(text, _find_leak_spans(text, system_prompt or "", leak_min_words) or leak_spans)
        leak_redacted = True

    return OutputGuardrailResult(text=text, blocked=False, pii_redacted_counts=pii_counts, leak_redacted=leak_redacted)


def _apply_replacements(combined: str, offset: int, replacements: list[tuple[int, int, str]]) -> str:
    """
    Apply a set of (start, end, replacement_text) spans to combined[offset:] only.

    Parameters: combined - lookback context plus the new text, concatenated.
                 offset - len(lookback context); only this suffix is returned.
                 replacements - spans in combined's coordinates (may start
                 before offset, for a match that began in the lookback
                 context); each is clipped to not touch combined[:offset].
    Returns: combined[offset:] with every replacement applied.
    Why clip instead of skip: a PII match or leak run that started in the
    already-emitted lookback context but continues into the new text must
    still have its new-text portion redacted, even though the already-sent
    portion cannot be recalled.
    """
    pieces: list[str] = []
    last = offset
    for start, end, replacement in sorted(replacements, key=lambda r: r[0]):
        if end <= offset:
            continue
        clipped_start = max(start, offset)
        if clipped_start < last:
            # Overlaps a replacement already applied (e.g. a PII match
            # inside a leak span) - skip it rather than double-replacing.
            continue
        pieces.append(combined[last:clipped_start])
        pieces.append(replacement)
        last = end
    pieces.append(combined[last:])
    return "".join(pieces)


# --- Section: streaming output guardrails — rolling buffer, same checks, no recall ---
def streaming_hold_back_chars(leak_min_words: int, average_word_chars: int = 8) -> int:
    """
    Compute the rolling-buffer hold-back length for streaming output guardrails.

    Parameters: leak_min_words - the tenant's configured leak_min_words.
                 average_word_chars - assumed average word length in
                 characters, used to size the leak-detection window.
    Returns: max(64, longest realistic PII candidate length, leak window
    character length). 64 is the floor. 40 approximates the
    longest PII candidate this module matches (a 19-digit, dash-grouped
    credit-card candidate, or a 30-character IBAN candidate). The leak window
    is leak_min_words * average_word_chars, so a full leak_min_words-word run
    is unlikely to be split by the streaming boundary.
    """
    longest_pii_candidate_chars = 40
    leak_window_chars = leak_min_words * average_word_chars
    return max(64, longest_pii_candidate_chars, leak_window_chars)


class StreamingOutputGuardrail:
    """
    Applies the same output PII and system-prompt-leak checks as
    apply_output_guardrails(), but to a token stream via a rolling buffer:
    text already sent to the caller can never be recalled, so each feed()
    call only emits the portion of the buffer that is safely past the
    hold-back tail, and never emits a PII or placeholder-token match that
    straddles the current cut point. Because a block action cannot un-send
    already-emitted text, a block degrades to redact + log on streams
    - `blocked`/`blocked_reason` are still
    recorded for the ledger even though the stream itself was not stopped.

    Known limitation: the hold-back is sized to guarantee catching one
    leak_min_words-long window, not an
    arbitrarily long verbatim run. A leak much longer than that window,
    streamed in pathologically tiny increments (well under one word per
    chunk), may have its earliest words emitted before enough lookback
    context accumulates to recognize them as part of the same run. Real
    provider streams emit whole tokens per chunk, which is comfortably within
    the guaranteed window; this is documented here rather than solved with
    unbounded lookback, which would defeat the bounded-memory point of a hold-back.
    """

    def __init__(
        self,
        own_data: set[tuple[str, str]],
        system_prompt: str | None,
        output_pii_action: str,
        output_leak_action: str,
        leak_min_words: int,
        placeholder_mapping: dict[str, str],
        observe: bool,
    ) -> None:
        """Parameters mirror apply_output_guardrails(), plus placeholder_mapping for inline restore as text is emitted."""
        self._buffer = ""
        self._own_data = own_data
        self._system_prompt = system_prompt or ""
        self._output_pii_action = output_pii_action
        self._output_leak_action = output_leak_action
        self._leak_min_words = leak_min_words
        self._mapping = placeholder_mapping
        self._observe = observe
        self._hold_back = streaming_hold_back_chars(leak_min_words)
        # The last _hold_back characters of raw text already handed to
        # _process(), kept purely as read-only lookback context so a PII
        # match or a leak's word-run that starts in one feed() call and
        # continues in the next is still detected as one match, not two
        # unrelated fragments neither of which is long enough to trigger on
        # its own. Text in this tail was already returned to the caller and
        # is never re-emitted or re-redacted.
        self._context_tail = ""
        self.pii_counts: dict[str, int] = {}
        self.leak_redacted = False
        self.blocked = False
        self.blocked_reason: str | None = None
        self.unrestored_count = 0

    def feed(self, chunk: str) -> str:
        """
        Feed newly arrived model text.

        Parameters: chunk - the next piece of text from the provider stream.
        Returns: the portion of the buffer now safe to emit to the caller
        (guardrail-processed and placeholder-restored); "" if nothing is safe
        to emit yet.
        """
        self._buffer += chunk
        if len(self._buffer) <= self._hold_back:
            return ""
        safe_len = len(self._buffer) - self._hold_back
        safe_len = self._shrink_to_avoid_split_matches(safe_len)
        if safe_len <= 0:
            return ""
        to_process = self._buffer[:safe_len]
        self._buffer = self._buffer[safe_len:]
        result = self._process(to_process)
        self._context_tail = (self._context_tail + to_process)[-self._hold_back :]
        return result

    def finish(self) -> str:
        """Flush the remaining hold-back buffer at stream end. Returns the final safe text."""
        remaining = self._buffer
        self._buffer = ""
        if not remaining:
            return ""
        result = self._process(remaining)
        self._context_tail = (self._context_tail + remaining)[-self._hold_back :]
        return result

    def _shrink_to_avoid_split_matches(self, safe_len: int) -> int:
        """Pull safe_len back to before any PII match or placeholder token that would otherwise be cut in half."""
        for _name, (start, end), _value in find_pii(self._buffer):
            if start < safe_len < end:
                safe_len = start
        for match in PLACEHOLDER_TOKEN_REGEX.finditer(self._buffer):
            if match.start() < safe_len < match.end():
                safe_len = match.start()
        return safe_len

    def _process(self, text: str) -> str:
        """
        Apply output PII/leak checks and placeholder restore to one
        safe-to-emit slice, using self._context_tail as lookback-only
        detection context (never re-emitted, never re-redacted).
        """
        context = self._context_tail
        combined = context + text
        offset = len(context)

        if self._observe:
            for name, (_start, end), _value in find_pii(combined):
                if end > offset:
                    self.pii_counts[name] = self.pii_counts.get(name, 0) + 1
            restored, unrestored = restore_placeholders(combined[offset:], self._mapping)
            self.unrestored_count += unrestored
            return restored

        replacements: list[tuple[int, int, str]] = []

        if self._output_pii_action != "off":
            foreign_matches = [
                (name, span)
                for name, span, value in find_pii(combined)
                if (name, _normalize_pii_value(name, value)) not in self._own_data
            ]
            new_matches = [(name, span) for name, span in foreign_matches if span[1] > offset]
            if new_matches:
                for name, _span in new_matches:
                    self.pii_counts[name] = self.pii_counts.get(name, 0) + 1
                if self._output_pii_action == "block":
                    # A streamed block cannot un-send prior text, so it
                    # always degrades to redact (see class docstring) -
                    # blocked/blocked_reason are still recorded for the ledger.
                    self.blocked = True
                    self.blocked_reason = self.blocked_reason or "pii"
                for name, (start, end) in new_matches:
                    replacements.append((start, end, f"[REDACTED_{name.upper()}]"))

        if self._system_prompt and self._leak_min_words > 0 and self._output_leak_action != "off":
            leak_spans = [
                span for span in _find_leak_spans(combined, self._system_prompt, self._leak_min_words) if span[1] > offset
            ]
            if leak_spans:
                self.leak_redacted = True
                if self._output_leak_action == "block":
                    self.blocked = True
                    self.blocked_reason = self.blocked_reason or "system_prompt_leak"
                for start, end in leak_spans:
                    replacements.append((start, end, "[REDACTED_SYSTEM_PROMPT]"))

        new_text = _apply_replacements(combined, offset, replacements)
        restored, unrestored = restore_placeholders(new_text, self._mapping)
        self.unrestored_count += unrestored
        return restored
