"""
Make a request smaller without changing its meaning.

- `trim_messages()`: whitespace and exact duplicate text; never the system
  message, negations or code.
- `trim_history()`: drops the oldest turns above `history_max_tokens`.
- `truncate_tool_output()`: shortens huge tool results, with a marker.
- Images: exact OpenAI token count, downscale, metadata stripped; the pixel
  limit is read from the file header before anything is decoded.
"""

import base64
import logging
import math
import re
from dataclasses import dataclass

import cv2
import litellm
import numpy as np

from src import messages as messages_module

logger = logging.getLogger("cordon.optimize")

try:
    import spacy

    _nlp = spacy.blank("en")
    _nlp.add_pipe("sentencizer")
except Exception:  # pragma: no cover - spaCy is a hard dependency, but trim must still fail open
    _nlp = None
    logger.warning("spaCy pipeline unavailable; sentence-level dedup will be skipped (fail open)")


# --- Section: prompt trim — whitespace normalization and exact-duplicate removal ---
_CODE_BLOCK_REGEX = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_REGEX = re.compile(r"`[^`\n]+`")


def _protect_code(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace fenced and inline code spans with opaque placeholders.

    Parameters: text - the message text to protect.
    Returns: (protected_text, placeholders) where placeholders maps each
    inserted marker back to the original code span.
    Why: trim must never touch code - deduplicating or reflowing code can
    silently change its meaning (e.g. two visually identical lines that
    differ only in whitespace significant to the language).
    """
    placeholders: dict[str, str] = {}
    counter = 0

    def replace(match: re.Match) -> str:
        """Replace one matched code span with a fresh opaque placeholder key."""
        nonlocal counter
        key = f"\x00CODE{counter}\x00"
        placeholders[key] = match.group(0)
        counter += 1
        return key

    text = _CODE_BLOCK_REGEX.sub(replace, text)
    text = _INLINE_CODE_REGEX.sub(replace, text)
    return text, placeholders


def _restore_code(text: str, placeholders: dict[str, str]) -> str:
    """Put protected code spans back. Parameters: text, placeholders (from _protect_code). Returns: the restored text."""
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, collapse 3+ newlines to a paragraph break, strip trailing line whitespace."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _dedupe_sentences_in_paragraph(paragraph: str) -> str:
    """
    Remove exact-duplicate sentences within one paragraph, keeping the first occurrence.

    Parameters: paragraph - one paragraph of text (no blank lines inside).
    Returns: the paragraph with later exact repeats of an earlier sentence
    removed. Falls back to the unmodified paragraph if spaCy is unavailable
    (fail open - a cost optimization skipping itself is never a correctness risk).
    """
    if _nlp is None or not paragraph.strip():
        return paragraph
    doc = _nlp(paragraph)
    seen: set[str] = set()
    kept: list[str] = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue
        if sentence_text in seen:
            continue
        seen.add(sentence_text)
        kept.append(sentence_text)
    return " ".join(kept) if kept else paragraph


def trim_text(text: str) -> str:
    """
    Normalize whitespace and remove exact-duplicate sentences/paragraphs.

    Parameters: text - one message's text content.
    Returns: the trimmed text. Stopwords and negations are never touched
    (no word-level editing happens at all); code blocks are protected and
    restored unchanged; meaning is preserved because the only operations are
    whitespace normalization and removal of sentences/paragraphs that are
    exact, verbatim repeats of earlier ones in the same message.
    """
    if not text:
        return text
    protected, placeholders = _protect_code(text)
    normalized = _normalize_whitespace(protected)

    paragraphs = normalized.split("\n\n")
    seen_paragraphs: set[str] = set()
    kept_paragraphs: list[str] = []
    for paragraph in paragraphs:
        key = paragraph.strip()
        if key and key in seen_paragraphs:
            continue
        if key:
            seen_paragraphs.add(key)
        kept_paragraphs.append(_dedupe_sentences_in_paragraph(paragraph))

    result = "\n\n".join(kept_paragraphs)
    return _restore_code(result, placeholders)


def _map_message_content(message: dict, fn) -> dict:
    """
    Return a new message with its `content` text (string, or text-type parts
    of a list content) replaced by fn(text). Unlike guardrails.py's
    messages.map_text_fields, this deliberately never touches tool_calls[].function.
    arguments - that is structured JSON, and deduplicating "sentences" inside
    a JSON string would corrupt it.
    """
    new_message = dict(message)
    content = message.get("content")
    if isinstance(content, str):
        new_message["content"] = fn(content)
    elif isinstance(content, list):
        new_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                new_part = dict(part)
                new_part["text"] = fn(part["text"])
                new_parts.append(new_part)
            else:
                new_parts.append(part)
        new_message["content"] = new_parts
    return new_message


def trim_messages(messages: list[dict]) -> list[dict]:
    """
    Apply trim_text() to every user/assistant message's content.

    Parameters: messages - the full request's messages.
    Returns: a new list; system messages are returned unchanged (never
    trimmed, so the stable prefix providers use for prompt caching is never
    touched); tool messages are also left alone here - their own truncation
    is truncate_tool_output()'s job, a different operation with a different
    budget.
    """
    result = []
    for message in messages:
        if messages_module.is_instruction(message) or message.get("role") == "tool":
            result.append(dict(message))
            continue
        result.append(_map_message_content(message, trim_text))
    return result


# --- Section: history trimming — drop oldest turns, keep system + last user message ---
def trim_history(messages: list[dict], history_max_tokens: int, litellm_name: str) -> list[dict]:
    """
    Drop the oldest non-system turns when the conversation exceeds history_max_tokens.

    Parameters: messages - the full request's messages.
                 history_max_tokens - the tenant's/global context.history_max_tokens.
                 litellm_name - the model used for token counting.
    Returns: a new list, always keeping every system message and the last
    user message; oldest turns in between are dropped first, one at a time,
    until the remaining messages fit the budget or nothing more can be
    dropped without losing the last user message.
    """
    if litellm.token_counter(model=litellm_name, messages=messages) <= history_max_tokens:
        return messages

    system_messages = [m for m in messages if messages_module.is_instruction(m)]
    non_system = [m for m in messages if not messages_module.is_instruction(m)]
    if not non_system:
        return messages

    user_indices = [i for i, m in enumerate(non_system) if m.get("role") == "user"]
    last_user_index = user_indices[-1] if user_indices else len(non_system) - 1

    must_keep = non_system[last_user_index:]
    droppable = list(non_system[:last_user_index])

    while droppable:
        candidate = system_messages + droppable + must_keep
        if litellm.token_counter(model=litellm_name, messages=candidate) <= history_max_tokens:
            break
        droppable.pop(0)

    return system_messages + droppable + must_keep


# --- Section: tool output truncation ---
def truncate_tool_output(messages: list[dict], tool_output_max_tokens: int, litellm_name: str) -> list[dict]:
    """
    Truncate `role: tool` message content above tool_output_max_tokens.

    Parameters: messages - the full request's messages.
                 tool_output_max_tokens - context.tool_output_max_tokens.
                 litellm_name - the model used for token counting.
    Returns: a new list; each oversized tool message's content is cut to
    approximately the token budget (by proportional character length - an
    approximation, not an exact token boundary, which is acceptable for a
    cost optimization) and suffixed with a
    "[... truncated N tokens by Cordon ...]" marker naming the excess.
    """
    result = []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "tool" or not isinstance(content, str):
            result.append(message)
            continue
        tokens = litellm.token_counter(model=litellm_name, text=content)
        if tokens <= tool_output_max_tokens:
            result.append(message)
            continue
        excess = tokens - tool_output_max_tokens
        ratio = tool_output_max_tokens / tokens
        approx_chars = max(1, int(len(content) * ratio))
        new_message = dict(message)
        new_message["content"] = f"{content[:approx_chars]} [... truncated {excess} tokens by Cordon ...]"
        result.append(new_message)
    return result


# --- Section: images — exact OpenAI token algorithm, header-only dimension reads ---
class ImageTooLargeError(Exception):
    """Raised when an image's pixel count exceeds the configured limit, or its format cannot be verified safely."""

    def __init__(self, width: int | None = None, height: int | None = None) -> None:
        """Parameters: width, height - the image's dimensions, if known."""
        self.width = width
        self.height = height
        super().__init__(f"image exceeds the pixel limit ({width}x{height})" if width else "image format could not be verified")


_DATA_URL_REGEX = re.compile(r"^data:(image/[\w.+-]+);base64,(.+)$", re.DOTALL)


def parse_data_url(url: str) -> tuple[bytes, str] | None:
    """
    Decode a base64 image data URL.

    Parameters: url - the `image_url.url` value from a message content part.
    Returns: (raw_bytes, mime_type), or None if `url` is not a base64 data
    URL (e.g. a remote http(s) URL, which Cordon never fetches - see SECURITY.md's SSRF section).
    """
    match = _DATA_URL_REGEX.match(url)
    if not match:
        return None
    mime_type, b64_data = match.groups()
    try:
        return base64.b64decode(b64_data, validate=False), mime_type
    except Exception:
        return None


def _read_png_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read width/height from a PNG's IHDR chunk (bytes 16-24), without decoding pixel data."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def _read_gif_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read width/height from a GIF's logical screen descriptor (bytes 6-10)."""
    if len(data) < 10 or data[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    width = int.from_bytes(data[6:8], "little")
    height = int.from_bytes(data[8:10], "little")
    return width, height


_JPEG_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _read_jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Walk JPEG markers to the first SOF segment, which carries height/width, without decoding pixel data."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(data)
    while i + 1 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in _JPEG_SOF_MARKERS:
            if i + 9 > n:
                return None
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            return width, height
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9 or i + 4 > n:
            return None
        segment_length = int.from_bytes(data[i + 2 : i + 4], "big")
        i += 2 + segment_length
    return None


def _read_webp_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read width/height from a WEBP container's VP8/VP8L/VP8X chunk header."""
    if len(data) < 30 or data[0:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk_type = data[12:16]
    if chunk_type == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk_type == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    if chunk_type == b"VP8 ":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    return None


def read_image_dimensions(data: bytes) -> tuple[int, int] | None:
    """
    Read an image's width/height from its header, without decoding pixel data.

    Parameters: data - raw image bytes.
    Returns: (width, height), or None if the format is not PNG/JPEG/GIF/WEBP
    (the formats OpenAI-compatible vision APIs accept).
    Why header-only: decoding pixel data to find dimensions would defeat the
    point of a pixel-limit check meant to run "before any expensive work"
    - a small compressed file can decode to an
    enormous pixel array (a decompression-bomb-style DoS).
    """
    for parser in (_read_png_dimensions, _read_jpeg_dimensions, _read_gif_dimensions, _read_webp_dimensions):
        result = parser(data)
        if result:
            return result
    return None


def iter_image_data_urls(messages: list[dict]):
    """
    Yield every `image_url.url` string found in any message's content parts.

    Parameters: messages - the full request's messages.
    Yields: url strings (base64 data URLs and remote URLs alike).
    """
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    yield url


def check_image_pixel_limits(messages: list[dict], max_image_pixels: int) -> None:
    """
    Enforce the pixel limit on every base64 image in a request, before any decoding.

    Parameters: messages - the full request's messages.
                 max_image_pixels - config.yaml's request_limits.max_image_pixels.
    Raises: ImageTooLargeError on the first image whose header reports a
    pixel count over the limit, or whose format cannot be verified at all
    (fail closed - this is one of step1_admission.py's always-on, security-relevant
    checks, not a cost optimization, so an unverifiable image is rejected
    rather than silently allowed through).
    Returns: None if every base64 image is within the limit. Remote image
    URLs (not base64 data URLs) are not checked here - Cordon never fetches
    them (see SECURITY.md, SSRF), so their size is the provider's concern.
    """
    for url in iter_image_data_urls(messages):
        parsed = parse_data_url(url)
        if parsed is None:
            continue
        raw, _mime = parsed
        dims = read_image_dimensions(raw)
        if dims is None:
            raise ImageTooLargeError()
        width, height = dims
        if width * height > max_image_pixels:
            raise ImageTooLargeError(width, height)


# --- Section: what an image costs, per model, as OpenAI bills it ---
# Measured live on 16 October 2026 (prompt tokens of a request with one image, minus the same request
# without it; exact to the token on all 19 measured requests): gpt-4o and gpt-4o-mini bill 512-px tiles (gpt-4o-mini
# at about 33x the tokens - its tokens are cheaper); the gpt-4.1 minis bill 32-px patches, capped at
# 3,072, times a multiplier, and ignore "detail". Other models: gpt-4o's rule (as LiteLLM assumes).
_TILE_RULES = {"gpt-4o": (85, 170), "gpt-4o-mini": (2833, 5667)}
_PATCH_RULES = {"gpt-4.1-mini": 1.62, "gpt-4.1-nano": 2.46}
_PATCH_SIDE, _MAX_PATCHES = 32, 3072
def _tile_count(width: float, height: float) -> int:
    """
    Compute the 512x512 tile count OpenAI's "high" detail algorithm would use.

    Parameters: width, height - image dimensions (already scaled or raw).
    Returns: ceil(width/512) * ceil(height/512), after first fitting the
    image within 2048x2048 (scale down only) and then scaling so the
    shortest side is 768px (scale down only) - OpenAI's documented algorithm.
    """
    w, h = float(width), float(height)
    if max(w, h) > 2048:
        scale = 2048 / max(w, h)
        w, h = w * scale, h * scale
    if min(w, h) > 768:
        scale = 768 / min(w, h)
        w, h = w * scale, h * scale
    return math.ceil(w / 512) * math.ceil(h / 512)


def _patch_count(width: float, height: float) -> int:
    """32x32 patches covering the image, after shrinking it to at most _MAX_PATCHES of them."""
    w, h = float(width), float(height)
    if math.ceil(w / _PATCH_SIDE) * math.ceil(h / _PATCH_SIDE) > _MAX_PATCHES:
        scale = math.sqrt(_MAX_PATCHES * _PATCH_SIDE ** 2 / (w * h))
        scale *= min(math.floor(w * scale / _PATCH_SIDE) / (w * scale / _PATCH_SIDE),
                     math.floor(h * scale / _PATCH_SIDE) / (h * scale / _PATCH_SIDE))
        w, h = w * scale, h * scale
    return math.ceil(w / _PATCH_SIDE) * math.ceil(h / _PATCH_SIDE)


def image_tokens(model: str, width: int, height: int, detail: str = "high") -> int:
    """
    The prompt tokens OpenAI bills for one image sent to this model.

    Parameters: model - an alias or LiteLLM name ("gpt-4o-mini", "openai/gpt-4o-mini").
                 width, height - image pixel dimensions. detail - "low", "high" or "auto".
    """
    name = model.split("/")[-1]
    if name in _PATCH_RULES:
        return int(_patch_count(width, height) * _PATCH_RULES[name])
    base, per_tile = _TILE_RULES.get(name, _TILE_RULES["gpt-4o"])
    return base if detail == "low" else base + per_tile * _tile_count(width, height)


# --- Section: image optimization — downscale, detail, grayscale (documents), re-encode ---
@dataclass
class ImageOptimizationResult:
    """
    Outcome of optimize_image_part().

    Fields: data_url - the (possibly re-encoded) image to send to the
    provider. detail - the detail level to request ("low" if Cordon
    downgraded it). tokens_before/tokens_after - OpenAI's own token cost at
    the original vs. new size/detail. savings_tokens - tokens_before -
    tokens_after, but only if the tile count actually decreased (or detail
    was downgraded to "low"); otherwise 0, even if tokens_before != tokens_after,
    because savings count only when the result has fewer tiles than
    OpenAI would produce itself. exif_stripped - True if the
    image was re-encoded (which always drops metadata).
    """

    data_url: str
    detail: str
    tokens_before: int
    tokens_after: int
    savings_tokens: int
    exif_stripped: bool = False


def decode_image(raw_bytes: bytes) -> np.ndarray | None:
    """Decode image bytes into a BGR numpy array via OpenCV. Returns None on failure (fail open for callers)."""
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


def downscale_image(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is at most max_side, via OpenCV INTER_AREA. Never upscales."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def grayscale_image(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale, kept as a 3-channel image for consistent downstream encoding. Reduces payload size, not tokens."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def encode_image_to_data_url(image: np.ndarray, mime_type: str, quality: int = 85) -> str:
    """
    Re-encode an image to a base64 data URL.

    Parameters: image - a BGR numpy array. mime_type - the original MIME
    type (PNG stays PNG; anything else becomes JPEG). quality - JPEG quality.
    Returns: a "data:<mime>;base64,<...>" string.
    Why this strips EXIF: re-encoding from a decoded pixel array never
    carries over the original file's metadata, including GPS EXIF tags.
    """
    use_png = mime_type == "image/png"
    ext = ".png" if use_png else ".jpg"
    params = [] if use_png else [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(ext, image, params)
    if not success:
        raise ValueError("failed to encode image")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    out_mime = "image/png" if use_png else "image/jpeg"
    return f"data:{out_mime};base64,{encoded}"


def optimize_image_part(
    image_part: dict,
    image_max_side: int,
    detail_low_for_photos: bool,
    is_document: bool,
    model: str,
) -> ImageOptimizationResult:
    """
    Optimize one image content part: downscale, optionally downgrade detail,
    optionally grayscale (documents only), and re-encode.

    Parameters: image_part - a `{"type": "image_url", "image_url": {...}}` content part.
                 image_max_side - the tenant's optimize.image_max_side.
                 detail_low_for_photos - the tenant's optimize.image_detail_low_for_photos.
                 model - the requested model, whose billing the saving is counted in.
                 is_document - True if this image was classified as a
                 document (routing.py's classification feeds this; a
                 document keeps "high" detail for OCR readability and gets
                 grayscaled for payload size instead).
    Returns: an ImageOptimizationResult. On any failure to parse/decode/
    re-encode, fails open: returns the original url unchanged with 0 savings
    (a cost-optimization failure is skipped and logged, never a request failure).
    """
    image_url_obj = image_part.get("image_url") or {}
    url = image_url_obj.get("url", "")
    requested_detail = image_url_obj.get("detail", "high")

    parsed = parse_data_url(url)
    if parsed is None:
        return ImageOptimizationResult(url, requested_detail, 0, 0, 0, False)

    raw, mime_type = parsed
    dims = read_image_dimensions(raw)
    if dims is None:
        return ImageOptimizationResult(url, requested_detail, 0, 0, 0, False)
    original_width, original_height = dims
    tokens_before = image_tokens(model, original_width, original_height, requested_detail)

    try:
        image = decode_image(raw)
        if image is None:
            raise ValueError("decode returned None")

        new_detail = requested_detail
        if detail_low_for_photos and not is_document and requested_detail != "low":
            new_detail = "low"

        resized = downscale_image(image, image_max_side)
        new_height, new_width = resized.shape[:2]

        payload_image = grayscale_image(resized) if is_document else resized

        tokens_after = image_tokens(model, new_width, new_height, new_detail)
        savings_tokens = max(0, tokens_before - tokens_after)

        new_data_url = encode_image_to_data_url(payload_image, mime_type)
    except Exception:
        logger.warning("image optimization skipped: processing failed (fail open)")
        return ImageOptimizationResult(url, requested_detail, tokens_before, tokens_before, 0, False)

    return ImageOptimizationResult(
        data_url=new_data_url,
        detail=new_detail,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        savings_tokens=savings_tokens,
        exif_stripped=True,
    )
