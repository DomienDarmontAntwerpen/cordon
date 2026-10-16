"""
Exact and semantic cache, isolated per tenant and encrypted at rest.

- Exact: a keyed hash over the masked request and every answer-shaping parameter.
- Semantic: a similar question (cosine >= `semantic_threshold`) with the same
  negations, numbers and names in the same order. Off by default.
- Never cached: holdout, unmasked personal data, tools, images, high
  temperature, time-sensitive questions, `X-Cordon-Cache: off`.
- Stores no message text: hashes, vectors and the encrypted answer only.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from src import messages as messages_module
from src.protection import guardrails, security
from src.savings import quality
from src.server.store import Store

logger = logging.getLogger("cordon.cache")


# --- Section: canonical serialization — stable hashing for cache keys ---
def _canonical_json(value: Any) -> str:
    """Serialize a value deterministically (sorted keys, no extra whitespace) for stable hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


# --- Section: cache eligibility — the never-written rules ---
@dataclass
class CacheEligibility:
    """Result of check_eligibility(). eligible - whether a cache read/write may happen at all; reason - why not, if ineligible."""

    eligible: bool
    reason: str | None = None


def _mentions_time_word(text: str) -> bool:
    """True if text contains any configured time word (any language), as a whole word, case-insensitive."""
    if not text:
        return False
    lowered = text.lower()
    for words in guardrails.TIME_WORDS.values():
        for word in words:
            if re.search(rf"\b{re.escape(word.lower())}\b", lowered):
                return True
    return False


def check_eligibility(
    holdout: bool,
    pii_detected_unmasked: bool,
    has_tools: bool,
    has_images: bool,
    temperature: float | None,
    cache_max_temperature: float,
    last_user_text: str,
    cache_header_off: bool,
) -> CacheEligibility:
    """
    Decide whether this request may read from or write to the cache at all.

    Parameters: holdout - whether this request was selected into the holdout group.
                 pii_detected_unmasked - True if PII was found in the request
                 but pii_action is not "mask" (so real PII would end up in the cache key material).
                 has_tools - whether the request includes `tools`.
                 has_images - whether any message contains an image part.
                 temperature - the request's temperature, or None.
                 cache_max_temperature - the tenant's optimize.cache_max_temperature.
                 last_user_text - the (masked) last user message's text, checked for time words.
                 cache_header_off - True if the caller sent `X-Cordon-Cache: off`.
    Returns: a CacheEligibility.
    Why a missing temperature counts as 1.0: omitting `temperature` means
    the provider's own default applies, which for every model in the pool is
    1.0 (effectively maximum randomness) - treating it as anything lower
    would cache a response that was actually sampled at high temperature.
    """
    if cache_header_off:
        return CacheEligibility(False, "disabled_by_header")
    if holdout:
        return CacheEligibility(False, "holdout")
    if pii_detected_unmasked:
        return CacheEligibility(False, "unmasked_pii")
    if has_tools:
        return CacheEligibility(False, "tools")
    if has_images:
        return CacheEligibility(False, "images")
    effective_temperature = 1.0 if temperature is None else temperature
    if effective_temperature > cache_max_temperature:
        return CacheEligibility(False, "temperature_too_high")
    if _mentions_time_word(last_user_text):
        return CacheEligibility(False, "time_sensitive")
    return CacheEligibility(True, None)


# --- Section: negation/number token extraction — the semantic cache's own-data-free guard ---
def extract_negation_and_number_tokens(text: str) -> list[str]:
    """
    Extract the set of negation words and numbers present in a text.

    Parameters: text - the (masked) last user message.
    Returns: a sorted list of lowercased negation words (from
    guardrails.NEGATIONS, every language) and numeric substrings found in
    text. Used as an exact-match guard on top of cosine similarity, so a
    semantic cache hit can never flip a "has an invoice" answer onto a "has
    no invoice" question, or a "costs 50" answer onto a "costs 500" question
    just because the two are semantically close.
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: set[str] = set()
    for words in guardrails.NEGATIONS.values():
        for word in words:
            if re.search(rf"\b{re.escape(word.lower())}\b", lowered):
                tokens.add(word.lower())
    for match in re.finditer(r"\d+(?:[.,]\d+)?", text):
        tokens.add(match.group())
    return sorted(tokens)


_WORD_REGEX = re.compile(r"[^\W\d_][\w'-]*")
_SENTENCE_SPLIT_REGEX = re.compile(r"[.?!]\s+")


def extract_name_tokens(text: str) -> list[str]:
    """
    Extract the capitalised words (names, places, products) of a text, in order.

    Parameters: text - the (masked) last user message.
    Returns: lowercased capitalised words that do not start a sentence
    (and not the pronoun "I"), in the order they appear.
    Why: an embedding sees "rehab center in Bartow County" and "rehab center
    in Towns County", or "cheaper in India than the UK" and "cheaper in the
    UK than India", as near-identical - measured on 40,430 labelled question
    pairs (notebooks/semantic_cache_threshold.ipynb), requiring the same
    names in the same order removed about a third of wrong semantic hits
    at the default threshold (34% on the held-out half). Order matters, so a swapped comparison never matches.
    """
    tokens = []
    for sentence in _SENTENCE_SPLIT_REGEX.split(text or ""):
        for word in _WORD_REGEX.findall(sentence)[1:]:
            if word[0].isupper() and word != "I":
                tokens.append(word.lower())
    return tokens


def semantic_guard_tokens(text: str) -> list[str]:
    """
    The exact-match guard a semantic cache hit needs on top of cosine similarity.

    Returns: the sorted negation words and numbers, a separator, then the
    ordered name tokens. Two questions may share a cached answer only if
    these lists are identical.
    """
    return extract_negation_and_number_tokens(text) + ["|names|"] + extract_name_tokens(text)


# --- Section: message splitting — isolate the last user message from "everything else" ---
def _split_last_user_message(messages: list[dict]) -> tuple[list[dict], str]:
    """
    Separate the last user message's text from the rest of the request.

    Parameters: messages - the masked request messages.
    Returns: (other_messages, last_user_text). other_messages is `messages`
    with the last user message's content blanked out (its role/position is
    kept so the context hash still reflects "a user turn happened here");
    last_user_text is that message's text (messages.message_text()), or "" if there
    is no user message.
    """
    last_user_index = messages_module.last_user_index(messages)
    if last_user_index is None:
        return list(messages), ""
    last_user_text = messages_module.message_text(messages[last_user_index].get("content"))
    other = list(messages)
    other[last_user_index] = {**other[last_user_index], "content": None}
    return other, last_user_text


# --- Section: Cache — exact and semantic, tenant-isolated, encrypted ---
class Cache:
    """
    Exact and semantic response cache. One instance per process, holding a
    Store, the data-encryption key and hash salt, and the tenant-independent
    thresholds (semantic_threshold, cache_ttl_seconds) from config.yaml.
    """

    def __init__(self, store: Store, data_key: bytes, hash_salt: str, semantic_threshold: float, cache_ttl_seconds: int) -> None:
        """Parameters: store, data_key (32 bytes), hash_salt, semantic_threshold, cache_ttl_seconds - see class docstring."""
        self._store = store
        self._encryption = security.CacheEncryption(data_key)
        self._hash_salt = hash_salt
        self._semantic_threshold = semantic_threshold
        self._cache_ttl_seconds = cache_ttl_seconds

    # --- exact cache key ---
    def exact_key(
        self,
        tenant_id: str,
        mode: str,
        requested_model: str,
        masked_messages: list[dict],
        temperature: float | None,
        tools: Any,
        response_format: Any,
        generation_params: dict | None = None,
    ) -> str:
        """
        Compute the exact-cache HMAC key for a request.

        Parameters: tenant_id, mode, requested_model, masked_messages
        (already PII-masked and optimize.py-processed), temperature, tools,
        response_format, generation_params - every other parameter that
        changes the answer (max_tokens, stop, top_p, seed, penalties): an
        answer cut off at max_tokens=3 must never be served to a request
        allowing 2,000.
        Returns: a tenant-scoped HMAC hex key (security.keyed_hash), so the
        same raw payload hashed for two different tenants never collides.
        """
        payload = {
            "mode": mode, "model": requested_model, "messages": masked_messages,
            "temperature": temperature, "tools": tools, "response_format": response_format,
            "generation": generation_params or {},
        }
        return security.keyed_hash(tenant_id, _canonical_json(payload), self._hash_salt)

    def get_exact(self, tenant_id: str, cache_key: str) -> str | None:
        """Return the cached answer text for an exact key, or None on a miss or decryption failure."""
        row = self._store.cache_get(cache_key, tenant_id)
        if row is None or row["cache_type"] != "exact":
            return None
        try:
            plaintext = self._encryption.decrypt(tenant_id, row["value_encrypted"])
        except Exception:
            logger.warning("exact cache decrypt failed (treated as a miss)")
            return None
        return plaintext.decode("utf-8")

    def put_exact(self, tenant_id: str, cache_key: str, answer_text: str) -> None:
        """Encrypt and store an answer under an exact key."""
        blob = self._encryption.encrypt(tenant_id, answer_text.encode("utf-8"))
        self._store.cache_put(cache_key, tenant_id, "exact", blob, vector=None, negation_set=None, ttl_seconds=self._cache_ttl_seconds)

    # --- semantic cache ---
    def context_hash(
        self,
        tenant_id: str,
        mode: str,
        requested_model: str,
        masked_messages: list[dict],
        temperature: float | None,
        tools: Any,
        response_format: Any,
        generation_params: dict | None = None,
    ) -> str:
        """
        Hash everything in the request except the last user message's text.

        Parameters: same as exact_key().
        Returns: a tenant-scoped HMAC hex hash of the system prompt, history,
        model, temperature, tools and response_format - used to find
        semantic-cache candidates that share every piece of context except
        the specific question being asked.
        """
        other_messages, _last_user_text = _split_last_user_message(masked_messages)
        payload = {
            "mode": mode, "model": requested_model, "messages": other_messages,
            "temperature": temperature, "tools": tools, "response_format": response_format,
            "generation": generation_params or {},
        }
        return security.keyed_hash(tenant_id, _canonical_json(payload), self._hash_salt)

    def get_semantic(
        self, tenant_id: str, context_hash: str, last_user_text: str
    ) -> str | None:
        """
        Look up a semantic cache hit.

        Parameters: tenant_id - owning tenant.
                     context_hash - from context_hash() above.
                     last_user_text - the masked last user message's text
                     (embedded here, and compared against each candidate's
                     guard tokens: negations, numbers, hashed names).
        Returns: the decrypted cached answer text, or None if there is no
        candidate meeting every condition: same context_hash, cosine
        similarity >= semantic_threshold between embeddings, and identical
        guard tokens (same negations and numbers, same names in the same order).
        """
        candidates = self._store.cache_entries_for_tenant(tenant_id, cache_type="semantic", context_hash=context_hash)
        if not candidates:
            return None
        query_tokens = self._stored_guard_tokens(tenant_id, last_user_text)
        query_vector = quality.embed_texts([last_user_text])[0]

        for row in candidates:
            stored_tokens = json.loads(row["negation_set_json"] or "[]")
            if stored_tokens != query_tokens:
                continue
            stored_vector = np.frombuffer(row["vector"], dtype=np.float32)
            similarity = quality.cosine_similarity(query_vector, stored_vector)
            if similarity >= self._semantic_threshold:
                try:
                    plaintext = self._encryption.decrypt(tenant_id, row["value_encrypted"])
                except Exception:
                    logger.warning("semantic cache decrypt failed (treated as a miss)")
                    continue
                return plaintext.decode("utf-8")
        return None

    def _stored_guard_tokens(self, tenant_id: str, last_user_text: str) -> list[str]:
        """
        The semantic guard tokens in the form that is stored: every token as a keyed hash.

        Why: names and numbers are fragments of the prompt, and only
        structured identifiers (email, IBAN, ...) are masked as PII - storing
        them in clear could put a person's name or an amount in the database.
        A tenant-scoped keyed hash keeps the equality check (same token, same
        position, same hash) and cannot be read back.
        """
        return [security.keyed_hash(tenant_id, token, self._hash_salt)[:16] for token in semantic_guard_tokens(last_user_text)]

    def put_semantic(self, tenant_id: str, context_hash: str, last_user_text: str, answer_text: str) -> None:
        """
        Store a semantic cache entry.

        Parameters: tenant_id, context_hash, last_user_text (embedded and
        tokenized, never stored as text itself), answer_text.
        Returns: None. The stored row holds only the embedding vector and
        the guard tokens (as keyed hashes) - never
        last_user_text or answer_text in clear (answer_text is AES-GCM encrypted).
        """
        vector = quality.embed_texts([last_user_text])[0].astype(np.float32)
        tokens = self._stored_guard_tokens(tenant_id, last_user_text)
        blob = self._encryption.encrypt(tenant_id, answer_text.encode("utf-8"))
        cache_key = security.keyed_hash(
            tenant_id, f"{context_hash}:{hashlib.sha256(vector.tobytes()).hexdigest()}", self._hash_salt
        )
        self._store.cache_put(
            cache_key, tenant_id, "semantic", blob, vector=vector.tobytes(), negation_set=tokens,
            ttl_seconds=self._cache_ttl_seconds, context_hash=context_hash,
        )
