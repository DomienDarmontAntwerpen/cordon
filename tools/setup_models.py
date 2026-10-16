"""
Downloads the embedding model and the spaCy model once after install, so
Cordon, the tests and the tools then run offline.

    .venv/bin/python -m tools.setup_models
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cordon.setup_models")

# --- Section: constants — the exact models the rest of Cordon expects on disk ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SPACY_MODEL_NAME = "en_core_web_sm"


def prefetch_embedding_model() -> None:
    """
    Download and cache the sentence-transformers embedding model.

    Parameters: none.
    Returns: None.
    Why: quality.py, cache.py and limits.py all share one embedding model
    instance; it must be cached locally before the server can start offline.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Downloading/loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vector = model.encode(["cordon setup check"])
    logger.info("Embedding model ready, vector dimension=%d", vector.shape[1])


def prefetch_spacy_model() -> None:
    """
    Install and verify the spaCy en_core_web_sm model package.

    Parameters: none.
    Returns: None.
    Why: optimize.py's prompt trim uses spaCy's blank("en") pipeline (no
    download needed), but en_core_web_sm is prefetched here as the shared,
    offline-available English model for any component that needs it.
    """
    import spacy

    try:
        spacy.load(SPACY_MODEL_NAME)
        logger.info("spaCy model already present: %s", SPACY_MODEL_NAME)
        return
    except OSError:
        pass

    logger.info("Installing spaCy model: %s", SPACY_MODEL_NAME)
    from spacy.cli.download import download as spacy_download

    spacy_download(SPACY_MODEL_NAME)
    spacy.load(SPACY_MODEL_NAME)
    logger.info("spaCy model installed and verified: %s", SPACY_MODEL_NAME)


def main() -> int:
    """
    Run both prefetch steps and report success or failure.

    Parameters: none.
    Returns: process exit code (0 on success, 1 on failure).
    Why: a single entry point keeps the documented setup command
    (`.venv/bin/python -m tools.setup_models`) in one place.
    """
    try:
        prefetch_embedding_model()
        prefetch_spacy_model()
    except Exception:
        logger.exception("Model setup failed")
        return 1
    logger.info("All models ready. Cordon can now run fully offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
