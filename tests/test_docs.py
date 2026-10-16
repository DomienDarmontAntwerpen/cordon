"""
Keeps the repository readable: every module, class and function has a
docstring, and the text is English only (patterns.yaml, which holds Dutch and
French detection phrases, is the one exception).

Run: .venv/bin/python -m pytest tests/test_docs.py -v
"""

import ast
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).parent.parent

# --- Section: docstring coverage ---
PYTHON_DIRS = ["src", "tools", "tests"]


def _python_files() -> list[pathlib.Path]:
    """Return every tracked .py file under the directories this check covers."""
    files = []
    for directory in PYTHON_DIRS:
        base = REPO_ROOT / directory
        if base.exists():
            files.extend(p for p in base.rglob("*.py") if ".venv" not in p.parts)
    return files


def _missing_docstrings(path: pathlib.Path) -> list[str]:
    """
    Parameters: path - a Python source file.
    Returns: a list of human-readable descriptions of nodes missing a docstring
    (module, class, or function/async function), empty if all are documented.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    missing = []
    if ast.get_docstring(tree) is None:
        missing.append(f"{path}: module docstring")
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node) is None:
                missing.append(f"{path}:{node.lineno}: {type(node).__name__} '{node.name}' has no docstring")
    return missing


def test_every_module_class_and_function_has_a_docstring():
    """Every module, class and function/async function under src/, tools/ and tests/ must have a docstring."""
    all_missing = []
    for path in _python_files():
        all_missing.extend(_missing_docstrings(path))
    assert not all_missing, "missing docstrings:\n" + "\n".join(all_missing)


# --- Section: English-only repository, with one documented exception ---
DUTCH_WORDS = ["het", "niet", "wordt", "deze", "zijn", "voor", "maar", "ook", "alleen", "kosten", "negeer", "vorige", "instructies"]
DUTCH_WORD_REGEX = re.compile(r"\b(" + "|".join(DUTCH_WORDS) + r")\b", re.IGNORECASE)

SCANNED_EXTENSIONS = {".py", ".md", ".yaml", ".html", ".js", ".css"}
# test_docs.py itself is excluded alongside patterns.yaml: it has to hold the
# literal Dutch word list as data in order to scan for it, the same reason
# patterns.yaml is allowed to hold Dutch/French detection phrases.
EXCLUDED_NAMES = {"patterns.yaml", "CORDON_PROMPT.md", "test_docs.py"}
EXCLUDED_DIR_PARTS = {".venv", "data"}


def _scanned_files() -> list[pathlib.Path]:
    """Return every file under the repo matching SCANNED_EXTENSIONS, minus the documented exclusions."""
    files = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in SCANNED_EXTENSIONS:
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        if EXCLUDED_DIR_PARTS & set(path.parts):
            continue
        if ".git" in path.parts:
            continue
        files.append(path)
    return files


def test_repository_is_english_only_outside_patterns_yaml():
    """No tracked file outside patterns.yaml (and this test) may contain the sample Dutch words as whole words."""
    violations = []
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in DUTCH_WORD_REGEX.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            violations.append(f"{path}:{line_no}: found Dutch word '{match.group()}'")
    assert not violations, "non-English text found outside patterns.yaml:\n" + "\n".join(violations)
