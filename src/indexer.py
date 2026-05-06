"""
Indexer module for COMP3011 Coursework 2.

Builds and persists an inverted index of the form::

    {
        word: {
            url: {"freq": int, "positions": [int, ...]},
            ...
        },
        ...
    }

Storage: JSON or Pickle (auto-detected from path extension). The on-disk
shape is wrapped in a versioned envelope so older formats are rejected
on load with a rebuild prompt.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Iterable, Optional, Union

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

LOGGER = logging.getLogger(__name__)

# Apostrophe is in the character class so contractions ("don't") stay intact.
TOKEN_RE = re.compile(r"[a-z0-9']+")

SUPPORTED_FORMATS: tuple[str, ...] = ("json", "pickle")

# Lower-case keys; _format_from_path case-folds before lookup.
_EXTENSION_TO_FORMAT: dict[str, str] = {
    ".json": "json",
    ".pkl": "pickle",
    ".pickle": "pickle",
}

# On-disk envelope version; bumping invalidates older index files on load.
INDEX_FORMAT_VERSION: int = 2

# Tags whose contents are markup, not prose; decomposed before extraction.
NON_CONTENT_TAGS: tuple[str, ...] = ("script", "style")

# Embedded list (no nltk.download required); contractions excluded so apostrophes survive.
STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an",
    "and", "or", "but", "if", "of", "in", "on", "at", "by", "to", "for",
    "from", "with", "as",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "not",
})

# PorterStemmer is stateless; share one instance across calls.
_STEMMER = PorterStemmer()


def tokenise(
    text: str,
    *,
    remove_stopwords: bool = True,
    stem: bool = True,
) -> list[str]:
    """Return the normalised token sequence for a piece of plain text.

    Applied symmetrically at index time and query time — the same
    ``remove_stopwords`` and ``stem`` flags must be used on both sides
    or queries silently miss matches.

    Pipeline: lowercase -> regex tokenise -> drop stopwords -> Porter stem.
    Positions are assigned to the post-filter stream (stopwords removed
    before positions stamp), which phrase queries rely on.

    Parameters
    ----------
    text:
        Plain text. Caller HTML-strips first if needed.
    remove_stopwords:
        If True (default), drop tokens in :data:`STOPWORDS` before stemming.
    stem:
        If True (default), apply Porter stemming.

    Returns
    -------
    list[str]
        Tokens in document order. Empty input yields an empty list.
    """
    tokens = TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [_STEMMER.stem(t) for t in tokens]
    return tokens


def _format_from_path(path: Union[str, Path]) -> str:
    """Resolve the on-disk format from a file path's extension.

    * ``.json`` -> ``"json"``
    * ``.pkl`` / ``.pickle`` -> ``"pickle"``
    * anything else -> ``"json"`` fallback

    Case-insensitive. The fallback is JSON (not Pickle) because
    ``pickle.load`` would execute arbitrary code on unknown extensions.
    """
    suffix = Path(path).suffix.lower()
    return _EXTENSION_TO_FORMAT.get(suffix, "json")


def _strip_html(html: str) -> str:
    """Convert raw HTML to plain text suitable for tokenisation.

    Decomposes ``<script>`` and ``<style>`` outright (their bodies are
    code, not content) and joins the remainder with a space so words
    don't run together across tag boundaries.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(list(NON_CONTENT_TAGS)):
        tag.decompose()
    return soup.get_text(separator=" ")


class Indexer:
    """Build, persist, and query an inverted index of crawled pages.

    Use :meth:`get_postings` for word lookups (handles case-fold and missing
    keys); :attr:`index` is exposed for callers needing bulk access.
    """

    def __init__(self) -> None:
        self.index: dict[str, dict[str, dict]] = {}
        # 0-token pages still count toward N for IDF, so doc_lengths records every URL.
        self.doc_lengths: dict[str, int] = {}

    # ------------------------------------------------------------------ public

    def build(self, pages: Iterable[tuple[str, str]]) -> None:
        """Populate :attr:`index` and :attr:`doc_lengths` from ``pages``.

        Existing state is discarded; repeated builds are idempotent.

        Parameters
        ----------
        pages:
            Iterable of ``(url, html)`` tuples. Streaming is supported.
        """
        self.index = {}
        self.doc_lengths = {}
        for url, html in pages:
            tokens = tokenise(_strip_html(html))
            self.doc_lengths[url] = len(tokens)
            for position, token in enumerate(tokens):
                postings = self.index.setdefault(token, {})
                entry = postings.setdefault(url, {"freq": 0, "positions": []})
                entry["freq"] += 1
                entry["positions"].append(position)
        LOGGER.info(
            "Indexed %d pages, %d unique words",
            len(self.doc_lengths),
            len(self.index),
        )

    def save(
        self, path: Union[str, Path], fmt: Optional[str] = None
    ) -> None:
        """Serialise :attr:`index` and :attr:`doc_lengths` to ``path``.

        JSON is human-readable (``indent=2``, ``sort_keys=True``); pickle
        is binary and faster to load.

        Parameters
        ----------
        path:
            Destination file. Parent directory is created if missing.
        fmt:
            ``"json"`` or ``"pickle"``. If omitted, inferred from the path
            extension via :func:`_format_from_path`.

        Raises
        ------
        ValueError
            If ``fmt`` is not in :data:`SUPPORTED_FORMATS`.
        """
        if fmt is None:
            fmt = _format_from_path(path)
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(
                f"unsupported format: {fmt!r}; allowed: {SUPPORTED_FORMATS}"
            )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_FORMAT_VERSION,
            "index": self.index,
            "doc_lengths": self.doc_lengths,
        }
        if fmt == "json":
            with target.open("w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
        else:
            with target.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info(
            "Wrote index (%d words, %s) to %s", len(self.index), fmt, target
        )

    def load(
        self, path: Union[str, Path], fmt: Optional[str] = None
    ) -> None:
        """Read a previously-saved index from ``path`` into :attr:`index`.

        Parameters
        ----------
        path:
            Source file produced by :meth:`save`.
        fmt:
            Storage format; if omitted, inferred from path extension
            (symmetric with :meth:`save`).

        Raises
        ------
        ValueError
            If ``fmt`` is unsupported, the file is in a pre-versioned
            format, or the version is not understood.
        FileNotFoundError
            If ``path`` does not exist.
        json.JSONDecodeError
            If a JSON file is corrupt.
        pickle.UnpicklingError
            If a pickle file is corrupt or fmt mismatch.
        """
        if fmt is None:
            fmt = _format_from_path(path)
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(
                f"unsupported format: {fmt!r}; allowed: {SUPPORTED_FORMATS}"
            )
        source = Path(path)
        if fmt == "json":
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            with source.open("rb") as handle:
                payload = pickle.load(handle)
        if not isinstance(payload, dict) or "version" not in payload:
            raise ValueError(
                f"{source} is in pre-3.2 index format (no version envelope). "
                "Rebuild with `build` to upgrade."
            )
        version = payload["version"]
        if version != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"unsupported index version {version!r} in {source}; "
                f"this build expects version {INDEX_FORMAT_VERSION}"
            )
        self.index = payload["index"]
        self.doc_lengths = payload["doc_lengths"]
        LOGGER.info("Loaded %d words from %s (%s)", len(self.index), source, fmt)

    def get_postings(self, word: str) -> dict:
        """Return the postings dict for ``word`` (empty dict if absent).

        The query is run through :func:`tokenise` so the lookup uses the
        same case-fold/stopword/stem rules as the index.

        Parameters
        ----------
        word:
            Free-form text. Multi-word inputs use only the first stem.

        Returns
        -------
        dict
            ``{url: {"freq": int, "positions": [int, ...]}}`` for the
            word's stem. Empty dict if the word tokenises to nothing
            or is absent.
        """
        tokens = tokenise(word)
        if not tokens:
            return {}
        return self.index.get(tokens[0], {})
