"""
Indexer module for COMP3011 Coursework 2.

Builds and persists an inverted index of the form:

    {
        word: {
            url: {"freq": int, "positions": [int, ...]},
            ...
        },
        ...
    }

Design notes:
    - Case-insensitive: all words stored lowercase.
    - Tokenisation via regex on [a-z0-9'] runs (keeps contractions like "don't").
    - Position is the 0-based index of the token within its page.
    - JSON storage so the file is human-readable (helpful for `print` debugging
      and for the markers to inspect the submitted index file).
"""

import json
import re

TOKEN_RE = re.compile(r"[a-z0-9']+")


class Indexer:
    """Build an inverted index from (url, html) pairs and save/load it."""

    def __init__(self) -> None:
        self.index: dict[str, dict[str, dict]] = {}

    def build(self, pages: list[tuple[str, str]]) -> None:
        """Populate self.index from raw HTML pages."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Serialise self.index to JSON on disk."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Read self.index back from a JSON file produced by save()."""
        raise NotImplementedError

    def get_postings(self, word: str) -> dict:
        """Return the postings dict for a single word (empty if not indexed)."""
        raise NotImplementedError
