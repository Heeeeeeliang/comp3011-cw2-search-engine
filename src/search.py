"""
Search module for COMP3011 Coursework 2.

Implements the two query commands the CLI exposes:

* ``print <word>`` — return the inverted-index entry for a single word.
* ``find <w1> <w2> ...`` — return URLs whose page contains *all* the
  given words (AND query, case-insensitive).

Design choices
--------------
* **Shared tokeniser with the indexer.** ``tokenise`` is imported from
  ``src.indexer`` so a query is normalised the same way pages were at
  build time. Without that symmetry, an upper-case query or a contraction
  silently misses matches that should hit. Day 3 will hang stopword
  removal and Porter stemming off this same function, and search
  inherits the change for free.
* **Set intersection over per-word posting URL-sets.** Cleaner than
  iterating one set and probing the others, and naturally handles the
  unknown-word case: an empty set in the list collapses the whole
  intersection to ``{}`` without a special branch.
* **Alphabetical sort, not insertion order.** TF-IDF ranking arrives in
  Day 3; until then a deterministic alphabetical order is the right
  contract — the CLI's output is stable across runs and the test suite
  can assert exact lists.
* **``SearchEngine`` is a thin layer.** All the data lives on the
  Indexer; this class is essentially a query planner. Keeping it small
  makes the Day 3 additions (TF-IDF, phrase queries) easier to slot in
  without disturbing the indexer's contract.
"""

from __future__ import annotations

import logging

from src.indexer import Indexer, tokenise

LOGGER = logging.getLogger(__name__)


class SearchEngine:
    """Run ``print``/``find`` queries against a populated :class:`Indexer`.

    Parameters
    ----------
    indexer:
        A built or loaded :class:`Indexer`. The engine holds a reference
        rather than a copy, so live updates to the index are visible
        without re-binding.
    """

    def __init__(self, indexer: Indexer) -> None:
        self.indexer: Indexer = indexer

    def print_word(self, word: str) -> dict:
        """Return the postings for a single word (empty dict if absent).

        Delegates to :meth:`Indexer.get_postings`, which already handles
        the lower-case fold. Kept on this class (rather than calling the
        indexer directly from the CLI) so all read-side concerns live in
        one place — useful when Day 3 adds TF-IDF metadata to the
        formatted output.
        """
        return self.indexer.get_postings(word)

    def find(self, query: str) -> list[str]:
        """Return URLs containing **every** word in ``query``.

        Parameters
        ----------
        query:
            Free-form user input. Tokenised with the same rules used at
            index time, so case, punctuation and contractions are
            normalised symmetrically.

        Returns
        -------
        list[str]
            URLs sorted alphabetically. Empty list when the query has no
            tokens or when at least one query word is absent from the
            index.
        """
        tokens = tokenise(query)
        if not tokens:
            return []
        url_sets = [
            set(self.indexer.get_postings(token).keys()) for token in tokens
        ]
        matched = set.intersection(*url_sets)
        LOGGER.debug(
            "find(%r) -> %d tokens, %d matches", query, len(tokens), len(matched)
        )
        return sorted(matched)
