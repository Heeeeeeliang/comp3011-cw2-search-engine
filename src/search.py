"""
Search module for COMP3011 Coursework 2.

Implements the two query commands:
    - print <word>           -> show the inverted-index entry for one word
    - find <w1> <w2> ...     -> list URLs containing ALL given words (AND query)

Design notes:
    - Multi-word queries use set intersection of per-word URL sets.
    - Empty query / unknown word is handled gracefully (returns []).
"""


class SearchEngine:
    """Run print/find queries against a populated Indexer."""

    def __init__(self, indexer) -> None:
        self.indexer = indexer

    def print_word(self, word: str) -> dict:
        """Return the postings for a single word (caller decides how to format)."""
        raise NotImplementedError

    def find(self, query: str) -> list[str]:
        """Return URLs containing every word in the query (case-insensitive)."""
        raise NotImplementedError
