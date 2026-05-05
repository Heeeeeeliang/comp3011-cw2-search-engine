"""Tests for the SearchEngine module.

Strategy
--------
Every test runs against a small in-memory Indexer built from hand-crafted
HTML. No real crawl cache, no network. The fixture lives at module
level (under ``conftest`` would be overkill) so the corpus content is
visible right next to the assertions that depend on it.
"""

from __future__ import annotations

import pytest

from src.indexer import Indexer
from src.search import SearchEngine


# --------------------------------------------------------------------- fixtures


# Hand-crafted four-page corpus. Word distribution is deliberate so each
# test can be reasoned about by reading this dict and the assertion side
# by side.
#
#   p1: the quick brown fox
#   p2: the lazy dog and the quick fox
#   p3: don't trust everything you see
#   p4: quick wisdom about life
#
# Word -> URLs that contain it:
#   "quick"  -> p1, p2, p4
#   "fox"    -> p1, p2
#   "the"    -> p1, p2
#   "brown"  -> p1
#   "lazy"   -> p2
#   "don't"  -> p3
#   "wisdom" -> p4
_PAGES: dict[str, str] = {
    "https://example.com/p1": (
        "<html><body><p>The quick brown fox</p></body></html>"
    ),
    "https://example.com/p2": (
        "<html><body><p>the lazy dog and the quick fox</p></body></html>"
    ),
    "https://example.com/p3": (
        "<html><body><p>Don't trust everything you see</p></body></html>"
    ),
    "https://example.com/p4": (
        "<html><body><p>quick wisdom about life</p></body></html>"
    ),
}


@pytest.fixture
def engine() -> SearchEngine:
    """A SearchEngine over the four-page corpus above."""
    indexer = Indexer()
    indexer.build(list(_PAGES.items()))
    return SearchEngine(indexer)


# ------------------------------------------------------------ TestSearchEngineInit


class TestSearchEngineInit:
    """Constructor wires up the indexer reference."""

    def test_holds_reference_to_indexer(self) -> None:
        idx = Indexer()
        engine = SearchEngine(idx)
        assert engine.indexer is idx

    def test_reflects_live_index_updates(self) -> None:
        # The engine must hold a *reference*, not a snapshot — when the
        # CLI rebuilds in place, search should see the new state.
        idx = Indexer()
        engine = SearchEngine(idx)
        idx.build([("u", "<p>hello</p>")])
        assert engine.find("hello") == ["u"]


# ------------------------------------------------------------------ TestPrintWord


class TestPrintWord:
    """The single-word lookup the CLI's ``print`` command uses."""

    def test_known_word_returns_postings(self, engine: SearchEngine) -> None:
        result = engine.print_word("brown")
        assert "https://example.com/p1" in result
        assert result["https://example.com/p1"]["freq"] == 1

    def test_unknown_word_returns_empty_dict(
        self, engine: SearchEngine
    ) -> None:
        assert engine.print_word("nonexistent") == {}

    def test_query_is_case_insensitive(self, engine: SearchEngine) -> None:
        assert engine.print_word("BROWN") == engine.print_word("brown")
        assert engine.print_word("Brown") == engine.print_word("brown")

    def test_returns_postings_for_word_in_multiple_pages(
        self, engine: SearchEngine
    ) -> None:
        result = engine.print_word("fox")
        assert set(result.keys()) == {
            "https://example.com/p1",
            "https://example.com/p2",
        }


# -------------------------------------------------------------- TestFindSingleWord


class TestFindSingleWord:
    """One-word queries: simple postings projection."""

    def test_match_returns_all_pages_containing_word(
        self, engine: SearchEngine
    ) -> None:
        assert engine.find("quick") == [
            "https://example.com/p1",
            "https://example.com/p2",
            "https://example.com/p4",
        ]

    def test_no_match_returns_empty_list(
        self, engine: SearchEngine
    ) -> None:
        assert engine.find("absent") == []

    def test_result_is_sorted_alphabetically(
        self, engine: SearchEngine
    ) -> None:
        result = engine.find("fox")
        assert result == sorted(result)


# --------------------------------------------------------------- TestFindMultiWord


class TestFindMultiWord:
    """Multi-word queries are AND, implemented as set intersection."""

    def test_intersection_of_two_words(self, engine: SearchEngine) -> None:
        # "quick" -> p1, p2, p4 ; "fox" -> p1, p2 ; AND -> p1, p2
        assert engine.find("quick fox") == [
            "https://example.com/p1",
            "https://example.com/p2",
        ]

    def test_intersection_of_three_words(self, engine: SearchEngine) -> None:
        # "the" is a stopword and is dropped at query-time, so the
        # effective query is ("quick", "fox") -> p1, p2. The test still
        # exercises a multi-token query path; what changed in 3.1 is
        # that stopwords no longer contribute to the intersection.
        assert engine.find("the quick fox") == [
            "https://example.com/p1",
            "https://example.com/p2",
        ]

    def test_intersection_can_collapse_to_single_page(
        self, engine: SearchEngine
    ) -> None:
        # "brown" only in p1 ; "quick" in p1,p2,p4 ; AND -> just p1
        assert engine.find("brown quick") == ["https://example.com/p1"]

    def test_unknown_word_makes_whole_query_empty(
        self, engine: SearchEngine
    ) -> None:
        # Even though "quick" matches three pages, "absent" matches none,
        # so the AND collapses to [].
        assert engine.find("quick absent") == []

    def test_query_is_case_insensitive(self, engine: SearchEngine) -> None:
        assert engine.find("QUICK Fox") == engine.find("quick fox")

    def test_punctuation_in_query_does_not_break_match(
        self, engine: SearchEngine
    ) -> None:
        # "quick, fox!" tokenises to ["quick", "fox"] same as the bare query.
        assert engine.find("quick, fox!") == engine.find("quick fox")


# --------------------------------------------------------------- TestFindEmptyQuery


class TestFindEmptyQuery:
    """Empty / whitespace / pure-punctuation queries return [] gracefully."""

    def test_empty_string_returns_empty_list(
        self, engine: SearchEngine
    ) -> None:
        assert engine.find("") == []

    def test_whitespace_only_returns_empty_list(
        self, engine: SearchEngine
    ) -> None:
        assert engine.find("   \t\n  ") == []

    def test_pure_punctuation_returns_empty_list(
        self, engine: SearchEngine
    ) -> None:
        # "!!! ??? ..." has no [a-z0-9'] runs, tokenises to [].
        assert engine.find("!!! ??? ...") == []


# -------------------------------------------------- TestQueryTokenisationConsistency


class TestQueryTokenisationConsistency:
    """Index-time and query-time tokenisation must produce the same tokens."""

    def test_contraction_query_matches_indexed_contraction(
        self, engine: SearchEngine
    ) -> None:
        # p3 contains "Don't" (capitalised). The indexer stores "don't".
        # A query for "Don't" must lower-case to "don't" and hit p3.
        assert engine.find("Don't") == ["https://example.com/p3"]

    def test_all_caps_query_normalises(self, engine: SearchEngine) -> None:
        assert engine.find("DON'T") == ["https://example.com/p3"]

    def test_mixed_query_with_contractions_and_words(
        self, engine: SearchEngine
    ) -> None:
        # Both terms are in p3.
        assert engine.find("don't trust") == ["https://example.com/p3"]
