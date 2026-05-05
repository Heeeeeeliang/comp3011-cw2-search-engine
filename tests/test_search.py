"""Tests for the SearchEngine module.

Strategy
--------
Every test runs against a small in-memory Indexer built from hand-crafted
HTML. No real crawl cache, no network. The fixture lives at module
level (under ``conftest`` would be overkill) so the corpus content is
visible right next to the assertions that depend on it.
"""

from __future__ import annotations

import math

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

    def test_result_orders_higher_tf_first(
        self, engine: SearchEngine
    ) -> None:
        # "fox" is in p1 (length 3) and p2 (length 4); both have freq 1.
        # tf(p1) = 1/3 > tf(p2) = 1/4 with identical idf, so p1 ranks
        # ahead of p2 even though p2 < p1 alphabetically would not
        # apply here. (Pre-3.2 this assertion was `result ==
        # sorted(result)` because the contract was alphabetical; that
        # only still passes here by coincidence.)
        assert engine.find("fox") == [
            "https://example.com/p1",
            "https://example.com/p2",
        ]


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


# -------------------------------------------------------------------- TestTFIDF


def _wrap(body: str) -> str:
    """Tiny HTML wrapper so the corpus literals stay readable."""
    return f"<html><body><p>{body}</p></body></html>"


class TestTFIDF:
    """Score formula and ranking behaviour."""

    def test_higher_freq_ranks_first(self) -> None:
        # All three "obscure"-bearing pages share idf = log(4/3); the
        # ranking is therefore driven entirely by tf, and the page that
        # is *all* "obscure" wins. Page p4 has no "obscure" and must
        # not appear in the result at all.
        idx = Indexer()
        idx.build([
            ("https://example.com/p1", _wrap("obscure word")),           # tf=1/2
            ("https://example.com/p2", _wrap("obscure other text")),     # tf=1/3
            ("https://example.com/p3", _wrap("obscure obscure obscure obscure obscure")),  # tf=1
            ("https://example.com/p4", _wrap("nothing here")),           # tf=0
        ])
        engine = SearchEngine(idx)
        assert engine.find("obscure") == [
            "https://example.com/p3",
            "https://example.com/p1",
            "https://example.com/p2",
        ]
        assert "https://example.com/p4" not in engine.find("obscure")

    def test_score_zero_for_term_in_every_doc(self) -> None:
        # df == N => idf = log(1) = 0 => score is 0 for every match.
        # The pages still appear in find_with_scores (the AND
        # intersection is non-empty), they just all carry score 0 and
        # fall back to the alphabetical tie-break.
        idx = Indexer()
        idx.build([
            ("https://example.com/a", _wrap("everywhere alpha")),
            ("https://example.com/b", _wrap("everywhere beta")),
            ("https://example.com/c", _wrap("everywhere gamma")),
        ])
        engine = SearchEngine(idx)
        scored = engine.find_with_scores("everywhere")
        assert [s for _, s in scored] == [0.0, 0.0, 0.0]
        assert [u for u, _ in scored] == [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        ]

    def test_single_word_score_matches_formula(self) -> None:
        # Hand-computed expectation locks the formula. Page "a" has
        # length 2 (alpha + beta), df(beta) = 1, N = 2.
        # tf(beta, a) = 1/2; idf(beta) = log(2/1) = log(2).
        # score = (1/2) * log(2).
        idx = Indexer()
        idx.build([
            ("a", _wrap("alpha beta")),
            ("b", _wrap("alpha gamma")),
        ])
        engine = SearchEngine(idx)
        scored = engine.find_with_scores("beta")
        assert scored == [("a", pytest.approx(math.log(2) / 2))]

    def test_multi_word_score_is_sum_of_term_scores(self) -> None:
        # Two-term query: "beta gamma". Page "a" has length 2, "beta"
        # freq 1; page "b" has length 2, "gamma" freq 1. Neither page
        # has BOTH terms, so AND-intersection is empty. We need a page
        # with BOTH to verify the sum.
        idx = Indexer()
        idx.build([
            ("a", _wrap("alpha beta gamma")),   # has beta + gamma; len 3
            ("b", _wrap("alpha beta delta")),   # has beta only;    len 3
            ("c", _wrap("alpha gamma epsilon")),# has gamma only;   len 3
        ])
        engine = SearchEngine(idx)
        scored = engine.find_with_scores("beta gamma")
        # Only page "a" has both. df(beta)=2, df(gamma)=2, N=3.
        # tf(beta, a)  = 1/3 ; idf(beta)  = log(3/2)
        # tf(gamma, a) = 1/3 ; idf(gamma) = log(3/2)
        # score = (1/3 + 1/3) * log(3/2) = (2/3) * log(3/2)
        expected = (2 / 3) * math.log(3 / 2)
        assert scored == [("a", pytest.approx(expected))]

    def test_query_token_missing_from_index_yields_empty(self) -> None:
        # Even a high-tf match on the other term must collapse: AND
        # semantics mean the unknown term breaks the intersection.
        idx = Indexer()
        idx.build([
            ("a", _wrap("alpha alpha alpha")),
            ("b", _wrap("alpha beta")),
        ])
        engine = SearchEngine(idx)
        assert engine.find("alpha unknownword") == []
        assert engine.find_with_scores("alpha unknownword") == []

    def test_pure_stopword_query_returns_empty(self) -> None:
        # "the" alone tokenises to [], so find_with_scores never gets
        # to the postings layer.
        idx = Indexer()
        idx.build([("a", _wrap("alpha"))])
        engine = SearchEngine(idx)
        assert engine.find_with_scores("the") == []
        assert engine.find("the") == []


# -------------------------------------------------------- TestRankingDeterminism


class TestRankingDeterminism:
    """Repeated queries return identical ordering; ties break by URL."""

    def test_repeated_query_returns_identical_ordering(self) -> None:
        # Set iteration order is stable within a CPython run, but it
        # is not stable across hash seeds in general. Running the same
        # query twice and asserting equality is a cheap regression
        # check that the explicit sort is in place. The 4th page
        # carries no "obscure" so df < N and idf is non-zero — without
        # it, every score collapses to 0 and "highest-tf wins" reduces
        # to alphabetical, which would mask a sort regression.
        idx = Indexer()
        idx.build([
            ("https://example.com/p1", _wrap("obscure word")),
            ("https://example.com/p2", _wrap("obscure other text")),
            ("https://example.com/p3", _wrap("obscure obscure obscure")),
            ("https://example.com/p4", _wrap("unrelated padding")),
        ])
        engine = SearchEngine(idx)
        first = engine.find("obscure")
        second = engine.find("obscure")
        assert first == second
        # And explicitly: highest-tf page first, then by tf desc.
        assert first[0] == "https://example.com/p3"

    def test_tied_scores_break_alphabetically(self) -> None:
        # Identical content => identical doc length, term freq, df,
        # idf — same score on every term. URLs are "zebra" and
        # "apple" so alphabetical tie-break must put apple first.
        idx = Indexer()
        idx.build([
            ("https://example.com/zebra", _wrap("topic content here")),
            ("https://example.com/apple", _wrap("topic content here")),
        ])
        engine = SearchEngine(idx)
        assert engine.find("topic") == [
            "https://example.com/apple",
            "https://example.com/zebra",
        ]

    def test_tied_zero_scores_still_break_alphabetically(self) -> None:
        # Tie-break must work even when every score is zero (term in
        # every document). This is the case where order would be most
        # tempting to leave to ``set`` iteration order.
        idx = Indexer()
        idx.build([
            ("https://example.com/charlie", _wrap("everywhere x")),
            ("https://example.com/alpha", _wrap("everywhere y")),
            ("https://example.com/bravo", _wrap("everywhere z")),
        ])
        engine = SearchEngine(idx)
        assert engine.find("everywhere") == [
            "https://example.com/alpha",
            "https://example.com/bravo",
            "https://example.com/charlie",
        ]


# -------------------------------------------------------------- TestFindWithScores


class TestFindWithScores:
    """Behaviour of the (url, score) projection used by tests + future print."""

    def test_returns_empty_list_for_no_match(self) -> None:
        idx = Indexer()
        idx.build([("a", _wrap("alpha"))])
        engine = SearchEngine(idx)
        assert engine.find_with_scores("missing") == []

    def test_returns_empty_list_for_empty_query(self) -> None:
        idx = Indexer()
        idx.build([("a", _wrap("alpha"))])
        engine = SearchEngine(idx)
        assert engine.find_with_scores("") == []

    def test_score_floats_are_finite(self) -> None:
        # Catch nan/inf regressions early — log(0) or division by 0
        # would surface here as math.isfinite() == False.
        idx = Indexer()
        idx.build([
            ("a", _wrap("alpha beta")),
            ("b", _wrap("beta gamma")),
        ])
        engine = SearchEngine(idx)
        for _, score in engine.find_with_scores("beta"):
            assert math.isfinite(score)

    def test_find_and_find_with_scores_agree_on_ordering(self) -> None:
        # find() must be a strict projection of find_with_scores —
        # never a separately-computed ranking.
        idx = Indexer()
        idx.build([
            ("a", _wrap("alpha beta")),
            ("b", _wrap("alpha beta beta")),
            ("c", _wrap("alpha")),
        ])
        engine = SearchEngine(idx)
        urls_via_find = engine.find("alpha")
        urls_via_scores = [u for u, _ in engine.find_with_scores("alpha")]
        assert urls_via_find == urls_via_scores
