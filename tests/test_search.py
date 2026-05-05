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
from src.search import SearchEngine, _parse_query


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


# ------------------------------------------------------------------ TestQueryParser


class TestQueryParser:
    """Behaviour of the module-level _parse_query helper.

    The parser turns a raw query string into a list of "atoms": single
    words (str) and phrases (tuple of words). Tokenisation (lower /
    stopwords / stemming) happens later, in find_with_scores; the parser
    is a pure syntactic split.
    """

    def test_empty_query_returns_empty_list(self) -> None:
        assert _parse_query("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert _parse_query("   \t\n  ") == []

    def test_single_word_returns_str(self) -> None:
        assert _parse_query("wisdom") == ["wisdom"]

    def test_multiple_unquoted_words_return_strs(self) -> None:
        assert _parse_query("good friends wisdom") == [
            "good",
            "friends",
            "wisdom",
        ]

    def test_single_phrase_returns_tuple(self) -> None:
        assert _parse_query('"good friends"') == [("good", "friends")]

    def test_phrase_followed_by_word(self) -> None:
        assert _parse_query('"good friends" wisdom') == [
            ("good", "friends"),
            "wisdom",
        ]

    def test_word_followed_by_phrase(self) -> None:
        assert _parse_query('wisdom "good friends"') == [
            "wisdom",
            ("good", "friends"),
        ]

    def test_multiple_phrases(self) -> None:
        assert _parse_query('"good friends" "lazy dog"') == [
            ("good", "friends"),
            ("lazy", "dog"),
        ]

    def test_quoted_single_word_collapses_to_str(self) -> None:
        # `"wisdom"` is syntactically a phrase but has nothing to
        # phrase-match on. The parser normalises it to a plain str so
        # that find('"wisdom"') and find('wisdom') produce identical
        # parser output (and TestPhraseQueries asserts the find()-level
        # equivalence the spec requires).
        assert _parse_query('"wisdom"') == ["wisdom"]

    def test_apostrophe_in_word_does_not_choke(self) -> None:
        # shlex.split(query, posix=True) -- the spec's first reach --
        # would treat the apostrophe in "don't" as an unbalanced
        # single-quote and raise ValueError. The customised lexer
        # (quotes='"') preserves the contraction unchanged so Day 3.1's
        # contraction-symmetry tests do not regress.
        assert _parse_query("don't trust") == ["don't", "trust"]

    def test_empty_quotes_dropped(self) -> None:
        assert _parse_query('""') == []

    def test_empty_quotes_with_word_drops_only_the_empty(self) -> None:
        assert _parse_query('"" wisdom') == ["wisdom"]

    def test_adjacent_quoted_runs_concatenate(self) -> None:
        # POSIX shell behaviour: `"a "b" c"` is one token because
        # adjacent quoted/unquoted runs concatenate. Once that single
        # token "a b c" is split on whitespace it has 3 words, so the
        # parser emits a tuple of 3. Asserted here directly so a future
        # shlex change is visible rather than silent.
        assert _parse_query('"a "b" c"') == [("a", "b", "c")]

    def test_unbalanced_quote_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _parse_query('"good friends')


# ------------------------------------------------------------- TestPhraseQueries


class TestPhraseQueries:
    """Phrase queries match consecutive positions in the post-filter stream.

    Position semantics inherit from Day 3.1: positions are stamped on
    the **filtered** token list, so a phrase query implicitly skips
    over stopwords in the source HTML. The tests below are calibrated
    against that contract.
    """

    def test_phrase_matches_consecutive_words(self) -> None:
        # "we are good friends" -> filtered tokens ["good", "friend"]
        # at positions 0, 1 (stopwords "we"/"are" drop before
        # positions are assigned). Phrase ("good", "friend") matches.
        idx = Indexer()
        idx.build([
            ("https://example.com/p1", _wrap("we are good friends")),
        ])
        engine = SearchEngine(idx)
        assert engine.find('"good friends"') == ["https://example.com/p1"]

    def test_phrase_does_not_match_non_consecutive_words(self) -> None:
        # "good people who are friends": after stopword removal/stem,
        # tokens are ["good", "peopl", "who", "friend"]. ("who" is NOT
        # in the curated stopword list -- only the personal pronouns
        # i/you/he/she/it/we/they are.) Phrase ("good", "friend")
        # therefore needs positions 0, 1 -- but "friend" is at 3, not
        # 1. No match.
        idx = Indexer()
        idx.build([
            ("https://example.com/p", _wrap("good people who are friends")),
        ])
        engine = SearchEngine(idx)
        assert engine.find('"good friends"') == []

    def test_phrase_and_word_query_intersects(self) -> None:
        # Mixed query: phrase + bare word, AND-intersected.
        #
        # p1: "good friends share wisdom"
        #     phrase OK (good@0, friend@1), wisdom present  -> match
        # p2: "good friends together always"
        #     phrase OK, no "wisdom"                         -> drop
        # p3: "good people share wisdom about friends"
        #     phrase fails (good@0 friend@5, not adjacent),
        #     wisdom present, but AND fails                  -> drop
        # p4: "good wisdom only here"
        #     no "friends" at all -> phrase candidate set    -> drop
        idx = Indexer()
        idx.build([
            ("https://example.com/p1", _wrap("good friends share wisdom")),
            ("https://example.com/p2", _wrap("good friends together always")),
            ("https://example.com/p3",
             _wrap("good people share wisdom about friends")),
            ("https://example.com/p4", _wrap("good wisdom only here")),
        ])
        engine = SearchEngine(idx)
        assert engine.find('"good friends" wisdom') == [
            "https://example.com/p1",
        ]

    def test_phrase_across_stopwords_works_correctly(self) -> None:
        """The likely-bug spot: position-after-filter has consequences.

        Page A text: "the good and friends here"
            Filtered tokens: ["good", "friend", "here"]  (the/and dropped)
            Positions:       [0,      1,        2]
            Phrase ("good", "friend") needs (p, p+1) -> 0, 1: HIT.

        Page B text: "the good wonderful friends here"
            "wonderful" is NOT a stopword.
            Filtered tokens: ["good", "wonder", "friend", "here"]
            Positions:       [0,      1,        2,        3]
            Phrase needs (p, p+1): 0, 1 -> friend? no, that's "wonder".
                                   No other "good". MISS.

        Page B is the canary -- dropping a stopword "completes" a
        phrase, but inserting a content word breaks it. This is the
        contract Day 3.1 set up; the test makes it observable.
        """
        idx = Indexer()
        idx.build([
            ("https://example.com/A", _wrap("the good and friends here")),
            ("https://example.com/B",
             _wrap("the good wonderful friends here")),
        ])
        engine = SearchEngine(idx)
        assert engine.find('"good friends"') == ["https://example.com/A"]

    def test_single_word_phrase_equivalent_to_unquoted(self) -> None:
        # Spec-required equivalence: `find "wisdom"` must behave
        # identically to `find wisdom`. Verified at both API levels.
        idx = Indexer()
        idx.build([
            ("https://example.com/p1", _wrap("alpha wisdom beta")),
            ("https://example.com/p2", _wrap("wisdom is everywhere")),
            ("https://example.com/p3", _wrap("nothing useful here")),
        ])
        engine = SearchEngine(idx)
        assert engine.find('"wisdom"') == engine.find("wisdom")
        assert (
            engine.find_with_scores('"wisdom"')
            == engine.find_with_scores("wisdom")
        )

    def test_phrase_query_with_stemming(self) -> None:
        # Plural query, singular page. Both stem to "friend"; phrase
        # ("good", "friend") matches against ["good", "friend", ...].
        # The stemming-symmetry tokenise() provides covers phrase
        # queries by construction -- but worth locking in explicitly
        # because phrase + stemming together is the subtle bit.
        idx = Indexer()
        idx.build([
            ("https://example.com/p", _wrap("good friend always")),
        ])
        engine = SearchEngine(idx)
        assert engine.find('"good friends"') == ["https://example.com/p"]

    def test_unbalanced_quote_in_find_returns_empty_list(self) -> None:
        # Graceful degradation: parser raises ValueError, find catches
        # and returns []. A user typo should not crash the REPL.
        idx = Indexer()
        idx.build([("https://example.com/p", _wrap("good friends"))])
        engine = SearchEngine(idx)
        assert engine.find('"good friends') == []
        assert engine.find_with_scores('"good friends') == []

    def test_phrase_matches_at_later_occurrence_of_first_term(self) -> None:
        # "good news good friends here" -> positions:
        # good=0, news/new=1, good=2, friend=3, here=4
        # start=0: friend at 1? no. start=2: friend at 3? YES.
        # Locks in the for-each-start iteration in _phrase_matches.
        idx = Indexer()
        idx.build([
            ("https://example.com/p", _wrap("good news good friends here")),
        ])
        engine = SearchEngine(idx)
        assert engine.find('"good friends"') == ["https://example.com/p"]

    def test_phrase_word_unknown_returns_empty(self) -> None:
        # AND-intersection prefilter sees an empty postings set for
        # the unknown stem and short-circuits to no candidates.
        idx = Indexer()
        idx.build([("https://example.com/p", _wrap("good friends here"))])
        engine = SearchEngine(idx)
        assert engine.find('"good unknownword"') == []

    def test_unquoted_intra_word_punctuation_splits_to_AND(self) -> None:
        # A single shlex token "quick,fox" tokenises to ["quick", "fox"]
        # via TOKEN_RE. The find() loop must add each as a separate
        # AND term -- this matches pre-3.3 behaviour where the whole
        # query was tokenised at once.
        idx = Indexer()
        idx.build([
            ("https://example.com/p1", _wrap("quick brown fox")),
            ("https://example.com/p2", _wrap("quick only")),
            ("https://example.com/p3", _wrap("fox only")),
        ])
        engine = SearchEngine(idx)
        assert engine.find("quick,fox") == ["https://example.com/p1"]

    def test_pure_stopword_phrase_returns_empty(self) -> None:
        # `"the and a"` -> tokenise -> [] -> phrase part skipped ->
        # url_sets empty -> return [].
        idx = Indexer()
        idx.build([("https://example.com/p", _wrap("good friends"))])
        engine = SearchEngine(idx)
        assert engine.find('"the and a"') == []

    def test_phrase_collapses_when_only_one_content_term_remains(self) -> None:
        # `"good and"` -> tokenise -> ["good"] (len 1). Collapses to a
        # plain single-term postings lookup; phrase matching never
        # runs. The page just needs to contain "good" somewhere.
        idx = Indexer()
        idx.build([
            ("https://example.com/p1", _wrap("good things happen")),
            ("https://example.com/p2", _wrap("nothing here")),
        ])
        engine = SearchEngine(idx)
        assert engine.find('"good and"') == ["https://example.com/p1"]

    def test_pure_stopword_unquoted_query_returns_empty(self) -> None:
        # Unquoted variant of test_pure_stopword_phrase_returns_empty.
        # `find("the and a")` -> tokenise each part -> all empty ->
        # url_sets empty -> [].
        idx = Indexer()
        idx.build([("https://example.com/p", _wrap("good friends"))])
        engine = SearchEngine(idx)
        assert engine.find("the and a") == []
        assert engine.find_with_scores("the and a") == []

    def test_mixed_case_phrase_normalises(self) -> None:
        # Spec-required edge case: `find "Good Friends"` and similar
        # case variants must produce identical results to the
        # lowercase form. Tokenise's lower() inside _parse_query
        # handles this -- but the test pins the contract.
        idx = Indexer()
        idx.build([
            ("https://example.com/p", _wrap("good friends here together")),
        ])
        engine = SearchEngine(idx)
        baseline = engine.find('"good friends"')
        assert engine.find('"Good Friends"') == baseline
        assert engine.find('"GOOD FRIENDS"') == baseline
        assert engine.find('"gOoD fRiEnDs"') == baseline
        # And the baseline is non-empty so we're asserting on a real
        # match, not just "all variants return []".
        assert baseline == ["https://example.com/p"]

    def test_phrase_outranks_non_phrase_in_combined_query(self) -> None:
        # Sanity check on phrase-aware ranking: among AND-matching
        # docs, the phrase pre-filter has already trimmed the
        # candidate set, and constituent-sum TF-IDF ranks them.
        # p1: "good friends" adjacent, "wisdom" present, len 4
        # p2: "good friends" adjacent, "wisdom" present, len 6
        # Both pass phrase + AND. Shorter doc has higher tf -> ranks first.
        idx = Indexer()
        idx.build([
            ("https://example.com/p1", _wrap("good friends share wisdom")),
            ("https://example.com/p2",
             _wrap("good friends together share quiet wisdom")),
        ])
        engine = SearchEngine(idx)
        ranked = engine.find('"good friends" wisdom')
        assert ranked == [
            "https://example.com/p1",
            "https://example.com/p2",
        ]
