"""
Search module for COMP3011 Coursework 2.

Implements the two query commands the CLI exposes:

* ``print <word>`` — return the inverted-index entry for a single word.
* ``find <w1> <w2> ...`` — return URLs containing *all* words (AND query),
  ranked by TF-IDF. Double-quoted runs are phrase queries that require
  positional adjacency in the post-filter token stream.

The tokeniser is shared with :mod:`src.indexer` so query-time and
index-time normalisation match.
"""

from __future__ import annotations

import logging
import math
import shlex

from src.indexer import Indexer, tokenise

LOGGER = logging.getLogger(__name__)


def _parse_query(query: str) -> list[str | tuple[str, ...]]:
    """Split a raw query into single-word and phrase parts.

    Double-quoted substrings become phrase tuples; bare words become strings.
    A single quoted word collapses to a string. Single-quote characters are
    not phrase delimiters, so contractions like ``don't`` survive intact.

    Parameters
    ----------
    query:
        Free-form user input.

    Returns
    -------
    list[str | tuple[str, ...]]
        Empty quoted runs are dropped.

    Raises
    ------
    ValueError
        If the query contains an unbalanced ``"``.
    """
    lexer = shlex.shlex(query, posix=True)
    lexer.whitespace_split = True
    lexer.quotes = '"'
    raw_tokens = list(lexer)

    parts: list[str | tuple[str, ...]] = []
    for raw in raw_tokens:
        words = raw.split()
        if not words:
            continue
        if len(words) == 1:
            parts.append(words[0])
        else:
            parts.append(tuple(words))
    return parts


class SearchEngine:
    """Run ``print``/``find`` queries against a populated :class:`Indexer`.

    Holds a reference (not a copy), so live index updates are visible.
    """

    def __init__(self, indexer: Indexer) -> None:
        self.indexer: Indexer = indexer

    def print_word(self, word: str) -> dict:
        """Return the postings for a single word (empty dict if absent).

        Delegates to :meth:`Indexer.get_postings`. Multi-word input uses
        only the first stem.
        """
        return self.indexer.get_postings(word)

    def find(self, query: str) -> list[str]:
        """Return URLs containing **every** word in ``query``, TF-IDF ranked.

        Use :meth:`find_with_scores` for the formula and to retrieve raw
        scores. Double-quoted runs introduce phrase queries.
        """
        return [url for url, _ in self.find_with_scores(query)]

    def find_with_scores(self, query: str) -> list[tuple[str, float]]:
        """Return ``(url, tfidf_score)`` pairs for the AND query.

        Formula::

            tf(t, d) = freq[t][d] / doc_lengths[d]
            idf(t)   = log(N / df[t])
            score(d) = sum( tf(t, d) * idf(t)  for t in score_terms )

        Phrase parts contribute their constituent terms' tf*idf sum.
        Ties break alphabetically by URL for deterministic output.
        Unbalanced quotes are caught and yield an empty list.

        Returns
        -------
        list[tuple[str, float]]
            Sorted by score descending, URL ascending. Empty when the
            query has no tokens or no document satisfies the AND.
        """
        try:
            parts = _parse_query(query)
        except ValueError as exc:
            LOGGER.debug("find(%r): unbalanced quote -- %s", query, exc)
            return []
        if not parts:
            return []

        # One set per atom; final result is their intersection.
        url_sets: list[set[str]] = []
        score_terms: list[str] = []

        for part in parts:
            if isinstance(part, str):
                terms = tokenise(part)
                if not terms:
                    # Pure-stopword/punctuation token; drop silently.
                    continue
                # Intra-word punctuation can split one shlex token into multiple terms.
                for term in terms:
                    url_sets.append(set(self.indexer.index.get(term, {}).keys()))
                    score_terms.append(term)
            else:
                phrase_terms = tokenise(" ".join(part))
                if not phrase_terms:
                    continue
                if len(phrase_terms) == 1:
                    # Phrase reduced to a single term after stopword removal.
                    term = phrase_terms[0]
                    url_sets.append(set(self.indexer.index.get(term, {}).keys()))
                    score_terms.append(term)
                    continue
                # AND-intersect first, then check positional adjacency (avoids full corpus scan).
                phrase_tuple = tuple(phrase_terms)
                candidates = set.intersection(*[
                    set(self.indexer.index.get(t, {}).keys())
                    for t in phrase_terms
                ])
                phrase_hits = {
                    url for url in candidates
                    if self._phrase_matches(url, phrase_tuple)
                }
                url_sets.append(phrase_hits)
                score_terms.extend(phrase_terms)

        if not url_sets:
            # All parts dropped (all stopwords/punctuation).
            return []

        matched_urls = set.intersection(*url_sets)
        scored = [(url, self._score(url, score_terms)) for url in matched_urls]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        LOGGER.debug(
            "find(%r) -> %d parts, %d matches",
            query,
            len(parts),
            len(matched_urls),
        )
        return scored

    def _phrase_matches(self, doc: str, phrase: tuple[str, ...]) -> bool:
        """Return True iff ``doc`` contains ``phrase`` at consecutive positions.

        ``phrase`` is a tuple of pre-tokenised terms. Positions follow the
        indexer's post-filter convention: stopwords are removed before
        positions are stamped, so ``("good", "friend")`` matches a page
        whose source was ``"the good and friends"``.

        Caller contract: ``len(phrase) >= 2`` and ``doc`` is in every
        term's postings (guaranteed by :meth:`find_with_scores`).
        """
        starts = self.indexer.index[phrase[0]][doc]["positions"]
        follow_position_sets = [
            set(self.indexer.index[term][doc]["positions"])
            for term in phrase[1:]
        ]
        for start in starts:
            if all(
                start + i + 1 in follow_position_sets[i]
                for i in range(len(phrase) - 1)
            ):
                return True
        return False

    def _score(self, url: str, query_terms: list[str]) -> float:
        """TF-IDF sum for ``url`` over ``query_terms``.

        ``query_terms`` must be pre-tokenised. ``url`` must be in every
        term's postings (guaranteed by AND-intersection in
        :meth:`find_with_scores`).
        """
        # Standard TF-IDF: score(d) = sum( freq[t][d]/|d| * log(N/df[t]) ).
        doc_len = self.indexer.doc_lengths[url]
        n_docs = len(self.indexer.doc_lengths)
        score = 0.0
        for term in query_terms:
            postings = self.indexer.index[term]
            tf = postings[url]["freq"] / doc_len
            idf = math.log(n_docs / len(postings))
            score += tf * idf
        return score
