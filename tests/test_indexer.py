"""Tests for the Indexer module.

Strategy
--------
Tests are entirely in-process with hand-crafted HTML strings; no crawl
cache or live network is touched, so the suite stays fast and
deterministic. Persistence tests use ``tmp_path`` to isolate writes.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import pytest

from src.indexer import (
    NON_CONTENT_TAGS,
    SUPPORTED_FORMATS,
    TOKEN_RE,
    Indexer,
    _strip_html,
    tokenise,
)


# --------------------------------------------------------------------- helpers


def _wrap(body: str) -> str:
    """Tiny HTML wrapper so tests don't need to type ``<html><body>...``."""
    return f"<html><body>{body}</body></html>"


# ---------------------------------------------------------------- module-level


class TestModuleConstants:
    """Sanity-check the module-level constants other code relies on."""

    def test_token_regex_matches_lowercase_and_digits_and_apostrophes(self) -> None:
        assert TOKEN_RE.findall("don't 123 ABC") == ["don't", "123"]

    def test_supported_formats_contains_json(self) -> None:
        assert "json" in SUPPORTED_FORMATS

    def test_non_content_tags_covers_script_and_style(self) -> None:
        assert "script" in NON_CONTENT_TAGS
        assert "style" in NON_CONTENT_TAGS


# -------------------------------------------------------------- TestTokenisation


class TestTokenisation:
    """Pure-text tokeniser behaviour, independent of HTML parsing."""

    def test_lowercases_input(self) -> None:
        assert tokenise("Hello WORLD") == ["hello", "world"]

    def test_preserves_apostrophes_in_contractions(self) -> None:
        assert tokenise("don't won't it's") == ["don't", "won't", "it's"]

    def test_extracts_alphanumeric_runs(self) -> None:
        assert tokenise("abc 123 mix4 7even") == ["abc", "123", "mix4", "7even"]

    def test_strips_punctuation(self) -> None:
        assert tokenise("hello, world! good?") == ["hello", "world", "good"]

    def test_empty_text_yields_empty_list(self) -> None:
        assert tokenise("") == []

    def test_whitespace_only_yields_empty_list(self) -> None:
        assert tokenise("   \n\t  ") == []

    def test_non_ascii_letters_dropped(self) -> None:
        # Documented limitation: TOKEN_RE matches only [a-z0-9']. The "é"
        # in "café" is not in that class, so it acts as a separator.
        assert tokenise("café naïve 你好 hello") == ["caf", "na", "ve", "hello"]

    def test_consecutive_apostrophes_kept_as_token(self) -> None:
        # Edge case: malformed input shouldn't crash.
        assert tokenise("''") == ["''"]


# ---------------------------------------------------------------- TestStripHtml


class TestStripHtml:
    """The HTML -> plain text helper that feeds tokenise()."""

    def test_extracts_visible_text(self) -> None:
        assert "hello" in _strip_html(_wrap("<p>hello</p>"))

    def test_decomposes_script_subtree(self) -> None:
        text = _strip_html(_wrap("<p>visible</p><script>alert(1)</script>"))
        assert "visible" in text
        assert "alert" not in text
        assert "1" not in text

    def test_decomposes_style_subtree(self) -> None:
        text = _strip_html(
            _wrap("<style>.x { color: red; }</style><p>seen</p>")
        )
        assert "seen" in text
        assert "color" not in text
        assert "red" not in text

    def test_separator_prevents_word_mash(self) -> None:
        text = _strip_html(_wrap("<p>foo</p><p>bar</p>"))
        # Adjacent block tags must not collapse into "foobar".
        assert "foo" in text
        assert "bar" in text
        assert "foobar" not in text

    def test_empty_html_yields_empty_text(self) -> None:
        assert _strip_html("").strip() == ""

    def test_attributes_not_extracted_as_text(self) -> None:
        # href values etc. must not leak into the token stream.
        text = _strip_html(_wrap('<a href="https://leaked.example">link</a>'))
        assert "link" in text
        assert "leaked" not in text


# ----------------------------------------------------------------- TestBuild


class TestBuild:
    """End-to-end indexing from HTML to inverted-index dict."""

    def test_single_page_basic_freq_and_positions(self) -> None:
        idx = Indexer()
        idx.build([("u", _wrap("<p>hello world hello</p>"))])
        assert idx.index["hello"]["u"] == {"freq": 2, "positions": [0, 2]}
        assert idx.index["world"]["u"] == {"freq": 1, "positions": [1]}

    def test_multiple_pages_share_word(self) -> None:
        idx = Indexer()
        idx.build(
            [
                ("a", _wrap("<p>foo bar</p>")),
                ("b", _wrap("<p>foo baz</p>")),
            ]
        )
        assert set(idx.index["foo"].keys()) == {"a", "b"}
        assert idx.index["foo"]["a"]["freq"] == 1
        assert idx.index["foo"]["b"]["freq"] == 1
        assert "bar" in idx.index and "baz" in idx.index

    def test_repeated_word_in_one_page_has_correct_positions(self) -> None:
        idx = Indexer()
        idx.build([("u", _wrap("<p>cat dog cat bird cat</p>"))])
        assert idx.index["cat"]["u"]["freq"] == 3
        assert idx.index["cat"]["u"]["positions"] == [0, 2, 4]

    def test_script_tag_excluded_from_index(self) -> None:
        idx = Indexer()
        idx.build(
            [("u", _wrap("<p>visible</p><script>alert(1)</script>"))]
        )
        assert "alert" not in idx.index
        assert "1" not in idx.index
        assert "visible" in idx.index

    def test_style_tag_excluded_from_index(self) -> None:
        idx = Indexer()
        idx.build(
            [("u", _wrap("<style>.x{color:red}</style><p>shown</p>"))]
        )
        assert "color" not in idx.index
        assert "red" not in idx.index
        assert "shown" in idx.index

    def test_mixed_case_normalised_to_lowercase(self) -> None:
        idx = Indexer()
        idx.build([("u", _wrap("<p>Hello HELLO hello</p>"))])
        assert idx.index["hello"]["u"]["freq"] == 3
        # No upper-case key should ever appear.
        assert all(key == key.lower() for key in idx.index)

    def test_empty_html_string_indexes_nothing(self) -> None:
        idx = Indexer()
        idx.build([("u", "")])
        assert idx.index == {}

    def test_html_with_no_body_text_indexes_nothing(self) -> None:
        # Tag soup with nothing tokenisable.
        idx = Indexer()
        idx.build([("u", "<html><head></head><body></body></html>")])
        assert idx.index == {}

    def test_empty_pages_iterable(self) -> None:
        idx = Indexer()
        idx.build([])
        assert idx.index == {}

    def test_build_accepts_generator(self) -> None:
        def _gen() -> object:
            yield ("u1", _wrap("<p>foo</p>"))
            yield ("u2", _wrap("<p>bar</p>"))

        idx = Indexer()
        idx.build(_gen())
        assert "foo" in idx.index
        assert "bar" in idx.index

    def test_rebuild_replaces_old_index(self) -> None:
        idx = Indexer()
        idx.build([("u1", _wrap("<p>old</p>"))])
        idx.build([("u2", _wrap("<p>new</p>"))])
        assert "old" not in idx.index
        assert "new" in idx.index
        assert "u1" not in idx.index.get("new", {})

    def test_logs_indexed_summary(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        idx = Indexer()
        with caplog.at_level(logging.INFO, logger="src.indexer"):
            idx.build(
                [
                    ("a", _wrap("<p>x y</p>")),
                    ("b", _wrap("<p>y z</p>")),
                ]
            )
        # Two pages, three unique words: x, y, z.
        assert "Indexed 2 pages, 3 unique words" in caplog.text


# -------------------------------------------------------------- TestPersistence


class TestPersistence:
    """Save/load round-trips and error paths."""

    @pytest.fixture
    def populated(self) -> Indexer:
        idx = Indexer()
        idx.build(
            [
                ("u1", _wrap("<p>alpha beta alpha</p>")),
                ("u2", _wrap("<p>beta gamma</p>")),
            ]
        )
        return idx

    def test_round_trip_preserves_index(
        self, populated: Indexer, tmp_path: Path
    ) -> None:
        path = tmp_path / "idx.json"
        populated.save(path)
        fresh = Indexer()
        fresh.load(path)
        assert fresh.index == populated.index

    def test_save_creates_missing_parent_dirs(
        self, populated: Indexer, tmp_path: Path
    ) -> None:
        path = tmp_path / "deep" / "nested" / "idx.json"
        assert not path.parent.exists()
        populated.save(path)
        assert path.exists()

    def test_save_accepts_string_path(
        self, populated: Indexer, tmp_path: Path
    ) -> None:
        path_str = str(tmp_path / "idx.json")
        populated.save(path_str)
        assert Path(path_str).exists()

    def test_load_accepts_string_path(
        self, populated: Indexer, tmp_path: Path
    ) -> None:
        path = tmp_path / "idx.json"
        populated.save(path)
        fresh = Indexer()
        fresh.load(str(path))
        assert fresh.index == populated.index

    def test_save_rejects_unknown_format(
        self, populated: Indexer, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="unsupported format"):
            populated.save(tmp_path / "x", fmt="yaml")

    def test_load_rejects_unknown_format(self, tmp_path: Path) -> None:
        idx = Indexer()
        with pytest.raises(ValueError, match="unsupported format"):
            idx.load(tmp_path / "x", fmt="yaml")

    def test_pickle_round_trip_preserves_index(
        self, populated: Indexer, tmp_path: Path
    ) -> None:
        path = tmp_path / "idx.pkl"
        populated.save(path, fmt="pickle")
        fresh = Indexer()
        fresh.load(path, fmt="pickle")
        assert fresh.index == populated.index

    def test_pickle_file_is_binary(
        self, populated: Indexer, tmp_path: Path
    ) -> None:
        # Pickle output must not be readable as UTF-8 text — otherwise we
        # would have accidentally written JSON to a .pkl file.
        path = tmp_path / "idx.pkl"
        populated.save(path, fmt="pickle")
        raw = path.read_bytes()
        assert raw[:2] == b"\x80\x05"  # pickle protocol-5 header

    def test_pickle_load_rejects_text_file(
        self, populated: Indexer, tmp_path: Path
    ) -> None:
        # Sanity: feeding a JSON file to fmt="pickle" must fail loudly,
        # not silently corrupt the in-memory index.
        path = tmp_path / "idx.json"
        populated.save(path, fmt="json")
        fresh = Indexer()
        with pytest.raises(pickle.UnpicklingError):
            fresh.load(path, fmt="pickle")

    def test_load_missing_file_raises_filenotfounderror(
        self, tmp_path: Path
    ) -> None:
        idx = Indexer()
        with pytest.raises(FileNotFoundError):
            idx.load(tmp_path / "does_not_exist.json")

    def test_load_malformed_json_raises_decode_error(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        idx = Indexer()
        with pytest.raises(json.JSONDecodeError):
            idx.load(bad)

    def test_saved_json_is_human_readable(
        self, populated: Indexer, tmp_path: Path
    ) -> None:
        path = tmp_path / "idx.json"
        populated.save(path)
        text = path.read_text(encoding="utf-8")
        # Indented (newlines) and contains a recognisable token key.
        assert "\n" in text
        assert '"alpha"' in text

    def test_saved_json_keys_are_sorted(
        self, populated: Indexer, tmp_path: Path
    ) -> None:
        path = tmp_path / "idx.json"
        populated.save(path)
        text = path.read_text(encoding="utf-8")
        # alpha < beta < gamma alphabetically; their first occurrences
        # in the file must follow that order.
        assert text.index('"alpha"') < text.index('"beta"') < text.index('"gamma"')

    def test_save_preserves_unicode(self, tmp_path: Path) -> None:
        # The tokeniser drops non-ASCII letters, but if the index ever
        # contained one (via direct assignment or a future tokeniser
        # change) ensure_ascii=False must keep it readable.
        idx = Indexer()
        idx.index = {"café": {"u": {"freq": 1, "positions": [0]}}}
        path = tmp_path / "u.json"
        idx.save(path)
        assert "café" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------- TestGetPostings


class TestGetPostings:
    """The single-word lookup helper."""

    @pytest.fixture
    def populated(self) -> Indexer:
        idx = Indexer()
        idx.build([("u", _wrap("<p>quote wisdom quote</p>"))])
        return idx

    def test_known_word_returns_postings(self, populated: Indexer) -> None:
        result = populated.get_postings("quote")
        assert result == {"u": {"freq": 2, "positions": [0, 2]}}

    def test_unknown_word_returns_empty_dict(self, populated: Indexer) -> None:
        assert populated.get_postings("absent") == {}

    def test_case_insensitive_lookup(self, populated: Indexer) -> None:
        assert populated.get_postings("QUOTE") == populated.get_postings("quote")
        assert populated.get_postings("Quote") == populated.get_postings("quote")

    def test_returned_dict_not_aliased(self, populated: Indexer) -> None:
        # Mutating the returned dict for a missing word must not pollute
        # the index. (The empty dict is a fresh literal each call.)
        populated.get_postings("missing")["u"] = {"freq": 999, "positions": []}
        assert populated.get_postings("missing") == {}
