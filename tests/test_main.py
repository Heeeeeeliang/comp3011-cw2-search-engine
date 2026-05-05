"""Tests for the SearchShell CLI module.

Strategy
--------
We instantiate :class:`SearchShell` directly with a :class:`io.StringIO`
stdout and drive it through :meth:`onecmd`, which dispatches a single
command without entering the REPL. This keeps tests fast and avoids any
stdin/readline subtlety. The crawler is monkey-patched so ``do_build``
runs against canned ``(url, html)`` pages rather than the live site.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Iterator

import pytest

import src.main as main_module
from src.crawler import CrawlError
from src.indexer import Indexer
from src.main import SearchShell, main
from src.search import SearchEngine


# --------------------------------------------------------------------- helpers


_FAKE_PAGES: list[tuple[str, str]] = [
    ("https://example.com/p1", "<html><body><p>hello world</p></body></html>"),
    ("https://example.com/p2", "<html><body><p>hello again</p></body></html>"),
]


class _FakeCrawler:
    """Stand-in for ``src.crawler.Crawler`` that yields canned pages."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def iter_pages(self) -> Iterator[tuple[str, str]]:
        return iter(_FAKE_PAGES)


class _RaisingCrawler:
    """Crawler whose ``iter_pages`` raises CrawlError on first iteration."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def iter_pages(self) -> Iterator[tuple[str, str]]:
        raise CrawlError("seed fetch failed")


def _make_shell() -> SearchShell:
    """Return a SearchShell whose stdout is a fresh StringIO."""
    return SearchShell(stdout=io.StringIO())


def _populated_shell() -> SearchShell:
    """A shell with an in-memory index already wired in (no disk I/O)."""
    indexer = Indexer()
    indexer.build(_FAKE_PAGES)
    shell = _make_shell()
    shell.indexer = indexer
    shell.search = SearchEngine(indexer)
    return shell


# --------------------------------------------------------------- TestRequireIndex


class TestRequireIndex:
    """The ``_require_index`` guard used by print/find."""

    def test_returns_false_and_warns_when_no_index(self) -> None:
        shell = _make_shell()
        ok = shell._require_index()
        assert ok is False
        assert "Run `build` or `load` first." in shell.stdout.getvalue()

    def test_returns_true_when_index_loaded(self) -> None:
        shell = _populated_shell()
        # Drain any prior output for a clean assertion.
        assert shell._require_index() is True


# ------------------------------------------------------------------ TestDoBuild


class TestDoBuild:
    """``do_build`` end-to-end with a mocked Crawler and tmp paths."""

    @pytest.fixture
    def patched_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> tuple[Path, Path]:
        json_path = tmp_path / "index.json"
        pkl_path = tmp_path / "index.pkl"
        monkeypatch.setattr(main_module, "INDEX_JSON", json_path)
        monkeypatch.setattr(main_module, "INDEX_PKL", pkl_path)
        return json_path, pkl_path

    def test_build_writes_both_formats_and_wires_search(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_paths: tuple[Path, Path],
    ) -> None:
        json_path, pkl_path = patched_paths
        monkeypatch.setattr(main_module, "Crawler", _FakeCrawler)

        shell = _make_shell()
        shell.do_build("")

        out = shell.stdout.getvalue()
        # Two pages, three unique words: hello, world, again.
        assert "Indexed 2 pages, 3 unique words" in out
        # The summary mentions the configured JSON path (forward slashes).
        assert json_path.as_posix() in out
        assert json_path.exists()
        assert pkl_path.exists()
        assert isinstance(shell.indexer, Indexer)
        assert isinstance(shell.search, SearchEngine)

    def test_build_handles_crawl_error_gracefully(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_paths: tuple[Path, Path],
    ) -> None:
        json_path, pkl_path = patched_paths
        monkeypatch.setattr(main_module, "Crawler", _RaisingCrawler)

        shell = _make_shell()
        shell.do_build("")

        assert "Build failed:" in shell.stdout.getvalue()
        assert "seed fetch failed" in shell.stdout.getvalue()
        # No index should have been wired, no files written.
        assert shell.indexer is None
        assert shell.search is None
        assert not json_path.exists()
        assert not pkl_path.exists()


# ------------------------------------------------------------------- TestDoLoad


class TestDoLoad:
    """``do_load`` prefers the pickle (faster) and falls back to JSON."""

    @pytest.fixture
    def patched_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> tuple[Path, Path]:
        json_path = tmp_path / "index.json"
        pkl_path = tmp_path / "index.pkl"
        # Both paths are patched: without patching the pickle constant
        # too, do_load would happily prefer the real data/index.pkl
        # committed on Day 3.2 and the test would silently load that
        # instead of the tmp fixture.
        monkeypatch.setattr(main_module, "INDEX_JSON", json_path)
        monkeypatch.setattr(main_module, "INDEX_PKL", pkl_path)
        return json_path, pkl_path

    def test_load_prefers_pickle_when_both_exist(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        json_path, pkl_path = patched_paths
        seed = Indexer()
        seed.build(_FAKE_PAGES)
        seed.save(json_path)  # auto-detect json
        seed.save(pkl_path)   # auto-detect pickle
        assert json_path.exists() and pkl_path.exists()

        shell = _make_shell()
        shell.do_load("")

        out = shell.stdout.getvalue()
        # Loaded path should be the pickle (faster), not the JSON.
        assert pkl_path.as_posix() in out
        assert json_path.as_posix() not in out
        assert "Loaded 3 words from" in out
        # Wired up the same as before; word lookup still works.
        assert isinstance(shell.indexer, Indexer)
        assert isinstance(shell.search, SearchEngine)
        assert shell.search.print_word("hello") != {}

    def test_load_falls_back_to_json_when_pickle_missing(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        json_path, pkl_path = patched_paths
        seed = Indexer()
        seed.build(_FAKE_PAGES)
        seed.save(json_path)  # only JSON written; pkl_path unused
        assert json_path.exists() and not pkl_path.exists()

        shell = _make_shell()
        shell.do_load("")

        out = shell.stdout.getvalue()
        # Reported source is the JSON fallback.
        assert json_path.as_posix() in out
        assert "Loaded 3 words from" in out
        assert shell.search.print_word("hello") != {}

    def test_load_missing_both_files_prints_helpful_message(
        self, patched_paths: tuple[Path, Path]
    ) -> None:
        json_path, pkl_path = patched_paths
        # Neither file was created; both `.exists()` checks return False.
        assert not json_path.exists() and not pkl_path.exists()

        shell = _make_shell()
        shell.do_load("")

        assert "No index found. Run `build` first." in shell.stdout.getvalue()
        assert shell.indexer is None
        assert shell.search is None


# ----------------------------------------------------------------- TestDoPrint


class TestDoPrint:
    """``print <word>`` emits JSON for known words, helpful errors otherwise."""

    def test_known_word_prints_json_postings(self) -> None:
        shell = _populated_shell()
        shell.do_print("hello")

        out = shell.stdout.getvalue()
        # Output must be valid JSON parseable by the marker.
        parsed = json.loads(out)
        assert "https://example.com/p1" in parsed
        assert parsed["https://example.com/p1"]["freq"] == 1

    def test_unknown_word_prints_not_in_index(self) -> None:
        shell = _populated_shell()
        shell.do_print("absent")
        assert "'absent' not in index." in shell.stdout.getvalue()

    def test_empty_arg_prints_usage(self) -> None:
        shell = _populated_shell()
        shell.do_print("")
        assert "usage: print <word>" in shell.stdout.getvalue()

    def test_whitespace_arg_prints_usage(self) -> None:
        shell = _populated_shell()
        shell.do_print("   \t\n  ")
        assert "usage: print <word>" in shell.stdout.getvalue()

    def test_no_index_prints_run_build_message(self) -> None:
        shell = _make_shell()
        shell.do_print("hello")
        assert "Run `build` or `load` first." in shell.stdout.getvalue()

    def test_extra_words_use_first_only(self) -> None:
        # `print hello world` looks up "hello" and ignores "world".
        shell = _populated_shell()
        shell.do_print("hello world")
        out = shell.stdout.getvalue()
        parsed = json.loads(out)
        assert "https://example.com/p1" in parsed


# ------------------------------------------------------------------ TestDoFind


class TestDoFind:
    """``find <words...>`` prints URLs or a no-match message."""

    def test_match_prints_each_url_on_its_own_line(self) -> None:
        shell = _populated_shell()
        shell.do_find("hello")

        lines = [l for l in shell.stdout.getvalue().splitlines() if l]
        assert lines == [
            "https://example.com/p1",
            "https://example.com/p2",
        ]

    def test_multi_word_intersection(self) -> None:
        shell = _populated_shell()
        shell.do_find("hello world")
        # "world" only on p1.
        assert shell.stdout.getvalue().strip() == "https://example.com/p1"

    def test_no_match_prints_no_pages_message(self) -> None:
        shell = _populated_shell()
        shell.do_find("absent")
        assert "No pages contain all of: absent" in shell.stdout.getvalue()

    def test_empty_arg_prints_usage(self) -> None:
        shell = _populated_shell()
        shell.do_find("")
        assert "usage: find" in shell.stdout.getvalue()

    def test_whitespace_arg_prints_usage(self) -> None:
        shell = _populated_shell()
        shell.do_find("   ")
        assert "usage: find" in shell.stdout.getvalue()

    def test_no_index_prints_run_build_message(self) -> None:
        shell = _make_shell()
        shell.do_find("anything")
        assert "Run `build` or `load` first." in shell.stdout.getvalue()


# ------------------------------------------------------------------- TestDoExit


class TestDoExit:
    """``exit`` and EOF must signal the cmd loop to terminate."""

    def test_do_exit_returns_true(self) -> None:
        assert _make_shell().do_exit("") is True

    def test_do_eof_returns_true(self) -> None:
        # do_EOF is aliased to do_exit; both pipe-end and Ctrl-D hit this.
        assert _make_shell().do_EOF("") is True


# ------------------------------------------------------------------- TestDefault


class TestDefault:
    """Unknown commands are reported, not silently ignored."""

    def test_unknown_command_emits_message(self) -> None:
        shell = _make_shell()
        shell.default("flibbertigibbet")
        out = shell.stdout.getvalue()
        assert "Unknown command" in out
        assert "flibbertigibbet" in out

    def test_dispatched_via_onecmd(self) -> None:
        # cmd.Cmd routes unknown verbs to default(); confirm the wiring.
        shell = _make_shell()
        shell.onecmd("nosuch arg")
        assert "Unknown command" in shell.stdout.getvalue()


# --------------------------------------------------------------------- TestMain


class TestMain:
    """The ``main()`` entry point hands off to cmdloop after configuring I/O."""

    def test_main_invokes_cmdloop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = {"count": 0}

        def fake_loop(self: SearchShell) -> None:
            called["count"] += 1

        monkeypatch.setattr(SearchShell, "cmdloop", fake_loop)
        main()
        assert called["count"] == 1

    def test_main_reconfigures_stdout_to_utf8(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, str] = {}

        class _FakeStdout:
            def reconfigure(self, encoding: str) -> None:
                captured["encoding"] = encoding

            def write(self, _s: str) -> int:
                return 0

            def flush(self) -> None:
                pass

        monkeypatch.setattr(sys, "stdout", _FakeStdout())
        monkeypatch.setattr(SearchShell, "cmdloop", lambda self: None)
        main()
        assert captured.get("encoding") == "utf-8"

    def test_main_skips_reconfigure_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # io.StringIO has no .reconfigure attribute; main() must tolerate.
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(SearchShell, "cmdloop", lambda self: None)
        main()  # must not raise

    def test_main_swallows_reconfigure_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Some captured stdouts have reconfigure but raise on call. The
        # CLI must still start; URLs are ASCII so cp1252 is acceptable.
        class _BadStdout:
            def reconfigure(self, encoding: str) -> None:
                raise OSError("not reconfigurable")

            def write(self, _s: str) -> int:
                return 0

            def flush(self) -> None:
                pass

        monkeypatch.setattr(sys, "stdout", _BadStdout())
        monkeypatch.setattr(SearchShell, "cmdloop", lambda self: None)
        main()  # must not raise
