"""
Command-line entry point for the COMP3011 Coursework 2 search tool.

Provides an interactive shell with four commands::

    build           crawl the site, build the index, save to disk
    load            load the previously-built index from disk
    print <word>    print the inverted-index entry for <word>
    find <words>    list pages containing ALL given words

Design choices
--------------
* **``cmd.Cmd`` for the shell.** Standard library, gives us a prompt,
  command dispatch, built-in ``help``, and clean ``EOF``/``exit``
  handling for free. No third-party dependency, fully testable via
  :meth:`onecmd` without entering the REPL.
* **All output goes through ``self.stdout``.** ``cmd.Cmd`` already
  routes its prompt and intro through ``self.stdout``; making the
  ``do_*`` methods do the same means tests can capture output by
  passing a :class:`io.StringIO`.
* **Logging on stderr, results on stdout.** Crawl progress (INFO logs)
  goes to stderr by default; the user's command results go to stdout.
  This keeps the demo video transcript readable: stderr can be
  redirected if it's noisy without losing the answer.
* **UTF-8 stdout reconfigure on Windows.** Windows consoles default to
  cp1252, which can mis-render any odd character that slips through
  (URLs are ASCII-only, but defence in depth is cheap). The reconfigure
  is best-effort because some non-TTY stdouts (notably ``pytest``'s
  capture) don't expose the method.
* **Index path is a constant, not user-configurable.** The brief
  pins the submission to ``data/index.json``; a CLI flag would only
  invite the marker to type a wrong path during evaluation.
"""

from __future__ import annotations

import cmd
import json
import logging
import sys
from pathlib import Path
from typing import IO, Optional

from src.crawler import CrawlError, Crawler
from src.indexer import Indexer
from src.search import SearchEngine

LOGGER = logging.getLogger(__name__)

SEED_URL: str = "https://quotes.toscrape.com/"
DATA_DIR: Path = Path("data")
INDEX_JSON: Path = DATA_DIR / "index.json"
INDEX_PKL: Path = DATA_DIR / "index.pkl"
CACHE_DIR: str = ".crawl_cache"


class SearchShell(cmd.Cmd):
    """Interactive ``cmd.Cmd`` shell wired to crawler/indexer/search.

    Parameters
    ----------
    stdin, stdout:
        Forwarded to :class:`cmd.Cmd`. Tests pass :class:`io.StringIO`
        instances to capture output; in production both default to the
        process streams.
    """

    intro: str = "Search engine ready. Type 'help' for commands, 'exit' to quit."
    prompt: str = "> "

    def __init__(
        self,
        stdin: Optional[IO[str]] = None,
        stdout: Optional[IO[str]] = None,
    ) -> None:
        super().__init__(stdin=stdin, stdout=stdout)
        self.indexer: Optional[Indexer] = None
        self.search: Optional[SearchEngine] = None

    # --------------------------------------------------------------- helpers

    def _say(self, message: str) -> None:
        """Write a single line to ``self.stdout``.

        Centralised so tests need only inspect one channel and so a
        future change (e.g. piping to a logger or colourising) lands in
        one place.
        """
        print(message, file=self.stdout)

    def _require_index(self) -> bool:
        """Print a hint and return ``False`` if no index is loaded yet."""
        if self.indexer is None or self.search is None:
            self._say("Run `build` or `load` first.")
            return False
        return True

    # ----------------------------------------------------------- shell hooks

    def do_build(self, _arg: str) -> None:
        """build : crawl the site, build the inverted index, save to disk."""
        crawler = Crawler(SEED_URL, cache_dir=CACHE_DIR)
        try:
            pages = list(crawler.iter_pages())
        except CrawlError as exc:
            self._say(f"Build failed: {exc}")
            return

        indexer = Indexer()
        indexer.build(pages)
        indexer.save(INDEX_JSON, fmt="json")
        indexer.save(INDEX_PKL, fmt="pickle")

        self.indexer = indexer
        self.search = SearchEngine(indexer)
        self._say(
            f"Indexed {len(pages)} pages, {len(indexer.index)} unique words"
            f" -> {INDEX_JSON.as_posix()}"
        )

    def do_load(self, _arg: str) -> None:
        """load : load the previously-saved index from disk.

        Prefers the pickle (3-5x faster on the live 213-page corpus
        per the Day 3.4 benchmark logged in GENAI_LOG.md) and falls
        back to the JSON file if the pickle is missing. The format
        is auto-detected from the chosen path's extension; both
        constants point to recognised suffixes so no explicit ``fmt``
        is needed at the call site.
        """
        source = INDEX_PKL if INDEX_PKL.exists() else INDEX_JSON
        indexer = Indexer()
        try:
            indexer.load(source)
        except FileNotFoundError:
            self._say("No index found. Run `build` first.")
            return

        self.indexer = indexer
        self.search = SearchEngine(indexer)
        self._say(
            f"Loaded {len(indexer.index)} words from {source.as_posix()}"
        )

    def do_print(self, arg: str) -> None:
        """print <word> : show the inverted-index entry for one word."""
        words = arg.split()
        if not words:
            self._say("usage: print <word>")
            return
        if not self._require_index():
            return

        word = words[0]
        assert self.search is not None  # _require_index guarantees this
        postings = self.search.print_word(word)
        if not postings:
            self._say(f"'{word}' not in index.")
            return
        self._say(json.dumps(postings, indent=2, ensure_ascii=False))

    def do_find(self, arg: str) -> None:
        """find <word1> <word2> ... : list pages containing ALL given words."""
        query = arg.strip()
        if not query:
            self._say("usage: find <word> [<word> ...]")
            return
        if not self._require_index():
            return

        assert self.search is not None  # _require_index guarantees this
        results = self.search.find(query)
        if not results:
            self._say(f"No pages contain all of: {query}")
            return
        for url in results:
            self._say(url)

    def do_exit(self, _arg: str) -> bool:
        """exit : quit the shell."""
        return True

    # Ctrl-D / pipe-end also exits cleanly.
    do_EOF = do_exit

    def default(self, line: str) -> None:
        """Friendly fallback for unknown commands."""
        self._say(
            f"Unknown command: {line!r}. Type 'help' for available commands."
        )


def main() -> None:
    """Entry point for ``python -m src.main``.

    Configures logging so the user can see crawl progress, best-effort
    upgrades stdout to UTF-8 (Windows consoles default to cp1252), and
    hands off to the cmd-loop. Kept as a function (rather than inline
    under ``if __name__``) so it can be tested directly.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            # Some captured/pipe stdouts reject reconfigure; the rest of
            # the CLI still works in cp1252 since URLs are ASCII.
            pass
    SearchShell().cmdloop()


if __name__ == "__main__":  # pragma: no cover
    main()
