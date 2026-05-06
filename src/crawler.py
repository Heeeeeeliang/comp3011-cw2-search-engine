"""
Crawler module for COMP3011 Coursework 2.

Politely BFS-crawls every same-domain page reachable from a seed URL and
returns raw ``(url, html)`` pairs. Parsing for indexing is deliberately the
Indexer's responsibility; the crawler stays HTTP-aware but content-agnostic.

Design choices
--------------
* **BFS over recursion.** Iterative BFS with an explicit queue gives us
  deterministic ordering and avoids Python's recursion limit on sites with
  long link chains.
* **Politeness measured between requests, not in fixed sleeps.** We record
  the wall-clock time of the last network request and only sleep for the
  remainder of the 6-second window. Cache hits never sleep.
* **Optional on-disk cache.** During development the assignment's 6-second
  delay makes a full crawl take 10+ minutes; caching responses keyed by
  URL hash lets us re-run ``build`` instantly. The cache lives outside the
  repo (``.crawl_cache/`` by default) so it never gets committed.
* **Errors per-page are warnings, not aborts.** A 404 or timeout on one
  page must not lose the rest of the crawl. The seed URL is the one
  exception: if it fails we have nothing to index, so we raise.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

POLITENESS_DELAY: float = 6.0  # seconds; assignment requires >= 6
DEFAULT_TIMEOUT: float = 10.0  # per-request timeout
DEFAULT_USER_AGENT: str = (
    "COMP3011-CW2-Crawler/1.0 "
    "(University of Leeds coursework; +https://quotes.toscrape.com)"
)

# URL paths we deliberately never enqueue (no useful text content for search).
SKIPPED_PATH_PREFIXES: tuple[str, ...] = ("/login", "/logout")


class CrawlError(Exception):
    """Raised when the crawl cannot start (e.g. seed URL unreachable)."""


class Crawler:
    """Politely BFS-crawl every same-domain page from a seed URL.

    Parameters
    ----------
    seed_url:
        Starting URL. The crawl stays within this URL's network location.
    delay:
        Minimum seconds between successive HTTP requests. Must be >= 6 to
        comply with the assignment's politeness requirement.
    timeout:
        Per-request HTTP timeout in seconds.
    user_agent:
        Sent in the ``User-Agent`` header. A descriptive UA is good net
        citizenship and helps site owners identify the crawler.
    cache_dir:
        Optional directory in which to cache responses on disk. When set,
        previously-fetched URLs are read from cache instead of re-fetched.
    session:
        Optional pre-configured ``requests.Session`` (mainly for tests).
    """

    def __init__(
        self,
        seed_url: str,
        delay: float = POLITENESS_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
        cache_dir: Optional[Union[str, Path]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        if delay < POLITENESS_DELAY:
            raise ValueError(
                f"delay must be >= {POLITENESS_DELAY}s to comply with the "
                "assignment's politeness policy"
            )
        if not urlparse(seed_url).netloc:
            raise ValueError(f"seed_url has no netloc: {seed_url!r}")

        self.seed_url: str = self._normalise(seed_url)
        self.delay: float = delay
        self.timeout: float = timeout
        self._domain: str = urlparse(seed_url).netloc

        self.session: requests.Session = session or requests.Session()
        self.session.headers["User-Agent"] = user_agent

        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._last_request_at: Optional[float] = None

    # ------------------------------------------------------------------ public

    def crawl(self) -> list[tuple[str, str]]:
        """Crawl the entire site and return every ``(url, html)`` pair.

        Returns
        -------
        list[tuple[str, str]]
            Materialised list of ``(url, html)`` pairs in BFS order. For
            streaming over a partial result (without buffering the whole
            corpus in memory) use :meth:`iter_pages` instead.

        Raises
        ------
        CrawlError
            If the seed URL itself cannot be fetched. Per-page failures
            after the seed are logged and skipped, not raised.
        """
        return list(self.iter_pages())

    def iter_pages(self) -> Iterator[tuple[str, str]]:
        """Yield ``(url, html)`` pairs lazily as pages are crawled.

        Streaming is useful when callers want to start indexing while the
        crawl is still running, or to display incremental progress.

        Yields
        ------
        tuple[str, str]
            ``(url, html)`` for each successfully fetched page in BFS
            order. Pages that 404 or time out (after the seed) are
            logged at WARNING and skipped without yielding.

        Raises
        ------
        CrawlError
            If the seed URL itself cannot be fetched on the first
            iteration of the BFS loop.
        """
        queue: deque[str] = deque([self.seed_url])
        seen: set[str] = {self.seed_url}
        is_first = True

        while queue:
            url = queue.popleft()
            try:
                html = self._fetch(url)
            except requests.RequestException as exc:
                if is_first:
                    raise CrawlError(f"could not fetch seed {url}: {exc}") from exc
                LOGGER.warning("skipping %s: %s", url, exc)
                continue
            finally:
                is_first = False

            yield url, html

            for link in self._extract_links(url, html):
                if link not in seen and self._should_follow(link):
                    seen.add(link)
                    queue.append(link)

    # ----------------------------------------------------------------- private

    def _fetch(self, url: str) -> str:
        """Fetch ``url`` (or its cached copy) and return decoded HTML.

        The politeness delay is applied only to live network requests, not
        cache hits.
        """
        cached = self._cache_read(url)
        if cached is not None:
            LOGGER.debug("cache hit: %s", url)
            return cached

        self._wait_for_politeness()
        LOGGER.info("GET %s", url)
        response = self.session.get(url, timeout=self.timeout)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        html = response.text
        self._cache_write(url, html)
        return html

    def _wait_for_politeness(self) -> None:
        """Sleep just long enough that the next request is at least
        ``self.delay`` seconds after the previous one. No-op on the first
        request."""
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _extract_links(self, base_url: str, html: str) -> Iterable[str]:
        """Yield absolute, fragment-stripped URLs from ``<a href>`` tags."""
        soup = BeautifulSoup(html, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if not href or href.startswith(("mailto:", "javascript:", "tel:", "#")):
                continue
            absolute = urljoin(base_url, href)
            yield self._normalise(absolute)

    def _should_follow(self, url: str) -> bool:
        """Return ``True`` iff ``url`` is in-domain and not on the skip list."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if parsed.netloc != self._domain:
            return False
        if any(parsed.path.startswith(p) for p in SKIPPED_PATH_PREFIXES):
            return False
        return True

    @staticmethod
    def _normalise(url: str) -> str:
        """Normalise URLs so trivially-different forms dedupe correctly.

        Drops the fragment (``#section``) and any trailing slashes. So
        ``https://x.com/`` becomes ``https://x.com`` and
        ``https://x.com/a/`` becomes ``https://x.com/a``.
        """
        clean, _ = urldefrag(url)
        stripped = clean.rstrip("/")
        # Guard against a path-less URL collapsing to empty (shouldn't happen
        # for valid http URLs, but be defensive).
        return stripped if stripped else clean

    # --- cache helpers --------------------------------------------------

    def _cache_path(self, url: str) -> Optional[Path]:
        """Return the on-disk cache path for ``url``, or ``None`` if disabled."""
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{digest}.html"

    def _cache_read(self, url: str) -> Optional[str]:
        """Return cached HTML for ``url`` if present, else ``None``."""
        path = self._cache_path(url)
        if path is None or not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _cache_write(self, url: str, html: str) -> None:
        """Persist ``html`` to the cache (no-op if caching disabled)."""
        path = self._cache_path(url)
        if path is None:
            return
        path.write_text(html, encoding="utf-8")
