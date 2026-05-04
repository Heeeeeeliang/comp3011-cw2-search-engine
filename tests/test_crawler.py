"""Tests for the Crawler module.

Strategy
--------
Every test mocks the ``requests.Session`` so we never hit the live site.
This keeps the suite fast (< 1s) and deterministic, and avoids burning
through politeness windows on every CI run. ``time.sleep`` is monkey-
patched in delay tests so the suite runs instantly even when verifying
6-second behaviour.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
import requests

from src.crawler import POLITENESS_DELAY, CrawlError, Crawler


# --------------------------------------------------------------------- helpers


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` used by Crawler._fetch."""

    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _page(*links: str, body: str = "hello world") -> str:
    """Build a tiny HTML doc that links to each given URL."""
    anchors = "\n".join(f'<a href="{href}">x</a>' for href in links)
    return f"<html><body>{body}{anchors}</body></html>"


def _make_session(pages: dict[str, _FakeResponse]) -> MagicMock:
    """Build a MagicMock Session that returns canned responses by URL.

    Any URL not in ``pages`` raises an HTTPError, simulating a 404/network
    problem so we can verify error-handling paths.
    """
    session = MagicMock(spec=requests.Session)
    session.headers = {}

    def _get(url, timeout):  # noqa: ARG001
        if url not in pages:
            raise requests.HTTPError(f"unknown URL: {url}")
        return pages[url]

    session.get.side_effect = _get
    return session


# ----------------------------------------------------------------------- tests


class TestCrawlerInit:
    """Constructor validation and configuration."""

    def test_rejects_delay_below_six_seconds(self) -> None:
        with pytest.raises(ValueError, match=r"delay must be >= 6"):
            Crawler("https://x.com/", delay=3)

    def test_accepts_minimum_delay(self) -> None:
        # Must not raise.
        Crawler("https://x.com/", delay=POLITENESS_DELAY)

    def test_rejects_seed_without_netloc(self) -> None:
        with pytest.raises(ValueError, match="netloc"):
            Crawler("not-a-url", delay=POLITENESS_DELAY)

    def test_sets_user_agent_on_session(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        Crawler("https://x.com/", delay=POLITENESS_DELAY, session=session)
        assert "COMP3011" in session.headers["User-Agent"]

    def test_seed_url_is_normalised(self) -> None:
        crawler = Crawler("https://x.com/", delay=POLITENESS_DELAY)
        assert crawler.seed_url == "https://x.com"


class TestSingleDomain:
    """Crawler must stay on the seed's network location."""

    def test_does_not_follow_off_domain_links(self) -> None:
        pages = {
            "https://example.com": _FakeResponse(_page("https://other.com/x")),
        }
        session = _make_session(pages)
        crawler = Crawler(
            "https://example.com/", delay=POLITENESS_DELAY, session=session
        )
        urls = [u for u, _ in crawler.iter_pages()]
        assert urls == ["https://example.com"]
        # No call should have been made to other.com
        for call in session.get.call_args_list:
            assert "other.com" not in call.args[0]

    def test_follows_in_domain_links(self) -> None:
        pages = {
            "https://example.com": _FakeResponse(_page("/page/2")),
            "https://example.com/page/2": _FakeResponse(_page()),
        }
        crawler = Crawler(
            "https://example.com/",
            delay=POLITENESS_DELAY,
            session=_make_session(pages),
        )
        urls = [u for u, _ in crawler.iter_pages()]
        assert urls == ["https://example.com", "https://example.com/page/2"]


class TestDeduplication:
    """Each unique URL must be visited at most once."""

    def test_does_not_visit_same_url_twice(self) -> None:
        pages = {
            "https://example.com": _FakeResponse(
                _page(
                    "/",
                    "/",
                    "https://example.com/",
                    "https://example.com",
                )
            ),
        }
        session = _make_session(pages)
        crawler = Crawler(
            "https://example.com/", delay=POLITENESS_DELAY, session=session
        )
        list(crawler.iter_pages())
        assert session.get.call_count == 1  # one fetch despite 4 self-links

    def test_treats_fragment_variants_as_same_url(self) -> None:
        pages = {
            "https://example.com": _FakeResponse(_page("/page#a", "/page#b")),
            "https://example.com/page": _FakeResponse(_page()),
        }
        session = _make_session(pages)
        crawler = Crawler(
            "https://example.com/", delay=POLITENESS_DELAY, session=session
        )
        list(crawler.iter_pages())
        assert session.get.call_count == 2  # root + /page only

    def test_treats_trailing_slash_variants_as_same_url(self) -> None:
        pages = {
            "https://example.com": _FakeResponse(_page("/page", "/page/")),
            "https://example.com/page": _FakeResponse(_page()),
        }
        session = _make_session(pages)
        crawler = Crawler(
            "https://example.com/", delay=POLITENESS_DELAY, session=session
        )
        list(crawler.iter_pages())
        assert session.get.call_count == 2


class TestSkipList:
    """Login/logout etc. must not be crawled even when linked."""

    def test_skips_login_path(self) -> None:
        pages = {
            "https://example.com": _FakeResponse(_page("/login", "/about")),
            "https://example.com/about": _FakeResponse(_page()),
        }
        crawler = Crawler(
            "https://example.com/",
            delay=POLITENESS_DELAY,
            session=_make_session(pages),
        )
        urls = [u for u, _ in crawler.iter_pages()]
        assert "https://example.com/login" not in urls
        assert "https://example.com/about" in urls

    def test_skips_non_http_schemes(self) -> None:
        # mailto/javascript/tel hrefs must not crash _extract_links.
        pages = {
            "https://example.com": _FakeResponse(
                _page("mailto:a@b.com", "javascript:alert(1)", "tel:+44")
            ),
        }
        crawler = Crawler(
            "https://example.com/",
            delay=POLITENESS_DELAY,
            session=_make_session(pages),
        )
        urls = [u for u, _ in crawler.iter_pages()]
        assert urls == ["https://example.com"]


class TestErrorHandling:
    """Failed fetches must not abort the entire crawl."""

    def test_seed_failure_raises_crawl_error(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.get.side_effect = requests.ConnectionError("boom")
        crawler = Crawler(
            "https://example.com/", delay=POLITENESS_DELAY, session=session
        )
        with pytest.raises(CrawlError, match="seed"):
            list(crawler.iter_pages())

    def test_non_seed_failure_is_skipped(self) -> None:
        # /broken is deliberately omitted from `pages` so the mock raises.
        pages = {
            "https://example.com": _FakeResponse(_page("/broken", "/ok")),
            "https://example.com/ok": _FakeResponse(_page()),
        }
        crawler = Crawler(
            "https://example.com/",
            delay=POLITENESS_DELAY,
            session=_make_session(pages),
        )
        urls = [u for u, _ in crawler.iter_pages()]
        assert "https://example.com/ok" in urls
        assert "https://example.com/broken" not in urls

    def test_timeout_on_non_seed_is_skipped(self) -> None:
        def _get(url, timeout):  # noqa: ARG001
            if url == "https://example.com":
                return _FakeResponse(_page("/slow"))
            raise requests.Timeout(f"timeout: {url}")

        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.get.side_effect = _get

        crawler = Crawler(
            "https://example.com/", delay=POLITENESS_DELAY, session=session
        )
        urls = [u for u, _ in crawler.iter_pages()]
        assert urls == ["https://example.com"]  # /slow skipped, no crash


class TestPolitenessDelay:
    """Verify the 6-second window is actually enforced."""

    def test_delay_applied_between_consecutive_requests(self, monkeypatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        pages = {
            "https://example.com": _FakeResponse(_page("/a")),
            "https://example.com/a": _FakeResponse(_page("/b")),
            "https://example.com/b": _FakeResponse(_page()),
        }
        crawler = Crawler(
            "https://example.com/",
            delay=POLITENESS_DELAY,
            session=_make_session(pages),
        )
        list(crawler.iter_pages())

        # 3 fetches => at least 2 sleeps, each close to (but <=) 6s
        positive_sleeps = [s for s in sleeps if s > 0]
        assert len(positive_sleeps) >= 2
        assert all(0 < s <= POLITENESS_DELAY for s in positive_sleeps)

    def test_no_sleep_before_first_request(self, monkeypatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        pages = {"https://example.com": _FakeResponse(_page())}
        crawler = Crawler(
            "https://example.com/",
            delay=POLITENESS_DELAY,
            session=_make_session(pages),
        )
        list(crawler.iter_pages())
        assert sleeps == []  # exactly zero sleeps for a single fetch

    def test_cache_hit_does_not_trigger_sleep(self, monkeypatch, tmp_path) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        pages = {"https://example.com": _FakeResponse(_page())}
        # First crawl populates cache (no sleep, only one fetch).
        Crawler(
            "https://example.com/",
            delay=POLITENESS_DELAY,
            session=_make_session(pages),
            cache_dir=tmp_path,
        ).crawl()
        sleeps.clear()

        # Second crawler with the same cache -> zero network calls, zero sleeps.
        empty_session = _make_session({})  # would raise on any get()
        list(
            Crawler(
                "https://example.com/",
                delay=POLITENESS_DELAY,
                session=empty_session,
                cache_dir=tmp_path,
            ).iter_pages()
        )
        assert sleeps == []
        empty_session.get.assert_not_called()


class TestCache:
    """Disk cache short-circuits network calls and is content-correct."""

    def test_cache_round_trip(self, tmp_path) -> None:
        original_html = _page(body="cached content!")
        pages = {"https://example.com": _FakeResponse(original_html)}
        session = _make_session(pages)

        # First crawl populates cache.
        list(
            Crawler(
                "https://example.com/",
                delay=POLITENESS_DELAY,
                session=session,
                cache_dir=tmp_path,
            ).iter_pages()
        )
        first_calls = session.get.call_count

        # Second crawl reads cache; no extra network calls.
        result = list(
            Crawler(
                "https://example.com/",
                delay=POLITENESS_DELAY,
                session=session,
                cache_dir=tmp_path,
            ).iter_pages()
        )
        assert session.get.call_count == first_calls
        assert result == [("https://example.com", original_html)]

    def test_cache_directory_is_created_on_demand(self, tmp_path) -> None:
        cache = tmp_path / "deep" / "nested" / "cache"
        assert not cache.exists()
        Crawler("https://x.com/", delay=POLITENESS_DELAY, cache_dir=cache)
        assert cache.is_dir()


class TestExtractLinks:
    """White-box checks on link extraction so the BFS gets the right inputs."""

    def test_resolves_relative_urls(self) -> None:
        crawler = Crawler("https://example.com/", delay=POLITENESS_DELAY)
        html = _page("/page/2", "about", "../up")
        links = list(crawler._extract_links("https://example.com/foo/", html))
        assert "https://example.com/page/2" in links
        assert "https://example.com/foo/about" in links
        assert "https://example.com/up" in links

    def test_ignores_anchors_without_href(self) -> None:
        crawler = Crawler("https://example.com/", delay=POLITENESS_DELAY)
        html = "<html><body><a>no href</a><a href='/x'>ok</a></body></html>"
        links = list(crawler._extract_links("https://example.com/", html))
        assert links == ["https://example.com/x"]


class TestShouldFollow:
    """Unit-level tests of the URL-acceptance predicate.

    These complement the integration tests above by exercising every
    branch of ``_should_follow`` directly, without going through the
    BFS or the link extractor (which has its own filtering).
    """

    @pytest.fixture
    def crawler(self) -> Crawler:
        return Crawler("https://example.com/", delay=POLITENESS_DELAY)

    def test_accepts_in_domain_https_url(self, crawler: Crawler) -> None:
        assert crawler._should_follow("https://example.com/page")

    def test_rejects_off_domain_url(self, crawler: Crawler) -> None:
        assert not crawler._should_follow("https://other.com/page")

    def test_rejects_non_http_scheme(self, crawler: Crawler) -> None:
        # ftp:// must be rejected by the scheme guard, not by extraction.
        assert not crawler._should_follow("ftp://example.com/file")

    def test_rejects_skipped_paths(self, crawler: Crawler) -> None:
        assert not crawler._should_follow("https://example.com/login")
        assert not crawler._should_follow("https://example.com/logout")
