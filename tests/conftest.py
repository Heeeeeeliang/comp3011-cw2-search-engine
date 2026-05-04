"""
Shared test fixtures for the search-engine test suite.

The crawler enforces a 6-second politeness delay between successive HTTP
requests. Without intervention, a 22-test crawler suite would spend ~30s
sleeping. We auto-patch ``time.sleep`` to a no-op for every test in the
suite so the run finishes in well under a second.

Tests that need to *verify* sleep behaviour (e.g. that the delay was
applied) override this fixture by calling ``monkeypatch.setattr`` themselves
within the test - their override wins for the duration of that test only.
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``time.sleep`` with a no-op for the duration of every test.

    This keeps the test suite fast even though the crawler's politeness
    window is 6 seconds. Tests that need to observe the actual sleep
    arguments (e.g. ``TestPolitenessDelay``) install their own monkeypatch,
    which transparently shadows this one.
    """
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
