"""COMP3011 Coursework 2 search engine package.

The package contains four cooperating modules:

* :mod:`src.crawler` — politely BFS-crawls a single domain.
* :mod:`src.indexer` — builds, persists, and exposes the inverted index.
* :mod:`src.search` — runs ``print``/``find`` queries (TF-IDF, phrase).
* :mod:`src.main` — interactive ``cmd.Cmd`` shell that ties the rest
  together.

Each module's own docstring is the authoritative source for its design
choices and trade-offs; this file is a navigation aid only.
"""
