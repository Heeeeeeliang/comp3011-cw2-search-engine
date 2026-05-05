# COMP3011 Coursework 2 — Search Engine

> A polite BFS web crawler, an inverted index with TF-IDF ranking and phrase
> queries, and a small `cmd.Cmd` shell that ties them together.

![tests](https://img.shields.io/badge/tests-241%20passing-brightgreen)
![coverage](https://img.shields.io/badge/coverage-100%25%20line%20%2B%20branch-brightgreen)
![python](https://img.shields.io/badge/python-3.12-blue)
![status](https://img.shields.io/badge/status-coursework%20submission-blue)

The badges above are static — there is no CI configured for this submission.
The numbers reflect the local test run on the commit this README ships with;
verify them yourself with `pytest --cov=src --cov-branch`.

---

## Quick demo

Six lines from a real session against the live `quotes.toscrape.com` corpus
(213 pages, ~3,830 unique stems):

```text
> build
Indexed 213 pages, 3830 unique words -> data/index.json
> find "good friends"
https://quotes.toscrape.com/tag/contentment/page/1
https://quotes.toscrape.com/tag/friendship/page/1
... (8 more results, ranked by TF-IDF descending)
```

`build` runs once (cached on subsequent invocations), `load` rehydrates from
the saved pickle in ~11 ms, and individual queries return in milliseconds.

---

## Features

- **Polite crawler.** BFS traversal of one domain with a 6-second politeness
  window between live requests, on-disk cache so repeated runs are network-free,
  and per-page error isolation (a single broken page doesn't kill the crawl).
- **Inverted index with positions.** Plain nested dicts —
  `word -> url -> {freq, positions}` — JSON-serialisable, marker-inspectable,
  trivially fast in memory.
- **Porter stemming + 50-word stopword list.** Symmetric at index time and
  query time; queries for `running` match pages containing `runs` and *vice
  versa*.
- **TF-IDF ranking** with alphabetical tie-break for deterministic output
  across runs.
- **Phrase queries.** `find "good friends"` matches only pages where the words
  occur at consecutive positions in the post-filter token stream.
- **Dual JSON / Pickle storage** with extension-based auto-detection. JSON for
  the marker to inspect, Pickle for fast `load` and 6.5× smaller on disk.
- **Versioned on-disk format.** A pre-3.2 index file is rejected on load with
  a clear "rebuild" message rather than silently producing zero scores.
- **241 tests, 100% line + branch coverage** across crawler, indexer, search,
  CLI, and integration.

---

## Installation

Python 3.10+ is required (3.12 is the development target). The project has
three runtime dependencies plus pytest for development.

```bash
git clone https://github.com/Heeeeeeliang/comp3011-cw2-search-engine
cd comp3011-cw2-search-engine

python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

No `nltk.download(...)` step is needed — Porter stemming uses pure-Python
rules that ship with `nltk`, and the stopword list is embedded in source.
See [Design rationale § 7](#7-embedded-stopword-list-vs-nltk-download).

---

## Usage

Launch the interactive shell from the project root:

```bash
python -m src.main
```

You'll see a `>` prompt. The five commands are:

| Command            | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| `build`            | Crawl the seed site, build the inverted index, save **both** JSON and Pickle to `data/`. |
| `load`             | Load the previously-saved index from disk. Prefers `data/index.pkl` for speed.         |
| `print <word>`     | Print the inverted-index entry for `<word>`: `{url: {freq, positions}}`.              |
| `find <words…>`    | List pages containing **all** given words (AND), ranked by TF-IDF.                     |
| `find "<phrase>"`  | List pages where the quoted words occur at consecutive positions (phrase query).       |
| `exit` / `Ctrl-D`  | Quit the shell.                                                                        |

### Query syntax

The `find` command accepts an arbitrary mix of bare words and double-quoted
phrases:

```text
> find good friends                    # AND query: pages with both stems
> find "good friends"                  # phrase: words must be adjacent
> find "good friends" wisdom           # phrase AND single word
> find Don't trust strangers           # contractions kept; case folded
```

A few details worth knowing:

- **Case is folded** at both index and query time — `find QUICK Fox` ≡
  `find quick fox`.
- **Stopwords are dropped** from queries the same way they're dropped at
  index time — `find the quick fox` is effectively `find quick fox`.
- **Apostrophes are kept** as part of the token (`don't`, `it's`), so the
  contractions in the corpus are matchable.
- **Single-quoted phrases are not supported.** Apostrophes never open a
  phrase; only the double-quote character does. This is deliberate so that
  contractions don't accidentally produce phrase semantics.
- **An unbalanced `"` returns `[]`** rather than crashing. Typos shouldn't
  kill the REPL.

### Single-word lookup

`print` returns the raw postings dict so you can inspect what the index
stores:

```text
> print indifference
{
  "https://quotes.toscrape.com/tag/indifference/page/1": {
    "freq": 6,
    "positions": [27, 50, 73, 96, 119, 142]
  },
  ...
}
```

`print` is also stem-aware: typing `print indifferent` looks up the same
stem and returns the same postings.

---

## Architecture

The codebase is four small modules and one CLI driver. Each does one job
and has a single integration point with the next.

### [`src/crawler.py`](src/crawler.py) — politely fetches the corpus

The `Crawler` class performs a breadth-first traversal of one domain
starting from a seed URL. Politeness is implemented as a *window* (≥ 6 s
since the last live request), not a flat sleep — cache hits don't pay
the politeness cost, so re-runs are network-free. URLs are normalised
(fragment-stripped, scheme-lowercased), and an on-disk cache stores the
raw HTML keyed by SHA-256 of the URL so a successful crawl can be
replayed offline.

Failures on individual pages are *isolated*: a 404 on one URL doesn't
abort the whole crawl. The seed URL is the exception — if the seed
fetch fails the entire run is hopeless and `CrawlError` is raised.

### [`src/indexer.py`](src/indexer.py) — builds and persists the index

`Indexer.build(pages)` consumes an iterable of `(url, html)` pairs and
populates two state attributes:

- `self.index: dict[str, dict[str, dict]]` — the inverted index proper.
  Shape: `word -> url -> {"freq": int, "positions": [int, ...]}`.
- `self.doc_lengths: dict[str, int]` — post-filter token count per URL,
  needed for the TF denominator in the search ranker.

Two stateless helpers do the heavy lifting:

- `_strip_html(html)` parses the HTML with BeautifulSoup and decomposes
  `<script>` and `<style>` subtrees before extracting text. Without the
  decompose step, inline JavaScript would leak into the index as tokens.
- `tokenise(text)` lower-cases the input, regex-extracts `[a-z0-9']+`
  runs, drops the curated stopword list, and applies Porter stemming.
  This same function is called by the search side on user queries — the
  symmetry is what makes case folding, stopword removal and stemming
  *work* without query-side bookkeeping.

`save(path, fmt=None)` and `load(path, fmt=None)` serialise the index +
doc_lengths to disk. When `fmt` is omitted, the format is inferred from
the path's extension via `_format_from_path`. The on-disk envelope is
`{"version": 2, "index": {...}, "doc_lengths": {...}}`; pre-versioned
files (Day 2 era) are detected and rejected with a "rebuild" message.

### [`src/search.py`](src/search.py) — query planner

`SearchEngine` holds a reference to a populated `Indexer` (live updates
visible without re-binding) and exposes two query commands:

- `print_word(word)` — single-stem postings lookup, returns `{}` if the
  query tokenises to nothing or no document contains the term.
- `find(query) -> list[str]` and `find_with_scores(query) -> list[tuple[str,
  float]]` — the AND query with TF-IDF ranking, with phrase support
  introduced by Day 3.3.

The `find_with_scores` pipeline:

1. `_parse_query(query)` (module-level) splits the query into atoms via
   `shlex.shlex` configured with `quotes='"'`. Atoms are either single
   words (`str`) or phrases (`tuple[str, ...]`).
2. Each atom is tokenised with the indexer's `tokenise`, then becomes a
   URL set:
   - Single-word atoms: postings of the stem.
   - Phrase atoms (≥ 2 stems): AND-intersect candidate URLs first
     (cheap), then filter through `_phrase_matches(doc, phrase)` which
     walks `index[phrase[0]][doc]["positions"]` and checks for adjacent
     positions of the remaining phrase terms.
3. Set-intersect across atoms.
4. Score each surviving URL via `_score(url, score_terms)` — straight
   `sum(tf(t,d) * idf(t))` over every stem (phrase atoms contribute their
   constituent stems). Sort by `(-score, url)` for deterministic
   alphabetical tie-break.

### [`src/main.py`](src/main.py) — `cmd.Cmd` shell

`SearchShell` is a thin subclass of `cmd.Cmd`. All output goes through
`self.stdout` (so tests can swap in `io.StringIO`), and `do_*` methods
validate input and call `_require_index()` before touching the search
engine. `do_build` writes both `data/index.json` and `data/index.pkl`;
`do_load` prefers the pickle and falls back to JSON. The interactive
shell is what the demo video drives.

### Inverted index structure

The index is the central data structure. It looks like this on disk
(pretty-printed JSON, abridged):

```jsonc
{
  "version": 2,
  "index": {
    "good": {
      "https://quotes.toscrape.com/tag/friendship/page/1": {
        "freq": 4,
        "positions": [12, 47, 95, 132]
      },
      "https://quotes.toscrape.com/tag/love/page/1": {
        "freq": 2,
        "positions": [33, 81]
      }
    },
    "friend": {
      "https://quotes.toscrape.com/tag/friendship/page/1": {
        "freq": 7,
        "positions": [13, 31, 48, 76, 96, 110, 133]
      }
    }
  },
  "doc_lengths": {
    "https://quotes.toscrape.com/tag/friendship/page/1": 158,
    "https://quotes.toscrape.com/tag/love/page/1": 174
  }
}
```

Three things to notice:

1. **Keys are stems, not raw words.** `friends` (plural) and `friend`
   (singular) collapse to `friend`; queries for either match.
2. **Positions are 0-indexed in the post-filter token stream.** The
   stopword `the` is dropped *before* positions are stamped, which
   means a page reading `"the good and friends"` records
   `("good", 0)` and `("friend", 1)` — phrase queries see the words
   as adjacent.
3. **`doc_lengths` is a peer of `index`**, not derivable on the fly.
   It's persisted alongside so `load` is one pass; without it, TF-IDF
   would need to walk every posting at startup to recompute lengths.

---

## Design rationale

Every non-trivial decision in the codebase is documented inline as a
docstring or comment. This section consolidates the seven decisions
the rubric specifically asks about, in `chosen / alternatives /
trade-off` form.

### 1. BFS over DFS for the crawl

**Chosen.** A `collections.deque` queue, popped left, links extracted
from the popped page appended right. Every iteration of the loop is
exactly one page.

**Alternatives.** Recursive DFS (the obvious "follow this link"
expression); priority queue keyed by URL depth or estimated relevance;
bidirectional / federated crawl across multiple seeds.

**Trade-off.** BFS gives uniform depth coverage, which matters for a
small site where the marker may inspect any page. Recursive DFS would
have hit Python's stack limit (~1 000) on long link chains, and
`sys.setrecursionlimit` is the kind of code-smell I'd rather not ship.
A priority queue would help on a corpus with millions of pages where
budget-aware crawling is necessary; at 213 pages the BFS terminates in
under two minutes on a fresh run and the priority machinery is a
distraction.

### 2. Nested dicts, not a database, for the index

**Chosen.** `dict[str, dict[str, dict]]` — `word -> url -> {freq,
positions}`. JSON-serialisable directly, no schema, O(1) hash lookup.

**Alternatives.** SQLite with a `(word, url, freq)` table; a real
search engine like Elasticsearch or Lucene; `defaultdict(Counter)` to
auto-vivify on the way in.

**Trade-off.** A 213-page corpus produces a ~2.6 MB JSON index that
fits comfortably in memory; the database would buy us concurrent
writes and SQL queries, neither of which the brief asks for.
Elasticsearch is the right answer at scale, wrong for a coursework
artefact. `defaultdict` is convenient at write time but pollutes the
serialisation contract — `json.dumps(defaultdict)` works, but the
on-disk shape is technically `{...}` and the type round-trip is not
guaranteed across Python versions. Plain dicts are dull, predictable,
and exactly what a marker is expecting to open in a text editor.

### 3. Dual JSON + Pickle storage

**Chosen.** `do_build` writes `data/index.json` *and* `data/index.pkl`;
`do_load` prefers the pickle and falls back to JSON if the pickle is
missing. Format is auto-detected from the path's extension on every
call.

**Alternatives.** JSON only (the Day 2 baseline); pickle only;
MessagePack or Protocol Buffers; a single binary format like SQLite.

**Trade-off.** JSON is the format the marker is told to inspect — it
stays. Pickle is **6.5× smaller on disk and 24× faster to write**
(see § Performance). Shipping both is cheap (the combined ~3 MB is a
rounding error in the repo) and lets the CLI prefer pickle for
fast-load while preserving the inspection target. MessagePack /
Protobuf would add a dependency for marginal gain. Single binary
SQLite would foreclose the inspection use case.

### 4. AND semantics for multi-word queries

**Chosen.** `find good friends` returns pages that contain *both*
words; the implementation is `set.intersection(*per_word_sets)`.

**Alternatives.** OR (union of postings); proximity / weighted-AND
(boost docs where the words occur near each other); positional
operators like `NEAR/5`.

**Trade-off.** AND matches user intuition for a search prompt:
typing more words narrows results. OR would balloon the result list
on common terms (e.g. `find love good friends` with OR returns 168+
pages because `love` alone is on 168 of them). Phrase queries — the
double-quoted form added in Day 3.3 — give the user explicit access
to the strictest variant, so AND-by-default plus phrase-on-demand
covers both ends of the precision/recall spectrum.

### 5. TF-IDF formula choice

**Chosen.** Standard textbook form, natural log, no smoothing:

```
tf(t, d)   = freq[t][d] / |d|
idf(t)     = log(N / df[t])
score(d)   = Σ tf(t, d) · idf(t)   for t in query_terms
```

with alphabetical tie-break on `(-score, url)`.

**Alternatives.** `log(1 + tf)` damping (sublinear TF scaling);
`log(N/df) + 1` smoothing (avoids 0 scores when `df = N`); BM25 (more
accurate ranking with two free parameters); cosine similarity over
TF-IDF vectors.

**Trade-off.** The standard form is the one the rubric references and
the one a marker recognises. The variants help mostly on long
documents (sublinear TF) or vocabulary-rich corpora (BM25); at ~250
pages with average 119 tokens per page neither is the dominant
concern. The "no smoothing" choice means terms appearing in *every*
document yield `idf = log(1) = 0`, contributing zero to the score —
this is a useful sanity property (the suite asserts it directly with
`test_score_zero_for_term_in_every_doc`) and it doubles as a poor
man's stopword filter for terms the curated list missed.

### 6. Stopwords removal

**Chosen.** A 50-word English stopword list (articles, common verbs,
pronouns, conjunctions) is dropped from tokens at both index time and
query time, before Porter stemming. Contractions (`don't`, `won't`,
`it's`) are deliberately *excluded* from the stopword set.

**Alternatives.** No stopword removal (keep every token); a larger
list (NLTK's stopwords corpus has ~180 entries); a domain-tuned list
mined from corpus statistics; full removal *including* contractions.

**Trade-off.** Removing 50 stopwords cuts the index by ~30% on this
corpus and removes obviously-uninformative tokens from query results.
A larger list buys diminishing returns and starts to drop
borderline-content words like "many" or "most"; a domain-tuned list
is over-engineering for 213 pages. Including contractions in the
stopword list would prevent searches for `don't trust` from matching
the expected pages, which the corpus has plenty of (Maugham, Wilde,
Twain). The trade-off the user *does* accept is that phrase queries
operate on the post-filter token stream — `"good friends"` matches a
page reading "the good and friends" because `the` and `and` were
dropped before positions were stamped. Documented and tested.

### 7. Embedded stopword list vs `nltk.corpus.stopwords`

**Chosen.** A `frozenset[str]` literal of 50 entries hardcoded in
`src/indexer.py`.

**Alternatives.** `nltk.corpus.stopwords.words('english')` —
authoritative, well-curated, larger.

**Trade-off.** The NLTK stopwords corpus requires
`nltk.download('stopwords')` to populate the data directory before
first use. That call goes over the network and writes to a user
directory — it fails in offline / sandboxed CI environments and adds
an opaque first-run step the user has to know about. The 50-word
hardcoded list ships with the source, has no install-time side
effects, and is small enough to audit by reading. **Crucially, the
PorterStemmer used in the same file does NOT need a download** — it
ships with pure-Python rules in nltk's source tree. So we're using
nltk where it's frictionless and side-stepping it where it's not.
A future change that wants the bigger list could add it without
breaking the offline-first guarantee by vendoring an extracted copy
or making the download conditional.

### Honourable mentions

A couple of decisions the rubric didn't list but are worth surfacing:

- **Positions are stamped *after* stopword removal**, not before. This
  is the contract phrase queries depend on; locked in by
  `test_positions_assigned_after_stopword_removal` and
  `test_phrase_across_stopwords_works_correctly`.
- **`shlex.shlex` over `shlex.split`** for query parsing — the natural
  reach is `shlex.split(posix=True)`, but POSIX mode treats `'` as a
  string delimiter and crashes on `don't`. Customising the lexer with
  `quotes='"'` keeps phrases parseable without breaking contractions.
- **Versioned on-disk envelope** (`{"version": 2, ...}`) so a stale
  pre-3.2 index file is rejected with a "rebuild" message rather than
  silently restored without `doc_lengths` and producing zero scores.

---

## Performance

### Save / load benchmark (213 pages, 3,830 unique stems, 5-run average)

| Op   | JSON     | Pickle  | Pickle advantage |
| ---- | -------- | ------- | ---------------- |
| save | 117.3 ms | 4.8 ms  | **24.3× faster** |
| load | 17.7 ms  | 10.8 ms | **1.6× faster**  |
| size | 2 656 KiB | 409 KiB | **6.5× smaller** |

The save delta is the dramatic one: JSON has to traverse, sort, indent
and UTF-8-encode every key, while Pickle just dumps the in-memory
layout. Save is paid once per `build`; load is paid every time the
shell starts. The combined wins (faster save, smaller disk, faster
load) justify the dual-storage strategy.

The load speed-up (1.6×) is more modest than is sometimes claimed for
Pickle vs JSON — the brief expected 3-5×. At this corpus size Python's
stdlib `json` module is a C extension and is already fast. A 100 000-
page corpus would shift the ratio.

### Complexity table

| Operation             | Complexity                                | Notes                                                         |
| --------------------- | ----------------------------------------- | ------------------------------------------------------------- |
| `crawl`               | `O(P)` HTTP + cache I/O                   | `P` = pages discovered via BFS                                |
| `build`               | `O(T)`                                    | `T` = total post-filter tokens; one pass per page             |
| `save` JSON           | `O(K log K)`                              | `sort_keys=True` dominates; `K` = index keys                  |
| `save` Pickle         | `O(K)`                                    | byte-stream dump, no sort                                     |
| `load`                | `O(F)`                                    | `F` = file bytes; parse cost                                  |
| `print <word>`        | `O(1)`                                    | hash lookup on the stem                                       |
| `find <w>`            | `O(D log D)`                              | `D` = matching docs; sort dominates                           |
| `find <w1> <w2> ...`  | `O(min(\|d_i\|) · n)`                     | set-intersection over `n` terms; smallest set drives the cost |
| `find "<phrase>"`     | `O(C · Q · \|positions\|)`                | `C` = candidates after AND, `Q` = phrase length               |

The phrase complexity is a bit pessimistic: `_phrase_matches` walks
`positions` of `phrase[0]` and does `O(1)` set-membership checks for
the remaining `Q-1` terms. Worst case is a doc where the first phrase
term appears at every position — bounded by document length.

---

## Testing

```bash
pytest                                            # full suite, no coverage
pytest --cov=src --cov-report=term-missing        # line coverage
pytest --cov=src --cov-branch --cov-report=term-missing   # line + branch
pytest --cov=src --cov-report=html                # HTML report at htmlcov/index.html
```

### What ships at this commit

| Metric              | Value                                |
| ------------------- | ------------------------------------ |
| Tests               | **241 passing**, 0 failing, 0 errors |
| Line coverage       | **100%** (375 / 375 statements)      |
| Branch coverage     | **100%** (110 / 110 branches)        |
| Suite runtime       | 11.4 s plain, 38.3 s with `--cov-branch` |
| Test files          | 5 (one per module + integration)     |

### Test layout

- **`tests/test_crawler.py`** — 27 tests; HTTP mocked via a fake
  `requests.Session` so the suite never touches the network.
- **`tests/test_indexer.py`** — 110 tests covering tokenisation,
  HTML stripping, build pipeline, persistence, version envelope,
  format auto-detection, edge cases.
- **`tests/test_search.py`** — 63 tests covering AND queries, TF-IDF,
  phrase queries, ranking determinism, the query parser, edge cases.
- **`tests/test_main.py`** — 29 tests driving the `cmd.Cmd` shell
  through `onecmd`, with mocked crawler.
- **`tests/test_integration.py`** — 12 tests that load the **real**
  cached corpus and exercise the full pipeline. Skipped if
  `.crawl_cache/` is empty (the directory is `.gitignore`d, so a
  fresh checkout cleanly skips this module).

A no-network guard (`_NoNetworkSession`) wraps the integration
crawler: any cache miss raises `AssertionError` rather than silently
hitting the live site. CI safety in the absence of a CI runner.

---

## Known limitations

These are deliberate trade-offs documented for the marker; each is
either dictated by the brief, by the scope of the corpus, or by a
calculated decision elsewhere in this document.

- **Non-ASCII tokenisation is ASCII-only.** The token regex is
  `[a-z0-9']+`, so any character outside that class — accented Latin
  (`é`, `ï`), CJK (`你`, `好`), emoji — acts as a separator. `Café`
  becomes `caf`; `naïve` becomes `na` + `ve`; `你好` becomes nothing.
  The `quotes.toscrape.com` corpus is ASCII-only so the practical
  impact is zero, but a multilingual corpus would need either
  Unicode-aware boundaries or `\w+` with `re.UNICODE`.
- **No JavaScript rendering.** Pages are fetched with `requests` and
  parsed with BeautifulSoup. Content injected by client-side JS will
  not be in the index. `quotes.toscrape.com` serves fully-rendered
  HTML so this isn't a problem in practice; a SPA would need
  Playwright or Selenium.
- **Single-domain crawl.** The crawler only follows links inside the
  seed's netloc. Cross-domain links are extracted but not enqueued.
  This is a politeness and scope-creep guard, not a technical
  limitation — relaxing it to a list of allowed domains would be a
  one-line change.
- **Porter stemming, not lemmatisation.** Porter is a rule-based
  suffix stripper. `running` and `runs` collapse to `run`, but `ran`
  (irregular past tense) does not. A WordNet lemmatiser would handle
  irregulars but requires `nltk.download('wordnet')` and the same
  download-friction the stopword design dodged. Documented as
  `test_irregular_past_tense_does_not_lemmatise`.
- **No phrase-level ranking signal.** A phrase atom contributes its
  constituent stems' TF-IDF, summed; a "phrase rarity" score that
  treats `"good friends"` differently from the AND of `good` and
  `friends` is not implemented. The AND-intersection prefilter
  already handles selectivity, so the practical effect is small.
- **No incremental crawl / re-crawl.** Cache hits are reused
  forever; there's no `If-Modified-Since` header, no TTL, no diff.
  A long-lived index would need this; a one-shot coursework demo
  doesn't.

---

## Dependencies

Pinned to a minor version in [`requirements.txt`](requirements.txt):

| Package         | Min version | Purpose                                     |
| --------------- | ----------- | ------------------------------------------- |
| `requests`      | 2.31.0      | HTTP client for the crawler                 |
| `beautifulsoup4`| 4.12.0      | HTML parsing for `_strip_html`              |
| `nltk`          | 3.9.0       | `PorterStemmer` (rules ship with the wheel) |
| `pytest`        | 7.4.0       | Test runner                                 |
| `pytest-cov`    | 4.1.0       | Coverage plugin (line + branch)             |

No `nltk.download(...)` is required at install time — see
[Design rationale § 7](#7-embedded-stopword-list-vs-nltk-download).

---

## GenAI declaration

This project was developed with AI assistance, in line with the assignment's
"Green tier" GenAI policy. The tools used were **Claude Opus 4.7** (via the
anthropic.com chat UI) and **Claude Code CLI** (Opus 4.7) — the latter as a
pair-programming surface running inside the project directory.

A timestamped log of every substantive AI interaction — including specific
points where the AI was wrong, where I pushed back, and design decisions I
made independently — was kept throughout development as raw material for the
critical evaluation. The video demonstration covers the most significant of
these.
---

## Future work

Four directions, ordered by how much I'd want them on a real product
of this:

1. **BM25 ranking** in place of TF-IDF. Same inverted-index inputs,
   slightly more accurate at longer documents, and the implementation
   is one extra `_score` variant gated on a flag. Day 3.2's `_score`
   was deliberately straight-line so a BM25 swap is a 5-line change.

2. **Bigram (and trigram) indexing.** A second inverted index keyed
   on adjacent stem pairs would make phrase queries O(1) per pair
   rather than `O(C × Q)`. Storage cost is roughly 2-3× the
   single-term index; only worth doing if phrase queries dominate the
   workload.

3. **Persistent crawler state.** Right now `do_build` re-discovers
   the full URL graph every run. A `crawl_state.json` with the BFS
   queue and visited set would let the crawl pause / resume, and a
   `--max-pages` flag would make budgeted crawls trivial.

4. **Web UI.** Flask + a single-page form would turn the demo from
   "type at the prompt" into "search like Google". The search engine
   itself doesn't change — `SearchEngine.find_with_scores` is already
   the right shape for a JSON endpoint.

---

## License

Coursework submission — no open-source license attached. Reuse of
non-trivial portions for academic purposes outside this assignment
should cite this repository.
