"""Comment audit: produces a concrete-data report on src/*.py."""
from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

SRC = Path("src")
FILES = sorted(p for p in SRC.glob("*.py"))


def _tokens(path: Path):
    src = path.read_text(encoding="utf-8")
    return list(tokenize.tokenize(io.BytesIO(src.encode("utf-8")).readline)), src


def _docstring_lines(src: str) -> set[int]:
    """Set of 1-based line numbers occupied by docstrings (module/class/function)."""
    tree = ast.parse(src)
    out: set[int] = set()

    def add(node):
        if not (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return
        ds = node.body[0]
        for ln in range(ds.lineno, ds.end_lineno + 1):
            out.add(ln)

    add(tree)
    for n in ast.walk(tree):
        add(n)
    return out


def per_file_metrics():
    rows = []
    for path in FILES:
        toks, src = _tokens(path)
        total_lines = src.count("\n") + (0 if src.endswith("\n") else 1)
        ds_lines = _docstring_lines(src)

        # Comment lines: count lines containing a COMMENT token
        comment_lines: set[int] = set()
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                comment_lines.add(tok.start[0])

        # Blank lines
        blank_lines: set[int] = set()
        for i, line in enumerate(src.splitlines(), start=1):
            if not line.strip():
                blank_lines.add(i)

        # Code lines = total - blank - docstring-only - comment-only
        # A line is "comment-only" if it has a COMMENT token and no other non-trivial token
        # We already know which lines have comments. We still want lines with code+comment to count as code.
        non_code = set(blank_lines) | set(ds_lines)
        # Determine pure-comment lines (comment is the only meaningful content)
        pure_comment_lines: set[int] = set()
        # group tokens by line
        line_tokens: dict[int, list] = {}
        for tok in toks:
            line_tokens.setdefault(tok.start[0], []).append(tok)
        for ln, tlist in line_tokens.items():
            kinds = {t.type for t in tlist}
            # ignore NL/NEWLINE/ENCODING/INDENT/DEDENT/ENDMARKER
            ignore = {tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING,
                      tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
            non_trivial = kinds - ignore
            if non_trivial == {tokenize.COMMENT}:
                pure_comment_lines.add(ln)

        non_code |= pure_comment_lines
        code_lines = total_lines - len(non_code)
        # Filter out lines we already counted as blank/docstring from pure_comment_lines for accurate display
        ratio = (len(comment_lines) / code_lines) if code_lines else 0.0
        rows.append(
            (path.as_posix(), total_lines, code_lines, len(ds_lines),
             len(comment_lines), len(pure_comment_lines), ratio)
        )
    return rows


def longest_comment_block_per_file():
    out = []
    for path in FILES:
        toks, src = _tokens(path)
        comment_line_set = {tok.start[0]: tok.string for tok in toks
                            if tok.type == tokenize.COMMENT}
        # We want PURE-comment lines (not trailing comments after code)
        line_tokens: dict[int, list] = {}
        for tok in toks:
            line_tokens.setdefault(tok.start[0], []).append(tok)
        pure: dict[int, str] = {}
        ignore = {tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING,
                  tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
        for ln, tlist in line_tokens.items():
            kinds = {t.type for t in tlist}
            if (kinds - ignore) == {tokenize.COMMENT}:
                pure[ln] = next(t.string for t in tlist if t.type == tokenize.COMMENT)

        # Find consecutive runs
        sorted_lines = sorted(pure)
        if not sorted_lines:
            out.append((path.as_posix(), None, 0, []))
            continue
        best_run = []
        cur_run = [sorted_lines[0]]
        for ln in sorted_lines[1:]:
            if ln == cur_run[-1] + 1:
                cur_run.append(ln)
            else:
                if len(cur_run) > len(best_run):
                    best_run = cur_run
                cur_run = [ln]
        if len(cur_run) > len(best_run):
            best_run = cur_run
        text = [pure[ln] for ln in best_run]
        out.append((path.as_posix(), (best_run[0], best_run[-1]), len(best_run), text))
    return out


def functions_with_more_comments_than_code():
    """Find functions where comment lines > code lines."""
    findings = []
    for path in FILES:
        src = path.read_text(encoding="utf-8")
        toks, _ = _tokens(path)
        tree = ast.parse(src)

        # Build per-line token classification
        line_tokens: dict[int, list] = {}
        for tok in toks:
            line_tokens.setdefault(tok.start[0], []).append(tok)
        ignore = {tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING,
                  tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}

        ds_lines = _docstring_lines(src)
        src_lines = src.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start = node.body[0].lineno  # skip the def header
            end = node.end_lineno
            code_count = 0
            comment_count = 0
            for ln in range(start, end + 1):
                if ln in ds_lines:
                    continue
                if ln not in line_tokens:
                    continue
                kinds = {t.type for t in line_tokens[ln]}
                non_triv = kinds - ignore
                if non_triv == {tokenize.COMMENT}:
                    comment_count += 1
                elif tokenize.COMMENT in kinds:
                    code_count += 1  # trailing comment line counts as code
                    comment_count += 1
                elif non_triv:
                    code_count += 1
            if comment_count > code_count and comment_count > 0:
                body_text = "\n".join(src_lines[node.lineno - 1:end])
                findings.append({
                    "file": path.as_posix(),
                    "func": node.name,
                    "code_lines": code_count,
                    "comment_lines": comment_count,
                    "body": body_text,
                })
    return findings


def longest_single_line_comments(n: int = 5):
    items = []
    for path in FILES:
        toks, _ = _tokens(path)
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                items.append((len(tok.string), path.as_posix(), tok.start[0], tok.string))
    items.sort(reverse=True)
    return items[:n]


def main():
    print("=" * 80)
    print("1. PER-FILE METRICS")
    print("=" * 80)
    rows = per_file_metrics()
    hdr = f"{'file':<22}{'total':>7}{'code':>7}{'docstr':>9}{'cmt':>6}{'pure#':>7}{'ratio':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        path, total, code, ds, cmt, pure, ratio = r
        print(f"{path:<22}{total:>7}{code:>7}{ds:>9}{cmt:>6}{pure:>7}{ratio:>9.3f}")
    print()
    print("Legend: docstr = lines inside docstrings; cmt = lines containing any '#';")
    print("        pure# = lines whose only content is '#' (no trailing-after-code);")
    print("        ratio = cmt / code.")

    print()
    print("=" * 80)
    print("2. LONGEST CONSECUTIVE COMMENT BLOCK PER FILE")
    print("=" * 80)
    for path, span, n_lines, text in longest_comment_block_per_file():
        if span is None:
            print(f"\n{path}: (no pure comment lines)")
            continue
        print(f"\n{path}  lines {span[0]}-{span[1]}  ({n_lines} consecutive lines)")
        for line in text:
            print(f"  {line}")

    print()
    print("=" * 80)
    print("3. FUNCTIONS WHERE comment_lines > code_lines")
    print("=" * 80)
    findings = functions_with_more_comments_than_code()
    if not findings:
        print("(none)")
    else:
        for f in findings:
            print(f"\n{f['file']}::{f['func']}  "
                  f"code={f['code_lines']}, comments={f['comment_lines']}")
            print("---")
            print(f["body"])
            print("---")

    print()
    print("=" * 80)
    print("4. FIVE LONGEST SINGLE-LINE COMMENTS")
    print("=" * 80)
    for length, path, lineno, text in longest_single_line_comments():
        print(f"\n{path}:{lineno}  ({length} chars)")
        print(f"  {text}")


if __name__ == "__main__":
    main()
