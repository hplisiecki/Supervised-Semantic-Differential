"""Reader for .wordpack files, format version 1.

Uses only the standard library and numpy. Every pack carries a self-test,
which holds this reader and every other reader of the format to the same output.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import threading
import unicodedata
import weakref
from array import array
from bisect import bisect_left
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

FORMAT_VERSION = 1

# Keys answered from a dict built when a pack opens. Rows are in frequency
# order, so these are the most frequent keys.
_TOP_KEYS = 50_000
_SAMPLE_STEP = 32


class WordPackError(ValueError):
    """A pack this reader cannot read, or a failed self-test."""


class Vocab(Sequence):
    """Keys in row order, with exact lookup.

    No dict of all keys is built: for 1.2M keys a Python list plus dict takes
    about 159 MB, the key bytes, offsets and `vocab_order` about 20 MB. The
    first _TOP_KEYS keys go into a dict at load (9 ms, 6 MB for 50k keys).
    Any other key is found by binary search over UTF-8 bytes (about 3 µs for
    1.3M keys), so every reader agrees on the order whatever its locale, and
    the answer is kept: a text pays one search per distinct rare word.
    """

    def __init__(self, blob: bytes, order: np.ndarray):
        self._blob = blob
        # Key r is blob[off[r] : off[r + 1] - 1], without its "\n". int32 holds
        # any vocab under 2 GB and takes half the RAM of int64.
        newlines = np.flatnonzero(np.frombuffer(blob, dtype=np.uint8) == 10)
        off = np.zeros(len(newlines) + 1, dtype=np.int32)
        off[1:] = newlines + 1
        # array.array, not numpy: the binary search reads one element at a time,
        # and a numpy scalar read costs several times an array.array read.
        self._off = array("i", off.tobytes())
        self._order = array("i", np.ascontiguousarray(order, dtype=np.int32).tobytes())
        if len(self._order) != len(self):
            raise WordPackError(
                f"vocab has {len(self)} keys but vocab_order has {len(self._order)} entries"
            )
        n = min(len(self), _TOP_KEYS)
        # Keys are stored in row order, so the top rows are one leading block of blob.
        try:
            top = blob[: self._off[n]].decode("utf-8")
        except UnicodeDecodeError as e:
            raise WordPackError(f"vocab is not valid UTF-8 ({e})") from e
        self._rows = dict(zip(top.split("\n"), range(n)))
        self._prefixes: dict[str, bool] = {}
        # Every _SAMPLE_STEP-th key in sorted order, as bytes. A search bisects
        # this list, then the one stretch of _SAMPLE_STEP keys it points to,
        # so both bisections run in C (about 2 MB for 1.3M keys).
        self._sample = [self._key_bytes(self._order[i]) for i in range(0, len(self._order), _SAMPLE_STEP)]

    def __len__(self) -> int:
        return len(self._off) - 1

    def __getitem__(self, row):
        if isinstance(row, slice):
            return [self[i] for i in range(*row.indices(len(self)))]
        if row < 0:
            row += len(self)
        if not 0 <= row < len(self):
            raise IndexError(row)
        return self._key_bytes(row).decode("utf-8")

    def __contains__(self, key) -> bool:
        return isinstance(key, str) and self.row(key) >= 0

    def _key_bytes(self, row) -> bytes:
        return self._blob[self._off[row] : self._off[row + 1] - 1]

    def _search(self, kb: bytes) -> int:
        """Position of the first sorted key >= kb."""
        # sample[j - 1] < kb <= sample[j], so the answer is after sorted
        # position (j - 1) * step and at most j * step.
        j = bisect_left(self._sample, kb)
        lo = (j - 1) * _SAMPLE_STEP + 1 if j else 0
        hi = min(j * _SAMPLE_STEP, len(self._order))
        return bisect_left(self._order, kb, lo, hi, key=self._key_bytes)

    def row(self, key: str) -> int:
        # No lock: a dict read or insert is atomic in CPython; two threads may
        # both search a new key and store the same answer.
        row = self._rows.get(key)
        if row is None:
            kb = key.encode("utf-8")
            i = self._search(kb)
            found = i < len(self._order) and self._key_bytes(self._order[i]) == kb
            row = self._order[i] if found else -1
            self._rows[key] = row
        return row

    def has_prefix(self, prefix: str) -> bool:
        hit = self._prefixes.get(prefix)
        if hit is None:
            kb = prefix.encode("utf-8")
            i = self._search(kb)
            hit = i < len(self._order) and self._key_bytes(self._order[i]).startswith(kb)
            self._prefixes[prefix] = hit
        return hit


def _l2_rows(x: np.ndarray) -> None:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    x /= norms


def dequantize(q: np.ndarray, abtt: np.ndarray | None, m: int) -> np.ndarray:
    """Int8 rows to float32 unit rows with ABTT `m` applied.

    `abtt` row 0 is the mean, rows 1..10 the principal components.
    """
    limit = 0 if abtt is None else len(abtt) - 1
    if not 0 <= m <= limit:
        raise ValueError(f"abtt must be between 0 and {limit}, got {m}")
    x = q.astype(np.float32)
    _l2_rows(x)
    if m > 0:
        x -= abtt[0]
        if m == 1:
            # Dot plus outer product: 51 ms against 66 ms for the matrix form
            # on a 50k x 300 prefix.
            c = abtt[1]
            x -= np.outer(x @ c, c)
        else:
            comps = abtt[1 : m + 1]
            x -= (x @ comps.T) @ comps
        _l2_rows(x)
    return x


class Tokens(NamedTuple):
    """Output of `tokenize`. Offsets index `text`, the input after NFC, fold and emphasis blanking."""

    text: str
    sents: list
    keys: list
    rows: list
    spans: list
    sent: list


def _class(ranges) -> str:
    """Regex class body for inclusive code-point ranges, as \\U escapes."""
    return "".join(f"\\U{a:08x}" if a == b else f"\\U{a:08x}-\\U{b:08x}" for a, b in ranges)


class Chars:
    """Character classes and maps from a pack's `chars` table.

    Taken from the pack, never from this runtime's Unicode data, so readers
    built on different Unicode versions tokenize the same text alike.
    """

    def __init__(self, table: dict):
        letter, digit, space = (_class(table[k]) for k in ("letter", "digit", "space"))
        # The same set as a plain string, for str.strip.
        self.spaces = "".join(chr(c) for a, b in table["space"] for c in range(a, b + 1))
        self._lower = {int(k): v for k, v in table["lower"].items()}
        self._upper = {int(k): v for k, v in table["upper"].items()}
        self._fold = {int(k): v for k, v in table["fold"].items()}
        L, S, NS = f"[{letter}]", f"[{space}]", f"[^{space}]"
        # Only a whole run of . ! ? can match, so the lookbehind changes no
        # match. Without it a run that no whitespace follows is tried again
        # from each of its positions, which takes quadratic time.
        self.sent_end = re.compile(rf"(?<![.!?])[.!?]+(?={S}|\Z)|\n")
        self.skip = re.compile(rf"https?://{NS}+|(?<!{NS})@{NS}+")
        # Skipped spans come first, so a star inside one is never matched as a star run.
        self.stars = re.compile(self.skip.pattern + r"|(\*+)|\n")
        self.chunk = re.compile(rf"{NS}+")
        self.unit = re.compile(rf"({L}+)|([{digit}]+)|.", re.DOTALL)
        self.letter = re.compile(L)
        self.special = re.compile(rf"[^{letter}'\-]")
        self.ends_alnum = re.compile(rf"[{letter}{digit}]\Z")
        self.mask_run = re.compile(rf"{L}[{letter}*]*")
        self.word = re.compile(rf"{L}+(?:['\-]{L}+)*")

    def lower(self, s: str) -> str:
        return s.translate(self._lower)

    def upper(self, s: str) -> str:
        return s.translate(self._upper)

    def fold(self, s: str) -> str:
        return s.translate(self._fold)


def _case_forms(form: str, case_sensitive: bool, chars: Chars) -> list[str]:
    lower = chars.lower(form)
    if not case_sensitive:
        return [lower]
    forms = [form]
    for f in (lower, chars.upper(form[:1]) + chars.lower(form[1:])):
        if f not in forms:
            forms.append(f)
    return forms


def resolve(form: str, vocab, case_sensitive: bool, chars: Chars) -> tuple[str, int]:
    """Key and row of a form under the case rule; (lowercased form, -1) when absent."""
    for f in _case_forms(form, case_sensitive, chars):
        row = vocab.row(f)
        if row >= 0:
            return f, row
    return chars.lower(form), -1


class _Opts(NamedTuple):
    vocab: object
    chars: Chars
    cs: bool
    clitics: tuple
    by_len: dict
    # Chunk text -> (keys, rows, spans relative to the chunk). A chunk's tokens
    # depend on nothing but its text and the pack, and most chunks repeat: in
    # measuring_hate_speech 9% of 985k chunks are distinct, so tokenizing a
    # corpus mostly reuses earlier answers.
    chunks: dict


# The chunk cache is emptied when it reaches this size. An entry takes about
# 455 bytes (measured on measuring_hate_speech), so the bound keeps it near
# 115 MB, which matters under Pyodide's 4 GB; frequent chunks come back within
# a few texts after a reset.
_CHUNK_CACHE_MAX = 250_000


def _opts(vocab, chars: Chars, case_sensitive: bool, clitics, masked) -> _Opts:
    by_len: dict[int, list[str]] = {}
    for w in masked:
        by_len.setdefault(len(w), []).append(w)
    return _Opts(vocab, chars, case_sensitive, tuple(clitics), by_len, {})


def tokenize(text: str, vocab, chars: Chars, case_sensitive: bool = False, clitics=(), masked=()) -> Tokens:
    """Sentences, keys, rows and spans of a text.

    `vocab` needs `row(key)` and `has_prefix(prefix)`; a `Vocab` has both.
    """
    return _tokenize(text, _opts(vocab, chars, case_sensitive, clitics, masked))


def _spaces(m: re.Match) -> str:
    return " " * len(m.group())


def _tokenize(text: str, o: _Opts) -> Tokens:
    chars, cache = o.chars, o.chunks
    text = _blank_emphasis(chars.fold(unicodedata.normalize("NFC", text)), chars)
    out = Tokens(text, [], [], [], [], [])
    keys, rows, spans, sent = out.keys, out.rows, out.spans, out.sent
    for si, (a, b) in enumerate(_sentences(text, chars)):
        out.sents.append((a, b))
        s = text[a:b]
        # Blanking skipped spans with spaces keeps every offset valid. Every
        # skipped span contains "@" or "://", so most sentences need no regex.
        if "@" in s or "://" in s:
            s = chars.skip.sub(_spaces, s)
        for chunk in chars.chunk.finditer(s):
            c = chunk.group()
            hit = cache.get(c)
            if hit is None:
                if len(cache) >= _CHUNK_CACHE_MAX:
                    cache.clear()
                toks = list(_chunk_tokens(c, o))
                # Most chunks are one token: kept as that (key, row, start, end),
                # appended without the per-field lists of the general case.
                hit = cache[c] = toks[0] if len(toks) == 1 else (
                    tuple(t[0] for t in toks),
                    tuple(t[1] for t in toks),
                    tuple((t[2], t[3]) for t in toks),
                )
            if len(hit) == 4:
                base = a + chunk.start()
                keys.append(hit[0])
                rows.append(hit[1])
                spans.append((base + hit[2], base + hit[3]))
                sent.append(si)
            elif hit[0]:
                base = a + chunk.start()
                keys.extend(hit[0])
                rows.extend(hit[1])
                spans.extend([(base + ts, base + te) for ts, te in hit[2]])
                sent.extend([si] * len(hit[0]))
    return out


def _blank_emphasis(text: str, chars: Chars) -> str:
    """`text` with paired emphasis stars replaced by spaces.

    Left in, the closing stars of `**bit**` would join the masked run and be
    read as masked letters. Spaces, not deletion, keep every offset in place.
    """
    if "*" not in text:
        return text
    # One-character tests. Both fail on "", so the text's ends count as whitespace.
    nonspace, alnum = chars.chunk.match, chars.ends_alnum.match
    out = list(text)
    open_runs: dict[int, int] = {}  # run length -> start of the open run of that length on this line
    for m in chars.stars.finditer(text):
        a, b = m.span()
        if m.lastindex is None:  # a skipped span or a line feed
            if text[a] == "\n":
                open_runs.clear()
            continue
        n = b - a
        before, after = text[a - 1 : a], text[b : b + 1]
        if n in open_runs and nonspace(before) and not alnum(after):
            for start in (open_runs.pop(n), a):
                out[start : start + n] = " " * n
        elif nonspace(after) and not alnum(before):
            open_runs[n] = a
    return "".join(out)


def _sentences(text: str, chars: Chars) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in chars.sent_end.finditer(text):
        _add_sentence(text, start, m.end(), spans, chars)
        start = m.end()
    _add_sentence(text, start, len(text), spans, chars)
    return spans


def _add_sentence(text: str, a: int, b: int, spans: list, chars: Chars) -> None:
    piece = text[a:b]
    lead = len(piece) - len(piece.lstrip(chars.spaces))
    end = len(piece.rstrip(chars.spaces))
    if end > lead:
        spans.append((a + lead, a + end))


def _chunk_tokens(c: str, o: _Opts):
    """Yield (key, row, start, end) for one chunk, offsets relative to the chunk."""
    units = [(m.start(), m.end(), m.lastindex) for m in o.chars.unit.finditer(c)]  # lastindex: 1 letters, 2 digits
    special = o.chars.special.search(c) is not None  # without one, no vocabulary match is possible
    i = 0
    while i < len(units):
        us, _, kind = units[i]
        if kind is None:
            cl = _stray_clitic(c, us, o)
            if cl is None:
                i += 1
                continue
            key, row = resolve(c[us : us + len(cl)], o.vocab, o.cs, o.chars)
            yield key, row, us, us + len(cl)
            while i < len(units) and units[i][0] < us + len(cl):
                i += 1
            continue
        if special:
            hit = _vocab_match(c, units, i, o)
            if hit:
                j, key, row = hit
                yield key, row, us, units[j - 1][1]
                i = j
                continue
        if kind == 2:
            i += 1
            continue
        hit = _masked_match(c, us, o) if o.by_len else None
        if hit:
            key, row, end = hit
            yield key, row, us, end
        else:
            end = o.chars.word.match(c, us).end()
            yield from _word_tokens(c, us, end, o)
        while i < len(units) and units[i][0] < end:
            i += 1


def _stray_clitic(c: str, pos: int, o: _Opts):
    """The clitic starting with ' at `pos`, if the chunk ends or a non-letter follows it.

    Catches the `'s` left after a vocabulary match, masked word or digit run
    (`covid-19's`, `90's`), which would otherwise leave a stray word `s`.
    """
    for cl in o.clitics:
        end = pos + len(cl)
        if cl[0] == "'" and o.chars.lower(c[pos:end]) == cl and not o.chars.letter.match(c, end):
            return cl
    return None


def _vocab_match(c: str, units, i: int, o: _Opts):
    """(index after the last unit, key, row) of the longest qualifying key from unit i, or None."""
    start = units[i][0]
    best = None
    for j in range(i, len(units)):
        cand = c[start : units[j][1]]
        if not any(o.vocab.has_prefix(f) for f in _case_forms(cand, o.cs, o.chars)):
            break
        if _qualifies(cand, o.chars):
            key, row = resolve(cand, o.vocab, o.cs, o.chars)
            if row >= 0:
                best = (j + 1, key, row)
    return best


def _qualifies(cand: str, chars: Chars) -> bool:
    if not chars.letter.search(cand) or not chars.special.search(cand):
        return False
    if chars.ends_alnum.search(cand):
        return True
    # A final period only with another before it: keeps `u.s.`, rejects junk keys like `it.`.
    return cand.endswith(".") and "." in cand[:-1]


def _masked_match(c: str, us: int, o: _Opts):
    run = o.chars.mask_run.match(c, us).group()
    if "*" not in run:
        return None
    low = o.chars.lower(run)
    if len(low) != len(run):
        return None
    hits = [w for w in o.by_len.get(len(run), ()) if all(r == "*" or r == x for r, x in zip(low, w))]
    if len(hits) != 1:
        return None
    key, row = resolve(hits[0], o.vocab, o.cs, o.chars)
    return key, row, us + len(run)


def _word_tokens(c: str, ws: int, we: int, o: _Opts):
    word = c[ws:we]
    cut = we
    for cl in o.clitics:
        if len(word) > len(cl) and o.chars.lower(word[-len(cl) :]) == cl:
            cut = we - len(cl)
            break
    yield from _stem_tokens(c, ws, cut, o)
    if cut < we:
        key, row = resolve(c[cut:we], o.vocab, o.cs, o.chars)
        yield key, row, cut, we


def _stem_tokens(c: str, ss: int, se: int, o: _Opts):
    stem = c[ss:se]
    key, row = resolve(stem, o.vocab, o.cs, o.chars)
    if row >= 0 or "-" not in stem:
        yield key, row, ss, se
        return
    pos = ss
    for part in stem.split("-"):
        if part:  # a part has no '-', so this recursion ends at the next level
            yield from _word_tokens(c, pos, pos + len(part), o)
        pos += len(part) + 1


SELFTEST_ABTT = (0, 1, 3)


def _token_fields(t: Tokens) -> dict:
    return {
        "sents": [list(s) for s in t.sents],
        "keys": list(t.keys),
        "spans": [list(s) for s in t.spans],
        "sent": list(t.sent),
        "rows": list(t.rows),
    }


def make_selftest(lines: list[str], tokenize_fn, rows_fn, n: int) -> dict:
    """Self-test content: token output per line, rows 0 and n-1 at SELFTEST_ABTT."""
    texts = [{"text": line, **_token_fields(tokenize_fn(line))} for line in lines]
    vectors = [
        {"row": r, "abtt": m, "values": [float(v) for v in rows_fn([r], m)[0]]}
        for r in sorted({0, n - 1})
        for m in SELFTEST_ABTT
    ]
    return {"texts": texts, "vectors": vectors}


_DTYPES = {"F32": "<f4", "I32": "<i4", "I8": "i1", "U8": "u1"}
_REQUIRED_TENSORS = {"abtt", "norms", "vocab_order", "matrix", "vocab", "stopwords", "masked", "chars", "selftest"}


def read_header(f, label: str) -> tuple[int, dict, dict]:
    """(start of tensor data, tensor entries, raw metadata) of a safetensors file."""
    f.seek(0)
    raw = f.read(8)
    if len(raw) != 8:
        raise WordPackError(f"{label}: file is truncated (no header)")
    (size,) = struct.unpack("<Q", raw)
    # A file that is not safetensors gives a huge size here; check it before reading.
    if size > os.fstat(f.fileno()).st_size - 8:
        raise WordPackError(f"{label}: not a safetensors file, or truncated (header)")
    try:
        tensors = json.loads(f.read(size))
    except ValueError as e:
        raise WordPackError(f"{label}: not a safetensors file (header is not JSON)") from e
    if not isinstance(tensors, dict):
        raise WordPackError(f"{label}: not a safetensors file (header is not an object)")
    meta = tensors.pop("__metadata__", {})
    if not isinstance(meta, dict):
        raise WordPackError(f"{label}: not a safetensors file (__metadata__ is not an object)")
    return 8 + size, tensors, meta


def _decode_meta(raw: dict, label: str) -> dict:
    meta = dict(raw)
    for key in ("case_sensitive", "clitics"):
        if key not in meta:
            raise WordPackError(f"{label}: missing metadata {key!r}")
    for key in ("clitics", "source"):
        if key in meta:
            try:
                meta[key] = json.loads(meta[key])
            except (TypeError, ValueError) as e:
                raise WordPackError(f"{label}: metadata {key!r} is not valid JSON") from e
    meta["case_sensitive"] = meta["case_sensitive"] == "true"
    return meta


def _check_meta(meta: dict, label: str) -> None:
    if meta.get("format") != "wordpack":
        raise WordPackError(f"{label} is not a wordpack (format={meta.get('format')!r})")
    if meta.get("format_version") != str(FORMAT_VERSION):
        raise WordPackError(
            f"{label} has format version {meta.get('format_version')}; this reader reads format version {FORMAT_VERSION}"
        )
    if meta.get("keys") != "surface":
        # A lemma-keyed pack needs a form table this reader does not apply; it would tokenize silently wrong.
        raise WordPackError(f"{label} has keys={meta.get('keys')!r}; this reader reads surface-keyed packs only")


def _lines(data: bytes) -> list[str]:
    return data.decode("utf-8").split("\n")[:-1]


def check_selftest(data: dict, tokenize_fn, rows_fn, label: str) -> None:
    """Raise WordPackError at the first difference from the pack's stored self-test."""
    prefix = f"{label}: self-test failed (reader format version {FORMAT_VERSION})"
    for i, entry in enumerate(data["texts"]):
        got = _token_fields(tokenize_fn(entry["text"]))
        for field, value in got.items():
            if value != entry[field]:
                raise WordPackError(
                    f"{prefix}: text {i} {entry['text']!r}, field {field!r}: expected {entry[field]}, got {value}"
                )
    for v in data["vectors"]:
        got = rows_fn([v["row"]], v["abtt"])[0]
        diff = float(np.abs(got - np.asarray(v["values"])).max(initial=0.0))
        if not diff <= 1e-5:  # `not <=`, so a NaN difference fails too
            raise WordPackError(f"{prefix}: row {v['row']} at abtt={v['abtt']} differs by {diff:.2e}")


class WordPack:
    """An open .wordpack file. Rows are read from the file on request."""

    def __init__(self, path):
        self.path = str(path)
        self._f = open(self.path, "rb")
        # Closes the file when the pack is collected; a weakref callback runs before the file object's own finalizer, so no unclosed-file warning.
        weakref.finalize(self, self._f.close)
        self._lock = threading.Lock()
        try:
            self._load()
        except BaseException:
            self._f.close()
            raise

    def _load(self) -> None:
        self._base, self._tensors, raw_meta = read_header(self._f, self.path)
        _check_meta(raw_meta, self.path)
        self.meta = _decode_meta(raw_meta, self.path)
        missing = _REQUIRED_TENSORS - self._tensors.keys()
        if missing:
            raise WordPackError(f"{self.path}: missing tensors {sorted(missing)}")
        for name, entry in self._tensors.items():
            self._check_entry(name, entry)
        # Tensors may be stored in any order, so the file's end is the largest tensor end.
        data_end = self._base + max(e["data_offsets"][1] for e in self._tensors.values())
        if data_end > os.fstat(self._f.fileno()).st_size:
            raise WordPackError(f"{self.path}: file is truncated")
        self.n, self.dims = self._tensors["matrix"]["shape"]
        self._matrix_offset = self._base + self._tensors["matrix"]["data_offsets"][0]
        self._abtt = np.ascontiguousarray(self._array("abtt"))
        try:
            self.keys = Vocab(self._bytes("vocab"), self._array("vocab_order"))
        except WordPackError as e:
            raise WordPackError(f"{self.path}: {e}") from e
        self.stopwords = _lines(self._bytes("stopwords"))
        self.masked = _lines(self._bytes("masked"))
        self.chars = Chars(json.loads(self._bytes("chars")))
        self._selftest = json.loads(self._bytes("selftest"))
        # One set of options per pack, so the chunk cache lives as long as the pack.
        self._opts = _opts(self.keys, self.chars, self.meta["case_sensitive"], self.meta["clitics"], self.masked)

    def _check_entry(self, name: str, entry) -> None:
        if not isinstance(entry, dict) or not {"dtype", "shape", "data_offsets"} <= entry.keys():
            raise WordPackError(f"{self.path}: tensor {name!r} needs dtype, shape and data_offsets")
        if not isinstance(entry["dtype"], str) or entry["dtype"] not in _DTYPES:
            raise WordPackError(f"{self.path}: tensor {name!r} has unknown dtype {entry['dtype']!r}")
        shape = entry["shape"]
        # bool is an int subclass, so it is excluded by name.
        if not (
            isinstance(shape, list)
            and all(type(v) is int and v >= 0 for v in shape)
            and (name != "matrix" or len(shape) == 2)
        ):
            raise WordPackError(f"{self.path}: tensor {name!r} has invalid shape {shape!r}")
        span = entry["data_offsets"]
        if not (
            isinstance(span, list)
            and len(span) == 2
            and all(type(v) is int and v >= 0 for v in span)
            and span[0] <= span[1]
        ):
            raise WordPackError(f"{self.path}: tensor {name!r} has invalid data_offsets {span!r}")
        if math.prod(shape) * np.dtype(_DTYPES[entry["dtype"]]).itemsize != span[1] - span[0]:
            raise WordPackError(f"{self.path}: tensor {name!r} shape {shape!r} does not match its {span[1] - span[0]} bytes")

    def _bytes(self, name: str) -> bytes:
        start, end = self._tensors[name]["data_offsets"]
        self._f.seek(self._base + start)
        data = self._f.read(end - start)
        if len(data) != end - start:
            raise WordPackError(f"{self.path}: file is truncated (tensor {name!r})")
        return data

    def _array(self, name: str) -> np.ndarray:
        entry = self._tensors[name]
        return np.frombuffer(self._bytes(name), dtype=_DTYPES[entry["dtype"]]).reshape(entry["shape"])

    def lookup(self, keys) -> np.ndarray:
        return np.array([self.keys.row(k) for k in keys], dtype=np.int64)

    def tokenize(self, text: str) -> Tokens:
        return _tokenize(text, self._opts)

    def read_rows(self, rows, abtt: int = 1) -> np.ndarray:
        """Float32 rows in the order asked, with ABTT `abtt`.

        Reads each contiguous run of sorted unique rows with one seek + read;
        no memory map, so it also works on Pyodide's lazy and worker file systems.
        """
        if not 0 <= abtt < len(self._abtt):
            raise ValueError(f"abtt must be between 0 and {len(self._abtt) - 1}, got {abtt}")
        idx = np.asarray(rows, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return np.empty((0, self.dims), dtype=np.float32)
        if idx.min() < 0 or idx.max() >= self.n:
            raise IndexError(f"rows must be between 0 and {self.n - 1}")
        uniq, inverse = np.unique(idx, return_inverse=True)
        q = np.empty((len(uniq), self.dims), dtype=np.int8)
        cuts = np.flatnonzero(np.diff(uniq) != 1) + 1
        starts = np.concatenate(([0], cuts))
        stops = np.concatenate((cuts, [len(uniq)]))
        with self._lock:  # one file position shared by all threads
            for a, b in zip(starts, stops):
                nbytes = int(b - a) * self.dims
                self._f.seek(self._matrix_offset + int(uniq[a]) * self.dims)
                data = self._f.read(nbytes)
                if len(data) != nbytes:
                    raise WordPackError(f"{self.path}: file is truncated (matrix)")
                q[a:b] = np.frombuffer(data, dtype=np.int8).reshape(-1, self.dims)
        return dequantize(q, self._abtt, abtt)[inverse.reshape(-1)]

    def selftest(self) -> None:
        label = f"{self.meta.get('name')}-{self.meta.get('version')} ({self.path})"
        check_selftest(self._selftest, self.tokenize, self.read_rows, label)

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> WordPack:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
