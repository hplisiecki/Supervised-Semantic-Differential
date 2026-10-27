"""Word embedding container: load, normalize, save."""

from __future__ import annotations

import copy
import gzip
import os
import pickle
import re
import sys
import warnings

import numpy as np

from ssdiff.lang_config import WORDPACK_TAGS
from ssdiff.utils.math import l2_normalize_rows_inplace

# Rows of a wordpack kept in RAM: the most frequent rows, and the only ones neighbour search sees. It equals
# pca_sweep._RESTRICT_VOCAB and cluster_top_neighbors' restrict_vocab
# default, so clusters and the sweep's choice of K match a full load; a
# smaller prefix would change them. Change the three together.
_RAM_TOP_N: int = 50_000


class Embeddings:
    """Stores word vectors and provides lookup / nearest-neighbor search.

    L2-normalised rows are a precondition for ``SSD``; use
    ``.normalize(l2=True, abtt=N)`` after loading a raw embedding, or
    rely on ``Embeddings.load``'s autodetect when reloading an already-
    normalised file. A ``.wordpack`` loads normalised.

    >>> emb = Embeddings.load("model.txt").normalize(l2=True, abtt=1)
    """

    def __init__(self, keys: list[str] | tuple[str, ...], vectors: np.ndarray) -> None:
        """Create an Embeddings instance from a word list and vector matrix.

        Parameters
        ----------
        keys : list or tuple of str
            Words in the vocabulary, in the order matching *vectors* rows.
        vectors : numpy.ndarray, shape (n_words, dim)
            2-D array of word vectors (will be cast to float32).
        """
        self.index_to_key: list[str] = list(keys)
        self.vectors: np.ndarray = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim == 2 and len(self.index_to_key) != self.vectors.shape[0]:
            raise ValueError(
                f"len(keys)={len(self.index_to_key)} != vectors.shape[0]={self.vectors.shape[0]}"
            )
        self.key_to_index: dict[str, int] = {w: i for i, w in enumerate(self.index_to_key)}
        self.vector_size: int = self.vectors.shape[1] if self.vectors.ndim == 2 else 0
        self._norms: np.ndarray | None = None
        self.l2_normalized: bool = False
        self.abtt: int = 0
        self._source_path: str | None = None
        # Partial-load state, set by _load_wordpack for a pack over
        # _RAM_TOP_N rows; always present so downstream code can read it
        # unconditionally.
        self._partial: bool = False
        self._local_row: dict[int, int] | None = None
        self._extras: np.ndarray | None = None
        self._prefix_size: int | None = None
        # Wordpack state, set by _load_wordpack; None for every other format.
        self._pack = None
        self.name: str | None = None
        self.version: str | None = None
        self.lang: str | None = None
        self.stopwords: list[str] | None = None
        self.neighbor_filter: re.Pattern | None = None

    def __getstate__(self) -> dict:
        """Return pickle state, omitting large recomputable attributes.

        Drops ``key_to_index`` (rebuilt from ``index_to_key`` on load)
        and ``_norms`` to keep the pickle small.
        """
        if self._pack is not None:
            raise RuntimeError(
                "A wordpack embedding cannot be pickled; load the .wordpack file again instead. "
                "A result fit on it is pickled without it; use result.attach(embeddings=path) after loading."
            )
        state = self.__dict__.copy()
        state.pop("key_to_index", None)
        state.pop("_norms", None)
        return state

    def __deepcopy__(self, memo: dict) -> Embeddings:
        new = object.__new__(type(self))
        memo[id(self)] = new
        if self._pack is not None:
            # Shares rows and the open pack: rows are never written in place
            # (normalize() swaps in a new array), so the copy stays independent.
            new.__dict__.update(self.__dict__)
        else:
            new.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return new

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle state, migrating legacy attribute names.

        Handles legacy renames introduced before v1.0
        (``_l2_normalized`` → ``l2_normalized``, ``_abtt_m`` /
        ``abtt_m`` → ``abtt``) and drops obsolete attributes from older
        pickles (``_normed_vectors``, ``_is_unit_normed``).
        """
        if "_l2_normalized" in state and "l2_normalized" not in state:
            state["l2_normalized"] = state.pop("_l2_normalized")
        if "_abtt_m" in state and "abtt_m" not in state and "abtt" not in state:
            state["abtt_m"] = state.pop("_abtt_m")
        if "abtt_m" in state and "abtt" not in state:
            state["abtt"] = state.pop("abtt_m")
        # Drop attributes from older pickles that no longer exist.
        state.pop("_normed_vectors", None)
        state.pop("_is_unit_normed", None)
        self.__dict__.update(state)
        if not hasattr(self, "key_to_index"):
            self.key_to_index = {w: i for i, w in enumerate(self.index_to_key)}
        if not hasattr(self, "_norms"):
            self._norms = None
        for attr, default in (
            ("_partial", False),
            ("_local_row", None),
            ("_extras", None),
            ("_prefix_size", None),
            ("_pack", None),
            ("name", None),
            ("version", None),
            ("lang", None),
            ("stopwords", None),
            ("neighbor_filter", None),
        ):
            if not hasattr(self, attr):
                setattr(self, attr, default)

    # ---- construction ----

    @classmethod
    def load(
        cls,
        path: str | os.PathLike = "default",
        *,
        verbose: bool = False,
        parallel: bool = False,
    ) -> Embeddings:
        """Load embeddings from a file, or a wordpack by tag or full name.

        Supports: .wordpack, .kv, .bin, .txt, .vec (and .gz variants of the
        last four). The .ssdembed format is not supported.

        ``path`` is read in this order:

        1. A string with no dot that is a wordpack tag (``"en"``): the tag's
           message is printed to stderr and its pack is loaded. The tags are
           listed in ``ssdiff.lang_config.WORDPACK_TAGS``.
        2. A path where a file exists: the file is loaded by its extension.
        3. A string shaped like a wordpack's full name
           (``"en-glove2024-dolma-1"``): that pack is loaded.
        4. Anything else raises ``FileNotFoundError`` listing the tags.

        A pack is downloaded once into a cache directory (``WORDPACK_CACHE``,
        or ``~/.wordpacks``) and loaded from
        there, offline once it is cached. A tag means the same pack for a
        given ssdiff version; a full name always means the same file. With no
        argument, the tag ``"default"`` is loaded. A ``pathlib.Path`` is never
        read as a tag or a full name, and neither is a string with a dot, so
        ``"./en"`` loads a local file named ``en``.

        A ``.wordpack`` passes its self-test, loads at ABTT
        ``m = 1`` with L2-normalised rows, keeps its 50,000 most frequent
        rows in RAM and reads any other row from the file when needed, so
        the file must stay in place while the embedding is in use.

        After loading another format, rows are checked once: if every row
        has L2 norm ≈ 1 (tolerance 1e-5), ``l2_normalized`` is set to True.

        Parameters
        ----------
        path : str or os.PathLike, default "default"
            A wordpack tag, a file path, or a wordpack's full name.
        verbose : bool, default False
        parallel : bool, default False

        Returns
        -------
        Embeddings
        """
        from ssdiff.wordpack import fetch

        is_name = isinstance(path, str) and "." not in path
        if is_name and path in WORDPACK_TAGS:
            file, message = WORDPACK_TAGS[path]
            print(message, file=sys.stderr)
            return cls._load_wordpack(fetch.resolve(file.removesuffix(".wordpack")))
        path = os.fspath(path)
        low = path.lower()
        if low.removesuffix(".gz").endswith(".ssdembed"):
            raise ValueError(
                f"'{path}': the .ssdembed format is deprecated and no longer loads. "
                "Use a .wordpack, or a raw embedding (.kv, .bin, .txt, .vec)."
            )
        if not os.path.isfile(path):
            if is_name and fetch.is_full_name(path):
                return cls._load_wordpack(fetch.resolve(path))
            raise FileNotFoundError(f"No wordpack tag or file named {path!r}. Tags: {', '.join(WORDPACK_TAGS)}.")
        if low.endswith(".wordpack"):
            return cls._load_wordpack(path)

        emb = _load_embeddings(path, verbose=verbose, parallel=parallel)
        fmt = os.path.splitext(low.removesuffix(".gz"))[1] or "non-wordpack"
        warnings.warn(
            f"Loading a {fmt} embedding: build the text from token lists "
            "(`SSD(tokens, embeddings=emb)`). "
            "A .wordpack embedding tokenizes strings itself.",
            UserWarning,
            stacklevel=2,
        )
        emb._source_path = path
        if not emb.l2_normalized and emb.vectors.shape[0] > 0:
            norms = np.sqrt(np.einsum("ij,ij->i", emb.vectors, emb.vectors))
            if np.all(np.abs(norms - 1.0) < 1e-5):
                emb.l2_normalized = True
        return emb

    @classmethod
    def _load_wordpack(cls, path: str) -> Embeddings:
        """Open a .wordpack. See ``load``."""
        from ssdiff.wordpack.reader import WordPack

        pack = WordPack(path)
        try:
            pack.selftest()
            prefix = min(pack.n, _RAM_TOP_N)
            vectors = pack.read_rows(np.arange(prefix), abtt=1)
        except BaseException:
            pack.close()
            raise

        emb = cls([], np.empty((0, pack.dims), dtype=np.float32))
        # The pack's sorted key table answers lookups; a dict over 1.2M keys
        # would cost about 159 MB.
        emb.index_to_key = pack.keys
        emb.key_to_index = None
        emb.vectors = vectors
        emb.vector_size = pack.dims
        emb.l2_normalized = True
        emb.abtt = 1
        emb._source_path = path
        emb._pack = pack
        emb._prefix_size = prefix
        if pack.n > prefix:
            emb._partial = True
            emb._local_row = {i: i for i in range(prefix)}
        meta = pack.meta
        emb.name = meta.get("name")
        emb.version = meta.get("version")
        emb.lang = meta.get("language")
        emb.stopwords = pack.stopwords
        nf = meta.get("neighbor_filter")
        emb.neighbor_filter = re.compile(nf) if nf else None
        return emb

    # ---- normalization ----

    def normalize(self, *, l2: bool = True, abtt: int = 1) -> Embeddings:
        """Normalize embeddings (in place; a wordpack gets a new array). Returns self for chaining.

        Parameters
        ----------
        l2 : L2-normalize rows. Skipped if already applied.
        abtt : Target number of top principal components to remove (ABTT).
            Absolute: if ABTT was already applied with a smaller value, only
            the remaining components are removed.  Equal value is a no-op.
            Smaller value than already applied raises ValueError (ABTT is
            irreversible).
            On a wordpack, any value from 0 to 10, up or down: the pack
            stores its rows before ABTT with the mean and top 10 components,
            so the rows are rebuilt for the new value.

        Rows are always L2-normalized again after ABTT, and l2_normalized
        is then true.
        """
        if self._pack is not None:
            if not l2:
                raise ValueError(
                    "A wordpack stores unit-length rows and SSD needs them; "
                    "l2=False is not available."
                )
            if abtt != self.abtt:
                # A new array, never a write into the old one: views made by
                # with_corpus share the old array and keep the old ABTT.
                self.vectors = self._pack.read_rows(np.arange(self._prefix_size), abtt=abtt)
                self.abtt = abtt
                self._norms = None
                self._extras = None
                if self._partial:
                    self._local_row = {i: i for i in range(self._prefix_size)}
            return self
        V = self.vectors
        if not V.flags.writeable:
            V = np.array(V)
            self.vectors = V

        # --- L2 normalization ---
        did_l2 = False
        if l2 and not self.l2_normalized:
            l2_normalize_rows_inplace(V)
            self.l2_normalized = True
            did_l2 = True

        # --- ABTT ---
        did_abtt = False
        if abtt > 0:
            if abtt < self.abtt:
                raise ValueError(
                    f"Cannot reduce ABTT from {self.abtt} to {abtt}. "
                    "ABTT is irreversible — reload the original embeddings."
                )
            if abtt == self.abtt:
                warnings.warn(
                    f"ABTT with m={abtt} already applied. Skipping.",
                    stacklevel=2,
                )
            else:
                delta = abtt - self.abtt
                V -= V.mean(axis=0)
                gram = V.T @ V
                eigvals, eigvecs = np.linalg.eigh(gram)
                m = min(delta, eigvecs.shape[1])
                top = np.ascontiguousarray(eigvecs[:, -m:].T, dtype=V.dtype)
                del eigvals, eigvecs, gram
                coeffs = V @ top.T
                _CHUNK = 100_000
                for j in range(m):
                    c = coeffs[:, j]
                    for s in range(0, len(V), _CHUNK):
                        e = min(s + _CHUNK, len(V))
                        V[s:e] -= c[s:e, None] * top[j]
                self.abtt = abtt
                did_abtt = True

        # --- Re-normalize after ABTT ---
        if did_abtt:
            l2_normalize_rows_inplace(V)
            self.l2_normalized = True

        # --- Update state ---
        if did_l2 or did_abtt:
            self._norms = None
            self.fill_norms()

        return self

    def _read_extras(self, corpus) -> tuple[list[int], np.ndarray | None]:
        """Find corpus tokens above the rank-aligned prefix and read their rows.

        Returns ``(extra_indices, rows)`` where ``rows[k]`` is the vector for
        original vocab index ``extra_indices[k]``, read from the row source
        (see :meth:`_read_rows`). Returns ``([], None)`` when the corpus needs
        nothing beyond what is already materialised.
        """
        from ssdiff.utils.lexicon import _texts_to_token_lists

        flat_docs = _texts_to_token_lists(corpus.docs)
        cap = self.vectors.shape[0]  # rows already in the base matrix
        extras: list[int] = []
        for token in dict.fromkeys(t for doc in flat_docs for t in doc):
            oi = self._row(token)
            if oi < 0 or oi < cap or oi in self._local_row:
                continue
            extras.append(oi)
        if not extras:
            return [], None
        return extras, self._read_rows(extras)

    def _read_rows(self, idx) -> np.ndarray:
        """Rows *idx* (full-vocabulary indices) from the wordpack, at this embedding's ABTT.

        The wordpack reader serialises reads under its own lock, because all
        threads share one file position. ``os.pread`` would avoid the lock but
        does not exist on Windows.
        """
        return self._pack.read_rows(idx, abtt=self.abtt)

    def _local_vector(self, local: int) -> np.ndarray:
        """Resolve a local row index across the base matrix and the side buffer.

        Rows ``< self.vectors.shape[0]`` live in the (possibly shared) base
        matrix; rows above it live in ``self._extras`` (filled by
        :meth:`with_corpus`).
        """
        base_n = self.vectors.shape[0]
        if local < base_n:
            return self.vectors[local]
        return self._extras[local - base_n]

    def with_corpus(self, corpus) -> Embeddings:
        """Return a corpus-extended *view* without mutating ``self``.

        Embeddings from any format other than a wordpack return ``self``
        unchanged. A wordpack returns a shallow view: its :meth:`normalize`
        swaps in a new array, and the view keeps the one a fit was made on.
        For a wordpack over ``_RAM_TOP_N`` rows, the view shares the base
        prefix matrix and vocabulary *by reference* (no copy) and carries its
        own materialised corpus rows in a private ``_extras`` buffer. The view
        shares the open pack, so ``self`` and any view can produce further
        views for other corpora; a view of a view keeps the rows its parent
        already loaded.
        """
        if self._pack is None:
            return self
        view = object.__new__(type(self))
        view.__dict__.update(self.__dict__)  # shares vectors/vocab/pack by ref, no rebuild
        if not self._partial:
            return view
        extras, rows = self._read_extras(corpus)

        # Override view-owned state (everything else is shared read-only).
        view._norms = None
        view._local_row = dict(self._local_row)
        if extras:
            base_n = self.vectors.shape[0]
            n_prev = 0 if self._extras is None else len(self._extras)
            view._extras = rows if self._extras is None else np.vstack([self._extras, rows])
            for k, oi in enumerate(extras):
                view._local_row[oi] = base_n + n_prev + k
        return view

    # ---- internal helpers ----

    def fill_norms(self) -> None:
        """Precompute L2 norms."""
        self._norms = np.sqrt(np.einsum("ij,ij->i", self.vectors, self.vectors))

    @property
    def vocab_size(self) -> int:
        """Number of words in the vocabulary."""
        return len(self.index_to_key)

    @property
    def dim(self) -> int:
        """Embedding dimensionality (alias for vector_size)."""
        return self.vector_size

    @property
    def norms(self) -> np.ndarray:
        """Per-row L2 norms, computed lazily and cached.

        Returns
        -------
        numpy.ndarray, shape (n_words,)
            Float32 array of L2 norms for each word vector.
        """
        if self._norms is None:
            self.fill_norms()
        return self._norms  # type: ignore[return-value]

    # ---- lookup ----

    def _row(self, word) -> int:
        """Vocabulary row of *word*, or -1 when absent."""
        if self._pack is not None:
            return self._pack.keys.row(word) if isinstance(word, str) else -1
        return self.key_to_index.get(word, -1)

    def __contains__(self, word: str) -> bool:
        return self._row(word) >= 0

    def __len__(self) -> int:
        return len(self.index_to_key)

    def __getitem__(self, word: str) -> np.ndarray:
        oi = self._row(word)
        if oi < 0:
            raise KeyError(word)
        if self._partial:
            local = self._local_row.get(oi)
            if local is not None:
                return self._local_vector(local)
            # Read on request and not cached, so the embedding stays read-only
            # and threads can share it.
            return self._read_rows([oi])[0]
        return self.vectors[oi]

    def __repr__(self) -> str:
        header = (
            f"Embeddings  V={len(self):,}  D={self.vector_size}  "
            f"l2={self.l2_normalized}  abtt={self.abtt}"
        )
        arrays = "  arrays:  .vectors  .index_to_key"
        methods = (
            "  methods: .load(...)  .save(...)  .normalize(...)  "
            ".similar_by_vector(...)  .get_vector(...)"
        )
        return "\n".join([header, arrays, methods])

    def _repr_html_(self) -> str:
        """Render a plain-text repr inside a ``<pre>`` block for Jupyter."""
        import html as _html
        return f"<pre>{_html.escape(repr(self))}</pre>"

    def get_vector(self, word: str, norm: bool = False) -> np.ndarray:
        """Return the vector for a word.

        Parameters
        ----------
        word : str
            Word to look up.
        norm : bool, default False
            If True, return the (already unit) row from ``self.vectors``.
            Raises ``RuntimeError`` when ``not self.l2_normalized``.

        Returns
        -------
        numpy.ndarray, shape (dim,)
            1-D vector for the requested word.

        Raises
        ------
        KeyError
            If *word* is not in the vocabulary.
        RuntimeError
            If ``norm=True`` and ``self.l2_normalized`` is False.
        """
        if norm and not self.l2_normalized:
            raise RuntimeError(
                "get_vector(norm=True) requires L2-normalised embeddings. "
                "Call .normalize(l2=True) first."
            )
        return self[word]

    # ---- persistence ----

    @staticmethod
    def _stem(path: str) -> str:
        """Return filename without any extensions (everything before the first dot)."""
        base = os.path.basename(path)
        dot = base.find(".")
        if dot > 0:
            base = base[:dot]
        return os.path.join(os.path.dirname(path), base)

    _FORMATS = {"kv", "bin", "txt"}

    def save(self, filename: str | None = None, fmt: str = "bin") -> None:
        """Save embeddings.

        Parameters
        ----------
        filename : Output path **without** extension.  Defaults to the stem
                   of the file this instance was loaded from; raises when
                   that would overwrite the source file.
        fmt : ``"bin"`` (default), ``"kv"`` (needs gensim), or ``"txt"``.

        Examples
        --------
        >>> emb.save("out/model_norm")              # → out/model_norm.bin
        >>> emb.save("out/model_norm", fmt="kv")     # → out/model_norm.kv
        >>> emb.save(fmt="txt")                      # → <source_stem>.txt
        """
        if self._pack is not None:
            raise RuntimeError(
                "A wordpack embedding cannot be saved; load the .wordpack file again instead."
            )
        if fmt not in self._FORMATS:
            raise ValueError(f"Unknown format {fmt!r}; choose from {sorted(self._FORMATS)}")
        if filename is None:
            if self._source_path is None:
                raise ValueError("filename is required (no source path to derive from)")
            filename = self._stem(self._source_path)
            if os.path.abspath(f"{filename}.{fmt}") == os.path.abspath(self._source_path):
                raise ValueError(
                    f"filename is required: the default would overwrite the source file {self._source_path!r}"
                )
        if self.l2_normalized or self.abtt > 0:
            warnings.warn(
                f"Saving as .{fmt} — normalization and ABTT metadata will be lost.",
                stacklevel=2,
            )
        path = f"{filename}.{fmt}"
        if fmt == "txt":
            self._save_text(path)
        elif fmt == "bin":
            self._save_binary(path)
        else:
            self._save_kv(path)

    def _save_kv(self, path: str) -> None:
        """Write embeddings to gensim ``KeyedVectors`` pickle format."""
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError(
                "gensim is required to save .kv files. "
                "Install it with: pip install ssdiff[gensim]"
            ) from None
        kv = KeyedVectors(vector_size=self.vector_size)
        kv.add_vectors(self.index_to_key, self.vectors)
        kv.save(path)

    def _save_binary(self, path: str) -> None:
        """Write embeddings to word2vec binary format (.bin)."""
        with open(path, "wb") as f:
            header = f"{len(self.index_to_key)} {self.vector_size}\n"
            f.write(header.encode("utf-8"))
            for i, word in enumerate(self.index_to_key):
                f.write(word.encode("utf-8"))
                f.write(b" ")
                f.write(self.vectors[i].tobytes())
                f.write(b"\n")

    def _save_text(self, path: str) -> None:
        """Write embeddings to word2vec text format (.txt/.vec)."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{len(self.index_to_key)} {self.vector_size}\n")
            for i, word in enumerate(self.index_to_key):
                vec_str = " ".join(f"{v:.6g}" for v in self.vectors[i])
                f.write(f"{word} {vec_str}\n")

    # ---- similarity ----

    def similar_by_vector(
        self,
        vector: np.ndarray,
        topn: int = 10,
        restrict_vocab: int | None = None,
    ) -> list[tuple[str, float]]:
        """Return (word, cosine) pairs, most similar first.

        Parameters
        ----------
        vector : numpy.ndarray, shape (dim,)
            Query vector.
        topn : int, default 10
            Number of nearest neighbors to return.
        restrict_vocab : int or None, default None
            If set, only search the first *restrict_vocab* words in the
            vocabulary. On a wordpack over ``_RAM_TOP_N`` rows, clamped to
            the rows held in RAM; values above that emit a ``UserWarning``
            and are clamped down.

        Returns
        -------
        list of (str, float)

        Raises
        ------
        RuntimeError
            If ``self.l2_normalized`` is False.
        """
        if not self.l2_normalized:
            raise RuntimeError(
                "similar_by_vector requires L2-normalised embeddings. "
                "Call .normalize(l2=True) first."
            )
        vec = np.asarray(vector, dtype=np.float32)
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-12:
            return []
        vec = vec / vec_norm

        if self._partial:
            cap = self._prefix_size  # rank-aligned prefix; fixed at load time
            if restrict_vocab is None:
                restrict_vocab = cap
            elif restrict_vocab > cap:
                warnings.warn(
                    f"restrict_vocab={restrict_vocab} clamped to {cap}: only the "
                    f"{cap} most frequent wordpack rows are held in RAM and searchable.",
                    stacklevel=2,
                )
                restrict_vocab = cap

        vecs = self.vectors
        if restrict_vocab is not None:
            vecs = vecs[:restrict_vocab]
        if len(vecs) == 0:
            return []

        sims = vecs @ vec
        count = min(topn, len(sims))
        top_idx = np.argpartition(-sims, min(count, len(sims) - 1))[:count]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        keys = self.index_to_key
        return [(keys[i], float(sims[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# File loaders (private)
# ---------------------------------------------------------------------------


def _seek_to_line_start(f, start: int, is_continuation: bool) -> None:
    """Position *f* at the first complete line in a byte region.

    When *is_continuation* is True, the region may start mid-line.
    If the byte before *start* is not a newline, the partial line is skipped.
    """
    if is_continuation:
        f.seek(start - 1)
        if f.read(1) != b"\n":
            f.readline()
    else:
        f.seek(start)


def _count_lines_in_region(args: tuple) -> int:
    """Worker: count data lines in a byte region of a text embedding file."""
    path, start, end, is_continuation = args
    count = 0
    with open(path, "rb") as f:
        _seek_to_line_start(f, start, is_continuation)
        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            if b" " not in raw:
                continue
            count += 1
    return count


def _parse_into_shared(args: tuple) -> list[str]:
    """Worker: parse lines and write vectors into shared memory block."""
    from multiprocessing.shared_memory import SharedMemory

    path, start, end, dim, shm_name, row_offset, total_rows, is_continuation = args
    words: list[str] = []
    shm = SharedMemory(name=shm_name, create=False)
    try:
        mat = np.ndarray((total_rows, dim), dtype=np.float32, buffer=shm.buf)
        row = row_offset
        with open(path, "rb") as f:
            _seek_to_line_start(f, start, is_continuation)
            while f.tell() < end:
                raw = f.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="ignore").rstrip()
                sp = line.find(" ")
                if sp < 0:
                    continue
                words.append(line[:sp])
                mat[row] = np.fromstring(line[sp + 1 :], dtype=np.float32, sep=" ")
                row += 1
    finally:
        shm.close()
    return words


def _load_text(path: str, binary: bool = False, verbose: bool = False, parallel: bool = False) -> Embeddings:
    """Load word2vec text or binary format.

    Dispatches to :func:`_load_word2vec_binary` when *binary* is True.
    For text files, uses a parallel multi-process path (two-pass: count
    then parse into shared memory) when ``parallel=True`` and the file is
    not gzip-compressed.  Falls back to a simple serial reader otherwise.
    """
    if binary:
        return _load_word2vec_binary(path, is_gz=path.lower().endswith(".gz"), verbose=verbose)

    is_gz = path.lower().endswith(".gz")
    opener = gzip.open if is_gz else open

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
        toks = first_line.split()
        has_header = len(toks) == 2 and toks[0].isdigit() and toks[1].isdigit()
        if has_header:
            total: int | None = int(toks[0])
            dim = int(toks[1])
        else:
            total = None
            dim = len(toks) - 1

    n_cpus = os.cpu_count() or 1
    if parallel and not is_gz and n_cpus > 1:
        from concurrent.futures import ProcessPoolExecutor
        from itertools import accumulate
        from multiprocessing.shared_memory import SharedMemory

        n_workers = min(n_cpus, 4)
        file_size = os.path.getsize(path)

        with open(path, "rb") as f:
            if has_header:
                f.readline()
            data_start = f.tell()

        region_size = (file_size - data_start) // n_workers
        boundaries = [data_start + i * region_size for i in range(n_workers)] + [file_size]
        byte_regions = list(zip(boundaries[:-1], boundaries[1:]))

        # Pass 1: count lines per region
        count_args = [
            (path, start, end, i > 0)
            for i, (start, end) in enumerate(byte_regions)
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            counts = list(pool.map(_count_lines_in_region, count_args))

        total_rows = sum(counts)
        if total_rows == 0:
            return Embeddings([], np.empty((0, dim), dtype=np.float32))

        offsets = [0] + list(accumulate(counts))

        # Pass 2: parse into shared memory
        nbytes = total_rows * dim * 4  # float32
        shm = SharedMemory(create=True, size=nbytes)
        try:
            parse_args = [
                (path, start, end, dim, shm.name, offsets[i], total_rows, i > 0)
                for i, (start, end) in enumerate(byte_regions)
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                word_lists = list(pool.map(_parse_into_shared, parse_args))

            shared_view = np.ndarray((total_rows, dim), dtype=np.float32, buffer=shm.buf)
            mat = np.array(shared_view)  # copy into regular heap array
        finally:
            shm.close()
            shm.unlink()

        all_words = [w for wl in word_lists for w in wl]
        return Embeddings(all_words, mat)

    # Serial path
    from ssdiff.utils import _progress

    words: list[str] = []
    capacity = total if has_header else 100_000
    mat = np.empty((capacity, dim), dtype=np.float32)
    row = 0

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        if has_header:
            f.readline()  # skip header (already parsed above)
        else:
            # First line is data — already parsed as toks, reuse it
            words.append(toks[0])
            mat[0] = np.fromstring(first_line[first_line.index(" ") + 1:], dtype=np.float32, sep=" ")
            row = 1
            f.readline()  # skip first line (already processed above)

        lines = _progress(f, verbose=verbose, total=total, desc="Loading embeddings")
        for line in lines:
            sp = line.find(" ")
            if sp < 0:
                continue
            words.append(line[:sp])
            if row >= mat.shape[0]:
                new_mat = np.empty((mat.shape[0] * 2, dim), dtype=np.float32)
                new_mat[:row] = mat[:row]
                mat = new_mat
            mat[row] = np.fromstring(line[sp + 1:], dtype=np.float32, sep=" ")
            row += 1

    return Embeddings(words, mat[:row])


def _load_word2vec_binary(path: str, is_gz: bool = False, verbose: bool = False) -> Embeddings:
    """Load word2vec binary format (.bin)."""
    from ssdiff.utils import _progress

    opener = gzip.open if is_gz else open
    words: list[str] = []

    with opener(path, "rb") as f:
        header = f.readline().decode("utf-8").strip()
        vocab_size, dim = (int(x) for x in header.split())

        mat = np.empty((vocab_size, dim), dtype=np.float32)
        for i in _progress(range(vocab_size), verbose=verbose,
                           total=vocab_size, desc="Loading embeddings"):
            word_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch == b" " or ch == b"\t":
                    break
                if ch == b"\n" or ch == b"":
                    continue
                word_bytes.extend(ch)
            word = word_bytes.decode("utf-8", errors="ignore")
            mat[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            words.append(word)

    return Embeddings(words, mat)


class _GensimUnpickler(pickle.Unpickler):
    """Custom unpickler that intercepts gensim class lookups.

    Redirects ``KeyedVectors`` / ``Word2VecKeyedVectors`` and any other
    gensim class to :class:`_GensimKVShim` so that ``.kv`` files created
    by gensim can be loaded without gensim installed.
    """

    def find_class(self, module: str, name: str) -> type:
        if "KeyedVectors" in name or "Word2VecKeyedVectors" in name:
            return _GensimKVShim
        if module.startswith(("gensim", "ssdiff")):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, ImportError):
                return _GensimKVShim
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, ImportError):
            # numpy 2.0 renamed ``numpy.core`` -> ``numpy._core``. Pickles
            # written by numpy < 2.0 (and frozen/PyInstaller builds that omit
            # the legacy compat stub) reference the old path; redirect it.
            if module.startswith("numpy.core"):
                return super().find_class("numpy._core" + module[len("numpy.core"):], name)
            raise


class _GensimKVShim:
    """Temporary shim that absorbs a gensim pickle state and converts it.

    Accepts any gensim ``KeyedVectors`` state dict via ``__setstate__``
    and normalises legacy attribute names (``index2word``, ``syn0``) so
    that :meth:`to_embeddings` can always produce a valid
    :class:`Embeddings` instance.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if hasattr(self, "index2word") and not hasattr(self, "index_to_key"):
            self.index_to_key = self.index2word
        if hasattr(self, "index2entity") and not hasattr(self, "index_to_key"):
            self.index_to_key = self.index2entity
        if hasattr(self, "syn0") and not hasattr(self, "vectors"):
            self.vectors = self.syn0

    def to_embeddings(self) -> Embeddings:
        keys = list(self.index_to_key)
        vecs = np.asarray(self.vectors, dtype=np.float32)
        return Embeddings(keys, vecs)


def _needs_sidecar(vecs) -> bool:
    """True if vectors are missing/empty and should be loaded from .npy sidecar."""
    if vecs is None:
        return True
    if hasattr(vecs, "shape") and vecs.shape[0] == 0:
        return True
    return False


def _load_pickle(path: str) -> Embeddings:
    """Load pickle-based embeddings (.kv format)."""
    base = path[:-len(".gz")] if path.lower().endswith(".gz") else path
    vectors_npy = base + ".vectors.npy"
    has_sidecar = os.path.exists(vectors_npy)

    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rb") as f:
        shim = _GensimUnpickler(f).load()

    if isinstance(shim, _GensimKVShim):
        if has_sidecar and _needs_sidecar(getattr(shim, "vectors", None)):
            shim.vectors = np.load(vectors_npy)
        return shim.to_embeddings()

    if isinstance(shim, Embeddings):
        if has_sidecar and _needs_sidecar(shim.vectors):
            shim.vectors = np.load(vectors_npy)
            shim.vector_size = shim.vectors.shape[1]
            shim._norms = None
        return shim

    # Duck-type: object from another package (e.g. ssdiff.embeddings.Embeddings)
    if hasattr(shim, "index_to_key") and hasattr(shim, "vectors"):
        vecs = shim.vectors
        if has_sidecar and _needs_sidecar(vecs):
            vecs = np.load(vectors_npy)
        return Embeddings(list(shim.index_to_key), np.asarray(vecs, dtype=np.float32))

    raise ValueError(f"Cannot load pickle embeddings: unexpected type {type(shim)}")


def _load_embeddings(path: str, *, verbose: bool = False, parallel: bool = False) -> Embeddings:
    """Load pre-trained word embeddings from file, auto-detecting format by extension.

    Dispatches to :func:`_load_pickle` for ``.kv`` files,
    :func:`_load_text` for ``.bin`` (binary word2vec) and ``.txt``/``.vec``
    (text word2vec), applying gzip decompression transparently for ``.gz``
    variants.
    """
    low = path.lower()
    ext = os.path.splitext(low)[1]

    if ext == ".kv" or low.endswith(".kv.gz"):
        return _load_pickle(path)
    if ext == ".bin" or low.endswith(".bin.gz"):
        return _load_text(path, binary=True, verbose=verbose, parallel=parallel)
    if ext in {".txt", ".vec"} or low.endswith(".txt.gz") or low.endswith(".vec.gz"):
        return _load_text(path, binary=False, verbose=verbose, parallel=parallel)
    if ext == ".gz":
        raise ValueError(
            f"Cannot determine embedding format for '{path}'. "
            "Rename to .txt.gz, .vec.gz, .bin.gz, or .kv.gz."
        )
    return _load_pickle(path)
