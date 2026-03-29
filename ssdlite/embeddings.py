"""Word embedding container: load, normalize, save."""

from __future__ import annotations

import gzip
import os
import pickle
import numpy as np

from ssdlite.utils.math import l2_normalize_rows_inplace


class Embeddings:
    """Stores word vectors and provides lookup / nearest-neighbor search.

    >>> emb = Embeddings.load("model.ssdembed")
    >>> emb.normalize(l2=True, abtt_m=1)
    >>> emb.save("model_norm")          # saves model_norm.ssdembed
    >>> emb.save(fmt="kv")              # saves model.kv  (needs gensim)
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
        self._normed_vectors: np.ndarray | None = None
        self._is_unit_normed: bool = False
        self._source_path: str | None = None

    # ---- construction ----

    @classmethod
    def load(cls, path: str, *, verbose: bool = False, parallel: bool = False) -> Embeddings:
        """Load embeddings from file. Auto-detects format by extension.

        Supports: .ssdembed, .kv, .bin, .txt, .vec (and .gz variants).

        Parameters
        ----------
        path : str
            Path to the embedding file.
        verbose : bool, default False
            If True, print progress information while loading.
        parallel : bool, default False
            If True, use multiprocess loading for .txt/.vec files.
            Ignored for other formats.

        Returns
        -------
        Embeddings
            A new Embeddings instance populated from the file.
        """
        emb = _load_embeddings(path, verbose=verbose, parallel=parallel)
        emb._source_path = path
        return emb

    # ---- normalization ----

    def normalize(self, *, l2: bool = True, abtt_m: int = 0, re_normalize: bool = True) -> Embeddings:
        """Normalize embeddings in-place. Returns self for chaining.

        Parameters
        ----------
        l2 : L2-normalize rows.
        abtt_m : Remove projection onto top-m principal components (ABTT).
        re_normalize : L2-normalize again after ABTT.
        """
        V = self.vectors
        if not V.flags.writeable:
            V = np.array(V)
            self.vectors = V

        if l2:
            l2_normalize_rows_inplace(V)

        if abtt_m > 0:
            V -= V.mean(axis=0)
            gram = V.T @ V
            eigvals, eigvecs = np.linalg.eigh(gram)
            m = min(abtt_m, eigvecs.shape[1])
            top = np.ascontiguousarray(eigvecs[:, -m:].T, dtype=V.dtype)
            del eigvals, eigvecs, gram
            coeffs = V @ top.T
            _CHUNK = 100_000
            for j in range(m):
                c = coeffs[:, j]
                for s in range(0, len(V), _CHUNK):
                    e = min(s + _CHUNK, len(V))
                    V[s:e] -= c[s:e, None] * top[j]

        if re_normalize:
            l2_normalize_rows_inplace(V)

        self._is_unit_normed = l2 or re_normalize
        self._norms = None
        self._normed_vectors = None
        self.fill_norms()
        return self

    # ---- internal helpers ----

    def fill_norms(self) -> None:
        """Precompute L2 norms."""
        self._norms = np.sqrt(np.einsum("ij,ij->i", self.vectors, self.vectors))
        self._normed_vectors = None

    @property
    def norms(self) -> np.ndarray:
        if self._norms is None:
            self.fill_norms()
        return self._norms  # type: ignore[return-value]

    def get_normed_vectors(self) -> np.ndarray:
        """Return L2-normalized vectors.

        Returns
        -------
        numpy.ndarray, shape (n_words, dim)
            Float array of unit-length row vectors. If the embeddings were
            already L2-normalized via :meth:`normalize`, returns the original
            vectors directly (no copy). Otherwise computes and caches
            normalized copies on first call.
        """
        if self._is_unit_normed:
            return self.vectors
        if self._normed_vectors is None:
            n = self.norms.copy()
            zero_mask = n < 1e-12
            n[zero_mask] = 1.0  # avoid division by zero
            normed = self.vectors / n[:, None]
            normed[zero_mask] = 0.0  # zero vectors stay zero
            self._normed_vectors = normed
        return self._normed_vectors

    # ---- lookup ----

    def __contains__(self, word: str) -> bool:
        return word in self.key_to_index

    def __len__(self) -> int:
        return len(self.index_to_key)

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.key_to_index[word]]

    def __repr__(self) -> str:
        return f"Embeddings({len(self)} words, {self.vector_size}d)"

    def get_vector(self, word: str, norm: bool = False) -> np.ndarray:
        """Return the vector for a word.

        Parameters
        ----------
        word : str
            Word to look up.
        norm : bool, default False
            If True, return the L2-normalized vector.

        Returns
        -------
        numpy.ndarray, shape (dim,)
            1-D vector for the requested word.

        Raises
        ------
        KeyError
            If *word* is not in the vocabulary.
        """
        idx = self.key_to_index[word]
        if norm:
            return self.get_normed_vectors()[idx]
        return self.vectors[idx]

    # ---- persistence ----

    @staticmethod
    def _stem(path: str) -> str:
        """Return filename without any extensions (everything before the first dot)."""
        base = os.path.basename(path)
        dot = base.find(".")
        if dot > 0:
            base = base[:dot]
        return os.path.join(os.path.dirname(path), base)

    _FORMATS = {"ssdembed", "kv", "bin", "txt"}

    def save(self, filename: str | None = None, fmt: str = "ssdembed") -> None:
        """Save embeddings.

        Parameters
        ----------
        filename : Output path **without** extension.  Defaults to the stem
                   of the file this instance was loaded from.
        fmt : ``"ssdembed"`` (default), ``"kv"`` (needs gensim), ``"bin"``,
              or ``"txt"``.

        Examples
        --------
        >>> emb.save("out/model_norm")              # → out/model_norm.ssdembed
        >>> emb.save("out/model_norm", fmt="kv")     # → out/model_norm.kv
        >>> emb.save(fmt="txt")                      # → <source_stem>.txt
        """
        if fmt not in self._FORMATS:
            raise ValueError(f"Unknown format {fmt!r}; choose from {sorted(self._FORMATS)}")
        if filename is None:
            if self._source_path is None:
                raise ValueError("filename is required (no source path to derive from)")
            filename = self._stem(self._source_path)
        path = f"{filename}.{fmt}"
        if fmt == "txt":
            self._save_text(path)
        elif fmt == "bin":
            self._save_binary(path)
        elif fmt == "kv":
            self._save_kv(path)
        else:
            self._save_pickle(path)

    def _save_kv(self, path: str) -> None:
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError(
                "gensim is required to save .kv files. "
                "Install it with: pip install ssdlite[gensim]"
            ) from None
        kv = KeyedVectors(vector_size=self.vector_size)
        kv.add_vectors(self.index_to_key, self.vectors)
        kv.save(path)

    def _save_pickle(self, path: str) -> None:
        npy_path = path + ".vectors.npy"
        np.save(npy_path, self.vectors)
        saved_vectors = self.vectors
        saved_source = self._source_path
        self.vectors = np.zeros((0, self.vector_size), dtype=np.float32)
        self._norms = None
        self._normed_vectors = None
        self._source_path = None
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            self.vectors = saved_vectors
            self._source_path = saved_source
            self._norms = None
            self._normed_vectors = None

    def _save_binary(self, path: str) -> None:
        with open(path, "wb") as f:
            header = f"{len(self.index_to_key)} {self.vector_size}\n"
            f.write(header.encode("utf-8"))
            for i, word in enumerate(self.index_to_key):
                f.write(word.encode("utf-8"))
                f.write(b" ")
                f.write(self.vectors[i].tobytes())
                f.write(b"\n")

    def _save_text(self, path: str) -> None:
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
            vocabulary (useful to limit results to most frequent words).

        Returns
        -------
        list of (str, float)
            Tuples of (word, cosine_similarity) sorted by descending
            similarity. Returns an empty list if the query is a zero vector.
        """
        vec = np.asarray(vector, dtype=np.float32)
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-12:
            return []
        vec = vec / vec_norm

        vecs = self.get_normed_vectors()
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
    """Load word2vec text or binary format."""
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

        for line in f:
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
    opener = gzip.open if is_gz else open
    words: list[str] = []

    with opener(path, "rb") as f:
        header = f.readline().decode("utf-8").strip()
        vocab_size, dim = (int(x) for x in header.split())

        mat = np.empty((vocab_size, dim), dtype=np.float32)
        for i in range(vocab_size):
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
    """Intercept gensim classes during unpickling."""

    def find_class(self, module: str, name: str) -> type:
        if "KeyedVectors" in name or "Word2VecKeyedVectors" in name:
            return _GensimKVShim
        if module.startswith(("gensim", "ssdiff")):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, ImportError):
                return _GensimKVShim
        return super().find_class(module, name)


class _GensimKVShim:
    """Temporary shim for gensim pickle state → Embeddings conversion."""

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
    """Load pickle-based embeddings: .ssdembed or .kv format."""
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
            shim._normed_vectors = None
        return shim

    # Duck-type: object from another package (e.g. ssdiff.embeddings.Embeddings)
    if hasattr(shim, "index_to_key") and hasattr(shim, "vectors"):
        vecs = shim.vectors
        if has_sidecar and _needs_sidecar(vecs):
            vecs = np.load(vectors_npy)
        return Embeddings(list(shim.index_to_key), np.asarray(vecs, dtype=np.float32))

    raise ValueError(f"Cannot load pickle embeddings: unexpected type {type(shim)}")


def _load_embeddings(path: str, *, verbose: bool = False, parallel: bool = False) -> Embeddings:
    """Load pre-trained word embeddings from file (auto-detect format)."""
    low = path.lower()
    ext = os.path.splitext(low)[1]

    if ext == ".ssdembed" or low.endswith(".ssdembed.gz"):
        return _load_pickle(path)
    if ext == ".kv" or low.endswith(".kv.gz"):
        return _load_pickle(path)
    if ext == ".bin" or low.endswith(".bin.gz"):
        return _load_text(path, binary=True, verbose=verbose, parallel=parallel)
    if ext in {".txt", ".vec"} or low.endswith(".txt.gz") or low.endswith(".vec.gz"):
        return _load_text(path, binary=False, verbose=verbose, parallel=parallel)
    if ext == ".gz":
        raise ValueError(
            f"Cannot determine embedding format for '{path}'. "
            "Rename to .txt.gz, .vec.gz, .bin.gz, or .ssdembed.gz."
        )
    return _load_pickle(path)
