# tests/test_basic_pipeline.py


from ssdiff import SSD
import numpy as np
from gensim.models import KeyedVectors

def _tiny_kv():
    dim = 5
    words = ['kraj', 'naród', 'państwo', 'piękny', 'silny', 'zły', 'dobry']  # 7 words
    kv = KeyedVectors(vector_size=dim)

    rng = np.random.default_rng(0)
    mat = rng.normal(size=(len(words), dim)).astype(np.float32)
    # normalize rows to unit length (nice for cosine sims)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)

    kv.add_vectors(words, mat)  # shapes now match: (7, 5)
    return kv


def test_ssd_smoke():
    kv = _tiny_kv()
    docs = [["kraj","piękny"], ["naród","silny"], ["państwo","dobry"], ["kraj","zły"]]
    y = np.array([1.0, 1.2, 1.1, 0.8])
    lex = {"kraj","naród","państwo"}
    ssd = SSD(kv=kv, docs=docs, y=y, lexicon=lex, l2_normalize_docs=True, use_unit_beta=True, N_PCA=3)
    # basic attributes exist
    assert hasattr(ssd, "beta")
    assert hasattr(ssd, "beta_unit")
    assert hasattr(ssd, "r2")
    # per-essay scores compute
    scores = ssd.ssd_scores()
    assert len(scores) == len(docs)
