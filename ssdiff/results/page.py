"""Results page: a result as the JSON object the HTML results page reads, and the page file.

The page itself lives in ``page_template.html``; R and the browser app carry
verbatim copies of it and write the same JSON (``REPORT_FORMAT`` below).
"""

from __future__ import annotations

import datetime
import json
import math
import os
from importlib import resources

import numpy as np

from ssdiff.results.core import _row_to_dict

REPORT_FORMAT = 1
_MARKER = "__SSD_DATA__"
_DROP_SNIPPET_COLS = ("text_lemmas",)  # never shown; repeats each whole document in full-doc mode


def _clean(v):
    """Plain JSON value: numpy scalars unwrapped, NaN/inf -> None, sets sorted, tuples -> lists."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, (set, frozenset)):
        return [_clean(x) for x in sorted(v, key=str)]
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    return str(v)


def _table(columns, rows) -> dict:
    columns = list(columns)
    return {"columns": columns, "rows": [[_clean(r.get(c)) for c in columns] for r in rows]}


def _view_table(view, drop=()) -> dict:
    return _table([c for c in view._columns if c not in drop], (_row_to_dict(r) for r in view))


def _maybe(compute):
    """A view that needs embeddings or a corpus raises RuntimeError without one (when its rows
    are computed, not when it is accessed); the page leaves it out."""
    try:
        return compute()
    except RuntimeError:
        return None


def _has_text(result) -> bool:
    corpus = getattr(result, "corpus", None)
    return corpus is not None and getattr(corpus, "pre_docs", None) is not None


def _unit_tables(leaf, snippets_per_side: int, snippets_per_cluster: int) -> dict:
    """Word, cluster and snippet tables of one leaf result (one unit of the page)."""
    t = {}
    words = _maybe(lambda: _view_table(leaf.words))
    if words is not None:
        t["words"] = words
    clusters = _maybe(lambda: _view_table(leaf.clusters))
    if clusters is not None:
        t["clusters"] = clusters
        t["cluster_words"] = _view_table(leaf.clusters.words)
    if clusters is not None and _has_text(leaf):
        from ssdiff.results.continuous_result import SnippetsView
        t["snippets"] = _view_table(leaf.snippets(top_per_side=snippets_per_side), drop=_DROP_SNIPPET_COLS)
        rows = [
            _row_to_dict(r)
            for side in ("pos", "neg")
            for r in leaf.cluster_snippets(side=side, top_per_cluster=snippets_per_cluster)(None)
        ]
        cols = [c for c in SnippetsView._columns if c not in _DROP_SNIPPET_COLS]
        t["cluster_snippets"] = _table(cols, rows)
    return t


def _text_ids(ids: np.ndarray, rankings, doc_texts: int | None) -> set[int]:
    """Doc ids whose text goes into the file: both ends of each ranking, ``doc_texts`` deep."""
    if doc_texts is None:
        return {int(i) for i in ids}
    if doc_texts <= 0:
        return set()
    out: set[int] = set()
    for values in rankings:
        order = np.argsort(np.asarray(values, dtype=float), kind="stable")
        out.update(int(i) for i in ids[order[:doc_texts]])
        out.update(int(i) for i in ids[order[-doc_texts:]])
    return out


def _doc_text(leaf, doc_id: int):
    from ssdiff.results.continuous_result import _lookup_raw_text
    raw = _lookup_raw_text(leaf, doc_id)
    return "\n\n".join(raw) if isinstance(raw, list) else raw


def _docs_table(leaf, columns, rows: list[dict], rankings, doc_texts) -> dict:
    """``rows`` are dicts with ``doc_id`` and the given columns; adds ``text``."""
    ids = np.array([r["doc_id"] for r in rows], dtype=int)
    keep = _text_ids(ids, rankings, doc_texts) if _has_text(leaf) else set()
    for r in rows:
        r["text"] = _doc_text(leaf, r["doc_id"]) if r["doc_id"] in keep else None
    return _table(list(columns) + ["text"], rows)


def _embeddings(result) -> dict | None:
    emb = getattr(result, "embeddings", None)
    if emb is None:
        pack = getattr(result, "_wordpack", None)
        if pack is None:
            return None
        return {"name": pack["name"], "version": pack["version"], "language": getattr(result, "lang", None),
                "n": None, "dims": None, "abtt": pack["abtt"]}
    name = emb.name
    if name is None and getattr(emb, "_source_path", None) is not None:
        name = os.path.basename(emb._source_path)
    return {"name": name, "version": emb.version, "language": emb.lang or getattr(result, "lang", None),
            "n": emb.vocab_size, "dims": emb.dim, "abtt": emb.abtt}


def _cluster_settings(leaf) -> dict | None:
    params = _maybe(lambda: leaf.clusters.pos.params)
    if params is None:
        return None
    return {k: v for k, v in params.items() if k != "side"}


def _settings(result, leaf) -> dict:
    return {
        "lexicon": sorted(result.lexicon) if result.lexicon else None,
        "window": result.window,
        "sif_a": result.sif_a,
        "use_full_doc": getattr(result, "use_full_doc", not result.lexicon),
        "fit": getattr(result, "fit_args", None),
        "clusters": _cluster_settings(leaf),
    }


def _continuous(result, kind, sps, spc, doc_texts):
    tables = {"stats": _view_table(result.stats), "fit_info": _view_table(result.fit_info),
              "test": _view_table(result.test)}
    if kind == "pca_ols" and len(result.sweep):
        tables["sweep"] = _view_table(result.sweep)
    covariates = result.covariates
    if len(covariates):
        tables["covariates"] = _view_table(covariates)
    unit = _unit_tables(result, sps, spc)
    cols = list(result.docs._columns)
    rows = [_row_to_dict(d) for d in result.docs]
    unit["docs"] = _docs_table(
        result, cols, rows,
        [[r["alignment_score"] for r in rows], [r["residual"] for r in rows]], doc_texts,
    )
    leaves = result.keys() if kind == "pls" else []
    units = [{"key": "main", "label": "β" if leaves else "", "pos_label": "+β", "neg_label": "−β",
              "tables": unit}]
    if leaves:
        # fit_pls(ssd_gradients=N): one more unit per rotated gradient, laid out as MultiPLS dimensions.
        tables["pls_info"] = _view_table(result.pls_info)
        doc_ids = [r["doc_id"] for r in rows]
        for key in leaves:
            leaf = result[key]
            dim = {"stats": _view_table(leaf.stats), **_unit_tables(leaf, sps, spc)}
            scores = np.asarray(leaf.alignment_scores)
            dim_rows = [{"doc_id": i, "alignment_score": s} for i, s in zip(doc_ids, scores)]
            dim["docs"] = _docs_table(leaf, ["doc_id", "alignment_score"], dim_rows, [scores], doc_texts)
            units.append({"key": key, "label": result._key_repr(key), "pos_label": "+", "neg_label": "−",
                          "tables": dim})
    return tables, units, result


def _groups(gr, kind, sps, spc, doc_texts):
    tables = {"stats": _view_table(gr.stats), "test": _view_table(gr.test), "pairs": _view_table(gr.pairs)}
    lab = lambda g: str(gr.group_labels.get(g, g))  # noqa: E731
    doc_ids = getattr(gr, "doc_ids", None)
    units = []
    for g1, g2 in gr.keys():
        leaf = gr[(g1, g2)]
        unit = _unit_tables(leaf, sps, spc)
        if doc_ids is not None:
            mask = leaf.groups_mask
            scores = np.asarray(leaf.alignment_scores)
            rows = [{"doc_id": int(i), "group": lab(g), "alignment_score": s}
                    for i, g, s in zip(doc_ids[mask], gr._groups[mask], scores)]
            unit["docs"] = _docs_table(leaf, ["doc_id", "group", "alignment_score"], rows, [scores], doc_texts)
        units.append({"key": leaf.contrast, "label": f"{lab(g1)} vs {lab(g2)}",
                      "pos_label": lab(g1), "neg_label": lab(g2), "tables": unit})
    return tables, units, gr[gr.keys()[0]]


def _multipls(mp, kind, sps, spc, doc_texts):
    tables = {"stats": _view_table(mp.stats), "pls_info": _view_table(mp.pls_info), "test": _view_table(mp.test)}
    doc_ids = getattr(mp, "doc_ids", None)
    units = []
    for key in mp.keys():
        leaf = mp[key]
        unit = {"stats": _view_table(leaf.stats), "outcomes": _view_table(leaf.outcomes),
                **_unit_tables(leaf, sps, spc)}
        if doc_ids is not None:
            scores = np.asarray(leaf.alignment_scores)
            rows = [{"doc_id": int(i), "alignment_score": s} for i, s in zip(doc_ids, scores)]
            unit["docs"] = _docs_table(leaf, ["doc_id", "alignment_score"], rows, [scores], doc_texts)
        units.append({"key": key, "label": mp._key_repr(key), "pos_label": "+", "neg_label": "−",
                      "tables": unit})
    return tables, units, mp[mp.keys()[0]]


def _kind(result) -> str:
    from ssdiff.results.continuous_result import PCAOLSResult, PLSResult
    from ssdiff.results.group_result import GroupResult
    from ssdiff.results.multi_pls_result import MultiPLSResult
    for cls, name in ((PLSResult, "pls"), (PCAOLSResult, "pca_ols"),
                      (GroupResult, "groups"), (MultiPLSResult, "multipls")):
        if isinstance(result, cls):
            return name
    raise TypeError(f"no results page for {type(result).__name__}")


def page_data(result, *, title=None, dataset=None, snippets_per_side=200,
              snippets_per_cluster=50, doc_texts=50) -> dict:
    """The results page's JSON object for ``result`` (plain values; ``json.dumps(..., allow_nan=False)`` works)."""
    import ssdiff

    kind = _kind(result)
    build = {"pls": _continuous, "pca_ols": _continuous, "groups": _groups, "multipls": _multipls}[kind]
    tables, units, first_leaf = build(result, kind, snippets_per_side, snippets_per_cluster, doc_texts)
    data = {
        "report_format": REPORT_FORMAT,
        "kind": kind,
        "title": title or "SSD results",
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "software": {"package": "ssdiff", "language": "python", "version": ssdiff.__version__},
        "embeddings": _clean(_embeddings(result)),
        "settings": _clean(_settings(result, first_leaf)),
        "tables": tables,
        "units": units,
    }
    if dataset is not None:
        data["dataset"] = _clean(dict(dataset))
    return data


def save_page(result, path="ssd_results.html", **kwargs) -> None:
    """Write ``result`` as a self-contained HTML results page (opens from disk, no network)."""
    data = page_data(result, **kwargs)
    # Every "<" becomes \u003c: no "</script>" can end the data element, and no "<!--<script"
    # can switch the HTML parser into a state where the real "</script>" no longer ends it.
    text = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = resources.files("ssdiff.results").joinpath("page_template.html").read_text(encoding="utf-8")
    with open(path, "w", encoding="utf-8") as f:
        f.write(template.replace(_MARKER, text, 1))
