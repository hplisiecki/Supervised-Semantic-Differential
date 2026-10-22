"""Results page: a result as the JSON object the HTML results page reads, and the page file.

The page itself lives in ``page_template.html``; the R package (ssdiff)
embeds the same file and writes the same JSON (``REPORT_FORMAT`` below).
"""

from __future__ import annotations

import datetime
import json
import math
import os
from importlib import resources

import numpy as np

from ssdiff.results.continuous_result import PCAOLSResult, PLSResult, RegressionResult
from ssdiff.results.contrast_result import ContrastResult
from ssdiff.results.core import _row_to_dict

REPORT_FORMAT = 2
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
    """A view that needs embeddings or a text raises RuntimeError without one (when its rows
    are computed, not when it is accessed); the page leaves it out."""
    try:
        return compute()
    except RuntimeError:
        return None


def _has_text(result) -> bool:
    text = getattr(result, "text", None)
    return text is not None and getattr(text, "pre_docs", None) is not None


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
    if raw is None and 0 <= doc_id < len(leaf.text.docs):
        return " ".join(leaf.text.docs[doc_id])
    return raw


def _docs_table(leaf, columns, rows: list[dict], rankings, doc_texts) -> dict:
    """``rows`` are dicts with ``doc_id`` and the given columns; adds ``text``."""
    ids = np.array([r["doc_id"] for r in rows], dtype=int)
    keep = _text_ids(ids, rankings, doc_texts) if getattr(leaf, "text", None) is not None else set()
    for r in rows:
        r["text"] = _doc_text(leaf, r["doc_id"]) if r["doc_id"] in keep else None
    return _table(list(columns) + ["text"], rows)


def _embeddings(result) -> dict | None:
    emb = getattr(result, "embeddings", None)
    if emb is None:
        pack = getattr(result, "_wordpack", None)
        if pack is None:
            return None
        return {"name": pack["name"], "version": pack["version"], "language": None,
                "n": None, "dims": None, "abtt": pack["abtt"]}
    name = emb.name
    if name is None and getattr(emb, "_source_path", None) is not None:
        name = os.path.basename(emb._source_path)
    return {"name": name, "version": emb.version, "language": emb.lang,
            "n": emb.vocab_size, "dims": emb.dim, "abtt": emb.abtt}


def _cluster_settings(leaf) -> dict | None:
    params = _maybe(lambda: leaf.clusters.pos.params)
    if params is None:
        return None
    return {k: v for k, v in params.items() if k != "side"}


def _settings(result, leaf) -> dict:
    return {"fit": getattr(result, "fit_args", None), "clusters": _cluster_settings(leaf)}


def _text_entry(result, kind, name=None) -> dict:
    """The ``texts`` entry of a result's text: its embeddings, concept settings, K, sweep and snippet flag.

    ``name`` is the text's name in a several-text fit (the keyword it was passed under).
    A contrast has no text: only its embeddings are filled."""
    if kind == "contrast":
        return {"name": None, "embeddings": _embeddings(result), "settings": None, "k": None,
                "sweep": None, "n_no_vector": None, "snippets": None}
    text = result.text
    sweep = result.sweep if kind == "pca_ols" else None
    return {
        "name": name or text.name,
        "embeddings": _embeddings(result),
        "settings": {
            "lexicon": sorted(text.lexicon) if text.lexicon else None,
            "window": text.window,
            "sif_a": text.sif_a,
            "use_full_doc": text.use_full_doc,
        },
        "k": result.fit_info.n_components if kind == "pca_ols" else None,
        "sweep": _view_table(sweep) if sweep is not None and len(sweep) else None,
        "n_no_vector": result.n_no_vector,
        "snippets": text.pre_docs is not None,
    }


def _scored_docs(leaf, doc_texts):
    """The docs table of a continuous leaf, and its rows as dicts."""
    rows = [_row_to_dict(d) for d in leaf.docs]
    table = _docs_table(
        leaf, list(leaf.docs._columns), rows,
        [[r["alignment_score"] for r in rows], [r["residual"] for r in rows]], doc_texts,
    )
    return table, rows


def _continuous(result, kind, sps, spc, doc_texts):
    tables = {"stats": _view_table(result.stats), "fit_info": _view_table(result.fit_info),
              "test": _view_table(result.test)}
    covariates = result.covariates
    if len(covariates):
        tables["covariates"] = _view_table(covariates)
    unit = _unit_tables(result, sps, spc)
    unit["docs"], rows = _scored_docs(result, doc_texts)
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


def _regression_leaves(result):
    """``(text name, key, leaf)`` per term leaf, in formula order, text by text."""
    if not result._by_text:
        return [(None, key, result[key]) for key in result.keys()]
    out = []
    for name, entry in result._leaves.items():
        if isinstance(entry, RegressionResult):
            out += [(name, key, leaf) for _, key, leaf in _regression_leaves(entry)]
        else:
            out.append((name, entry._term, entry))
    return out


def _regression(result, sps, spc, doc_texts):
    from ssdiff.backends.regression import label_of

    leaves = _regression_leaves(result)
    first = leaves[0][2]
    tables = {"stats": _view_table(result.stats), "fit_info": _view_table(first.fit_info),
              "test": _view_table(result.test)}
    if len(result.covariates):
        tables["covariates"] = _view_table(result.covariates)
    names = list(dict.fromkeys(name for name, _, _ in leaves))
    # A text's own row (label = its name) is not a term's test.
    term_labels = [t.label for t in result.test if t.label is not None and t.label not in names]
    units = []
    for name, key, leaf in leaves:
        unit = {"stats": _view_table(leaf.stats), "fit_info": _view_table(leaf.fit_info),
                **_unit_tables(leaf, sps, spc)}
        unit["docs"], _ = _scored_docs(leaf, doc_texts)
        units.append({"key": key, "label": key, "pos_label": "+β", "neg_label": "−β",
                      "text": names.index(name), "test": label_of(key, term_labels), "tables": unit})
    return tables, units, first


def _text_leaves(result):
    """``(name, first leaf)`` per text of a result; ``name`` is ``None`` unless the result has several texts."""
    if not isinstance(result, RegressionResult):
        return [(None, result)]
    first = {}
    for name, _, leaf in _regression_leaves(result):
        first.setdefault(name, leaf)
    return list(first.items())


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


def _contrast(result, kind, sps, spc, doc_texts):
    a, b = result.labels
    tables = {"stats": _view_table(result.stats)}
    units = [{"key": "contrast", "label": f"{a} \u2212 {b}", "pos_label": a, "neg_label": b,
              "tables": _unit_tables(result, sps, spc)}]
    return tables, units, result


def _kind(result) -> str:
    from ssdiff.results.group_result import GroupResult
    from ssdiff.results.multi_pls_result import MultiPLSResult
    for cls, name in ((PLSResult, "pls"), (PCAOLSResult, "pca_ols"), (RegressionResult, "pca_ols"),
                      (GroupResult, "groups"), (MultiPLSResult, "multipls"),
                      (ContrastResult, "contrast")):
        if isinstance(result, cls):
            return name
    raise TypeError(f"no results page for {type(result).__name__}")


def page_data(result, *, title=None, dataset=None, snippets_per_side=200,
              snippets_per_cluster=50, doc_texts=50) -> dict:
    """The results page's JSON object for ``result`` (plain values; ``json.dumps(..., allow_nan=False)`` works)."""
    import ssdiff

    kind = _kind(result)
    if isinstance(result, RegressionResult):
        tables, units, first_leaf = _regression(result, snippets_per_side, snippets_per_cluster, doc_texts)
    else:
        build = {"pls": _continuous, "pca_ols": _continuous, "groups": _groups, "multipls": _multipls,
                 "contrast": _contrast}[kind]
        tables, units, first_leaf = build(result, kind, snippets_per_side, snippets_per_cluster, doc_texts)
    data = {
        "report_format": REPORT_FORMAT,
        "kind": kind,
        "title": title or "SSD results",
        "created": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "software": {"package": "ssdiff", "language": "python", "version": ssdiff.__version__},
        "texts": _clean([_text_entry(leaf, kind, name) for name, leaf in _text_leaves(result)]),
        "settings": _clean(_settings(result, first_leaf)),
        "tables": tables,
        "units": units,
    }
    for u in units:
        u.setdefault("text", 0)
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
