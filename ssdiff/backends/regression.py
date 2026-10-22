"""Formula helpers for fit_regression — text terms ``ssd(...)`` in a formula.

A text term is written ``ssd()`` (the text of the calling object) or
``ssd(name)`` (a named text). Before glmm sees the formula, each text term is
replaced by a placeholder variable (``_ssd_text`` / ``_ssd_<name>``), whose
labels are read with ``glmm.term_labels``; null formulas for the model tests
are built from those labels, never by expanding ``*`` here.

Implements:
- Finding, appending and validating ``ssd(...)`` terms
- Placeholders and their expansion into PCA component columns
- Null formulas (text terms dropped, random effects and offsets kept)
- Grouping glmm coefficient names into one group per text term
"""

from __future__ import annotations

import re
from itertools import permutations

_SSD = re.compile(r"\bssd\(\s*(\w*)\s*\)")

# ---------------------------------------------------------------------------
# Finding text terms
# ---------------------------------------------------------------------------

def text_names(formula: str) -> list[str | None]:
    """Names inside ``ssd(...)`` in order of first appearance; ``None`` for ``ssd()``."""
    names: list[str | None] = []
    for m in _SSD.finditer(formula):
        name = m.group(1) or None
        if name not in names:
            names.append(name)
    return names


def with_text_term(formula: str) -> str:
    """Append `` + ssd()`` when the formula has no ``ssd(...)`` term."""
    if _SSD.search(formula):
        return formula
    return f"{formula} ssd()" if formula.rstrip().endswith("~") else f"{formula} + ssd()"


def _groups(formula: str, opener: str) -> list[str]:
    """Top-level parenthesized groups (parentheses included) that start at *opener*.

    *opener* is a regex matching what precedes the ``(``; nested parentheses
    are counted, so ``(1 + (x) | id)`` is one group.
    """
    out = []
    i = 0
    rx = re.compile(opener)
    while (m := rx.search(formula, i)):
        start = m.start()
        depth = 0
        for j in range(m.end() - 1, len(formula)):
            depth += formula[j] == "("
            depth -= formula[j] == ")"
            if depth == 0:
                break
        out.append(formula[start:j + 1])
        i = j + 1
    return out


def _random_groups(formula: str) -> list[str]:
    """Random-effect groups ``( … | … )`` as written, from the right-hand side."""
    rhs = formula.split("~", 1)[-1]
    return [g for g in _groups(rhs, r"\(") if "|" in g]


def check_text_terms(formula: str) -> None:
    """Raise ``ValueError`` for ``ssd(...)`` inside a ``( … | … )`` group."""
    for g in _random_groups(formula):
        if _SSD.search(g):
            raise ValueError(
                f"ssd(...) cannot be used inside a random-effect term: {g}"
            )

# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------

def placeholder(name: str | None) -> str:
    """Placeholder variable for a text term."""
    return "_ssd_text" if name is None else f"_ssd_{name}"


def to_placeholders(formula: str) -> str:
    """Replace every ``ssd(...)`` term by its placeholder."""
    return _SSD.sub(lambda m: placeholder(m.group(1) or None), formula)


def expand(formula_ph: str, columns: dict[str, list[str]]) -> str:
    """Replace each placeholder by ``(col1 + col2 + …)``."""
    pattern = "|".join(sorted(map(re.escape, columns), key=len, reverse=True))
    return re.sub(
        rf"(?<!\w)(?:{pattern})(?!\w)",
        lambda m: "(" + " + ".join(columns[m.group(0)]) + ")",
        formula_ph,
    )

# ---------------------------------------------------------------------------
# Null formulas
# ---------------------------------------------------------------------------

def null_formula(formula_ph: str, drop: set[str]) -> str:
    """The formula without the fixed terms in *drop*.

    Response, intercept removal, random-effect groups and ``offset(...)``
    terms are carried over as written; they are not fixed-term labels.
    """
    import glmm  # imported on use, so `import ssdiff` does not load glmm

    labels, has_intercept = glmm.term_labels(formula_ph)
    keep = [lab for lab in labels if lab not in drop]
    if not keep and not has_intercept:
        raise ValueError(
            f"'{formula_ph}' has no fixed term left once the text terms are "
            "dropped and the intercept is removed"
        )
    lhs = formula_ph.split("~", 1)[0].strip()
    rhs = formula_ph.split("~", 1)[-1]
    fixed = ["0"] * (not has_intercept) + (keep or ["1"])
    extras = _random_groups(formula_ph) + _groups(rhs, r"\boffset\s*\(")
    head = f"{lhs} ~ " if lhs else "~ "
    return head + " + ".join(fixed + extras)


def text_labels(labels: list[str], ph: str) -> list[str]:
    """Labels whose ``:``-separated variables include the placeholder *ph*."""
    return [lab for lab in labels if ph in lab.split(":")]

# ---------------------------------------------------------------------------
# Coefficient names
# ---------------------------------------------------------------------------

def term_groups(
    names: list[str], components: dict[str, tuple[str, int]]
) -> dict[str, list[tuple[int, int]]]:
    """Group coefficient names by text term.

    *components* maps a component column to ``(shown text term, component
    index)``. A name with a component column among its ``:``-separated parts
    gets as key the name with that part replaced by the shown term; its value
    is ``(component index, position in names)``, sorted by component index.
    """
    groups: dict[str, list[tuple[int, int]]] = {}
    for pos, name in enumerate(names):
        parts = name.split(":")
        for k, part in enumerate(parts):
            if part in components:
                shown, idx = components[part]
                key = ":".join(parts[:k] + [shown] + parts[k + 1:])
                groups.setdefault(key, []).append((idx, pos))
                break
    return {key: sorted(v) for key, v in groups.items()}


def label_of(key: str, labels_shown: list[str]) -> str | None:
    """The label that produced the leaf *key*, or ``None``.

    Same number of ``:`` parts, paired one to one: text parts (``ssd(...)``)
    must be equal, every other key part must start with its label variable
    (``sexM`` with ``sex``). When several labels fit (``age`` and ``ageGroup``
    for ``ageGroupB``), the one with the longest variables wins.
    """
    parts = key.split(":")
    fits = []
    for lab in labels_shown:
        lparts = lab.split(":")
        if len(lparts) != len(parts):
            continue
        for perm in permutations(lparts):
            if all(
                kp == lp if lp.startswith("ssd(") else kp.startswith(lp)
                for kp, lp in zip(parts, perm)
            ):
                fits.append(lab)
                break
    return max(fits, key=len, default=None)
