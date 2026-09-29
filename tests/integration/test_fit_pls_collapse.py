"""fit_pls(k='auto') always runs exactly one k=1 split_exact confirmatory
call — the diagnostic chain is never reused as the reported p-value."""
from __future__ import annotations

from unittest.mock import patch

import plskit


def test_fit_pls_auto_kstar1_runs_one_confirmatory(ssd_instance):
    """Even when k_star == 1, one split_exact confirmatory call runs."""
    real_ct = plskit.pls1_confirmatory_test
    real_fk = plskit.pls1_find_k_optimal
    calls = []

    def counting_ct(*a, **kw):
        calls.append(kw.get("method"))
        return real_ct(*a, **kw)

    def force_kstar1(*a, **kw):
        r = real_fk(*a, **kw)
        object.__setattr__(r, "k_star", 1)
        return r

    with patch("plskit.pls1_confirmatory_test", new=counting_ct), \
         patch("plskit.pls1_find_k_optimal", new=force_kstar1):
        res = ssd_instance.fit_pls(
            k="auto", k_max=3, n_splits=20, random_state=0,
        )

    assert res.fit_info.n_components == 1
    assert res.fit_info.p_at_k == 1
    assert calls == ["split_exact"], (
        f"expected exactly one split_exact confirmatory call, got {calls}"
    )
