"""When find_k picks k_star=1, fit_pls reuses chain[0] — no extra
confirmatory call."""
from __future__ import annotations

from unittest.mock import patch

import plskit


def test_fit_pls_auto_kstar1_reuses_chain_p(ssd_instance):
    """If k_star == 1, the second pls1_confirmatory_test must NOT run."""
    real_ct = plskit.pls1_confirmatory_test
    real_fk = plskit.pls1_find_k_optimal
    call_count = {"ct": 0}

    def counting_ct(*a, **kw):
        call_count["ct"] += 1
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
    assert call_count["ct"] == 0, (
        f"expected 0 confirmatory calls (chain[0] should be reused), "
        f"got {call_count['ct']}"
    )
