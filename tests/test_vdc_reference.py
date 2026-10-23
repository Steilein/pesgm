"""Referência do VDC que respeita a rampa do site (``vdc.reference: ramp_feasible``).

O EDF antigo viola a rampa agregada do site (DC_A1 cai 37,8 MW numa hora, DC_A2
sobe 40 MW, contra 20 MW/h), e o VDC contínuo pagava c^flex só para torná-la
executável. Com a referência nova:

* a energia, as janelas e a capacidade de cada job são as mesmas;
* a rampa por job e a rampa agregada do site são respeitadas nos dois sentidos;
* com preço plano, o VDC contínuo reproduz a referência e não paga nada.

Rede mínima como em ``test_vdc_rigid.py``: um gerador por barra de site, sem
linhas. Resolve com HiGHS em segundos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_profile, set_by_path                             # noqa: E402
from constraints import add_flex_cost, add_site_capacity, add_site_ramp   # noqa: E402
from layers import (_edf_schedule, _reference_by_job, _site_table,        # noqa: E402
                    add_vdc, generate_workloads)

PERFIL = [ROOT / "config" / "realistic.yaml", ROOT / "config" / "journal.yaml"]


def _cfg() -> dict:
    return set_by_path(load_profile(PERFIL), "vdc.reference", "ramp_feasible")


def test_referencia_respeita_rampas_janelas_e_energia():
    cfg = _cfg()
    wl, _ = generate_workloads(cfg)
    r = float(cfg["vdc"]["ramp_limit_pu"])
    for d, s in _site_table(cfg).iterrows():
        cap = float(s["flex_mw_it"])
        site_wl = [w for w in wl if w["native"] == d]
        por_job = _reference_by_job(site_wl, cap, cfg)
        X = sum(por_job.values())
        tol = 1e-6 * cap
        assert X.max() <= cap + tol
        assert np.abs(np.diff(X)).max() <= r * cap + tol, f"{d}: rampa agregada"
        for w in site_wl:
            x = por_job[w["id"]]
            fora = np.r_[x[:w["arrival"]], x[w["deadline"] + 1:]]
            assert np.abs(fora).max(initial=0.0) <= tol, f"{w['id']} fora da janela"
            assert abs(x.sum() - w["E_it_mwh"]) <= 1e-6 * max(1.0, w["E_it_mwh"])
            assert np.abs(np.diff(np.r_[0.0, x])).max() <= r * cap + tol, f"{w['id']}: rampa do job"


def test_edf_antigo_viola_a_rampa_e_a_nova_nao():
    """Guarda do motivo da mudança: se o EDF passar a respeitar a rampa, este
    teste avisa que a opção perdeu a razão de existir."""
    cfg = _cfg()
    wl, _ = generate_workloads(cfg)
    r = float(cfg["vdc"]["ramp_limit_pu"])
    violou = False
    for d, s in _site_table(cfg).iterrows():
        cap = float(s["flex_mw_it"])
        site_wl = [w for w in wl if w["native"] == d]
        violou |= np.abs(np.diff(_edf_schedule(site_wl, cap))).max() > r * cap * (1 + 1e-6)
    assert violou


@pytest.fixture(scope="module")
def plano():
    cfg = _cfg()
    sites = _site_table(cfg)
    n = pypsa.Network()
    n.set_snapshots(range(24))
    for d, s in sites.iterrows():
        n.add("Bus", s["bus"])
        n.add("Generator", f"g_{d}", bus=s["bus"], p_nom=1000.0,
              marginal_cost=pd.Series(30.0, index=n.snapshots))
    add_vdc(n, cfg)

    def ef(n, sns):
        add_site_capacity(n, sns, cfg)
        add_site_ramp(n, sns, cfg)
        add_flex_cost(n, sns, cfg)

    n.optimize(solver_name="highs", extra_functionality=ef)
    return n


def test_preco_plano_reproduz_a_referencia_sem_pagar(plano):
    links = plano.links[plano.links.carrier == "VDC"]
    for site, ref in plano.meta["vdc"]["reference_elec_mw"].items():
        P = plano.links_t.p0[links.index[links.site == site]].sum(axis=1).values
        assert np.clip(np.asarray(ref) - P, 0.0, None).sum() < 1e-3, f"{site} pagou c^flex"
