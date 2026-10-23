"""Variante rígida dos workloads do VDC (layers.rigid_jobs + constraints.add_vdc_rigid).

Rede mínima: as três barras dos sites, cada uma com um gerador próprio e sem
linhas entre elas. Com o preço igual em todas as horas não há o que ganhar
deslocando, e a solução tem de ser a própria referência; com energia mais
barata num site e em poucas horas, os jobs migráveis têm de ir inteiros para
lá. Em todos os casos cada job roda num único site com a forma de referência.
Resolve com HiGHS em segundos.
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
from constraints import add_flex_cost, add_site_capacity, add_site_ramp, add_vdc_rigid  # noqa: E402
from layers import (_edf_schedule, _edf_schedule_by_job, _site_table,       # noqa: E402
                    add_vdc, generate_workloads)

PERFIL = [ROOT / "config" / "realistic.yaml", ROOT / "config" / "journal.yaml"]
CHEAP_SITE, CHEAP_HOURS = "DC_A3", range(9, 13)


def _cfg(spatial: bool = True) -> dict:
    cfg = load_profile(PERFIL)
    cfg = set_by_path(cfg, "vdc.workloads.rigid", True)
    return set_by_path(cfg, "vdc.spatial", spatial)


def _solve(cfg: dict, cheap: bool):
    sites = _site_table(cfg)
    n = pypsa.Network()
    n.set_snapshots(range(24))
    for d, s in sites.iterrows():
        n.add("Bus", s["bus"])
        mc = pd.Series(30.0, index=n.snapshots)
        if cheap and d == CHEAP_SITE:
            mc.iloc[list(CHEAP_HOURS)] = 0.0
        n.add("Generator", f"g_{d}", bus=s["bus"], p_nom=1000.0, marginal_cost=mc)
    add_vdc(n, cfg)

    def ef(n, sns):
        add_site_capacity(n, sns, cfg)
        add_site_ramp(n, sns, cfg)
        add_vdc_rigid(n, sns, cfg)
        add_flex_cost(n, sns, cfg)

    # Gap zero: a 0,01 % padrão o HiGHS aceita deslocar um job pequeno (custo de
    # dezenas de dólares num objetivo de ~400 mil), e o teste da referência falha
    # por tolerância, não por modelo.
    n.optimize(solver_name="highs", extra_functionality=ef, mip_rel_gap=0.0)
    return n


def _served(n, job: str) -> pd.DataFrame:
    """MW de TI entregues ao job, por site (colunas = links do job)."""
    lk = n.links.index[n.links.workload == job]
    return -n.links_t.p1[lk]


@pytest.fixture(scope="module")
def plano():
    return _solve(_cfg(), cheap=False)


@pytest.fixture(scope="module")
def incentivo():
    return _solve(_cfg(), cheap=True)


def test_forma_por_job_soma_a_referencia_do_site():
    cfg = load_profile(PERFIL)
    wl, _ = generate_workloads(cfg)
    for d, s in _site_table(cfg).iterrows():
        site_wl = [w for w in wl if w["native"] == d]
        por_job = _edf_schedule_by_job(site_wl, float(s["flex_mw_it"]))
        assert np.allclose(sum(por_job.values()), _edf_schedule(site_wl, float(s["flex_mw_it"])))


def _confere_rigidez(n):
    jobs = n.meta["vdc"]["rigid"]
    for j, info in jobs.items():
        sv = _served(n, j)
        ativos = [c for c in sv if sv[c].abs().max() > 1e-6]
        assert len(ativos) == 1, f"{j} dividido entre sites: {ativos}"
        prof = sv[ativos[0]].values
        shape = np.asarray(info["shape"])
        s = int(np.flatnonzero(prof > 1e-6)[0])
        assert s in info["starts"]
        esperado = np.zeros(24)
        esperado[s:s + len(shape)] = shape
        assert np.allclose(prof, esperado, atol=1e-6), f"{j} fora da forma de referência"


def test_cada_job_inteiro_num_site_com_a_forma_de_referencia(plano, incentivo):
    _confere_rigidez(plano)
    _confere_rigidez(incentivo)


def test_sem_incentivo_nada_se_desloca(plano):
    """Preço plano: deslocar só custaria c^flex, então a carga de cada site
    reproduz a referência (trocas de horário entre jobs do mesmo site que deixam
    o site igual são equivalentes e não pagam nada)."""
    links = plano.links[plano.links.carrier == "VDC"]
    for site, ref in plano.meta["vdc"]["reference_elec_mw"].items():
        P = plano.links_t.p0[links.index[links.site == site]].sum(axis=1).values
        # 1e-4 MWh: a forma rígida descarta o resíduo do EDF abaixo de 0,1 kW
        # (DC_A2_B1 tem 9e-7 MW na última hora), que aparece aqui x PUE.
        assert np.clip(np.asarray(ref) - P, 0.0, None).sum() < 1e-4, f"{site} deslocou carga"


def test_com_incentivo_migraveis_vao_inteiros_para_o_site_barato(incentivo):
    jobs = incentivo.meta["vdc"]["rigid"]
    wl = {w["id"]: w for w in incentivo.meta["vdc"]["workloads"]}
    movidos = 0
    for j in jobs:
        if wl[j]["type"] != "migratable":
            continue
        sv = _served(incentivo, j)
        site = incentivo.links.at[[c for c in sv if sv[c].abs().max() > 1e-6][0], "site"]
        movidos += site == CHEAP_SITE and wl[j]["native"] != CHEAP_SITE
    assert movidos >= 1


def test_fdc_rigido_fica_no_site_nativo():
    n = _solve(_cfg(spatial=False), cheap=True)
    _confere_rigidez(n)
    assert n.links.is_native.astype(bool).all()
