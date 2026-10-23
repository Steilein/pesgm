"""Eletrolisador de três estados (layers.add_h2 + constraints.add_h2_states).

Rede de duas barras sem linha entre elas: o hidrogênio é o único caminho da
energia barata a montante (horas 0-2 e 6-7) para a carga a jusante. Entre as
duas janelas o eletrolisador tem de escolher entre ficar em standby (paga
P^sb pela energia cara da barra a montante) e desligar (paga uma nova partida
a frio). O teste confere a escolha nos dois regimes de custo e as restrições
que definem os estados. Resolve com HiGHS em poucos segundos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from constraints import add_h2_states, h2_state_solution  # noqa: E402
from layers import add_h2                                  # noqa: E402

T = 8
CHEAP = [1, 1, 1, 0, 0, 0, 1, 1]          # renovável disponível a montante


def _cfg(startup_cost: float, mode: str = "three_state") -> dict:
    return {
        "corridor": {"bus_up": "U", "bus_down": "D"},
        "h2": {"p_nom_mw": 100.0, "fc_p_nom_mw": 100.0, "store_mwh": 1000.0,
               "eta_electrolyzer": 0.8, "eta_fuelcell": 0.6, "cyclic": True,
               "marginal_cost": 2.0, "mode": mode, "min_load": 0.16,
               "standby_load": 0.05, "startup_cost": startup_cost},
    }


def _solve(cfg: dict):
    n = pypsa.Network()
    n.set_snapshots(range(T))
    n.add("Bus", "U")
    n.add("Bus", "D")
    n.add("Generator", "vre", bus="U", p_nom=300.0, marginal_cost=0.0,
          p_max_pu=pd.Series(CHEAP, index=n.snapshots, dtype=float))
    # Preços escolhidos para que a decisão entre as janelas seja de fato a do
    # estado: produzir H2 com energia cara não compensa (80 x 0,48 = 38 < 100),
    # e manter a carga mínima custa 16 x 3 x (100 - 38) = 2957 $, mais que o
    # standby (5 x 3 x 100 = 1500 $). Sobra standby contra religar (K^su).
    n.add("Generator", "peaker_up", bus="U", p_nom=50.0, marginal_cost=100.0)
    n.add("Generator", "peaker_down", bus="D", p_nom=500.0, marginal_cost=80.0)
    n.add("Load", "load", bus="D", p_set=40.0)
    add_h2(n, cfg)
    n.optimize(solver_name="highs", extra_functionality=lambda n, s: add_h2_states(n, s, cfg))
    return n, h2_state_solution(n)


@pytest.fixture(scope="module")
def caro():
    """Partida a frio cara (2000 $) contra standby de 5 MW x 3 h x 100 $/MWh = 1500 $."""
    return _solve(_cfg(2000.0))


@pytest.fixture(scope="module")
def barato():
    """Partida a frio barata (100 $): desligar e religar sai mais em conta."""
    return _solve(_cfg(100.0))


def test_estados_mutuamente_exclusivos_e_carga_minima(caro):
    n, st = caro
    on, sb = np.array(st["z_on"]), np.array(st["z_sb"])
    assert (on + sb <= 1).all()
    p = n.links_t.p0["H2_P2H"].values
    assert (p[on == 1] >= 16.0 - 1e-6).all()          # P^min = 0,16 x 100
    assert np.allclose(p[on == 0], 0.0, atol=1e-6)     # só produz ligado
    p_sb = n.links_t.p0["H2_P2H_sb"].values
    assert np.allclose(p_sb, 5.0 * sb, atol=1e-6)      # P^sb = 0,05 x 100


def test_standby_nao_produz_hidrogenio(caro):
    n, _ = caro
    assert np.allclose(n.links_t.p1["H2_P2H_sb"].values, 0.0, atol=1e-9)


def test_partida_cara_mantem_em_standby_entre_as_janelas(caro):
    _, st = caro
    assert st["z_sb"][3:6] == [1, 1, 1]
    assert sum(st["z_su"]) == 0          # ciclo do dia sem nenhuma hora desligada


def test_partida_barata_desliga_entre_as_janelas(barato):
    _, st = barato
    assert sum(st["z_sb"]) == 0
    assert st["z_on"][3:6] == [0, 0, 0]
    assert sum(st["z_su"]) == 1          # uma religação, na hora 6


def test_partida_cobrada_no_objetivo(barato):
    """Mesmo despacho com e sem o custo de partida: a diferença é K^su x partidas."""
    n, st = barato
    custo_partidas = 100.0 * sum(st["z_su"])
    custos = (n.generators_t.p * n.generators.marginal_cost).sum().sum() \
        + (n.links_t.p0 * n.links.marginal_cost).sum().sum()
    assert float(n.objective) == pytest.approx(custos + custo_partidas, rel=1e-6)


def test_partida_ciclica_conta_a_virada_do_dia():
    """Desligado na última hora e ligado na primeira é uma partida."""
    n, st = _solve(_cfg(100.0))
    on = np.array(st["z_on"]) + np.array(st["z_sb"])
    esperado = int(((on - np.roll(on, 1)) > 0).sum())
    assert sum(st["z_su"]) == esperado


def test_modo_continuo_nao_cria_binarias():
    cfg = _cfg(100.0, mode="continuous")
    n, st = _solve(cfg)
    assert st is None and "H2_P2H_sb" not in n.links.index
