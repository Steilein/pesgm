"""
third_order.py — o termo de terceira ordem contra explicações candidatas.

Reimplementa, a partir dos dados, os dois testes da Sec. V-B do artigo de
periódico, cujos números estavam no texto sem script que os produzisse:

1. **Catorze candidatos exógenos.** Cada um medido no caso base (A) do dia e,
   portanto, anterior ao efeito: horas congestionadas, utilização e fluxo médio
   do corredor, corte, partidas, carga, VRE disponível e sua razão com a carga,
   spread médio e spread nas horas congestionadas, os benefícios isolados dos
   três recursos e a razão hidrogênio/VPL. Correlação de postos (Spearman) com
   o coeficiente de terceira ordem do dia.

2. **Resíduo horário de terceira ordem.** A decomposição de Möbius aplicada ao
   cronograma horário de cada recurso em vez do custo: para o recurso r,
   ``p_r^{123} - p_r^{par1} - p_r^{par2} + p_r^{r}``, resumido pela soma dos
   valores absolutos (MWh). Mede quanto do que r faz só se explica pelos três
   juntos.

Uso::

    python papers/journal/scripts/third_order.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from interaction_stats import mobius_bounds  # noqa: E402

RES = ROOT / os.environ.get("RESULTS_DIR", "results_journal")
# Casos por máscara (src/runner.py): B=VPL, C=VDC, E=H2, D=VPL+VDC, F=VPL+H2, G=VDC+H2, H=todos
SINGLE = {"VPL": "B", "VDC": "C", "H2": "E"}
PAIRS = {"VPL": ("D", "F"), "VDC": ("D", "G"), "H2": ("F", "G")}


def third_order(df: pd.DataFrame) -> pd.Series:
    tab = mobius_bounds(df[df.variation == "base"])
    t = tab[tab.order == 3].set_index("day_key")
    return t["I"] / 1e3


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    b = df[df.variation == "base"].pivot_table(index="day_key", columns="case", values="cost_total")
    a = df[(df.variation == "base") & (df.case == "A")].set_index("day_key")
    out = pd.DataFrame({
        "congested_h": a.corridor_congested_hours,
        "corridor_util": a.corridor_utilization,
        "corridor_flow": a.corridor_flow_mean_mw,
        "curtailment": a.curtailment_pct,
        "startups": a.startups_total,
        "load": a.load_mwh,
        "vre_avail": a.vre_available_mwh,
        "vre_over_load": a.vre_available_mwh / a.load_mwh,
        "spread_mean": a.lmp_spread_mean,
        "spread_congested": a.lmp_spread_congested_mean,
        "dVPL": b.A - b.B,
        "dVDC": b.A - b.C,
        "dH2": b.A - b.E,
    })
    out["dH2_over_dVPL"] = out.dH2 / out.dVPL
    return out


def _net(case: str, day: str):
    import pypsa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pypsa.Network(str(RES / f"{day}__{case}__base.nc"))


def schedule(n, resource: str) -> np.ndarray:
    """Cronograma horário do recurso, empilhado por componente (MW)."""
    if resource == "VPL":
        su = n.storage_units.index
        return (n.storage_units_t.p_dispatch[su] - n.storage_units_t.p_store[su]).values
    if resource == "VDC":
        links = n.links[n.links.carrier == "VDC"]
        p = n.links_t.p0[links.index]
        return p.T.groupby(links.site).sum().T.values          # carga flexível por site
    if resource == "H2":
        p = n.links_t.p0[["H2_P2H", "H2_H2P"]].copy()
        if "H2_P2H_sb" in n.links.index:      # eletrolisador de três estados: + standby
            p["H2_P2H"] += n.links_t.p0["H2_P2H_sb"]
        return p.values
    raise ValueError(resource)


def residuals(days: list[str]) -> pd.DataFrame:
    rows = []
    for d in days:
        nets = {c: _net(c, d) for c in "BCDEFGH"}
        row = {"day_key": d}
        for r in ("VPL", "VDC", "H2"):
            p1, p2 = PAIRS[r]
            res = (schedule(nets["H"], r) - schedule(nets[p1], r) - schedule(nets[p2], r)
                   + schedule(nets[SINGLE[r]], r))
            row[r] = float(np.abs(res).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("day_key")


def main() -> None:
    df = pd.read_csv(RES / "metrics.csv")
    i3 = third_order(df)
    cand = candidates(df).loc[i3.index]
    print(f"terceira ordem: n={len(i3)}, media {i3.mean():+.2f}, mediana {i3.median():+.2f} k$/dia")
    rho = {c: spearmanr(cand[c], i3).statistic for c in cand}
    rho = pd.Series(rho).sort_values(key=np.abs, ascending=False)
    print("\nCandidatos exogenos (Spearman com I_3):")
    print(rho.round(2).to_string())
    print(f"\nmaior |rho|: {rho.abs().max():.2f}")
    print("dias extremos:", i3.sort_values().round(2).to_dict())
    print("horas congestionadas:", cand.congested_h.round(1).to_dict())

    res = residuals(list(i3.index))
    print("\nResiduo horario de terceira ordem (MWh) e Spearman com I_3:")
    for r in res:
        print(f"  {r}: rho = {spearmanr(res[r], i3).statistic:+.2f}")
    j = res.join(i3.rename("I3")).sort_values("I3")
    print(j.round(1).to_string())
    out = RES / "third_order.md"
    out.write_text("# Terceira ordem contra candidatos\n\nGerado por "
                   "`papers/journal/scripts/third_order.py`.\n\n## Candidatos exógenos\n\n"
                   + rho.round(3).to_frame("rho").to_markdown() + "\n\n## Resíduo horário\n\n"
                   + j.round(2).to_markdown() + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
