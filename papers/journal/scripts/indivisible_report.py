"""
indivisible_report.py — workloads contínuos contra rígidos (Sec. V, "Indivisible Workloads").

Lê a grade principal do periódico (``results_journal/``, workloads contínuos) e
a sensibilidade de ``src/indivisible.py`` (``results_journal_rigid/``, workloads
rígidos e o FDC dos dois tipos), e compara, nos doze dias mensais:

* valor do FDC e do VDC sobre o caso A, e a fatia local (FDC/VDC);
* interação VPL x VDC e as que envolvem o hidrogênio, com a cota rigorosa;
* perda relativa de valor: da VPL com o VDC presente e do VDC com a VPL;
* partidas térmicas evitadas pelo VDC e energia deslocada.

Os casos sem data center flexível (A, B, E, F) são os mesmos nos dois modelos:
a referência é solução da variante rígida (ver ``layers.rigid_jobs``).

Uso::

    python papers/journal/scripts/indivisible_report.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from interaction_stats import mobius_bounds  # noqa: E402

MAIN = ROOT / os.environ.get("RESULTS_DIR", "results_journal")
RIGID = ROOT / os.environ.get("RIGID_DIR", "results_journal_rigid")
MONTHLY = [f"m{m:02d}" for m in range(1, 13)]


def _grids() -> dict[str, pd.DataFrame]:
    main = pd.read_csv(MAIN / "metrics.csv")
    main = main[(main.variation == "base") & main.day_key.isin(MONTHLY)]
    rig = pd.read_csv(RIGID / "metrics.csv")
    rig = rig[rig.day_key.isin(MONTHLY)]
    fixed = main[main.case.isin(list("ABEF"))]
    rigid = pd.concat([fixed, rig[rig.variation == "rigid"]], ignore_index=True)
    out = {"continuous": main.copy(), "rigid": rigid.assign(variation="base")}
    out["continuous_local"] = rig[rig.variation == "local"]
    out["rigid_local"] = rig[rig.variation == "rigid_local"]
    return out


def summary(g: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for model in ("continuous", "rigid"):
        df = g[model]
        c = df.pivot_table(index="day_key", columns="case", values="cost_total") / 1e3
        su = df.pivot_table(index="day_key", columns="case", values="startups_total")
        loc = g[f"{model}_local"].set_index("day_key").cost_total / 1e3
        days = c.dropna().index.intersection(loc.index)
        c, su, loc = c.loc[days], su.loc[days], loc.loc[days]
        dB, dC, dD = (c.A - c.B), (c.A - c.C), (c.A - c.D)
        I = dD - dB - dC
        tab = mobius_bounds(df[df.day_key.isin(days)])
        v = tab.groupby("subset")["verdict"].value_counts().unstack(fill_value=0)
        cm = df[df.case == "C"].set_index("day_key").loc[days]
        rows[model] = {
            "dias": len(days),
            "FDC_k": (c.A - loc).mean(), "VDC_k": dC.mean(), "fatia_local_pct": 100 * (c.A - loc).mean() / dC.mean(),
            "VPL_k": dB.mean(), "ambos_k": dD.mean(), "I_VPLxVDC_k": I.mean(),
            "perda_VPL_pct": 100 * -I.mean() / dB.mean(), "perda_VDC_pct": 100 * -I.mean() / dC.mean(),
            "VPLxVDC_subst": int(v.loc["VPLxVDC"].get("substitute", 0)),
            "VPLxVDC_compl": int(v.loc["VPLxVDC"].get("complement", 0)),
            "I_VDCxH2_k": tab[tab.subset == "VDCxH2"].I.mean() / 1e3,
            "VDCxH2_subst": int(v.loc["VDCxH2"].get("substitute", 0)),
            "I_3_k": tab[tab.subset == "VPLxVDCxH2"].I.mean() / 1e3,
            "partidas_evitadas_VDC": (su.A - su.C).mean(),
            "desloc_espacial_MWh": cm.vdc_spatial_shift_mwh.mean(),
            "desloc_temporal_MWh": cm.vdc_temporal_shift_mwh.mean(),
            "pagamento_flex_k": cm.cost_flex.mean() / 1e3,
        }
    return pd.DataFrame(rows)


def main() -> None:
    s = summary(_grids())
    s["rigido_sobre_continuo"] = s.rigid / s.continuous
    print(s.round(3).to_string())
    out = RIGID / "indivisible.md"
    out.write_text("# Workloads contínuos contra rígidos (doze dias mensais)\n\nGerado por "
                   "`papers/journal/scripts/indivisible_report.py`.\n\n" + s.round(3).to_markdown()
                   + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
