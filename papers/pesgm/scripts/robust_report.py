"""
robust_report.py — resultados dos testes de ``src/robustness_pesgm.py``.

Cada variante só resolveu os casos que ela muda; os outros vêm da grade
principal (``results_pesgm/``) ou da varredura do corredor
(``results_pesgm_corridor/``). Aqui eles se juntam e passam pela mesma
``mobius_bounds`` do artigo. Saída: ``results_pesgm_robust/robust_report.md``.

Uso::

    python papers/pesgm/scripts/robust_report.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from interaction_stats import mobius_bounds  # noqa: E402

MONTHLY = [f"m{m:02d}" for m in range(1, 13)]
D4 = ["spring_wind", "summer_peak", "solar_ramp", "median"]
SUBSETS = ["VPLxVDC", "VPLxH2", "VDCxH2", "VPLxVDCxH2"]


def _read(d: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / d / "metrics.csv").drop_duplicates("scenario", keep="last")


def _summary(tab: pd.DataFrame) -> pd.DataFrame:
    g = tab.groupby("subset").agg(mean_k=("I", "mean"), median_k=("I", "median"))
    g[["mean_k", "median_k"]] /= 1e3
    v = tab.groupby(["subset", "verdict"]).size().unstack(fill_value=0)
    return g.join(v).reindex(SUBSETS).round(3)


def main() -> None:
    warnings.simplefilter("ignore", RuntimeWarning)
    main_ = _read("results_pesgm")
    main_ = main_[(main_.variation == "base")]
    rob = _read("results_pesgm_robust")
    cor = _read("results_pesgm_corridor")
    L = ["# Testes de robustez do PES GM (parecer simulado, 23/10/2026)", "",
         "Gerado por `papers/pesgm/scripts/robust_report.py`.", ""]

    base_m = main_[main_.day_key.isin(MONTHLY)]
    ref = mobius_bounds(base_m)
    L += ["## Referência (grade principal, doze dias mensais)", "", _summary(ref).to_markdown(), ""]

    def value(df: pd.DataFrame, a: str, b: str) -> float:
        c = df.groupby("case").cost_total.mean()
        return (c[a] - c[b]) / 1e3

    for var, keep in (("vpl_emergent", list("ACEG")), ("h2_su10", list("ABCD")), ("h2_sb1", list("ABCD"))):
        new = rob[rob.variation == var]
        if new.empty:
            L += [f"## {var}: sem resultados", ""]
            continue
        # Só os dias em que a variante tem os casos completos, e a referência
        # nesses MESMOS dias: comparar 8 dias com 12 confunde variante e amostra.
        dias = sorted(d for d, g in new.groupby("day_key") if set(g.case) >= set("ABCDEFGH") - set(keep))
        comb = pd.concat([base_m[base_m.case.isin(keep)], new.assign(variation="base")])
        comb = comb[comb.day_key.isin(dias)]
        same = base_m[base_m.day_key.isin(dias)]
        tab = mobius_bounds(comb)
        L += [f"## {var} ({len(dias)} dias completos)", "", _summary(tab).to_markdown(), "",
              "Referência nos mesmos dias:", "", _summary(mobius_bounds(same)).to_markdown(), "",
              f"Valor do VPL sozinho: {value(comb, 'A', 'B'):.2f} k$/dia "
              f"(grade principal {value(same, 'A', 'B'):.2f}); do H2 sozinho: {value(comb, 'A', 'E'):.2f} "
              f"(grade principal {value(same, 'A', 'E'):.2f}).", ""]
        if var.startswith("h2"):
            e = new[new.case == "E"]
            L += [f"Eletrolisador (caso E): partidas a frio {int(e.h2_startups.sum())}, horas produzindo "
                  f"{e.h2_on_h.mean():.1f}, em standby {e.h2_standby_h.mean():.1f}, standby/eletrólise "
                  f"{100 * e.h2_standby_mwh.sum() / e.h2_p2h_mwh.sum():.1f} %.", ""]

    # teste topológico: célula a montante
    fc = rob[rob.variation.str.startswith("fc_up")]
    if not fc.empty:
        rows = []
        for d in D4:
            for lvl in (600, 900, 1200, 1500):
                if lvl == 900:
                    ab = main_[(main_.day_key == d) & main_.case.isin(["A", "B"])]
                else:
                    ab = cor[(cor.day_key == d) & (cor.variation == f"s{lvl}") & cor.case.isin(["A", "B"])]
                ef = fc[(fc.day_key == d) & (fc.variation == f"fc_up_s{lvl}")]
                comb = pd.concat([ab, ef]).assign(variation="x")
                t = mobius_bounds(comb)
                if t.empty:                      # instância ainda incompleta
                    continue
                t = t[t.subset == "VPLxH2"]
                if t.empty:
                    continue
                r = t.iloc[0]
                c = comb.set_index("case").cost_total
                rows.append({"dia": d, "MW": lvl, "I_k": r.I / 1e3, "lo_k": r.I_lo / 1e3, "hi_k": r.I_hi / 1e3,
                             "veredito": r.verdict, "dH2_k": (c["A"] - c["E"]) / 1e3, "dVPL_k": (c["A"] - c["B"]) / 1e3})
        L += ["## fc_up: célula a combustível na barra 318 (VPL–H2 contra o corredor)", "",
              pd.DataFrame(rows).round(2).to_markdown(index=False), ""]

    out = ROOT / "results_pesgm_robust" / "robust_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
