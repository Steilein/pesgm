"""
corridor_report.py — interação VPL x H2 contra a duração do congestionamento.

Lê a varredura de ``src/corridor_sweep.py`` (``results_journal_corridor/``,
limites de 600, 1200 e 1500 MW) e o caso base de 900 MW
(``results_journal/``), nos quatro dias extremos, e testa a previsão da Sec.
II-C: com o corredor congestionado por mais horas, o VPL perde o desvio no
tempo e o hidrogênio não, então o coeficiente do par deveria subir em direção
a zero (deixar de substituir).

Reporta, por dia e limite: horas congestionadas no caso A, benefício do VPL, do
H2 e dos dois, o coeficiente com a cota e a sua fração do benefício conjunto; e
as correlações de postos com a duração --- juntando tudo e dentro de cada dia,
onde só o limite muda.

Uso::

    python papers/journal/scripts/corridor_report.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from interaction_stats import mobius_bounds  # noqa: E402

D4 = ["spring_wind", "summer_peak", "solar_ramp", "median"]


def grid() -> pd.DataFrame:
    base = pd.read_csv(ROOT / os.environ.get("RESULTS_DIR", "results_journal") / "metrics.csv")
    base = base[(base.variation == "base") & base.day_key.isin(D4) & base.case.isin(list("ABEF"))]
    base = base.assign(variation="s900")
    sweep = pd.read_csv(ROOT / os.environ.get("CORRIDOR_DIR", "results_journal_corridor") / "metrics.csv")
    return pd.concat([base, sweep], ignore_index=True)


def table(df: pd.DataFrame) -> pd.DataFrame:
    tab = mobius_bounds(df)
    tab = tab[tab.subset == "VPLxH2"].set_index(["day_key", "variation"])
    c = df.pivot_table(index=["day_key", "variation"], columns="case", values="cost_total") / 1e3
    h = df[df.case == "A"].set_index(["day_key", "variation"]).corridor_congested_hours
    out = pd.DataFrame({
        "limite_MW": [int(v[1:]) for _, v in c.index],
        "horas_cong": h.reindex(c.index),
        "dVPL_k": c.A - c.B, "dH2_k": c.A - c.E, "dAmbos_k": c.A - c.F,
        "I_k": tab.I.reindex(c.index) / 1e3,
        "I_lo_k": tab.I_lo.reindex(c.index) / 1e3, "I_hi_k": tab.I_hi.reindex(c.index) / 1e3,
        "veredito": tab.verdict.reindex(c.index),
    }, index=c.index)
    out["I_sobre_conjunto_pct"] = 100 * out.I_k / out.dAmbos_k
    return out.dropna(subset=["I_k"]).sort_values(["day_key", "limite_MW"])


def main() -> None:
    t = table(grid())
    print(t.round(2).to_string())
    rows = {"todos": (t.horas_cong, t.I_k, t.I_sobre_conjunto_pct)}
    for d, g in t.groupby(level="day_key"):
        rows[d] = (g.horas_cong, g.I_k, g.I_sobre_conjunto_pct)
    cor = pd.DataFrame({k: {"n": len(h), "rho_I": spearmanr(h, i).statistic,
                            "rho_I_rel": spearmanr(h, r).statistic}
                        for k, (h, i, r) in rows.items()}).T
    print("\nCorrelacao de postos com as horas congestionadas (positiva = previsao):")
    print(cor.round(2).to_string())
    out = ROOT / os.environ.get("CORRIDOR_DIR", "results_journal_corridor") / "corridor_report.md"
    out.write_text("# Interação VPL x H2 contra a duração do congestionamento\n\nGerado por "
                   "`papers/journal/scripts/corridor_report.py` (quatro dias extremos; "
                   "900 MW é o caso base).\n\n" + t.round(2).to_markdown() + "\n\n## Correlações\n\n"
                   + cor.round(3).to_markdown() + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
