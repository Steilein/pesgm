"""
overlap_hours.py — sobreposição das horas de atuação de cada par de recursos contra I.

Teste da Sec. II-E do artigo de periódico ("When Should Benefits Fail to
Add?"). A moldura do grafo espaço-tempo prevê substituição tanto maior quanto
mais os caminhos dos dois recursos coincidem no corte; a grandeza que ela nomeia
é a sobreposição, medida aqui diretamente no caso em que o par atua junto
(D = VPL+VDC, F = VPL+H2, G = VDC+H2), dia a dia:

* VPL ativo na hora t: potência líquida de algum terminal acima de 1 % da
  nominal;
* VDC ativo na hora t: carga flexível total dos sites afastada mais de 5 % da
  sua média diária;
* H2 ativo na hora t: eletrolisador produzindo ou célula gerando acima de 1 %
  da nominal (o standby não conta como atuação);
* sobreposição: índice de Jaccard dos dois conjuntos de horas.

A sobreposição é correlacionada (Spearman e Pearson) com o coeficiente do par no
mesmo dia: por par (16 dias) e com os três pares juntos (48), em todos os dias e
só nos conclusivos.

Uso::

    python papers/journal/scripts/overlap_hours.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from interaction_stats import mobius_bounds  # noqa: E402

RES = ROOT / os.environ.get("RESULTS_DIR", "results_journal")
PCT_P = 0.01        # fração da potência nominal (VPL, H2)
VDC_TOL = 0.05      # fração da média diária da carga flexível
PAIRS = {"VPLxVDC": ("D", "VPL", "VDC"), "VPLxH2": ("F", "VPL", "H2"), "VDCxH2": ("G", "VDC", "H2")}


def active(n, resource: str) -> set[int]:
    if resource == "VPL":
        su = n.storage_units[n.storage_units.carrier == "VPL"]
        on = n.storage_units_t.p[su.index].abs().gt(PCT_P * su.p_nom, axis=1).any(axis=1)
    elif resource == "VDC":
        links = n.links[n.links.carrier == "VDC"]
        flex = n.links_t.p0[links.index].sum(axis=1)
        on = (flex - flex.mean()).abs() > VDC_TOL * flex.mean()
    else:
        p = n.links_t.p0
        on = (p["H2_P2H"] > PCT_P * n.links.at["H2_P2H", "p_nom"]) | \
             (p["H2_H2P"] > PCT_P * n.links.at["H2_H2P", "p_nom"])
    return set(np.flatnonzero(on.to_numpy()))


def _stats(t: pd.DataFrame) -> dict:
    conc = t[t.verdict != "indeterminate"]
    return {"n": len(t), "spearman": spearmanr(t.jaccard, t.I).statistic,
            "pearson": pearsonr(t.jaccard, t.I).statistic,
            "n_concl": len(conc), "spearman_concl": spearmanr(conc.jaccard, conc.I).statistic}


def main() -> None:
    import pypsa

    df = pd.read_csv(RES / "metrics.csv")
    tab = mobius_bounds(df[df.variation == "base"])
    rows = []
    for subset, (case, r1, r2) in PAIRS.items():
        pair = tab[tab.subset == subset].set_index("day_key")
        for d in pair.index:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n = pypsa.Network(str(RES / f"{d}__{case}__base.nc"))
            a, b = active(n, r1), active(n, r2)
            rows.append({"pair": subset, "day_key": d, "h1": len(a), "h2": len(b),
                         "jaccard": len(a & b) / len(a | b) if a | b else np.nan,
                         "I": pair.at[d, "I"], "verdict": pair.at[d, "verdict"]})
    t = pd.DataFrame(rows)
    stats = pd.DataFrame({p: _stats(g) for p, g in t.groupby("pair")}).T
    stats.loc["pooled"] = _stats(t)
    rng = t.groupby("pair").jaccard.agg(["min", "max"])
    print(t.round(3).to_string(index=False))
    print("\nJaccard por par:\n" + rng.round(2).to_string())
    print("\nCorrelacoes com I:\n" + stats.round(2).to_string())
    out = RES / "overlap_hours.md"
    out.write_text(
        "# Sobreposição das horas de atuação de cada par contra I\n\nGerado por "
        "`papers/journal/scripts/overlap_hours.py` (variação base; caso D, F ou G).\n\n"
        + t.round(3).to_markdown(index=False) + "\n\n## Correlações\n\n"
        + stats.round(3).to_markdown() + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
