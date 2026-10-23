"""
plot_mobius_col.py — mapa de interações em UMA coluna (2x2), para o PES GM.

A figura do periódico (``src/interaction_stats.py --mobius``) tem quatro
painéis lado a lado em largura de duas colunas. Em cinco páginas uma figura
de página inteira custa demais; aqui os mesmos coeficientes, as mesmas cotas
e o mesmo eixo vertical comum vão para uma grade 2x2 de uma coluna.

Os números vêm de ``mobius_bounds`` aplicado a ``results_pesgm/metrics.csv``
(``RESULTS_DIR`` troca a grade) --- a mesma função do periódico.

Uso::

    PLOTS_LANG=en python papers/pesgm/scripts/plot_mobius_col.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from interaction_stats import mobius_bounds  # noqa: E402
from plotstyle import GRID, INK, INK2, PALETTE, T, W1, ieee_style, save  # noqa: E402

ORDER = ["VPLxVDC", "VPLxH2", "VDCxH2", "VPLxVDCxH2"]
TITLE = {s: s.replace("x", "–").replace("H2", "H$_2$") for s in ORDER}


def main() -> None:
    import matplotlib.pyplot as plt

    tab = mobius_bounds(pd.read_csv(ROOT / os.environ.get("RESULTS_DIR", "results_pesgm") / "metrics.csv"))
    tab = tab[tab["variation"] == "base"]
    ieee_style()
    fig, axes = plt.subplots(2, 2, figsize=(W1, 2.75), sharey=True, sharex=True)
    for ax, s in zip(axes.flat, ORDER):
        t = tab[tab.subset == s].sort_values("I").reset_index(drop=True)
        x = np.arange(len(t))
        cor = np.where(t["verdict"] == "substitute", PALETTE[3],
                       np.where(t["verdict"] == "complement", PALETTE[0], INK2))
        ax.vlines(x, t["I_lo"] / 1e3, t["I_hi"] / 1e3, color=GRID, lw=0.8, zorder=1)
        ax.scatter(x, t["I"] / 1e3, s=7, c=cor, zorder=2, linewidths=0)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_title(TITLE[s], fontsize=7, pad=2)
        ax.set_xlim(-1, len(t))
        n_c = int((t["verdict"] != "indeterminate").sum())
        ax.text(0.04, 0.95, f"{n_c}/{len(t)}", transform=ax.transAxes,
                fontsize=6, color=INK2, va="top")
    for ax in axes[1]:
        ax.set_xlabel(T("instância (ordenada por $I$)"))
    for ax in axes[:, 0]:
        ax.set_ylabel(T("$I$ (k\\$/dia)"))
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
    save(fig, ROOT / "figures_pesgm", "fig_mobius_col")


if __name__ == "__main__":
    main()
