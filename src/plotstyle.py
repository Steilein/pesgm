"""
plotstyle.py — primitivas de estilo das figuras, compartilhadas pelos artigos.

Extraído de ``papers/gtdla/scripts/plots.py`` em 20/09/2026. O motivo é o mesmo
que levou ``check_tex.py`` para ``src/``: quatro módulos do núcleo
(``interaction_stats``, ``cross_sensitivity``, ``startup_overlap``, ``vdc_grid``)
importavam ``ieee_style``, ``save`` e a paleta de dentro de
``papers/gtdla/scripts/``. Era ``src/`` dependendo da camada de artigo --- a
inversão que a separação do commit b9da297 existe para evitar --- e bloqueava
as figuras do artigo de periódico, que não podem depender do artigo submetido.

O que fica AQUI é o que não tem nada a ver com um artigo específico: dimensões
de coluna IEEE, paleta, rcParams, gravação em PDF+PNG e o mecanismo de idioma.

O que fica NO ARTIGO é tudo que nomeia o conteúdo dele: a tabela de tradução
dos rótulos, as cores por caso, os grupos de despacho. Cada artigo registra a
sua tabela com ``register_translations``; ``src/`` não guarda rótulo de figura
de artigo nenhum.

Uso::

    from plotstyle import INK, PALETTE, T, W1, ieee_style, save
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pypsa  # noqa: E402

# --- Idioma dos rótulos: PLOTS_LANG=pt (padrão) | en (figuras do artigo,
# gravadas em <figuras>/en/)
LANG = os.environ.get("PLOTS_LANG", "pt").lower()

# Rótulos das figuras que o PRÓPRIO núcleo desenha (interaction_stats,
# cross_sensitivity, startup_overlap, vdc_grid). Estavam na tabela de
# papers/gtdla/scripts/plots.py, o que fazia as figuras do núcleo dependerem
# do artigo submetido para sair em inglês --- o acoplamento era mais fundo que
# o import. Ficam aqui, com os mesmos valores.
#
# A duplicação com a tabela daquele artigo é deliberada e auto-policiada:
# ``register_translations`` levanta erro se os valores divergirem, então uma
# alteração num lado sem o outro quebra o build em vez de mudar o PDF em
# silêncio.
_TR: dict[str, str] = {
    "instância": "instance",
    "instância (ordenada por $I$)": "instance (sorted by $I$)",
    "$I$ (k\\$/dia)": "$I$ (k\\$/day)",
    "(ΔB + ΔC) − ΔD  (k$/dia)": "I = ΔD − (ΔB + ΔC)  (k$/day)",
    "VPL (MW/terminal)": "VPL (MW per terminal)",
    "média": "mean",
    "substituição conclusiva": "conclusive substitution",
    "$I$ médio": "mean $I$",
    "k\\$/dia": "k\\$/day",
    "partidas evitadas": "start-ups avoided",
    "qualquer um evitaria": "either alone would avoid",
    "só a LVT": "VPL only",
    "só o CDV": "VDC only",
    "nestas unidades, {a} das {b} partidas evitadas seriam evitadas por qualquer um dos recursos":
        "on these units, {a} of the {b} avoided start-ups would be avoided by either resource alone",
}


def register_translations(tr: dict[str, str]) -> None:
    """Acrescenta entradas à tabela de tradução usada por ``T``.

    Cada artigo registra as suas. Colisão entre artigos levanta erro em vez de
    sobrescrever em silêncio: dois artigos que traduzem o mesmo rótulo de forma
    diferente é justamente o tipo de coisa que passa despercebida no PDF.
    """
    for k, v in tr.items():
        if k in _TR and _TR[k] != v:
            raise ValueError(f"traducao conflitante para {k!r}: {_TR[k]!r} vs {v!r}")
    _TR.update(tr)


def T(s: str) -> str:
    """Traduz um rótulo quando PLOTS_LANG=en; devolve o original caso contrário."""
    return _TR.get(s, s) if LANG == "en" else s


# --- Paleta categórica validada (ordem fixa; ver skill dataviz / palette.md)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
W1, W2 = 3.5, 7.16   # polegadas: uma coluna e duas colunas do IEEEtran


def ieee_style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"], "axes.edgecolor": INK2, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
        "legend.frameon": False, "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig, fdir: Path, name: str) -> None:
    if LANG == "en":
        fdir = fdir / "en"
        fdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fdir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf")


def load_net(rdir: Path, name: str) -> pypsa.Network | None:
    p = rdir / f"{name}.nc"
    return pypsa.Network(str(p)) if p.exists() else None
