"""
startup_overlap.py — quais partidas térmicas os dois recursos evitam em comum.

A interação I entre VPL e VDC nao vem de um recurso expulsar o outro do
corredor: os volumes de atuacao quase nao mudam quando os dois operam juntos.
Vem de ambos evitarem PARTE DAS MESMAS partidas térmicas. Em agregado isso
aparece como 3,83 + 2,25 = 6,08 partidas evitadas isoladamente contra 5,50
evitadas em conjunto.

Este módulo abre essa contagem por unidade, para mostrar quais acionamentos
especificamente cada recurso dispensa e quais os dois dispensariam. Usa a mesma
definição de partida do resto do trabalho — a variável ``start_up`` das unidades
térmicas, que inclui a transição da condição inicial —, de modo que os números
fechem com ``startups_total`` em ``metrics.csv``.

Uso::

    python src/startup_overlap.py              # mede e grava o relatorio
    python src/startup_overlap.py --fig median # + figura de um dia
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT  # noqa: E402

RESULTS = "results_realistic"
# Caso do artigo -> nome do NetCDF. Dois esquemas de nome convivem: a escada
# (L1..L5), usada na grade do perfil realista, e A-D, usado na grade de gap
# apertado. São os MESMOS quatro casos: conferido, os custos coincidem em 0,005 %.
SCHEMES = {
    "ladder": {"1": "{d}__L1__s900", "2": "{d}__L2__s900",
               "4": "{d}__L4__vdc", "5": "{d}__L5__full"},
    "abcd": {"1": "{d}__A__base", "2": "{d}__B__base",
             "4": "{d}__C__base", "5": "{d}__D__base"},
}
CASE_FILE = SCHEMES["ladder"]
MONTHLY = [f"m{m:02d}" for m in range(1, 13)]


def unit_startups(path: Path) -> pd.Series | None:
    """Partidas por unidade térmica, na definição canônica de ``metrics.startups``."""
    import pypsa

    if not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = pypsa.Network(str(path))
    g = n.generators
    thermal = g.index[g["is_thermal"]]
    return n.generators_t.start_up[thermal].sum()


def day_table(day: str) -> pd.DataFrame | None:
    """Partidas por unidade nos quatro casos de um dia."""
    cols = {}
    for case, pat in CASE_FILE.items():
        s = unit_startups(ROOT / RESULTS / f"{pat.format(d=day)}.nc")
        if s is None:
            return None
        cols[case] = s
    t = pd.DataFrame(cols).fillna(0.0)
    return t[t.sum(axis=1) > 0]


def decompose(t: pd.DataFrame) -> dict:
    """Partidas evitadas por cada recurso e a parcela que ambos evitariam."""
    av = (t["1"] - t["2"]).clip(lower=0)      # evitadas pelo VPL sozinho
    ad = (t["1"] - t["4"]).clip(lower=0)      # evitadas pelo VDC sozinho
    both = np.minimum(av, ad)                 # a mesma unidade, os dois evitariam
    # Cada recurso tambem CRIA partidas em unidades que o caso base nao acionava:
    # o despacho e reorganizado, nao apenas reduzido. Sem contabilizar isso, a
    # soma bruta das evitadas superestima o efeito liquido.
    cv = (t["2"] - t["1"]).clip(lower=0)
    cd = (t["4"] - t["1"]).clip(lower=0)
    return {"base": t["1"].sum(), "vpl": t["2"].sum(), "vdc": t["4"].sum(),
            "joint": t["5"].sum(),
            "avoided_vpl": av.sum(), "created_vpl": cv.sum(),
            "avoided_vdc": ad.sum(), "created_vdc": cd.sum(),
            "avoided_joint": (t["1"] - t["5"]).clip(lower=0).sum(),
            "overlap_unit": both.sum(),
            "only_vpl": (av - both).sum(), "only_vdc": (ad - both).sum()}


def plot_units(days: list[str], out_name: str = "fig11_overlap",
               fdir: Path | None = None) -> None:
    """Figura: por unidade térmica, quem dispensa cada partida, somado nos dias.

    Agregar em vez de escolher um dia evita tanto o cherry-picking quanto a
    impressão de que o overlap é um artefato de uma jornada específica.
    """
    import matplotlib.pyplot as plt
    from plotstyle import INK, INK2, PALETTE, T, W1, ieee_style, save

    av = ad = both = None
    for d in days:
        t = day_table(d)
        if t is None or t.empty:
            continue
        a = (t["1"] - t["2"]).clip(lower=0)
        b = (t["1"] - t["4"]).clip(lower=0)
        o = np.minimum(a, b)
        av = a if av is None else av.add(a, fill_value=0)
        ad = b if ad is None else ad.add(b, fill_value=0)
        both = o if both is None else both.add(o, fill_value=0)
    if av is None:
        print("sem dados")
        return
    av, ad, both = (s.fillna(0) for s in (av, ad, both))
    keep = av.index[(av + ad) > 0]
    if not len(keep):
        print("nenhuma partida evitada")
        return
    order = (both + av + ad).loc[keep].sort_values(ascending=False).index[:18]

    ieee_style()
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    x = np.arange(len(order))
    b = both.loc[order].values
    ov = (av.loc[order] - both.loc[order]).values
    od = (ad.loc[order] - both.loc[order]).values
    # cinza escuro para a categoria que importa, azul e ambar para as demais:
    # evita o par vermelho/verde, inseguro para daltonismo
    ax.bar(x, b, color=INK2, label=T("qualquer um evitaria"))
    ax.bar(x, ov, bottom=b, color=PALETTE[0], label=T("só a LVT"))
    ax.bar(x, od, bottom=b + ov, color=PALETTE[3], label=T("só o CDV"))
    ax.set_xticks(x)
    ax.set_xticklabels([str(u).replace("_", " ") for u in order], rotation=90, fontsize=5)
    ax.set_ylabel(T("partidas evitadas"))
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    shared, total = b.sum(), (b + ov + od).sum()
    ax.set_title(T("nestas unidades, {a} das {b} partidas evitadas seriam evitadas por qualquer um dos recursos")
                 .format(a=f"{shared:.0f}", b=f"{total:.0f}"), fontsize=6, color=INK)
    save(fig, fdir or ROOT / "figures", out_name)


def main() -> None:
    global RESULTS, CASE_FILE

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", nargs="*", default=MONTHLY)
    ap.add_argument("--fig", action="store_true", help="gera a figura agregada")
    ap.add_argument("--results", default=RESULTS, help="diretorio da grade")
    ap.add_argument("--scheme", default="ladder", choices=list(SCHEMES),
                    help="esquema de nome dos cenarios: ladder (L1..L5) ou abcd")
    ap.add_argument("--figdir", default=None, help="onde gravar a figura")
    a = ap.parse_args()
    RESULTS, CASE_FILE = a.results, SCHEMES[a.scheme]

    rows = []
    for d in a.days:
        t = day_table(d)
        if t is None:
            print(f"{d}: NetCDF ausente")
            continue
        rows.append({"day": d, **decompose(t)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    m = df.drop(columns="day").mean()
    print(df.round(2).to_string(index=False))
    print("\nmedias por dia:")
    print(m.round(2).to_string())
    print(f"\ninteracao em contagem = {m['avoided_joint'] - (m['avoided_vpl'] + m['avoided_vdc']):+.2f} partidas/dia")
    print(f"overlap medido por unidade = {m['overlap_unit']:.2f} partidas/dia")

    out = ROOT / RESULTS / "startup_overlap.md"
    out.write_text(
        "# Partidas térmicas evitadas: quem evita o quê\n\n"
        "Gerado por `src/startup_overlap.py`, definição canônica de partida\n"
        "(variável `start_up` das unidades térmicas).\n\n"
        + df.round(2).to_markdown(index=False) + "\n\n## Médias por dia\n\n"
        + m.round(2).to_frame("valor").to_markdown() + "\n", encoding="utf-8")
    print(f"\n-> {out}")

    if a.fig:
        plot_units(a.days, fdir=(ROOT / a.figdir) if a.figdir else None)


if __name__ == "__main__":
    main()
