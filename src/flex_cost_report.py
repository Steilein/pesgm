"""
flex_cost_report.py — tabela e figura da varredura do custo de flexibilidade.

Lê ``results_flexcost/metrics.csv`` (casos 3, 4 e 5 por c^flex, doze dias
mensais) e toma os casos 1 e 2 de ``results_robust/`` (mesmo perfil, mesmo gap
de 0,01 %). Para cada c^flex:

* Δ3, Δ4, Δ5 --- economia sobre o caso 1 (k$/dia);
* parcela local Δ3/Δ4 --- quanto do valor do VDC dispensa a transferência;
* I = Δ5 − (Δ2 + Δ4), com a cota direcional por instância;
* valor da VPL dada a flexibilidade do DC, Δ5 − Δ4, e do VDC dada a VPL, Δ5 − Δ2;
* energia deslocada no tempo e no espaço, e o custo de flexibilidade pago.

Uso::

    python src/flex_cost_report.py              # tabela em results_flexcost/flex_cost.md
    PLOTS_LANG=en python src/flex_cost_report.py --fig figures_realistic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT  # noqa: E402
from robustness_report import _err  # noqa: E402

MD = [f"m{m:02d}" for m in range(1, 13)]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    sw = pd.read_csv(ROOT / "results_flexcost" / "metrics.csv")
    sw = sw[sw.day_key.isin(MD)]
    rob = pd.read_csv(ROOT / "results_robust" / "metrics.csv")
    ab = rob[rob.day_key.isin(MD) & (rob.variation == "base") & rob.case.isin(["A", "B"])]
    return sw, ab.set_index(["day_key", "case"])


def per_instance(sw: pd.DataFrame, ab: pd.DataFrame, fc: float, premium: bool = False) -> pd.DataFrame:
    tag = f"fc{int(fc):02d}" + ("p" if premium else "")
    rows = []
    for day in MD:
        g = sw[sw.day_key == day]
        c3 = g[g.variation == f"dclocal_fc{int(fc):02d}"]
        c4 = g[(g.variation == tag) & (g.case == "C")]
        c5 = g[(g.variation == tag) & (g.case == "D")]
        if c4.empty or c5.empty or (day, "A") not in ab.index or (day, "B") not in ab.index:
            continue
        r1, r2, r4, r5 = ab.loc[(day, "A")], ab.loc[(day, "B")], c4.iloc[0], c5.iloc[0]
        k1, k2, k4, k5 = (float(r.cost_total) for r in (r1, r2, r4, r5))
        e = {k: _err(r) for k, r in zip("1245", (r1, r2, r4, r5))}
        I = k2 + k4 - k1 - k5
        lo, hi = I - (e["2"] + e["4"]), I + (e["1"] + e["5"])
        rows.append({
            "day": day,
            "d2": (k1 - k2) / 1e3, "d4": (k1 - k4) / 1e3, "d5": (k1 - k5) / 1e3,
            "d3": (k1 - float(c3.iloc[0].cost_total)) / 1e3 if not c3.empty else np.nan,
            "I": I / 1e3, "I_lo": lo / 1e3, "I_hi": hi / 1e3,
            "verdict": "subst" if hi < 0 else "compl" if lo > 0 else "indet",
            "flex4": float(r4.cost_flex) / 1e3, "flex5": float(r5.cost_flex) / 1e3,
            "sp4": float(r4.vdc_spatial_shift_mwh), "tm4": float(r4.vdc_temporal_shift_mwh),
            "tm3": float(c3.iloc[0].vdc_temporal_shift_mwh) if not c3.empty else np.nan,
            "su4": float(r1.startups_total) - float(r4.startups_total),
        })
    return pd.DataFrame(rows)


def summary(sw: pd.DataFrame, ab: pd.DataFrame) -> pd.DataFrame:
    out = []
    # c^flex vem do rótulo da variação (fc10, fc10p): metrics.csv não guarda as tags livres
    v = sw.variation[sw.variation.str.fullmatch(r"fc\d\d")]
    fcs = sorted({float(x[2:]) for x in v})
    for fc, prem in [(f, False) for f in fcs] + [(10.0, True)]:
        t = per_instance(sw, ab, fc, prem)
        if t.empty:
            continue
        m = t.mean(numeric_only=True)
        out.append({
            "c_flex": fc, "premium": prem, "n": len(t),
            "D2": m.d2, "D3": m.d3, "D4": m.d4, "D5": m.d5,
            "local_%": 100 * m.d3 / m.d4 if m.d4 else np.nan,
            "I": m.I, "I/D2_%": 100 * m.I / m.d2, "I/D4_%": 100 * m.I / m.d4 if m.d4 else np.nan,
            "VPL|DC": m.d5 - m.d4, "VDC|VPL": m.d5 - m.d2,
            "subst": int((t.verdict == "subst").sum()), "compl": int((t.verdict == "compl").sum()),
            "flex_paid4": m.flex4, "spatial4": m.sp4, "temporal4": m.tm4, "temporal3": m.tm3,
            "startups_avoided4": m.su4,
        })
    return pd.DataFrame(out)


def figure(tab: pd.DataFrame, fdir: Path, name: str = "fig12_flexcost") -> None:
    import matplotlib.pyplot as plt
    from plotstyle import INK, INK2, PALETTE, W1, ieee_style, register_translations, save, T

    register_translations({
        "preço de oferta $c^{\\mathrm{flex}}$ (\\$/MWh)": "offer price $c^{\\mathrm{flex}}$ (\\$/MWh)",
        "valor operacional (k\\$/dia)": "operating value (k\\$/day)",
        "CDV ($\\Delta_4$)": "VDC ($\\Delta_4$)",
        "CDF ($\\Delta_3$)": "FDC ($\\Delta_3$)",
        "CDV com LVT ($\\Delta_5-\\Delta_2$)": "VDC given VPL ($\\Delta_5-\\Delta_2$)",
        "$I$ (k\\$/dia)": "$I$ (k\\$/day)",
    })
    ieee_style()
    t = tab[~tab.premium].sort_values("c_flex")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 2.9), sharex=True,
                                 gridspec_kw={"height_ratios": [1.6, 1]})
    series = [("D4", "CDV ($\\Delta_4$)", PALETTE[0]),
              ("VDC|VPL", "CDV com LVT ($\\Delta_5-\\Delta_2$)", PALETTE[1]),
              ("D3", "CDF ($\\Delta_3$)", PALETTE[2])]
    for col, lbl, c in series:
        a1.plot(t.c_flex, t[col], color=c, lw=1.4, marker="o", ms=3.2, label=T(lbl))
    a1.set_ylabel(T("valor operacional (k\\$/dia)"))
    a1.set_ylim(bottom=0)
    a1.legend(loc="upper right", handlelength=1.6)
    a2.plot(t.c_flex, t.I, color=INK, lw=1.2, marker="o", ms=3.2)
    a2.axhline(0, color=INK2, lw=0.6)
    for x, y, s, n in zip(t.c_flex, t.I, t.subst, t.n):
        a2.annotate(f"{s}/{n}", (x, y), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=5.5, color=INK2)
    a2.set_ylim(min(-1.2, float(t.I.min()) - 0.3), 0.15)
    a2.set_ylabel(T("$I$ (k\\$/dia)"))
    a2.set_xlabel(T("preço de oferta $c^{\\mathrm{flex}}$ (\\$/MWh)"))
    fig.align_ylabels()
    save(fig, fdir, name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default=None)
    args = ap.parse_args(argv)
    sw, ab = load()
    tab = summary(sw, ab)
    pd.set_option("display.width", 200)
    print(tab.round(2).to_string(index=False))
    out = ROOT / "results_flexcost" / "flex_cost.md"
    out.write_text("# Varredura do custo de flexibilidade (doze dias mensais, k$/dia)\n\n"
                   "Gerado por `src/flex_cost_report.py`.\n\n" + tab.round(2).to_markdown(index=False) + "\n",
                   encoding="utf-8")
    print(f"escrito: {out}")
    if args.fig:
        figure(tab, ROOT / args.fig)


if __name__ == "__main__":
    main()
