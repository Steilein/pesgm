"""
cross_sensitivity.py — sensibilidade cruzada tamanho do VPL × fração flexível f_d.

A grade principal (``config/scenarios.yaml``) varia um eixo por vez. Os
resultados com Gurobi (gap 0,05 %) indicaram que a interação VPL–VDC é
majoritariamente de substituição e que ela cresce com o tamanho do VPL. Para
mapear a fronteira entre substituição e complementaridade é preciso cruzar os
dois eixos de dimensionamento:

    P_VPL ∈ {100, 200, 400} MW por terminal  ×  f_d ∈ {0,2; 0,5; 0,8}.

A complementaridade de cada par (P, f) usa os casos já resolvidos na grade
principal — A (base), B(P) e C(f) — e o caso D(P, f). Dos nove D por dia,
cinco já existem (base, vpl100, vpl400, f02, f08); este módulo resolve os
quatro restantes por dia (16 resoluções) e monta a tabela 3 × 3 por dia e a
média entre dias:

    comp(P, f) = [ΔBen_D(P, f)] − [ΔBen_B(P) + ΔBen_C(f)]
               = C_B(P) + C_C(f) − C_D(P, f) − C_A        (custos)

Positivo = complementares; negativo = substitutos. O ruído do gap é a soma
dos gaps absolutos dos quatro casos envolvidos.

Saídas: ``results/<dia>__D__vpl<P>_f<ff>.nc`` (novos casos, também gravados em
``metrics.csv`` com ``axis = vpl_x_fd``), ``results/cross_vpl_fd.md`` e
``figures/fig8_cross_vpl_fd.{pdf,png}``.

Uso::

    python src/cross_sensitivity.py                 # resolve o que falta e gera tabela + figura
    python src/cross_sensitivity.py --no-solve      # só tabela + figura a partir do metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, figures_dir, load_config, results_dir  # noqa: E402

VPL_MW = [100, 200, 400]
FD = [0.2, 0.5, 0.8]
FD_LABEL = {0.2: "f02", 0.5: "f05", 0.8: "f08"}


def _label_b(cfg: dict, p: int, f: float | None = None) -> str:
    """Rótulo do caso B(P, f): a carga inflexível dos DCs em B depende de f_d."""
    if f is None or abs(f - float(cfg["vdc"]["flex_fraction"])) < 1e-9:
        return "base" if p == int(cfg["vpl"]["p_nom_mw"]) else f"vpl{p}"
    return FD_LABEL[f] if p == int(cfg["vpl"]["p_nom_mw"]) else f"vpl{p}_{FD_LABEL[f]}"


def _label_a(cfg: dict, f: float) -> str:
    return "base" if abs(f - float(cfg["vdc"]["flex_fraction"])) < 1e-9 else FD_LABEL[f]


def _label_c(cfg: dict, f: float) -> str:
    return "base" if abs(f - float(cfg["vdc"]["flex_fraction"])) < 1e-9 else FD_LABEL[f]


def _label_d(cfg: dict, p: int, f: float) -> str:
    lb, lc = _label_b(cfg, p), _label_c(cfg, f)
    if lb == "base" and lc == "base":
        return "base"
    if lb == "base":
        return lc
    if lc == "base":
        return lb
    return f"vpl{p}_{FD_LABEL[f]}"


def solve_missing(cfg: dict, solver: str) -> list[Path]:
    """Resolve os casos D(P, f) ainda inexistentes em ``results/``."""
    from runner import run_case

    rdir = results_dir(cfg)
    hours = float(cfg["vpl"]["hours"])
    base_kw = dict(vre_share=float(cfg["vre"]["target_share"]),
                   reserve_from_flex=bool(cfg["reserve"]["from_flex"]))
    out = []
    for day in cfg["days"]:
        for p in VPL_MW:
            for f in FD:
                for case, layers in (("D", {"VPL", "VDC"}), ("B", {"VPL"})):
                    label = _label_d(cfg, p, f) if case == "D" else _label_b(cfg, p, f)
                    name = f"{day}__{case}__{label}"
                    if (rdir / f"{name}.nc").exists():
                        continue
                    print(f"[{name}] resolvendo…", flush=True)
                    out.append(run_case(day, layers=layers, vpl_mwh=p * hours, dc_flex=f, solver=solver,
                                        cfg=cfg, name=name, tags={"axis": "vpl_x_fd", "variation": label}, **base_kw))
    return out


def cross_table(cfg: dict, df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Tabelas 3 × 3 de complementaridade (custo) e de ruído do gap, por dia, e a média."""
    df = df.set_index("scenario")

    def get(day, case, label, col="cost_total"):
        return float(df.loc[f"{day}__{case}__{label}", col])

    comp, noise = {}, {}
    for day in cfg["days"]:
        tab = pd.DataFrame(index=[f"{p} MW" for p in VPL_MW], columns=[f"f_d = {f:.1f}" for f in FD], dtype=float)
        nz = tab.copy()
        for p in VPL_MW:
            for f in FD:
                names = {"A": ("A", _label_a(cfg, f)), "B": ("B", _label_b(cfg, p, f)), "C": ("C", _label_c(cfg, f)),
                         "D": ("D", _label_d(cfg, p, f))}
                try:
                    c = {k: get(day, *v) for k, v in names.items()}
                    g = {k: get(day, *v, col="mip_gap") for k, v in names.items()}
                except KeyError:
                    continue
                tab.loc[f"{p} MW", f"f_d = {f:.1f}"] = c["B"] + c["C"] - c["D"] - c["A"]
                nz.loc[f"{p} MW", f"f_d = {f:.1f}"] = sum(c[k] * g[k] for k in "ABCD")
        comp[day], noise[day] = tab, nz
    mean = sum(comp.values()) / len(comp)
    return comp, noise, mean


def write_report(cfg: dict, df: pd.DataFrame, path: Path) -> pd.DataFrame:
    comp, noise, mean = cross_table(cfg, df)
    lines = ["# Sensibilidade cruzada — tamanho do VPL × fração flexível f_d\n",
             "Complementaridade (ΔB + ΔC) − ΔD em $/dia (benefício conjunto menos soma dos isolados). "
             "Positivo = complementares; negativo = substitutos. Casos D fora da diagonal da grade principal "
             "resolvidos por `src/cross_sensitivity.py`; A, B(P) e C(f) vêm da grade principal.\n"]
    lines.append("## Média entre os quatro dias\n")
    lines.append(mean.round(0).to_markdown() + "\n")
    # razão sinal/ruído médio
    nz_mean = sum(noise.values()) / len(noise)
    lines.append("Ruído médio do gap (soma dos gaps absolutos dos quatro casos):\n")
    lines.append(nz_mean.round(0).to_markdown() + "\n")
    # benefício do VDC isolado para normalizar
    d = df.set_index("scenario")
    lines.append("## Complementaridade relativa ao benefício isolado do VDC (%)\n")
    rel = mean.copy()
    for f in FD:
        ben_c = np.mean([float(d.loc[f"{day}__A__{_label_a(cfg, f)}", "cost_total"]) - float(d.loc[f"{day}__C__{_label_c(cfg, f)}", "cost_total"])
                         for day in cfg["days"]])
        rel[f"f_d = {f:.1f}"] = 100 * mean[f"f_d = {f:.1f}"] / ben_c if ben_c > 0 else np.nan
    lines.append(rel.round(1).to_markdown() + "\n")
    lines.append("## Por dia\n")
    for day, tab in comp.items():
        lines.append(f"### {day}\n")
        lines.append(tab.round(0).to_markdown() + "\n")
        lines.append("Ruído do gap:\n\n" + noise[day].round(0).to_markdown() + "\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return mean


def fig8(cfg: dict, df: pd.DataFrame, fdir: Path) -> None:
    import matplotlib.pyplot as plt
    from plotstyle import INK, INK2, T, W2, ieee_style, save

    ieee_style()
    comp, noise, mean = cross_table(cfg, df)
    panels = list(comp.items()) + [(T("média"), mean)]
    vmax = max(np.nanmax(np.abs(t.values)) for _, t in panels) / 1e3
    fig, axes = plt.subplots(1, len(panels), figsize=(W2, 1.9), sharey=True)
    for ax, (day, tab) in zip(axes, panels):
        z = tab.values.astype(float) / 1e3
        im = ax.imshow(z, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if np.isnan(z[i, j]):
                    continue
                strong = day != T("média") and abs(tab.values[i, j]) > 2 * noise[day].values[i, j]
                ax.text(j, i, f"{z[i, j]:+.1f}", ha="center", va="center", fontsize=6,
                        fontweight="bold" if strong else "normal",
                        color="white" if abs(z[i, j]) > 0.55 * vmax else INK)
        ax.set_xticks(range(len(FD))); ax.set_xticklabels([f"{f:.1f}" for f in FD])
        ax.set_yticks(range(len(VPL_MW))); ax.set_yticklabels([f"{p}" for p in VPL_MW])
        ax.set_title(day.replace("_", " "), fontsize=7, color=INK2, pad=3)
        ax.grid(False)
        ax.set_xlabel(r"$f_d$")
    axes[0].set_ylabel(T("VPL (MW/terminal)"))
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(T("(ΔB + ΔC) − ΔD  (k$/dia)"), fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, fdir, "fig8_cross_vpl_fd")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sensibilidade cruzada VPL × f_d")
    ap.add_argument("--solver", default=None)
    ap.add_argument("--no-solve", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    solver = args.solver or cfg["solver"]["name"]
    if not args.no_solve:
        new = solve_missing(cfg, solver)
        print(f"{len(new)} casos novos resolvidos")
    df = pd.read_csv(results_dir(cfg) / "metrics.csv")
    mean = write_report(cfg, df, results_dir(cfg) / "cross_vpl_fd.md")
    fig8(cfg, df, figures_dir(cfg))
    print(mean.round(0).to_string())


if __name__ == "__main__":
    main()
