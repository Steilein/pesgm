"""
soc_check.py — sensibilidade dos resultados centrais do VPL ao SOC inicial.

O caso principal usa ``cyclic_state_of_charge``: o modelo impõe SOC_24 = SOC_0
mas deixa o NÍVEL livre, de modo que o otimizador escolhe a condição inicial
mais conveniente para cada dia. Não há criação de energia no ciclo, mas há uma
liberdade que o operador real não tem — no início do horizonte a energia
armazenada é um dado, não uma decisão.

Este módulo repete os resultados centrais do VPL no perfil realista com
``vpl.soc_initial = 0,5`` (metade da capacidade, fechando o ciclo no mesmo
valor) e reporta o efeito sobre as quatro grandezas que o artigo publica:

* Delta_2, o benefício do VPL, nos 12 dias mensais;
* o alívio líquido do corredor (MW), na mesma base;
* o deslocamento de capacidade de corredor (varredura 900 / 700 / 500 MW);
* a interação I, com o caso 5.

A referência sem VPL (L1) não depende do SOC e é reaproveitada de
``results_realistic/``.

Uso::

    python src/soc_check.py             # resolve e reporta
    python src/soc_check.py --report-only
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_profile, load_scenarios  # noqa: E402
from extra_runs import MONTHLY_DAYS, _base_kw  # noqa: E402

RESULTS = "results_realistic"
CORRIDORS = (900.0, 700.0, 500.0)          # mesma varredura do deslocamento
SOC0 = 0.5


def run(solver: str | None, force: bool) -> None:
    """Resolve L2 (VPL) e L5 (ambas) com SOC inicial fixo, nos 12 dias mensais."""
    from runner import run_case

    cfg = load_profile(ROOT / "config" / "realistic.yaml")
    scen = load_scenarios()
    solver = solver or cfg["solver"]["name"]
    rdir = ROOT / RESULTS

    jobs = [("L2", CORRIDORS), ("L5", (900.0,))]
    total = sum(len(c) for _, c in jobs) * len(MONTHLY_DAYS)
    k = 0
    for step, corridors in jobs:
        spec = scen["ladder"][step]
        for mw in corridors:
            for m, date in MONTHLY_DAYS.items():
                key = f"m{m:02d}"
                k += 1
                name = f"{key}__{step}__soc50_s{mw:.0f}"
                if not force and (rdir / f"{name}.nc").exists():
                    print(f"[{k}/{total}] {name}: ja resolvido", flush=True)
                    continue
                print(f"[{k}/{total}] {name}", flush=True)
                cfg_m = dict(cfg)
                cfg_m["days"] = {**cfg["days"], key: date}
                over = {**(spec.get("over") or {}),
                        "vpl.soc_initial": SOC0,
                        "corridor.s_nom_mw": mw}
                try:
                    run_case(key, layers=set(spec["layers"]), solver=solver, cfg=cfg_m,
                             overrides=over, name=name,
                             tags={"axis": "soc_realistic", "variation": f"soc50_s{mw:.0f}",
                                   "case": step, "month": m},
                             **_base_kw(cfg))
                except Exception:
                    print(f"  FALHOU em {name}:", flush=True)
                    traceback.print_exc()


def report(path: Path) -> None:
    """Compara SOC livre x SOC fixo nas grandezas centrais do VPL."""
    df = pd.read_csv(ROOT / RESULTS / "metrics.csv")
    mdays = [f"m{m:02d}" for m in MONTHLY_DAYS]

    def pick(case: str, axis: str, var: str, mw: float) -> pd.DataFrame:
        s = df[(df["case"] == case) & (df["axis"] == axis) & (df["variation"] == var)
               & (df["corridor_limit_mw"] == mw) & (df["day_key"].isin(mdays))]
        return s.set_index("day_key")

    # Referência sem VPL a 900 MW: nos 12 dias mensais ela vem da varredura de
    # deslocamento (axis ``defer_monthly``), não da escada.
    L1 = pick("L1", "defer_monthly", "s900", 900.0)
    if L1.empty:
        L1 = pick("L1", "ladder_monthly", "base", 900.0)

    rows = []
    for label, axis, var in (("SOC free (published)", "defer_monthly", "s900"),
                             ("SOC fixed at 50%", "soc_realistic", "soc50_s900")):
        L2 = pick("L2", axis, var, 900.0)
        idx = L1.index.intersection(L2.index)
        if idx.empty:
            continue
        rows.append({
            "case": label, "days": len(idx),
            "dB k$/day": (L1.loc[idx, "cost_total"] - L2.loc[idx, "cost_total"]).mean() / 1e3,
            "net relief MW": L2.loc[idx, "vpl_corridor_relief_mw"].mean(),
            "curtail %": L2.loc[idx, "curtailment_pct"].mean(),
        })
    t = pd.DataFrame(rows)

    # deslocamento: custo de operar corredor reduzido COM o VPL contra 900 MW SEM
    dsp = []
    for label, axis, pat in (("SOC free (published)", "defer_monthly", "s{:.0f}"),
                             ("SOC fixed at 50%", "soc_realistic", "soc50_s{:.0f}")):
        for mw in CORRIDORS:
            L2 = pick("L2", axis, pat.format(mw), mw)
            idx = L1.index.intersection(L2.index)
            if idx.empty:
                continue
            delta = (L2.loc[idx, "cost_total"] - L1.loc[idx, "cost_total"]) / 1e3
            dsp.append({"case": label, "corridor MW": mw, "displaced MW": 900 - mw,
                        "periods": len(idx), "mean k$/day": delta.mean(),
                        "worst k$/day": delta.max(),
                        "all periods cheaper": bool((delta < 0).all())})
    d = pd.DataFrame(dsp)

    L = ["# Sensibilidade ao SOC inicial do VPL\n",
         "Gerado por `src/soc_check.py`. O caso publicado usa ciclo diário fechado com",
         "nível inicial livre; aqui o nível é fixado em 50 % da capacidade, fechando o",
         "ciclo no mesmo valor. A referência sem VPL (L1) não depende do SOC.\n",
         "## 1. Benefício e alívio (12 dias mensais, corredor de 900 MW)\n",
         t.round(1).to_markdown(index=False) if len(t) else "_sem resultados_", "",
         "## 2. Deslocamento de capacidade de corredor\n",
         d.round(1).to_markdown(index=False) if len(d) else "_sem resultados_", ""]
    path.write_text("\n".join(L), encoding="utf-8")
    print(t.round(1).to_string(index=False) if len(t) else "sem resultados")
    print()
    print(d.round(1).to_string(index=False) if len(d) else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        run(a.solver, a.force)
    report(ROOT / RESULTS / "soc_check.md")


if __name__ == "__main__":
    main()
