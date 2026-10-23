"""
batch_share_sweep.py — a divisão temporal/espacial do VDC contra ``batch_share``.

O artigo reporta que 61 % do benefício do VDC vem de adiar DENTRO do site e
39 % de transferir entre sites, e chama essa proporção de resultado mais robusto
do estudo por ser estável em três configurações de sistema. Essas configurações,
porém, compartilham o mesmo gerador de workloads com ``batch_share = 0,50`` — a
fração da energia flexível alocada em jobs batch, que só têm flexibilidade
temporal. É esse parâmetro, e não a topologia do sistema, que governa a divisão.

Este módulo varre ``vdc.workloads.batch_share`` e mede a divisão resultante,
resolvendo os degraus L3 (VDC sem transferência) e L4 (VDC completo) nos dias
representativos. A referência sem VDC (L1) não depende do parâmetro e é
reaproveitada.

Uso::

    python src/batch_share_sweep.py
    python src/batch_share_sweep.py --report-only
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_profile, load_scenarios  # noqa: E402
from extra_runs import _base_kw  # noqa: E402

RESULTS = "results_realistic"
SHARES = (0.25, 0.75)          # 0,50 ja existe como linha de base da escada
STEPS = ("L3", "L4")


def run(days: list[str] | None, solver: str | None, force: bool) -> None:
    from runner import run_case

    cfg = load_profile(ROOT / "config" / "realistic.yaml")
    scen = load_scenarios()
    solver = solver or cfg["solver"]["name"]
    rdir = ROOT / RESULTS
    days = days or ["spring_wind", "summer_peak", "solar_ramp", "median"]

    total = len(SHARES) * len(STEPS) * len(days)
    k = 0
    for share in SHARES:
        for step in STEPS:
            spec = scen["ladder"][step]
            for day in days:
                k += 1
                tag = f"bs{int(share*100):02d}"
                name = f"{day}__{step}__{tag}"
                if not force and (rdir / f"{name}.nc").exists():
                    print(f"[{k}/{total}] {name}: ja resolvido", flush=True)
                    continue
                print(f"[{k}/{total}] {name}  (batch_share={share})", flush=True)
                over = {**(spec.get("over") or {}),
                        "vdc.workloads.batch_share": float(share)}
                try:
                    run_case(day, layers=set(spec["layers"]), solver=solver, cfg=cfg,
                             overrides=over, name=name,
                             tags={"axis": "batch_share", "variation": tag, "case": step},
                             **_base_kw(cfg))
                except Exception:
                    print(f"  FALHOU em {name}:", flush=True)
                    traceback.print_exc()


def report(path: Path) -> None:
    df = pd.read_csv(ROOT / RESULTS / "metrics.csv")
    days = ["spring_wind", "summer_peak", "solar_ramp", "median"]

    def pick(case, axis, var):
        s = df[(df.case == case) & (df.axis == axis) & (df.variation == var)
               & (df.day_key.isin(days))]
        return s.set_index("day_key")["cost_total"]

    L1 = pick("L1", "ladder", "base")
    rows = []
    combos = [(0.50, "ladder", "dclocal", "vdc")]
    combos += [(s, "batch_share", f"bs{int(s*100):02d}", f"bs{int(s*100):02d}") for s in SHARES]
    for share, axis, v3, v4 in sorted(combos):
        L3, L4 = pick("L3", axis, v3), pick("L4", axis, v4)
        idx = L1.index.intersection(L3.index).intersection(L4.index)
        if idx.empty:
            continue
        d3 = (L1 - L3)[idx].mean()
        d4 = (L1 - L4)[idx].mean()
        rows.append({"batch_share": round(share,2), "days": len(idx),
                     "dC_local k$/day": d3 / 1e3, "dC_full k$/day": d4 / 1e3,
                     "temporal %": 100 * d3 / d4, "spatial %": 100 * (1 - d3 / d4)})
    t = pd.DataFrame(rows)
    if len(t):   # arredondar a tabela inteira transformaria 0,25 em 0,2
        t = t.round({"dC_local k$/day": 1, "dC_full k$/day": 1,
                     "temporal %": 0, "spatial %": 0})

    L = ["# Divisão temporal/espacial do VDC contra `batch_share`\n",
         "Gerado por `src/batch_share_sweep.py`. `batch_share` é a fração da energia",
         "flexível alocada em jobs batch (janela longa, site nativo, só temporais).",
         "A divisão de valor que o artigo reporta como estável entre configurações de",
         "sistema é, na verdade, governada por este parâmetro.\n",
         t.to_markdown(index=False) if len(t) else "_sem resultados_", ""]
    path.write_text("\n".join(L), encoding="utf-8")
    print(t.to_string(index=False) if len(t) else "sem resultados")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--solver", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        run(a.days, a.solver, a.force)
    report(ROOT / RESULTS / "batch_share.md")


if __name__ == "__main__":
    main()
