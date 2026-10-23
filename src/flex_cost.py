"""
flex_cost.py — varredura do custo de flexibilidade do data center (c^flex).

Por que existe. Na formulação submetida ao GTD LA o deslocamento no tempo saía
de graça: o custo c_{j,d} do Link é zero no site nativo, de modo que o FDC
(caso 3) não pagava nada e só a transferência entre sítios (caso 4) pagava
5-20 $/MWh. Na reunião de 23/09/2026 o Flavio apontou que, para o UC, os dois
recursos precisam ter OPEX não nulo, e o Clodomiro pediu o MESMO custo de uso
para deslocar no próprio site e para transferir, de modo que a diferença entre
os casos 3 e 4 meça só o benefício da transferência.

O custo passa a incidir sobre a energia retirada do cronograma de referência
(``constraints.add_flex_cost``) e é varrido de 0 a 35 $/MWh de site --- o
último ponto fica logo abaixo do custo de combustível da térmica marginal
(~37 $/MWh), a proposta discutida na reunião. c^flex é lido como o preço de
oferta do data center em um programa de resposta da demanda.

Resolve, nos doze dias mensais e para cada c^flex, os três casos que dependem
dele: FDC (C, ``vdc.spatial = false``), VDC (C) e ambos (D). Os casos 1 e 2 não
têm data center flexível e são lidos de ``results_robust/``, resolvidos com o
mesmo perfil e o mesmo gap de 0,01 % --- condição para a cota de I valer.

Blocos:
    sweep  c^flex em FC_VALUES, sem prêmio de migração (custo único)
    prem   c^flex = 10 com o prêmio de migração original (5-20 $/MWh) somado

Uso::

    python src/flex_cost.py --months m01 m02 --only sweep --time-limit 1800
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, results_dir  # noqa: E402
from extra_runs import MONTHLY_DAYS, _base_kw  # noqa: E402
from robustness_tight import build_cfg  # noqa: E402

RESULTS = "results_flexcost"
FC_VALUES = [0, 5, 10, 15, 20, 30, 35]      # $/MWh de energia do site
FC_MAIN = 10
CASES = {"C_dclocal": ({"VDC"}, {"vdc.spatial": False}, "C"),
         "C": ({"VDC"}, {}, "C"),
         "D": ({"VPL", "VDC"}, {}, "D")}


def _months(cfg: dict) -> list[str]:
    return cfg.get("_months") or [f"m{m:02d}" for m in MONTHLY_DAYS]


def _run(cfg: dict, solver: str, fc: float, premium: bool, cases=CASES) -> None:
    from runner import run_case

    rdir = results_dir(cfg)
    tag = f"fc{int(fc):02d}" + ("p" if premium else "")
    lo, hi = cfg["vdc"]["workloads"]["migration_cost_range"]
    mig = [lo, hi] if premium else [0.0, 0.0]
    for m, date in MONTHLY_DAYS.items():
        key = f"m{m:02d}"
        if key not in _months(cfg):
            continue
        for label, (layers, extra, case) in cases.items():
            variation = f"dclocal_{tag}" if label == "C_dclocal" else tag
            name = f"{key}__{case}__{variation}"
            if (rdir / f"{name}.nc").exists():
                continue
            cfg_m = dict(cfg, days={**cfg["days"], key: date})
            # A matriz de migração é sorteada ANTES dos workloads com o mesmo
            # gerador; zerar a faixa consome os mesmos sorteios e preserva os jobs.
            ov = {"vdc.flex_cost": float(fc), "vdc.workloads.migration_cost_range": mig, **extra}
            print(f"-> {name}", flush=True)
            run_case(key, layers=layers, solver=solver, cfg=cfg_m, overrides=ov, name=name,
                     tags={"axis": "flex_cost", "variation": variation, "month": m,
                           "flex_cost": float(fc), "mig_premium": bool(premium)},
                     **_base_kw(cfg))


def run_sweep(cfg: dict, solver: str) -> None:
    for fc in cfg.get("_fc") or FC_VALUES:
        _run(cfg, solver, fc, premium=False)


def run_prem(cfg: dict, solver: str) -> None:
    # Sem transferência o prêmio não atua: o FDC com prêmio é o próprio fc10.
    _run(cfg, solver, FC_MAIN, premium=True, cases={k: v for k, v in CASES.items() if k != "C_dclocal"})


ROBUST = "results_flexcost_robust"


def _seed_robust(src: Path, dst: Path) -> None:
    """Semeia a grade de robustez com o que NÃO depende de c^flex.

    Casos A e B não têm data center flexível: copiam-se de ``results_robust/``
    (mesmo perfil, mesmo gap). Os casos mensais C/D e o FDC mensal vêm da
    varredura em c^flex = FC_MAIN, renomeados para as chaves que
    ``robustness_report`` e ``paper_numbers`` esperam (``base``/``dclocal``).
    """
    import shutil

    import pandas as pd

    dst.mkdir(exist_ok=True)
    rows = []
    rob = pd.read_csv(src / "metrics.csv")
    for _, r in rob[rob.case.isin(["A", "B"])].iterrows():
        if not (dst / f"{r.scenario}.nc").exists():
            shutil.copy2(src / f"{r.scenario}.nc", dst / f"{r.scenario}.nc")
        rows.append(r)
    sw = pd.read_csv(ROOT / RESULTS / "metrics.csv")
    tag = f"fc{FC_MAIN:02d}"
    for _, r in sw[sw.variation.isin([tag, f"dclocal_{tag}"])].iterrows():
        dcl = r.variation.startswith("dclocal")
        new = f"{r.day_key}__C__dclocal" if dcl else f"{r.day_key}__{r.case}__base"
        if not (dst / f"{new}.nc").exists():
            shutil.copy2(ROOT / RESULTS / f"{r.scenario}.nc", dst / f"{new}.nc")
        r = r.copy()
        r["scenario"], r["variation"] = new, ("dclocal" if dcl else "base")
        r["axis"] = "ladder_monthly" if dcl else "monthly"
        rows.append(r)
    seeded = pd.DataFrame(rows)
    seeded["cost_flex"] = seeded["cost_flex"].fillna(0.0)   # A e B não têm o termo
    path = dst / "metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        seeded = pd.concat([old[~old.scenario.isin(seeded.scenario)], seeded], ignore_index=True)
    seeded.to_csv(path, index=False)
    print(f"semeado {dst.name}: {len(rows)} linhas", flush=True)


def run_robust(cfg: dict, solver: str) -> None:
    """Refaz a grade de robustez (quatro dias extremos) com c^flex = FC_MAIN."""
    import robustness_tight as rt

    cfg = dict(cfg, paths={**cfg["paths"], "results": ROBUST})
    cfg["vdc"] = {**cfg["vdc"], "flex_cost": float(FC_MAIN),
                  "workloads": {**cfg["vdc"]["workloads"], "migration_cost_range": [0.0, 0.0]}}
    _seed_robust(ROOT / "results_robust", ROOT / ROBUST)
    for b in ("base", "grid", "reson", "hist", "soc50", "resref"):
        print(f"--- robust/{b}", flush=True)
        rt.BLOCKS[b](cfg, solver)
    # O prêmio de migração substitui a sensibilidade mig05/mig20 da versão submetida
    from runner import run_case
    rdir = results_dir(cfg)
    for day in cfg["days"]:
        for case, layers in (("C", {"VDC"}), ("D", {"VPL", "VDC"})):
            name = f"{day}__{case}__prem"
            if (rdir / f"{name}.nc").exists():
                continue
            run_case(day, layers=layers, solver=solver, cfg=cfg,
                     overrides={"vdc.workloads.migration_cost_range": [5.0, 20.0]}, name=name,
                     tags={"axis": "migration_cost", "variation": "prem"}, **_base_kw(cfg))


def _copy_ab(src: Path, dst: Path, keep) -> None:
    """Copia de ``src`` os casos A e B selecionados por ``keep(row)``."""
    import shutil

    import pandas as pd

    dst.mkdir(exist_ok=True)
    df = pd.read_csv(src / "metrics.csv")
    rows = df[df.case.isin(["A", "B"]) & df.apply(keep, axis=1)].copy()
    for s in rows.scenario:
        if not (dst / f"{s}.nc").exists():
            shutil.copy2(src / f"{s}.nc", dst / f"{s}.nc")
    rows["cost_flex"] = 0.0
    path = dst / "metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        rows = pd.concat([old[~old.scenario.isin(rows.scenario)], rows], ignore_index=True)
    rows.to_csv(path, index=False)
    print(f"semeado {dst.name}: {len(rows)} linhas", flush=True)


def _with_fc(cfg: dict, results: str) -> dict:
    cfg = dict(cfg, paths={**cfg["paths"], "results": results})
    cfg["vdc"] = {**cfg["vdc"], "flex_cost": float(FC_MAIN),
                  "workloads": {**cfg["vdc"]["workloads"], "migration_cost_range": [0.0, 0.0]}}
    return cfg


def run_pen(cfg: dict, solver: str) -> None:
    """Tabela III: penetração 30/50/70 % no dimensionamento de [9] (config/base.yaml)."""
    from common import load_config
    from runner import run_case

    base = _with_fc(load_config(), "results_flexcost_pen")
    D4 = ["median", "solar_ramp", "spring_wind", "summer_peak"]
    _copy_ab(ROOT / "results", ROOT / "results_flexcost_pen",
             lambda r: (r.axis in ("base", "vre_share", "monthly")) and r.variation in ("base", "vre30", "vre70"))
    rdir = results_dir(base)
    jobs = [(d, v, s, {"axis": "base" if v == "base" else "vre_share", "variation": v})
            for v, s in (("base", 0.50), ("vre30", 0.30), ("vre70", 0.70)) for d in D4]
    jobs += [(f"m{m:02d}", "base", 0.50, {"axis": "monthly", "variation": "base", "month": m})
             for m in MONTHLY_DAYS]
    for day, var, share, tags in jobs:
        for case, layers in (("C", {"VDC"}), ("D", {"VPL", "VDC"})):
            name = f"{day}__{case}__{var}"
            if (rdir / f"{name}.nc").exists():
                continue
            c = base
            if day.startswith("m") and day[1:].isdigit():
                c = dict(base, days={**base["days"], day: MONTHLY_DAYS[int(day[1:])]})
            kw = {**_base_kw(base), "vre_share": share}
            print(f"-> pen {name}", flush=True)
            try:
                run_case(day, layers=layers, solver=solver, cfg=c, name=name, tags=tags, **kw)
            except Exception:
                traceback.print_exc()


def run_ablation(cfg: dict, solver: str) -> None:
    """Tabela I: estágios cumulativos S0-S4 (src/ablation.py), casos C e D."""
    import ablation
    from common import load_profile

    ablation.RESULTS = "results_flexcost_ablation"
    _copy_ab(ROOT / "results_ablation", ROOT / ablation.RESULTS, lambda r: r.axis == "ablation")
    orig = load_profile

    def patched(path, _orig=orig):
        return _with_fc(_orig(path), ablation.RESULTS)

    ablation.load_profile = patched
    try:
        ablation.run(None, ["C", "D"], solver, force=False)
    finally:
        ablation.load_profile = orig


BLOCKS = {"sweep": run_sweep, "prem": run_prem, "robust": run_robust,
          "pen": run_pen, "ablation": run_ablation}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Varredura do custo de flexibilidade do data center.")
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--months", nargs="*", default=None)
    ap.add_argument("--fc", nargs="*", type=float, default=None, help=f"padrão: {FC_VALUES}")
    ap.add_argument("--only", nargs="*", default=None, help=f"blocos: {' '.join(BLOCKS)}")
    args = ap.parse_args(argv)

    cfg = build_cfg(time_limit=args.time_limit)
    cfg["paths"] = {**cfg["paths"], "results": RESULTS}
    cfg["_months"] = args.months
    cfg["_fc"] = args.fc
    print(f"saída {results_dir(cfg)} | meses {args.months or 'todos'} | c^flex {args.fc or FC_VALUES}",
          flush=True)
    for b in args.only or list(BLOCKS):
        print(f"\n=== bloco {b} ===", flush=True)
        try:
            BLOCKS[b](cfg, args.solver)
        except Exception:
            print(f"FALHOU no bloco {b}:", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
