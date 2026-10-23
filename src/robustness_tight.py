"""
robustness_tight.py — testes de robustez sobre o sistema CORRIGIDO, com gap apertado.

Por que existe. A Tabela de robustez do artigo e a distribuição por instância são
resolvidas sobre o sistema SEM as correções estruturais — aquele em que, pela
Tabela II, o elemento ativo é a linha de coleta C29 e não o corredor. A objeção
é direta: a evidência instância a instância vem de uma rede na qual o par alivia
a restrição errada. Este módulo refaz esses testes sobre o sistema corrigido.

Por que o gap muda. A regra de conclusividade compara |I| com a soma dos quatro
gaps do resolvedor. No sistema corrigido |I| é MENOR (0,8-1,4 k$/dia contra 8,3
a 30 % no sistema sem correção: o corredor está dimensionado e a escassez
disputada é menor), enquanto o ruído a 0,05 % fica em 2,3-3,6 k$/dia. Nessa
razão quase toda instância seria indeterminada. Com custo médio de 2.059 k$/dia,
o gap precisa cair para ~0,01 % para que a meia-largura da cota fique abaixo de
|I|. É a única alavanca: o ruído é truncamento do resolvedor, não amostragem.

O que resolve (quatro dias extremos salvo indicação, quatro casos, ~200 MILP):

    base     16   grade principal, a referência das demais
    grid     64   f_d x0,4 e x1,6 e janela batch x0,5 e x2 (quatro casos cada)
    reson    12   LVT e CDV autorizados a fornecer reserva (B, C, D)
    prevday   4   dia anterior, caso A, para herdar o estado térmico
    hist     16   estados térmicos herdados do dia anterior
    soc50     8   estado de carga inicial fixo em 50 % (B, D)
    resref    8   mesmo requisito de reserva em todos os casos (C, D)
    mig05    8    custo de migração x0,5 (C, D)
    mig20    8    custo de migração x2   (C, D)
    monthly  48   os doze dias mensais, quatro casos (dias próprios, não os extremos)

Grava em ``results_robust/`` — separado de ``results_realistic/`` de propósito:
misturar tolerâncias no mesmo metrics.csv falsearia a cota, que só é válida se
os quatro MILP de uma instância vierem do mesmo gap.

Uso::

    python src/robustness_tight.py --solver gurobi           # grade completa
    python src/robustness_tight.py --only base --days median # amostra, para medir tempo
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_profile, results_dir  # noqa: E402
from extra_runs import (_base_kw, run_history, run_migcost,  # noqa: E402
                        run_reserve_ref, run_soc)

D4 = ["median", "solar_ramp", "spring_wind", "summer_peak"]
RESULTS = "results_robust"
GAP = 0.0001                     # 0,01 %: ver o cabeçalho
CASES = {"A": set(), "B": {"VPL"}, "C": {"VDC"}, "D": {"VPL", "VDC"}}


def build_cfg(days: list[str] | None = None, gap: float = GAP,
              time_limit: float | None = None) -> dict:
    """Perfil recalibrado, com o gap apertado, os quatro dias extremos e saída própria."""
    cfg = load_profile(ROOT / "config" / "realistic.yaml")
    cfg["paths"] = {**cfg["paths"], "results": RESULTS}
    # Quatro threads e árvore em disco a partir de 0,5 GB: a grade foi morta
    # duas vezes pelo guarda de memória da máquina, e nenhum dos dois parâmetros
    # altera o ótimo — só o pico de consumo.
    cfg["solver"] = {**cfg["solver"], "mip_rel_gap": float(gap), "threads": 4,
                     "options": {"NodefileStart": 0.25}}
    if time_limit:
        # Um dia mensal chegou a meio milhão de nós com 122 mil na fila, a 0,01 %,
        # e foi ele que estourou a memória. Limitar o tempo devolve a instância
        # com o gap que ela alcançou -- a cota usa o gap efetivo, não o alvo.
        cfg["solver"]["time_limit"] = float(time_limit)
    keep = days or D4
    cfg["days"] = {k: v for k, v in cfg["days"].items() if k in keep}
    missing = [d for d in keep if d not in cfg["days"]]
    if missing:
        raise SystemExit(f"dias ausentes no perfil: {missing}")
    return cfg


def run_base(cfg: dict, solver: str) -> None:
    """Grade principal: quatro casos em cada dia, sem variação."""
    from runner import run_case

    rdir = results_dir(cfg)
    for day in cfg["days"]:
        for case, layers in CASES.items():
            name = f"{day}__{case}__base"
            if (rdir / f"{name}.nc").exists():
                continue
            run_case(day, layers=layers, solver=solver, cfg=cfg, name=name,
                     tags={"axis": "base", "variation": "base"}, **_base_kw(cfg))


def run_reson(cfg: dict, solver: str) -> None:
    """LVT e CDV autorizados a fornecer reserva girante (B, C, D)."""
    from runner import run_case

    rdir = results_dir(cfg)
    kw = {**_base_kw(cfg), "reserve_from_flex": True}
    for day in cfg["days"]:
        for case in ("B", "C", "D"):
            name = f"{day}__{case}__reson"
            if (rdir / f"{name}.nc").exists():
                continue
            run_case(day, layers=CASES[case], solver=solver, cfg=cfg, name=name,
                     tags={"axis": "reserve_from_flex", "variation": "reson"}, **kw)


def run_grid_axes(cfg: dict, solver: str) -> None:
    """Eixos de sensibilidade que se aplicam aos QUATRO casos.

    A distribuição por instância só admite instâncias completas — quatro MILP com
    a mesma variação —, e são estes eixos que as produzem: fração flexível $f_d$
    e largura da janela batch. Ficam de fora, de propósito: ``vre_share``, porque
    varrer 30--70 % reintroduz o regime que a correção de penetração removeu, e
    ``vpl_size``/``vpl_mode``, que só existem em B e D.
    """
    from common import get_by_path, load_scenarios
    from runner import run_grid

    scen = load_scenarios()
    axes = {}
    for k, spec in scen["axes"].items():
        if k not in ("dc_flex", "window_scale"):
            continue
        # O perfil corrigido já adota f_d = 0,2 e janela x0,5: essas "variações"
        # reproduzem a base MILP a MILP e só inflariam a grade e a distribuição.
        atual = get_by_path(cfg, spec["param"])
        vals = {lab: v for lab, v in spec["values"].items()
                if not (isinstance(v, (int, float)) and abs(float(v) - float(atual)) <= 1e-9)}
        axes[k] = dict(spec, values=vals)
    run_grid(cfg, dict(scen, axes=axes), solver, days=list(cfg["days"]))


def run_monthly_subset(cfg: dict, solver: str) -> None:
    """Os doze dias mensais, ou o subconjunto pedido em ``--months``.

    Os dias mensais não estão em ``config/realistic.yaml``: vivem na constante
    ``MONTHLY_DAYS`` de ``extra_runs`` e são injetados no cfg a cada chamada.
    Rodar por partes é o que mantém cada processo curto o bastante para não ser
    morto pelo guarda de memória.
    """
    from extra_runs import MONTHLY_DAYS
    from runner import run_case

    rdir = results_dir(cfg)
    want = cfg.get("_months") or [f"m{m:02d}" for m in MONTHLY_DAYS]
    for m, date in MONTHLY_DAYS.items():
        key = f"m{m:02d}"
        if key not in want:
            continue
        for case, layers in CASES.items():
            name = f"{key}__{case}__base"
            if (rdir / f"{name}.nc").exists():
                continue
            cfg_m = dict(cfg, days={**cfg["days"], key: date})
            run_case(key, layers=layers, solver=solver, cfg=cfg_m, name=name,
                     tags={"axis": "monthly", "variation": "base", "month": m},
                     **_base_kw(cfg))


def run_dclocal(cfg: dict, solver: str) -> None:
    """Caso 3 do artigo: CDV sem transferência entre sítios, nos dias mensais.

    Não é uma das quatro pontas de I — serve para repartir o benefício do CDV
    entre deslocamento no tempo e no espaço (Tabela III). Entra na grade apertada
    só para que a tabela dos cinco casos venha toda da mesma tolerância.
    """
    from extra_runs import MONTHLY_DAYS
    from runner import run_case

    rdir = results_dir(cfg)
    want = cfg.get("_months") or [f"m{m:02d}" for m in MONTHLY_DAYS]
    for m, date in MONTHLY_DAYS.items():
        key = f"m{m:02d}"
        if key not in want or (rdir / f"{key}__C__dclocal.nc").exists():
            continue
        cfg_m = dict(cfg, days={**cfg["days"], key: date})
        run_case(key, layers={"VDC"}, solver=solver, cfg=cfg_m,
                 overrides={"vdc.spatial": False}, name=f"{key}__C__dclocal",
                 tags={"axis": "ladder_monthly", "variation": "dclocal", "month": m},
                 **_base_kw(cfg))


BLOCKS = {
    "base": run_base,
    "grid": run_grid_axes,
    "reson": run_reson,
    "hist": run_history,        # resolve prevday e hist
    "soc50": run_soc,
    "resref": run_reserve_ref,
    "mig": run_migcost,         # mig05 e mig20
    "monthly": run_monthly_subset,   # doze dias mensais (ou --months)
    "dclocal": run_dclocal,          # caso 3, para a Tabela III sair toda no mesmo gap
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Robustez no sistema corrigido, gap apertado.")
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--gap", type=float, default=GAP)
    ap.add_argument("--days", nargs="*", default=None, help=f"padrão: {' '.join(D4)}")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="limite de tempo por MILP, em segundos (padrão: o do config)")
    ap.add_argument("--months", nargs="*", default=None,
                    help="subconjunto dos dias mensais (ex.: m01 m02 m03)")
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"blocos a resolver: {' '.join(BLOCKS)}")
    args = ap.parse_args(argv)

    cfg = build_cfg(args.days, args.gap, args.time_limit)
    cfg["_months"] = args.months
    blocks = args.only or list(BLOCKS)
    unknown = [b for b in blocks if b not in BLOCKS]
    if unknown:
        raise SystemExit(f"bloco desconhecido: {unknown}; use {list(BLOCKS)}")

    print(f"perfil recalibrado | gap {args.gap:.2%} | dias {list(cfg['days'])} "
          f"| saída {results_dir(cfg)}", flush=True)
    for b in blocks:
        print(f"\n=== bloco {b} ===", flush=True)
        try:
            BLOCKS[b](cfg, args.solver)
        except Exception:        # um bloco inviável não aborta os demais
            print(f"FALHOU no bloco {b}:", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
