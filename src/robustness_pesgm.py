"""
robustness_pesgm.py — testes pedidos pelo parecer simulado do PES GM (23/10/2026).

Três perguntas, cada uma com só os casos que mudam (os demais vêm da grade
principal ``results_pesgm/`` ou da varredura ``results_pesgm_corridor/``):

* ``vpl_emergent`` (comentário 4): o VPL sem a binária de direção do modo
  híbrido, as duas baterias livres. Casos com VPL (B, D, F, H), doze dias
  mensais. A, C, E e G não têm VPL e não mudam.
* ``h2_su10`` e ``h2_sb1`` (comentário 2B): partida a frio do eletrolisador
  10x mais barata (2,37 k$) e standby de 1 % (limite inferior de Saretta et
  al.). Casos com H2 (E, F, G, H), doze dias mensais.
* ``fc_up`` (comentário 5, teste topológico): a célula a combustível na barra
  a MONTANTE (318). A cadeia deixa de ser caminho paralelo ao corredor e vira
  armazenamento no terminal exportador, que depende do corredor como o VPL.
  O argumento de caminhos prevê a resposta OPOSTA à do caso base quando o
  corredor aperta. Casos E e F nos quatro dias extremos, corredor a 600, 900,
  1200 e 1500 MW (A e B não têm H2 e vêm da grade/varredura).

Uso (um processo por variante e dia, por causa da memória)::

    python src/robustness_pesgm.py --variant vpl_emergent --days m01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_profile, results_dir  # noqa: E402

RESULTS = "results_pesgm_robust"
MONTHLY = [f"m{m:02d}" for m in range(1, 13)]
D4 = ["spring_wind", "summer_peak", "solar_ramp", "median"]
LAYERS = {"B": {"VPL"}, "D": {"VPL", "VDC"}, "E": {"H2"}, "F": {"VPL", "H2"},
          "G": {"VDC", "H2"}, "H": {"VPL", "VDC", "H2"}}
# variante -> (casos, sobrescritas, dias padrão, níveis do corredor ou None)
VARIANTS = {
    "vpl_emergent": (["B", "D", "F", "H"], {"vpl.mode": "emergent"}, MONTHLY, None),
    "h2_su10": (["E", "F", "G", "H"], {"h2.startup_cost": 2370.0}, MONTHLY, None),
    "h2_sb1": (["E", "F", "G", "H"], {"h2.standby_load": 0.01}, MONTHLY, None),
    "fc_up": (["E", "F"], {"h2.fc_bus": "318"}, D4, [600, 900, 1200, 1500]),
}


def profile(threads: int = 4) -> dict:
    cfg = load_profile([ROOT / "config" / f for f in ("realistic.yaml", "journal.yaml", "pesgm.yaml")])
    cfg["paths"] = {**cfg["paths"], "results": RESULTS}
    cfg["solver"] = {**cfg["solver"], "threads": int(threads), "options": {"NodefileStart": 0.25}}
    return cfg


def main() -> None:
    from runner import run_case

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--solver", default="gurobi")
    a = ap.parse_args()

    cases, over, days, levels = VARIANTS[a.variant]
    cfg = profile(a.threads)
    rdir = results_dir(cfg)
    rdir.mkdir(exist_ok=True)
    base = {"vre_share": float(cfg["vre"]["target_share"]),
            "vpl_mwh": float(cfg["vpl"]["p_nom_mw"]) * float(cfg["vpl"]["hours"]),
            "dc_flex": float(cfg["vdc"]["flex_fraction"]),
            "reserve_from_flex": bool(cfg["reserve"]["from_flex"])}
    for day in a.days or days:
        for level in (levels or [None]):
            for case in cases:
                o = dict(over)
                variation = a.variant
                if level is not None:
                    o["corridor.s_nom_mw"] = float(level)
                    variation = f"{a.variant}_s{level}"
                name = f"{day}__{case}__{variation}"
                if (rdir / f"{name}.nc").exists():
                    print(f"[{name}] já existe — pulando", flush=True)
                    continue
                try:
                    run_case(day, layers=LAYERS[case], solver=a.solver, cfg=cfg, overrides=o, name=name,
                             tags={"axis": "robust_pesgm", "variation": variation}, **base)
                except Exception as exc:  # registra e segue
                    print(f"[{name}] FALHOU: {exc}", flush=True)


if __name__ == "__main__":
    main()
