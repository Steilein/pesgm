"""
corridor_sweep.py — o par VPL x H2 contra a duração do congestionamento.

Teste da previsão da Sec. II-C do artigo de periódico. O caminho do VPL é um
desvio no tempo, (u,t) -> (u,t') -> (d,t'), e só existe se houver uma hora t'
em que o corredor não esteja saturado; o do hidrogênio não cruza linha nenhuma.
À medida que o congestionamento se alonga, o VPL perde o desvio e o hidrogênio
não --- os dois deveriam deixar de ser substitutos. A variação natural entre os
dias (0 a 11 h congestionadas) não isola isso, porque os dias diferem em tudo;
aqui só o limite térmico do corredor muda, nos mesmos dias.

Resolve os casos que o par exige (A, B, E, F; o VDC não entra) nos quatro dias
extremos, para cada limite em ``LEVELS``. O nível de 900 MW é o caso base e é
lido de ``results_journal/``. Saída em ``results_journal_corridor/``.

Uso::

    python src/corridor_sweep.py --days spring_wind --runs A B --levels 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_profile, results_dir  # noqa: E402

RESULTS = "results_journal_corridor"
D4 = ["spring_wind", "summer_peak", "solar_ramp", "median"]
LEVELS = [600, 1200, 1500]          # MW; 900 é o caso base (results_journal/)
CASES = {"A": set(), "B": {"VPL"}, "E": {"H2"}, "F": {"VPL", "H2"}}


def profile(threads: int = 2, extra: list[str] | None = None, results: str = RESULTS) -> dict:
    # ``extra``: perfis sobrepostos depois de journal.yaml (o PES GM usa
    # config/pesgm.yaml); ``results``: destino. Os padrões são os do periódico.
    perfis = [ROOT / "config" / "realistic.yaml", ROOT / "config" / "journal.yaml"]
    cfg = load_profile(perfis + [ROOT / e for e in (extra or [])])
    cfg["paths"] = {**cfg["paths"], "results": results}
    # Mesma receita de indivisible.py: poucas threads e árvore em disco, porque
    # a máquina tem ~2 GB livres e o guarda de memória já matou campanhas.
    cfg["solver"] = {**cfg["solver"], "threads": int(threads), "options": {"NodefileStart": 0.25}}
    return cfg


def main() -> None:
    from runner import run_case

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", nargs="*", default=D4)
    ap.add_argument("--runs", nargs="*", default=list(CASES), choices=list(CASES))
    ap.add_argument("--levels", nargs="*", type=int, default=LEVELS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--extra-profile", nargs="*", default=None,
                    help="perfis sobrepostos depois de journal.yaml (ex.: config/pesgm.yaml)")
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args()

    cfg = profile(a.threads, a.extra_profile, a.results)
    rdir = results_dir(cfg)
    rdir.mkdir(exist_ok=True)
    base = {"vre_share": float(cfg["vre"]["target_share"]),
            "vpl_mwh": float(cfg["vpl"]["p_nom_mw"]) * float(cfg["vpl"]["hours"]),
            "dc_flex": float(cfg["vdc"]["flex_fraction"]),
            "reserve_from_flex": bool(cfg["reserve"]["from_flex"])}
    for day in a.days:
        for level in a.levels:
            for case in a.runs:
                name = f"{day}__{case}__s{level}"
                if (rdir / f"{name}.nc").exists():
                    print(f"[{name}] já existe — pulando", flush=True)
                    continue
                try:
                    run_case(day, layers=CASES[case], solver=a.solver, cfg=cfg,
                             overrides={"corridor.s_nom_mw": float(level)}, name=name,
                             tags={"axis": "corridor", "variation": f"s{level}"}, **base)
                except Exception as exc:  # registra e segue
                    print(f"[{name}] FALHOU: {exc}", flush=True)


if __name__ == "__main__":
    main()
