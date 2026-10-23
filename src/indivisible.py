"""
indivisible.py — sensibilidade à divisibilidade das cargas dos data centers.

Por que existe. No modelo do artigo cada workload é uma quantidade contínua
``x_{j,d,t}``: pode ser repartido entre sites na mesma hora, modulado em
potência e interrompido à vontade dentro da janela. Data centers reais não são
assim --- um job roda num lugar, com os dados que lá estão, e migrar custa e
tem limite. O Flavio apontou (out/2026) que um revisor vai cobrar isso.

A resposta é cercar o resultado pelo outro extremo. Na variante rígida
(``vdc.workloads.rigid``, ver ``layers.rigid_jobs`` e
``constraints.add_vdc_rigid``) cada job conserva a forma exata que tem no
cronograma de referência e só pode ser deslocado inteiro no tempo, dentro da
janela, e/ou transferido inteiro para outro site elegível: sem divisão entre
sites, sem modulação, sem preempção, migração tudo-ou-nada. O modelo contínuo é
a cota superior da flexibilidade; o rígido, a inferior para os mesmos jobs. Se
a substituição e a assimetria sobrevivem nos dois, a hipótese de divisibilidade
não é o que produz o resultado.

A referência (deslocamento nulo, site nativo) é uma solução da variante rígida,
de modo que os casos sem data center flexível (A, B, E, F) não mudam e são lidos
de ``results_journal/``. Aqui se resolvem, nos doze dias mensais:

    C__rigid_local   FDC rígido (só deslocamento no tempo, no site nativo)
    C__rigid         VDC rígido
    D__rigid         VPL + VDC rígido
    G__rigid         VDC rígido + H2
    H__rigid         os três, VDC rígido
    C__local         FDC contínuo, no mesmo perfil, para a comparação da fatia local

Uso::

    python src/indivisible.py                      # tudo, em série
    python src/indivisible.py --days m01 m02 m03   # um lote (rodar lotes em paralelo)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_profile, results_dir  # noqa: E402

RESULTS = "results_journal_rigid"
MONTHLY = [f"m{m:02d}" for m in range(1, 13)]
RIGID = {"vdc.workloads.rigid": True}
LOCAL = {"vdc.spatial": False}
# rótulo -> (camadas, sobrescritas, caso, variação)
RUNS = {
    "C_rigid_local": ({"VDC"}, {**RIGID, **LOCAL}, "C", "rigid_local"),
    "C_rigid": ({"VDC"}, RIGID, "C", "rigid"),
    "D_rigid": ({"VPL", "VDC"}, RIGID, "D", "rigid"),
    "G_rigid": ({"VDC", "H2"}, RIGID, "G", "rigid"),
    "H_rigid": ({"VPL", "VDC", "H2"}, RIGID, "H", "rigid"),
    "C_local": ({"VDC"}, LOCAL, "C", "local"),
}


def profile(threads: int = 4, extra: list[str] | None = None, results: str = RESULTS) -> dict:
    # ``extra``: perfis sobrepostos depois de journal.yaml (o PES GM usa
    # config/pesgm.yaml); ``results``: destino. Os padrões são os do periódico.
    perfis = [ROOT / "config" / "realistic.yaml", ROOT / "config" / "journal.yaml"]
    cfg = load_profile(perfis + [ROOT / e for e in (extra or [])])
    cfg["paths"] = {**cfg["paths"], "results": results}
    # Poucas threads e árvore em disco a partir de 0,25 GB, a receita de
    # robustness_tight.py depois que o guarda de memória matou a grade duas
    # vezes (aqui também, em 22/10/2026, com 2 GB livres na máquina). Não muda
    # o ótimo, só o pico de memória; os tempos desta sensibilidade não são
    # reportados.
    cfg["solver"] = {**cfg["solver"], "threads": int(threads), "options": {"NodefileStart": 0.25}}
    return cfg


def main() -> None:
    from runner import run_case

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", nargs="*", default=MONTHLY)
    ap.add_argument("--runs", nargs="*", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--extra-profile", nargs="*", default=None,
                    help="perfis sobrepostos depois de journal.yaml (ex.: config/pesgm.yaml)")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    cfg = profile(a.threads, a.extra_profile, a.results)
    rdir = results_dir(cfg)
    rdir.mkdir(exist_ok=True)
    base = {"vre_share": float(cfg["vre"]["target_share"]),
            "vpl_mwh": float(cfg["vpl"]["p_nom_mw"]) * float(cfg["vpl"]["hours"]),
            "dc_flex": float(cfg["vdc"]["flex_fraction"]),
            "reserve_from_flex": bool(cfg["reserve"]["from_flex"])}
    for day in a.days:
        for label in a.runs:
            layers, over, case, variation = RUNS[label]
            name = f"{day}__{case}__{variation}"
            if (rdir / f"{name}.nc").exists():
                print(f"[{name}] já existe — pulando", flush=True)
                continue
            try:
                run_case(day, layers=layers, solver=a.solver, cfg=cfg, overrides=over, name=name,
                         tags={"axis": "indivisible", "variation": variation}, **base)
            except Exception as exc:  # registra e segue
                print(f"[{name}] FALHOU: {exc}", flush=True)


if __name__ == "__main__":
    main()
