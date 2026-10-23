"""
ablation.py — ablação das correções estruturais do sistema de teste.

A Tabela II do artigo aplica quatro correções ao RTS-GMLC estressado para que
ele represente um corredor de exportação. Cada uma tem justificativa física e
uma medição que a motiva, mas o conjunto é extenso, e com todas aplicadas de
uma vez não se distingue uma propriedade geral do mecanismo de uma propriedade
induzida pela própria calibração.

Este módulo resolve a grade partindo do sistema SEM correção e aplicando-as
CUMULATIVAMENTE, uma de cada vez, na ordem em que a Tabela II as apresenta:

    S0  nenhuma          caminho paralelo CA-1 presente, coleta original,
                         penetração 50 %, composição original do parque
    S1  + radial         remove o caminho paralelo de exportação
    S2  + coleta         reforça a rede interna (exceto o corredor)
    S3  + penetração     reduz o alvo anual de VRE de 50 % para 35 %
    S4  + composição     repondera as tecnologias e torna a GD não-cortável
                         (equivale ao perfil ``config/realistic.yaml``)

Tudo o que não pertence às quatro correções — capacidade e comprimento do
corredor, dimensionamento do VPL e do VDC, dias representativos, tolerância do
solver — é mantido FIXO entre os estágios, de modo que a diferença entre duas
linhas consecutivas é atribuível a uma única mudança.

Para cada estágio resolvem-se os quatro casos (A base, B VPL, C VDC, D ambos)
nos dias representativos, e reporta-se o que o parecer pede: congestionamento
do corredor, curtailment, valor de cada camada e a interação I.

Uso::

    python src/ablation.py                    # grade completa (~80 MILP)
    python src/ablation.py --days spring_wind # subconjunto
    python src/ablation.py --report-only      # so regenera o relatorio
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_profile  # noqa: E402

# Estágios cumulativos. Cada entrada acrescenta overrides aos do estágio
# anterior; ``vre_share`` acompanha porque ``run_case`` recebe a penetração
# como argumento explícito, não pelo dicionário de configuração.
STAGES: list[tuple[str, str, dict, float]] = [
    ("S0", "none", {
        "stress.radial_export": False,
        "stress.collection_factor": None,
        "vre.carrier_weights": None,
        "vre.rtpv_must_take": False,
    }, 0.50),
    ("S1", "+radial export", {"stress.radial_export": True}, 0.50),
    ("S2", "+collection reinforcement", {
        "stress.collection_factor": 3.0,
        "stress.reinforce_scope": "all",
    }, 0.50),
    ("S3", "+penetration 35%", {}, 0.35),
    ("S4", "+carrier mix", {
        "vre.carrier_weights": {"Wind": 0.35, "Solar PV": 1.60, "Solar RTPV": 2.50},
        "vre.rtpv_must_take": True,
    }, 0.35),
]

CASES = {"A": set(), "B": {"VPL"}, "C": {"VDC"}, "D": {"VPL", "VDC"}}
RESULTS = "results_ablation"


def stage_overrides() -> list[tuple[str, str, dict, float]]:
    """Acumula os overrides de cada estágio sobre os anteriores."""
    acc: dict = {}
    out = []
    for sid, label, over, share in STAGES:
        acc = {**acc, **over}
        out.append((sid, label, dict(acc), share))
    return out


def run(days: list[str] | None, cases: list[str], solver: str | None, force: bool) -> None:
    """Resolve a grade de ablação, gravando em ``results_ablation/``."""
    from runner import run_case  # import tardio: carrega PyPSA só quando vai resolver

    cfg0 = load_profile(ROOT / "config" / "realistic.yaml")
    cfg0["paths"] = {**cfg0["paths"], "results": RESULTS}
    # dias representativos: os extremos, excluindo as chaves mensais m01..m12
    # (cuidado: "median" também começa por "m" e NÃO é um dia mensal)
    days = days or [d for d in cfg0["days"] if not re.fullmatch(r"m\d\d", d)]
    solver = solver or cfg0["solver"]["name"]
    rdir = ROOT / RESULTS

    total = len(stage_overrides()) * len(days) * len(cases)
    k = 0
    for sid, label, over, share in stage_overrides():
        for day in days:
            for case in cases:
                k += 1
                name = f"{day}__{case}__{sid}"
                if not force and (rdir / f"{name}.nc").exists():
                    print(f"[{k}/{total}] {name}: ja resolvido", flush=True)
                    continue
                print(f"[{k}/{total}] {name}  ({label}, VRE {share:.0%})", flush=True)
                try:
                    run_case(
                        day=day, vre_share=share, layers=CASES[case],
                        vpl_mwh=float(cfg0["vpl"]["p_nom_mw"]) * float(cfg0["vpl"]["hours"]),
                        dc_flex=float(cfg0["vdc"]["flex_fraction"]),
                        solver=solver,
                        reserve_from_flex=bool(cfg0["reserve"]["from_flex"]),
                        cfg=cfg0, overrides=over, name=name,
                        tags={"axis": "ablation", "variation": sid},
                    )
                except Exception:                      # um estágio inviável não aborta a grade
                    print(f"  FALHOU em {name}:", flush=True)
                    traceback.print_exc()


def active_constraints(day: str = "spring_wind", case: str = "A") -> pd.DataFrame:
    """Restrições de transmissão ativas em cada estágio, no caso base.

    Responde à pergunta que a tabela agregada não responde: quando o corredor
    não satura, o que o VPL está aliviando? Devolve, por estágio, quantas linhas
    saturam e qual é a mais carregada, o que distingue "aliviar a restrição
    errada" de "não haver restrição".
    """
    import pypsa

    rows = []
    for sid, *_ in STAGES:
        p = ROOT / RESULTS / f"{day}__{case}__{sid}.nc"
        if not p.exists():
            continue
        n = pypsa.Network(str(p))
        load = n.lines_t.p0.abs().div(n.lines["s_nom"], axis=1)
        sat = (load >= 0.99).sum()                       # horas saturadas por linha
        top = load.mean().idxmax()
        rows.append({"stage": sid, "lines saturated": int((sat > 0).sum()),
                     "most loaded": top, "mean util": load[top].mean(),
                     "hours saturated": int(sat[top]),
                     "is corridor": top in ("CB-1",)})
    return pd.DataFrame(rows)


def report(path: Path) -> None:
    """Gera ``results_ablation/ablation.md`` a partir das métricas resolvidas."""
    df = pd.read_csv(ROOT / RESULTS / "metrics.csv")
    order = [s[0] for s in STAGES]
    label = {s[0]: s[1] for s in STAGES}

    rows = []
    for sid in order:
        g = df[df["variation"] == sid]
        if g.empty:
            continue
        piv = g.pivot_table(index="day_key", columns="case", values="cost_total")
        gap = g.pivot_table(index="day_key", columns="case", values="mip_gap")
        if not set("ABCD") <= set(piv.columns):
            continue
        piv = piv.dropna()
        dB, dC, dD = piv["A"] - piv["B"], piv["A"] - piv["C"], piv["A"] - piv["D"]
        I = dD - (dB + dC)
        noise = (gap.reindex(piv.index) * piv).sum(axis=1)
        base = g[g["case"] == "A"]
        rows.append({
            "stage": sid, "correction": label[sid], "days": len(piv),
            "congested h": base["corridor_congested_hours"].mean(),
            "curtail %": base["curtailment_pct"].mean(),
            "dB k$": dB.mean() / 1e3, "dC k$": dC.mean() / 1e3, "dD k$": dD.mean() / 1e3,
            "I k$": I.mean() / 1e3, "noise k$": noise.mean() / 1e3,
            "I<0": f"{int((I < 0).sum())}/{len(I)}",
        })
    t = pd.DataFrame(rows)

    L = ["# Ablação das correções estruturais do sistema de teste\n",
         "Gerado por `src/ablation.py`. As quatro correções da Tabela II são aplicadas",
         "cumulativamente sobre o RTS-GMLC estressado; tudo o mais (corredor, VPL, VDC,",
         "dias, tolerância) permanece fixo, de modo que a diferença entre duas linhas",
         "consecutivas é atribuível a uma única mudança.\n",
         "`dB`/`dC`/`dD` são os benefícios das camadas sobre o caso base e",
         "`I = dD - (dB + dC)`; `noise` é a soma dos gaps absolutos dos quatro MILP,",
         "isto é, a meia-largura da cota rigorosa de I.\n",
         t.round(1).to_markdown(index=False) if len(t) else "_sem resultados_", ""]

    try:
        ac = active_constraints()
        L += ["## Restrição de transmissão ativa por estágio\n",
              "Caso base, dia `spring_wind`. Mostra o que o VPL alivia quando o corredor",
              "não satura: em S0 a restrição ativa é a coleta interna, não o tronco.\n",
              ac.round(2).to_markdown(index=False), ""]
    except Exception as exc:                 # relatório agregado não depende dos NetCDF
        L += [f"_restrições ativas não computadas: {exc}_", ""]

    path.write_text("\n".join(L), encoding="utf-8")
    print(t.round(1).to_string(index=False) if len(t) else "sem resultados")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--solver", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    if not a.report_only:
        run(a.days, a.cases, a.solver, a.force)
    report(ROOT / RESULTS / "ablation.md")


if __name__ == "__main__":
    main()
