"""
extra_runs.py — resoluções adicionais pedidas na revisão do artigo.

0. Condições iniciais herdadas (``--history``): o dia anterior a cada dia
   representativo é resolvido no caso A e o estado final das térmicas
   (ligada/desligada, potência, horas no estado) é usado como condição inicial
   dos quatro casos do dia de interesse, no lugar da regra "min_up >= 8 h começa
   ligada". Testa se as métricas de ciclagem dependem da condição de fronteira.
1. Sensibilidade ao custo de migração (``--migcost``): custo do VDC x0,5 e x2;
   ``--migsweep`` estende para sete pontos (x0 a x8) e estima por interpolação o
   custo em que o benefício do VDC se anula.
1b. Modo híbrido do VPL (``--hybrid``): regra de chaveamento só nas horas de
   alto uso, BESS livres no resto do dia (200 e 400 MW/terminal).
2. Decomposição do VPL (``--decomp``): caso B com apenas o BESS de montante
   (``vplup``) e apenas o de jusante (``vpldown``), nos quatro dias
   representativos. Valor do pareamento = ΔB − (ΔB_up + ΔB_down).
2. Dias mensais (``--monthly``): um dia útil por mês, o de energia de demanda
   líquida mais próxima da mediana mensal (excluídos os quatro dias já usados),
   casos A–D na variação base. Dá uma estimativa anual não viesada pela escolha
   dos dias extremos.

Os cenários entram em ``metrics.csv`` com ``axis = vpl_decomp`` e
``axis = monthly`` e ficam fora da tabela principal do ``summary.md``.

Uso::

    python src/extra_runs.py --decomp --monthly --solver gurobi
    python src/extra_runs.py --report          # só tabelas, a partir do metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_config, load_scenarios, results_dir  # noqa: E402

MONTHLY_DAYS = {
    1: "2020-01-03", 2: "2020-02-10", 3: "2020-03-18", 4: "2020-04-01", 5: "2020-05-26", 6: "2020-06-05",
    7: "2020-07-17", 8: "2020-08-24", 9: "2020-09-11", 10: "2020-10-09", 11: "2020-11-30", 12: "2020-12-07",
}
CASES = {"A": set(), "B": {"VPL"}, "C": {"VDC"}, "D": {"VPL", "VDC"}}


def _base_kw(cfg: dict) -> dict:
    return dict(vre_share=float(cfg["vre"]["target_share"]),
                vpl_mwh=float(cfg["vpl"]["p_nom_mw"]) * float(cfg["vpl"]["hours"]),
                dc_flex=float(cfg["vdc"]["flex_fraction"]),
                reserve_from_flex=bool(cfg["reserve"]["from_flex"]))


def run_decomp(cfg: dict, solver: str) -> None:
    from runner import run_case

    rdir = results_dir(cfg)
    for day in cfg["days"]:
        for label, term in (("vplup", "up"), ("vpldown", "down")):
            name = f"{day}__B__{label}"
            if (rdir / f"{name}.nc").exists():
                continue
            run_case(day, layers={"VPL"}, solver=solver, cfg=cfg, overrides={"vpl.terminals": term},
                     name=name, tags={"axis": "vpl_decomp", "variation": label}, **_base_kw(cfg))


def run_monthly(cfg: dict, solver: str) -> None:
    from runner import run_case

    rdir = results_dir(cfg)
    for m, date in MONTHLY_DAYS.items():
        key = f"m{m:02d}"
        for case, layers in CASES.items():
            name = f"{key}__{case}__base"
            if (rdir / f"{name}.nc").exists():
                continue
            cfg_m = dict(cfg)
            cfg_m["days"] = {**cfg["days"], key: date}
            run_case(key, layers=layers, solver=solver, cfg=cfg_m, name=name,
                     tags={"axis": "monthly", "variation": "base", "month": m}, **_base_kw(cfg))


def initial_state_from(nc_path: Path) -> dict:
    """Estado final das térmicas de uma rede resolvida: status, potência e
    quantas horas consecutivas cada unidade terminou o dia no mesmo estado."""
    import pypsa

    n = pypsa.Network(str(nc_path))
    st = n.generators_t.status
    p = n.generators_t.p
    out = {}
    for g in st.columns:
        v = (st[g].values > 0.5)
        last = bool(v[-1])
        streak = 1
        for k in range(len(v) - 2, -1, -1):
            if bool(v[k]) == last:
                streak += 1
            else:
                break
        out[g] = dict(on=last, streak=int(streak), p=float(p[g].iloc[-1]) if g in p else 0.0)
    return out


def apply_initial_state(state: dict):
    """Devolve um ``post_build(n, cfg)`` que injeta o estado inicial herdado."""

    def _hook(n, cfg):
        for g, s in state.items():
            if g not in n.generators.index:
                continue
            n.generators.loc[g, "up_time_before"] = s["streak"] if s["on"] else 0
            n.generators.loc[g, "down_time_before"] = 0 if s["on"] else s["streak"]
            n.generators.loc[g, "p_init"] = s["p"] if s["on"] else 0.0
        n.meta["initial_state"] = "inherited from previous day (case A)"

    return _hook


def run_history(cfg: dict, solver: str) -> None:
    """Resolve o dia anterior (caso A) e os quatro casos do dia com o estado herdado."""
    from datetime import datetime, timedelta

    from runner import run_case

    rdir = results_dir(cfg)
    for day, date in cfg["days"].items():
        prev = (datetime.fromisoformat(date) - timedelta(days=1)).date().isoformat()
        prev_name = f"{day}__A__prevday"
        prev_nc = rdir / f"{prev_name}.nc"
        if not prev_nc.exists():
            cfg_p = dict(cfg, days={**cfg["days"], f"{day}_prev": prev})
            run_case(f"{day}_prev", layers=set(), solver=solver, cfg=cfg_p, name=prev_name,
                     tags={"axis": "history", "variation": "prevday"}, **_base_kw(cfg))
        state = initial_state_from(prev_nc)
        for case, layers in CASES.items():
            name = f"{day}__{case}__hist"
            if (rdir / f"{name}.nc").exists():
                continue
            run_case(day, layers=layers, solver=solver, cfg=cfg, name=name,
                     tags={"axis": "history", "variation": "hist"},
                     post_build=apply_initial_state(state), **_base_kw(cfg))


def run_migcost(cfg: dict, solver: str) -> None:
    """Sensibilidade ao custo de migração do VDC (x0,5 e x2)."""
    from runner import run_case

    rdir = results_dir(cfg)
    lo, hi = cfg["vdc"]["workloads"]["migration_cost_range"]
    for label, k in (("mig05", 0.5), ("mig20", 2.0)):
        for day in cfg["days"]:
            for case, layers in (("C", {"VDC"}), ("D", {"VPL", "VDC"})):
                name = f"{day}__{case}__{label}"
                if (rdir / f"{name}.nc").exists():
                    continue
                run_case(day, layers=layers, solver=solver, cfg=cfg,
                         overrides={"vdc.workloads.migration_cost_range": [lo * k, hi * k]},
                         name=name, tags={"axis": "migration_cost", "variation": label}, **_base_kw(cfg))


# Multiplicadores da varredura fina do custo de migração. 0,5 e 2,0 já foram
# resolvidos por ``run_migcost`` (rótulos mig05/mig20) e são reaproveitados.
MIGSWEEP = {"mig000": 0.0, "mig025": 0.25, "mig05": 0.5, "migbase": 1.0,
            "mig20": 2.0, "mig40": 4.0, "mig80": 8.0}


def run_migsweep(cfg: dict, solver: str) -> None:
    """Varredura fina do custo de migração do VDC (``--migsweep``).

    Estende ``run_migcost`` de dois para sete pontos (x0 a x8) para estimar por
    interpolação o custo de migração em que o benefício do VDC se anula. O ponto
    x0 (migração gratuita) é a cota superior do valor da camada: tudo que
    sobra é o limite físico imposto pelas janelas, pela capacidade de site e
    pela rampa agregada, não pelo preço de mover a carga.

    ``migbase`` (x1) e os casos já resolvidos em ``base`` são o mesmo cenário:
    não se re-resolve, o relatório lê a linha de ``axis = base``.
    """
    from runner import run_case

    rdir = results_dir(cfg)
    lo, hi = cfg["vdc"]["workloads"]["migration_cost_range"]
    for label, k in MIGSWEEP.items():
        if label == "migbase":
            continue                                  # já existe como variação base
        for day in cfg["days"]:
            for case, layers in (("C", {"VDC"}), ("D", {"VPL", "VDC"})):
                name = f"{day}__{case}__{label}"
                if (rdir / f"{name}.nc").exists():
                    continue
                run_case(day, layers=layers, solver=solver, cfg=cfg,
                         overrides={"vdc.workloads.migration_cost_range": [lo * k, hi * k]},
                         name=name, tags={"axis": "migration_cost", "variation": label}, **_base_kw(cfg))


def run_hybrid(cfg: dict, solver: str) -> None:
    """Casos B e D no modo híbrido do VPL (``--hybrid``).

    Regra de chaveamento só nas horas de alto uso (sigma_t = 1); fora do pico os
    dois BESS ficam livres. Fecha a escala de restrição do VPL:
    ``dedicated`` (sempre) > ``explicit`` (pico e vale) > ``hybrid`` (só pico) >
    ``emergent`` (nunca). Resolvido também com 400 MW/terminal para comparar com
    ``run_leituras400``.
    """
    from runner import run_case

    rdir = results_dir(cfg)
    p_base = float(cfg["vpl"]["p_nom_mw"]) * float(cfg["vpl"]["hours"])
    for label, mwh, axis in (("hybrid", p_base, "vpl_mode"),
                             ("vpl400hybrid", 400 * float(cfg["vpl"]["hours"]), "vpl_mode_size")):
        for day in cfg["days"]:
            for case, layers in (("B", {"VPL"}), ("D", {"VPL", "VDC"})):
                name = f"{day}__{case}__{label}"
                if (rdir / f"{name}.nc").exists():
                    continue
                kw = dict(_base_kw(cfg))
                kw["vpl_mwh"] = mwh
                run_case(day, layers=layers, solver=solver, cfg=cfg,
                         overrides={"vpl.mode": "hybrid"},
                         name=name, tags={"axis": axis, "variation": label}, **kw)


def run_ladder(cfg: dict, solver: str, scen: dict | None = None) -> None:
    """Escada de flexibilidade de 6 degraus (``--ladder``), definida em ``scenarios.yaml``.

    Cada degrau acrescenta um grau de liberdade ao anterior, de forma que a
    diferença entre degraus consecutivos isola o valor daquele grau — em vez do
    desenho A/B/C/D, em que VPL e VDC entram inteiros e a decomposição do VDC em
    tempo e espaço fica implícita.

    Os cenários entram em ``metrics.csv`` com ``axis = ladder`` e ``case`` em
    ``L1``–``L6`` (a tag sobrescreve o rótulo A–D que ``run_case`` derivaria do
    conjunto de camadas, já que dois degraus podem ter as mesmas camadas e
    diferir só nos ``overrides``).
    """
    from runner import run_case

    scen = scen or load_scenarios()
    rdir = results_dir(cfg)
    for step, spec in scen["ladder"].items():
        for day in cfg["days"]:
            name = f"{day}__{step}__{spec['label']}"
            if (rdir / f"{name}.nc").exists():
                continue
            run_case(day, layers=set(spec["layers"]), solver=solver, cfg=cfg,
                     overrides=dict(spec.get("over") or {}), name=name,
                     tags={"axis": "ladder", "variation": spec["label"], "case": step},
                     **_base_kw(cfg))


WIRE_FACTORS = (1.10, 1.20, 1.40, 1.80)


def run_ladder_monthly(cfg: dict, solver: str, scen: dict | None = None) -> None:
    """Escada de flexibilidade nos 12 dias mensais (``--laddermonthly``).

    Os quatro dias representativos são extremos escolhidos a dedo (pico anual,
    maior rampa, maior excedente eólico, mediana) e superestimam o valor anual da
    flexibilidade. A comparação econômica entre o VPL e o reforço de corredor não
    pode repousar nessa base: um fator de ~0,78 entre as duas bases é suficiente
    para inverter o sinal de uma razão benefício/custo próxima de 1.

    Reaproveita os cenários mensais já resolvidos onde os degraus coincidem com a
    grade A–D: **L1 = caso A** (sem camadas) e **L4 = caso C** (VDC completo).
    Só L2, L3, L5 e L6 precisam ser resolvidos.
    """
    from runner import run_case

    scen = scen or load_scenarios()
    rdir = results_dir(cfg)
    reuse = {"L1": "A", "L4": "C"}          # já existem como <mês>__<caso>__base
    for step, spec in scen["ladder"].items():
        if step in reuse:
            continue
        for m, date in MONTHLY_DAYS.items():
            key = f"m{m:02d}"
            name = f"{key}__{step}__{spec['label']}"
            if (rdir / f"{name}.nc").exists():
                continue
            cfg_m = dict(cfg)
            cfg_m["days"] = {**cfg["days"], key: date}
            run_case(key, layers=set(spec["layers"]), solver=solver, cfg=cfg_m,
                     overrides=dict(spec.get("over") or {}), name=name,
                     tags={"axis": "ladder_monthly", "variation": spec["label"],
                           "case": step, "month": m},
                     **_base_kw(cfg))


def run_wire_curve(cfg: dict, solver: str) -> None:
    """Curva de oferta do alívio de corredor (``--wirecurve``).

    Resolve o caso base com ``s_nom`` do corredor multiplicado por cada fator de
    ``WIRE_FACTORS``, em todos os dias representativos. Duas leituras:

    * o **$/MW decrescente** mostra onde o reforço deixa de compensar;
    * o **patamar** (fator em que as horas congestionadas chegam a zero) é o
      *teto do valor de aliviar o corredor* naquele dia — nenhuma quantidade de
      flexibilidade que atue só sobre o congestionamento pode render mais que
      isso.

    O segundo número é o que dá sentido ao VPL: se ΔB excede o patamar, o
    benefício do par de BESS não vem de substituir transmissão, e chamá-lo de
    linha virtual descreve mal a origem do ganho.
    """
    from runner import run_case

    rdir = results_dir(cfg)
    for day in cfg["days"]:
        for f in WIRE_FACTORS:
            name = f"{day}__W__calib{int(f * 100)}"
            if (rdir / f"{name}.nc").exists():
                continue
            run_case(day, layers=set(), solver=solver, cfg=cfg,
                     overrides={"corridor.limit_corridor_factor": f}, name=name,
                     tags={"axis": "wire_calib", "variation": f"f{f:.2f}"}, **_base_kw(cfg))


def run_wire_equivalent(cfg: dict, solver: str, tol_mw: float = 5.0, max_iter: int = 8) -> None:
    """Reforço do corredor equivalente ao VPL híbrido (``--wireequiv``).

    Busca, por bissecção sobre ``corridor.limit_corridor_factor``, o acréscimo de
    ``s_nom`` que reproduz o mesmo custo operacional do degrau L2 (VPL híbrido)
    naquele dia. O resultado — "o par de BESS vale X MW de linha" — é a única
    forma de comparar as duas alternativas na mesma unidade, e é o número que
    falta em ``economics.py`` §5 para fechar a conta econômica do VPL.

    Cada avaliação é um MILP completo; a bissecção custa ``max_iter`` resoluções
    por dia. Os cenários intermediários entram em ``metrics.csv`` com
    ``axis = wire_equiv`` para que a busca seja auditável.
    """
    from runner import run_case

    rdir = results_dir(cfg)
    df = pd.read_csv(rdir / "metrics.csv").set_index("scenario")
    out = []
    for day in cfg["days"]:
        try:
            target = float(df.loc[f"{day}__L2__vpl", "cost_total"])
            base_cost = float(df.loc[f"{day}__L1__base", "cost_total"])
        except KeyError:
            print(f"[wireequiv] {day}: rode --ladder antes (faltam L1/L2).")
            continue
        s_nom = float(df.loc[f"{day}__L1__base", "corridor_limit_mw"])

        lo, hi = 1.0, 2.0            # fator sobre s_nom; 2.0 = corredor dobrado
        best = None
        for it in range(max_iter):
            mid = 0.5 * (lo + hi)
            name = f"{day}__W{it}__wireeq"
            if not (rdir / f"{name}.nc").exists():
                run_case(day, layers=set(), solver=solver, cfg=cfg,
                         overrides={"corridor.limit_corridor_factor": mid}, name=name,
                         tags={"axis": "wire_equiv", "variation": f"f{mid:.4f}"}, **_base_kw(cfg))
            cost = float(pd.read_csv(rdir / "metrics.csv").set_index("scenario").loc[name, "cost_total"])
            # custo cai monotonicamente com o reforço: se ainda está acima do
            # alvo, o reforço é insuficiente
            if cost > target:
                lo = mid
            else:
                hi = mid
                best = mid
            if (hi - lo) * s_nom < tol_mw:
                break
        if best is not None:
            out.append(dict(day=day, fator=best, mw_equivalente=(best - 1.0) * s_nom,
                            dB_hibrido=base_cost - target, alcancavel=True))
        else:
            # Nenhum reforço dentro de [1, hi] igualou o custo do L2: o benefício
            # do VPL excede o teto do alívio de corredor. Não é falha da busca —
            # é o resultado. Registra o teto para que o relatório possa dizê-lo.
            ceiling = base_cost - cost
            out.append(dict(day=day, fator=np.nan, mw_equivalente=np.nan,
                            dB_hibrido=base_cost - target, alcancavel=False,
                            teto_reforco=ceiling, mw_testados=(hi - 1.0) * s_nom))
            print(f"[wireequiv] {day}: VPL vale {base_cost - target:,.0f} $/dia, acima do teto de "
                  f"{ceiling:,.0f} $/dia obtido com +{(hi - 1.0) * s_nom:.0f} MW de corredor. "
                  "O benefício do VPL não é substituível por transmissão.")
    if out:
        t = pd.DataFrame(out)
        t.loc["média"] = t.drop(columns="day").mean(numeric_only=True)
        t.loc["média", "day"] = "média"
        md = ["# Reforço de corredor equivalente ao VPL híbrido",
              "",
              "`fator` = multiplicador de `s_nom` do corredor que iguala o custo",
              "operacional do degrau L2; `mw_equivalente` = capacidade de linha que",
              "o par de BESS substitui.",
              "",
              t.round(1).to_markdown(index=False),
              ""]
        (results_dir(cfg) / "wire_equivalent.md").write_text("\n".join(md), encoding="utf-8")
        print(t.round(1).to_string(index=False))


def run_native(cfg: dict, solver: str) -> None:
    """A e B com o baseline 'native' (mesmos jobs no site nativo, EDF desde a chegada)."""
    from runner import run_case

    rdir = results_dir(cfg)
    for day in cfg["days"]:
        for case, layers in (("A", set()), ("B", {"VPL"})):
            name = f"{day}__{case}__native"
            if (rdir / f"{name}.nc").exists():
                continue
            run_case(day, layers=layers, solver=solver, cfg=cfg, overrides={"vdc.baseline": "native"},
                     name=name, tags={"axis": "baseline", "variation": "native"}, **_base_kw(cfg))


def run_soc(cfg: dict, solver: str) -> None:
    """B e D com o SOC inicial dos BESS ancorado em 50 % de e_nom."""
    from runner import run_case

    rdir = results_dir(cfg)
    for day in cfg["days"]:
        for case, layers in (("B", {"VPL"}), ("D", {"VPL", "VDC"})):
            name = f"{day}__{case}__soc50"
            if (rdir / f"{name}.nc").exists():
                continue
            run_case(day, layers=layers, solver=solver, cfg=cfg, overrides={"vpl.soc_initial": 0.5},
                     name=name, tags={"axis": "soc_initial", "variation": "soc50"}, **_base_kw(cfg))


def run_leituras400(cfg: dict, solver: str) -> None:
    """VPL de 400 MW sob as três leituras: emergente, explícita e dedicada.

    No porte base a interação fica dentro do ruído do gap nas três leituras, então
    a comparação só é informativa onde o sinal de competição é mensurável, isto é,
    com VPL grande. A leitura emergente de 400 MW já vem da grade (``vpl400``);
    aqui resolvem-se as duas variantes com regra de chaveamento.
    """
    from runner import run_case

    rdir = results_dir(cfg)
    hours = float(cfg["vpl"]["hours"])
    kw = _base_kw(cfg)
    kw["vpl_mwh"] = 400.0 * hours
    for day in cfg["days"]:
        for case, layers in (("B", {"VPL"}), ("D", {"VPL", "VDC"})):
            for tag, mode in (("vpl400dedic", "dedicated"), ("vpl400explic", "explicit")):
                name = f"{day}__{case}__{tag}"
                if (rdir / f"{name}.nc").exists():
                    continue
                run_case(day, layers=layers, solver=solver, cfg=cfg,
                         overrides={"vpl.mode": mode, "vpl.p_nom_mw": 400},
                         name=name, tags={"axis": "vpl_mode_size", "variation": tag}, **kw)


def run_reserve_ref(cfg: dict, solver: str) -> None:
    """C e D com o mesmo requisito de reserva dos casos A e B.

    Em A e B a carga flexivel do data center entra como Load plana e conta em
    L_t; em C e D ela passa por Links e nao conta, o que torna R_t menor nesses
    casos. Esta sensibilidade soma de volta o equivalente plano, igualando R_t
    nos quatro casos, para separar o valor da flexibilidade do efeito de a carga
    flexivel nao gerar requisito de reserva. A e B nao mudam (nao tem Links VDC).
    """
    from runner import run_case

    rdir = results_dir(cfg)
    for day in cfg["days"]:
        for case, layers in (("C", {"VDC"}), ("D", {"VPL", "VDC"})):
            name = f"{day}__{case}__resref"
            if (rdir / f"{name}.nc").exists():
                continue
            run_case(day, layers=layers, solver=solver, cfg=cfg,
                     overrides={"reserve.common_reference": True},
                     name=name, tags={"axis": "reserve_ref", "variation": "resref"},
                     **_base_kw(cfg))


def report(cfg: dict) -> None:
    df = pd.read_csv(results_dir(cfg) / "metrics.csv").set_index("scenario")
    lines = ["# Resoluções adicionais — decomposição do VPL e dias mensais\n"]

    # 1. decomposição
    rows = []
    for day in cfg["days"]:
        try:
            A = df.loc[f"{day}__A__base", "cost_total"]
            B = df.loc[f"{day}__B__base", "cost_total"]
            up = df.loc[f"{day}__B__vplup", "cost_total"]
            dn = df.loc[f"{day}__B__vpldown", "cost_total"]
            vc_b = df.loc[f"{day}__B__base", "vpl_virtual_mwh"]
        except KeyError:
            continue
        dB, dUp, dDn = A - B, A - up, A - dn
        rows.append(dict(day=day, dB=dB, dB_up=dUp, dB_down=dDn, pairing=dB - (dUp + dDn),
                         pairing_pct=100 * (dB - (dUp + dDn)) / dB if dB else float("nan"), virtual_mwh=vc_b))
    dec = pd.DataFrame(rows)
    if not dec.empty:
        dec.loc["média"] = dec.drop(columns="day").mean(numeric_only=True)
        dec.loc["média", "day"] = "média"
        lines.append("## 1. Decomposição do benefício do VPL (caso B, 50 % VRE), $/dia\n")
        lines.append("ΔB = benefício do par; ΔB_up / ΔB_down = benefício de cada BESS sozinho (arbitragem nodal isolada); "
                     "pairing = ΔB − (ΔB_up + ΔB_down) = valor atribuível ao pareamento nos dois terminais.\n")
        lines.append(dec.round(0).to_markdown(index=False) + "\n")

    # 1b. baseline native
    rows = []
    for day in cfg["days"]:
        try:
            A, B = df.loc[f"{day}__A__base", "cost_total"], df.loc[f"{day}__B__base", "cost_total"]
            C, D = df.loc[f"{day}__C__base", "cost_total"], df.loc[f"{day}__D__base", "cost_total"]
            An, Bn = df.loc[f"{day}__A__native", "cost_total"], df.loc[f"{day}__B__native", "cost_total"]
            sA, sAn = df.loc[f"{day}__A__base", "startups_total"], df.loc[f"{day}__A__native", "startups_total"]
            sC = df.loc[f"{day}__C__base", "startups_total"]
        except KeyError:
            continue
        rows.append(dict(day=day, dC_flat=A - C, dC_native=An - C, dD_flat=A - D, dD_native=An - Bn + Bn - D,
                         I_flat=(A - D) - ((A - B) + (A - C)), I_native=(An - D) - ((An - Bn) + (An - C)),
                         startups_A_flat=sA, startups_A_native=sAn, startups_C=sC))
    nat = pd.DataFrame(rows)
    if not nat.empty:
        nat.loc["média"] = nat.drop(columns="day").mean(numeric_only=True)
        nat.loc["média", "day"] = "média"
        lines.append("## 1b. Sensibilidade ao baseline dos casos A/B: flat vs. native, $/dia\n")
        lines.append("flat = energia dos workloads distribuída uniformemente; native = mesmos jobs no site nativo, EDF desde a chegada.\n")
        lines.append(nat.round(0).to_markdown(index=False) + "\n")

    # 1c. condições iniciais herdadas
    rows = []
    for day in cfg["days"]:
        try:
            base = {c: df.loc[f"{day}__{c}__base"] for c in "ABCD"}
            hist = {c: df.loc[f"{day}__{c}__hist"] for c in "ABCD"}
        except KeyError:
            continue
        rows.append(dict(
            day=day,
            su_A=base["A"].startups_total, su_A_h=hist["A"].startups_total,
            su_D=base["D"].startups_total, su_D_h=hist["D"].startups_total,
            dsu_base=base["A"].startups_total - base["D"].startups_total,
            dsu_hist=hist["A"].startups_total - hist["D"].startups_total,
            I_base=(base["A"].cost_total - base["D"].cost_total)
            - ((base["A"].cost_total - base["B"].cost_total) + (base["A"].cost_total - base["C"].cost_total)),
            I_hist=(hist["A"].cost_total - hist["D"].cost_total)
            - ((hist["A"].cost_total - hist["B"].cost_total) + (hist["A"].cost_total - hist["C"].cost_total)),
        ))
    h = pd.DataFrame(rows)
    if not h.empty:
        h.loc["média"] = h.drop(columns="day").mean(numeric_only=True)
        h.loc["média", "day"] = "média"
        lines.append("## 1c. Condições iniciais: regra padrão vs. herdadas do dia anterior\n")
        lines.append("su = partidas; dsu = partidas evitadas por D; I = interação de custo ($/dia).\n")
        lines.append(h.round(1).to_markdown(index=False) + "\n")

    # 1d. custo de migração
    rows = []
    for day in cfg["days"]:
        r = {"day": day}
        ok = True
        for lbl, tag in (("base", "base"), ("x0,5", "mig05"), ("x2", "mig20")):
            try:
                A = df.loc[f"{day}__A__base", "cost_total"]
                r[f"dC {lbl}"] = A - df.loc[f"{day}__C__{tag}", "cost_total"]
                r[f"dD {lbl}"] = A - df.loc[f"{day}__D__{tag}", "cost_total"]
            except KeyError:
                ok = False
        if ok:
            rows.append(r)
    mg = pd.DataFrame(rows)
    if not mg.empty:
        mg.loc["média"] = mg.drop(columns="day").mean(numeric_only=True)
        mg.loc["média", "day"] = "média"
        lines.append("## 1d. Sensibilidade ao custo de migração do VDC ($/dia)\n")
        lines.append(mg.round(0).to_markdown(index=False) + "\n")

    # 1e. SOC inicial ancorado
    rows = []
    for day in cfg["days"]:
        try:
            A = df.loc[f"{day}__A__base", "cost_total"]
            B, D = df.loc[f"{day}__B__base", "cost_total"], df.loc[f"{day}__D__base", "cost_total"]
            Bs, Ds = df.loc[f"{day}__B__soc50", "cost_total"], df.loc[f"{day}__D__soc50", "cost_total"]
            C = df.loc[f"{day}__C__base", "cost_total"]
        except KeyError:
            continue
        rows.append(dict(day=day, dB_livre=A - B, dB_soc50=A - Bs, dD_livre=A - D, dD_soc50=A - Ds,
                         I_livre=(A - D) - ((A - B) + (A - C)),
                         I_soc50=(A - Ds) - ((A - Bs) + (A - C))))
    soc = pd.DataFrame(rows)
    if not soc.empty:
        soc.loc["média"] = soc.drop(columns="day").mean(numeric_only=True)
        soc.loc["média", "day"] = "média"
        lines.append("## 1e. SOC inicial do VPL: livre (ciclo fechado) vs. ancorado em 50 %\n")
        lines.append(soc.round(0).to_markdown(index=False) + "\n")

    # 2. dias mensais
    mon = df[df["axis"] == "monthly"] if "axis" in df else df.iloc[0:0]
    if not mon.empty:
        piv = mon.pivot_table(index="day_key", columns="case", values=["cost_total", "curtailment_mwh", "startups_total", "co2_t"])
        cost = piv["cost_total"]
        tab = pd.DataFrame({
            "A": cost["A"], "ΔB": cost["A"] - cost["B"], "ΔC": cost["A"] - cost["C"], "ΔD": cost["A"] - cost["D"],
        })
        tab["I"] = tab["ΔD"] - (tab["ΔB"] + tab["ΔC"])
        gaps = mon.pivot_table(index="day_key", columns="case", values="mip_gap")
        tab["ruído"] = (gaps * cost).sum(axis=1)
        tab["curt_A_MWh"] = piv["curtailment_mwh"]["A"]
        tab["curt_red_D_%"] = 100 * (1 - piv["curtailment_mwh"]["D"] / piv["curtailment_mwh"]["A"])
        tab["startups_A"] = piv["startups_total"]["A"]
        tab["startups_D"] = piv["startups_total"]["D"]
        tab.loc["média"] = tab.mean(numeric_only=True)
        lines.append("## 2. Dias mensais (um dia útil mediano por mês), variação base, $/dia\n")
        lines.append(tab.round(1).to_markdown() + "\n")
        a = cost["A"].sum(); d = cost["D"].sum()
        lines.append(f"Redução de custo D vs A, soma dos 12 dias: {100 * (a - d) / a:.1f} %; "
                     f"ΔB {100 * (a - cost['B'].sum()) / a:.1f} %; ΔC {100 * (a - cost['C'].sum()) / a:.1f} %. "
                     f"Curtailment D vs A: {100 * (1 - piv['curtailment_mwh']['D'].sum() / piv['curtailment_mwh']['A'].sum()):.1f} %; "
                     f"partidas D vs A: {100 * (1 - piv['startups_total']['D'].sum() / piv['startups_total']['A'].sum()):.1f} %. "
                     f"I médio {tab.loc['média', 'I']:.0f} $/dia; I < 0 em {(tab['I'].iloc[:-1] < 0).sum()} de {len(tab) - 1} dias; "
                     f"|I| > 2×ruído em {((tab['I'].iloc[:-1].abs()) > 2 * tab['ruído'].iloc[:-1]).sum()} dias.\n")
    out = results_dir(cfg) / "extra_runs.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default=None)
    ap.add_argument("--decomp", action="store_true")
    ap.add_argument("--monthly", action="store_true")
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--migcost", action="store_true")
    ap.add_argument("--migsweep", action="store_true",
                    help="varredura fina do custo de migração (x0 a x8)")
    ap.add_argument("--hybrid", action="store_true",
                    help="modo híbrido do VPL (regra só no pico), 200 e 400 MW/terminal")
    ap.add_argument("--ladder", action="store_true",
                    help="escada de flexibilidade de 6 degraus (L1–L6)")
    ap.add_argument("--laddermonthly", action="store_true",
                    help="escada de 6 degraus nos 12 dias mensais (base anual nao enviesada)")
    ap.add_argument("--wirecurve", action="store_true",
                    help="curva de oferta do alivio de corredor (+10/20/40/80 %%)")
    ap.add_argument("--wireequiv", action="store_true",
                    help="bissecção do reforço de corredor equivalente ao VPL híbrido")
    ap.add_argument("--soc", action="store_true")
    ap.add_argument("--resref", action="store_true",
                    help="C e D com o mesmo R_t de A e B")
    ap.add_argument("--leituras", action="store_true",
                    help="VPL 400 MW nas variantes explícita e dedicada")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    solver = args.solver or cfg["solver"]["name"]
    if args.decomp:
        run_decomp(cfg, solver)
    if args.monthly:
        run_monthly(cfg, solver)
    if args.native:
        run_native(cfg, solver)
    if args.history:
        run_history(cfg, solver)
    if args.migcost:
        run_migcost(cfg, solver)
    if args.migsweep:
        run_migsweep(cfg, solver)
    if args.hybrid:
        run_hybrid(cfg, solver)
    if args.ladder:
        run_ladder(cfg, solver)
    if args.laddermonthly:
        run_ladder_monthly(cfg, solver)
    if args.wirecurve:
        run_wire_curve(cfg, solver)
    if args.wireequiv:
        run_wire_equivalent(cfg, solver)
    if args.soc:
        run_soc(cfg, solver)
    if args.resref:
        run_reserve_ref(cfg, solver)
    if args.leituras:
        run_leituras400(cfg, solver)
    report(cfg)


if __name__ == "__main__":
    main()
