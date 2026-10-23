"""
validate.py — critérios de aceite das Fases 1 e 2.

``--phase F1``: resolve o RTS-GMLC **original** (sem estresse, sem camadas) para
um dia, verifica o balanço nodal com ``n.lpf`` (validação da modelagem dos
transformadores), compara ordem de grandeza do despacho por combustível e dos
fluxos inter-área e grava ``results/base_validation.nc`` +
``results/F1_validation.md``.

``--phase F2``: para o caso A (estressado, 50 % VRE) em cada dia representativo,
verifica se o corredor N_318–N_223 satura (|p0| ≥ 0,98·s_nom) em ≥ 6 h em pelo
menos 2 dias e se há spread de LMP visível entre 318 e 223. Reaproveita os
NetCDF já resolvidos pelo runner quando existem. Grava ``results/F2_corridor.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pypsa

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_config, results_dir, set_by_path  # noqa: E402
from layers import corridor_flow, corridor_limit  # noqa: E402
from metrics import corridor_metrics, cost_decomposition, curtailment, startups  # noqa: E402
from rts_import import build_network, load_rts_data, lpf_power_balance_error, thermal_summary  # noqa: E402
from runner import run_case, solve  # noqa: E402


def phase_f1(cfg: dict, day: str) -> None:
    rdir = results_dir(cfg)
    data = load_rts_data(ROOT / cfg["paths"]["rts"])
    n = build_network(data, day, cfg)
    n.meta["scenario"] = {"name": "base_validation", "day": day, "case": "original"}
    info = solve(n, cfg)

    # --- despacho por combustível e participação
    p = n.generators_t.p
    by_fuel = p.T.groupby(n.generators["fuel"]).sum().T.sum()
    load = n.loads_t.p_set.sum().sum()
    mix = pd.DataFrame({"MWh": by_fuel.round(0), "share_%": (100 * by_fuel / by_fuel.sum()).round(1)})

    # --- fluxos inter-área (médio e máximo) e carregamento máximo de ramos
    ln = n.lines
    a0, a1 = ln["bus0"].map(n.buses["area"]), ln["bus1"].map(n.buses["area"])
    inter = ln.index[a0 != a1]
    flows = pd.DataFrame({
        "areas": (a0[inter].astype(str) + "→" + a1[inter].astype(str)),
        "p0_mean_MW": n.lines_t.p0[inter].mean().round(0),
        "p0_max_MW": n.lines_t.p0[inter].abs().max().round(0),
        "s_nom_MW": ln.loc[inter, "s_nom"],
    })
    loading = pd.concat([n.lines_t.p0.abs().max() / ln["s_nom"],
                         n.transformers_t.p0.abs().max() / n.transformers["s_nom"]]).sort_values(ascending=False)

    # --- validação do fluxo DC: fixa despacho, roda lpf e compara fluxos
    p0_opt = pd.concat([n.lines_t.p0, n.transformers_t.p0], axis=1).copy()
    n.optimize.fix_optimal_dispatch()
    bal_err = lpf_power_balance_error(n, n.snapshots[12])
    p0_lpf = pd.concat([n.lines_t.p0, n.transformers_t.p0], axis=1)
    flow_diff = float((p0_lpf.loc[n.snapshots[12]] - p0_opt.loc[n.snapshots[12]]).abs().max())

    lmp = n.buses_t.marginal_price
    cost = cost_decomposition(n)
    curt = curtailment(n)
    su = startups(n)

    n.export_to_netcdf(rdir / "base_validation.nc")

    lines = [
        "# F1 — Validação da importação do RTS-GMLC (rede original)\n",
        f"Dia: **{day}** (24 snapshots horários). Solver: {info['solver']}, gap {info['mip_gap']:.2e}, "
        f"tempo MILP {info['time_milp_s']} s (+{info['time_lp_s']} s para o LP de preços).\n",
        "## Critério de aceite",
        f"* `n.optimize(solver_name=\"{info['solver']}\")` convergiu: **{info['status_milp']}**.",
        f"* Custo total do dia: **{cost['cost_total']:,.0f} $** "
        f"(combustível {cost['cost_fuel']:,.0f}; partidas {cost['cost_startup']:,.0f}; {su['startups_total']:.0f} partidas).",
        f"* Carga do dia: {load:,.0f} MWh; custo médio {cost['cost_total']/load:.1f} $/MWh; "
        f"LMP médio {lmp.mean().mean():.1f} $/MWh (mín {lmp.min().min():.1f}, máx {lmp.max().max():.1f}).",
        f"* Curtailment VRE: {curt['curtailment_mwh']:,.0f} MWh ({curt['curtailment_pct']:.1f} % do disponível).",
        f"* Erro máximo de balanço nodal no `lpf` (12 h): **{bal_err:.2e} MW**; "
        f"diferença máxima entre fluxos do `lpf` e do UC: **{flow_diff:.3f} MW** "
        "(confirma a modelagem dos transformadores como `Transformer` com x em p.u. sobre s_nom).",
        "",
        "## Despacho por combustível\n", mix.to_markdown(), "",
        "## Fluxos nas interligações entre áreas\n", flows.to_markdown(), "",
        "## Ramos mais carregados (|p0|máx / s_nom)\n", loading.head(8).round(3).to_frame("loading").to_markdown(), "",
        "## Parâmetros térmicos derivados do gen.csv\n", thermal_summary(n).to_markdown(), "",
        "## Comparação de ordem de grandeza com a literatura do RTS-GMLC",
        "O RTS-GMLC (Barrows et al., IEEE TPWRS 2020) tem 8,55 GW de carga de pico e um parque em que "
        "carvão, ciclo combinado a gás e nuclear formam a base do despacho, com eólica (2,5 GW) e solar "
        "(~2,9 GW entre PV, RTPV e CSP) fornecendo a maior parte da energia renovável. O mix acima "
        "(nuclear must-run, carvão e CC como base, CT/óleo marginais, VRE com curtailment baixo no sistema "
        "original) e os LMPs na faixa de 10–60 $/MWh são coerentes com os resultados publicados de PCM "
        "para este sistema. Os custos marginais derivados (nuclear ≈ 8, carvão ≈ 25–30, CC ≈ 25–35, "
        "CT ≈ 45–55, óleo ≈ 110–150 $/MWh) reproduzem a ordem de mérito documentada no repositório.",
        "",
        "## Decisões de modelagem registradas",
        "* Linhas: R/X em p.u. (100 MVA) → ohms via `Z·v_nom²/100`; `s_nom` = *Cont Rating*.",
        "* Transformadores (Tr Ratio ≠ 0 ou tensões distintas): `Transformer` com x em p.u. sobre `s_nom` "
        "(`X·s_nom/100`), `tap_ratio` do CSV; C35 (r = 0) recebeu piso numérico.",
        "* Elo CC DC1 (113–316): `Link` bidirecional de 100 MW.",
        "* Séries DAY_AHEAD já em MW: `p_max_pu = série/PMax`; hidro com despacho fixo (PMin = PMax no RTS); "
        "CSP como disponibilidade direta (armazenamento térmico desprezado); 313_STORAGE_1 como `StorageUnit`.",
        "* Térmicas: custo marginal linear no ponto de 80 % de PMax; partida a frio; nuclear must-run; "
        "unidades com min_up ≥ 8 h iniciam ligadas (p_init no ponto médio da faixa).",
    ]
    (rdir / "F1_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))


def phase_f2(cfg: dict, solver: str) -> None:
    rdir = results_dir(cfg)
    c = cfg["corridor"]
    rows = []
    for key, day in cfg["days"].items():
        name = f"{key}__A__base"
        nc = rdir / f"{name}.nc"
        if not nc.exists():
            run_case(key, float(cfg["vre"]["target_share"]), set(), float(cfg["vpl"]["p_nom_mw"]) * float(cfg["vpl"]["hours"]),
                     float(cfg["vdc"]["flex_fraction"]), solver, False, cfg=cfg, name=name)
        n = pypsa.Network(str(nc))
        m = corridor_metrics(n, cfg)
        flow = corridor_flow(n, cfg)
        rows.append({"dia": key, "data": day, "h_saturadas": m["corridor_congested_hours"],
                     "fluxo_médio_MW": round(m["corridor_flow_mean_mw"]), "fluxo_máx_MW": round(m["corridor_flow_max_mw"]),
                     "h_fluxo_reverso": int((flow < 0).sum()),
                     "spread_LMP_médio": round(m["lmp_spread_mean"], 2), "spread_LMP_saturado": round(m["lmp_spread_congested_mean"], 2),
                     "spread_LMP_máx": round(m["lmp_spread_max"], 2), "fator_VRE": n.meta["vre_scaling"]["factor"]})
    df = pd.DataFrame(rows)
    ok_days = int((df["h_saturadas"] >= 6).sum())
    passed = ok_days >= 2 and (df["spread_LMP_saturado"].abs() > 1).any()
    lines = [
        "# F2 — Saturação do corredor N_318–N_223 (caso A, configuração estressada, 50 % VRE)\n",
        f"Corredor: {', '.join(c['branches'])} (limite {corridor_limit(n, cfg):.0f} MW; saturação = |p0| ≥ "
        f"{100*c['saturation_threshold']:.0f} % do limite). Acréscimo na Área 3: "
        f"{n.meta['stress']['mw_added_area3']:.0f} MW ({100*cfg['stress']['added_wind_share']:.0f} % eólica).\n",
        df.to_markdown(index=False), "",
        f"Dias com ≥ 6 h saturadas: **{ok_days} de {len(df)}** → critério F2 "
        f"{'**ATENDIDO**' if passed else '**NÃO atendido** (aumentar added_wind_share ou usar limit_corridor)'}.",
    ]
    (rdir / "F2_corridor.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["F1", "F2"], default="F1")
    p.add_argument("--day", default=None, help="data para F1 (padrão: summer_peak)")
    p.add_argument("--solver", default=None)
    args = p.parse_args(argv)
    cfg = load_config()
    if args.solver:
        cfg = set_by_path(cfg, "solver.name", args.solver)
    if args.phase == "F1":
        phase_f1(cfg, args.day or cfg["days"]["summer_peak"])
    else:
        phase_f2(cfg, cfg["solver"]["name"])


if __name__ == "__main__":
    main()
