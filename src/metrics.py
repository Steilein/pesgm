"""
metrics.py — Fase 6: pós-processamento de cada rede resolvida.

``compute_metrics(n, cfg)`` extrai, de um ``pypsa.Network`` já otimizado (ou
recarregado de ``results/<cenario>.nc``), a linha de métricas descrita na
tabela do §9 do brief:

* custo total e decomposição (combustível, partidas, migração, degradação,
  penalidade de curtailment);
* curtailment VRE (MWh e %);
* número de partidas por combustível;
* FRUS / FRDS médios (eqs. 5.1–5.2 da tese);
* MWh operados em Pmin (ciclagem) e horas em Pmin;
* horas congestionadas do corredor e spread de LMP entre os terminais;
* capacidade virtual do VPL (MW médio e MWh a jusante em horas saturadas);
* horas em serviço VTL vs. arbitragem (variante explícita);
* energia VDC deslocada no espaço e no tempo, conclusão dos workloads e
  respeito à capacidade de site;
* emissões de CO₂; tempo de solução e gap.

``marginal_value_table`` monta a tabela central do artigo: valor marginal
ΔCusto(B−A), ΔCusto(C−A), ΔCusto(D−A) e complementaridade definida sobre os
BENEFÍCIOS (economias, ΔBen = −ΔCusto): ΔBen_D − (ΔBen_B + ΔBen_C) =
(ΔCusto_B + ΔCusto_C) − ΔCusto_D. Positivo = complementares (juntos economizam
mais do que a soma dos isolados); negativo = substitutos. Como as colunas
``delta_*`` são variações de CUSTO (negativas quando há economia), o sinal da
complementaridade é o oposto de ``delta_D − (delta_B + delta_C)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layers import VPL_DOWN, VPL_UP, corridor_flow, corridor_limit, vre_available  # noqa: E402

PMIN_TOL = 1e-3   # tolerância relativa para "operando em Pmin"


# -----------------------------------------------------------------------------
# Blocos de métricas
# -----------------------------------------------------------------------------
def cost_decomposition(n: pypsa.Network) -> dict:
    """Decompõe o objetivo em parcelas interpretáveis ($)."""
    g = n.generators
    p = n.generators_t.p
    mc = n.get_switchable_as_dense("Generator", "marginal_cost")
    thermal = g.index[g["is_thermal"]]
    vre = g.index[g["is_vre"]]

    fuel = float((p[thermal] * mc[thermal]).sum().sum())
    startup = float((n.generators_t.start_up[thermal] * g.loc[thermal, "start_up_cost"]).sum().sum())
    shutdown = float((n.generators_t.shut_down[thermal] * g.loc[thermal, "shut_down_cost"]).sum().sum())
    curtail_penalty = float((p[vre] * mc[vre]).sum().sum())   # negativo por construção (−0,1 $/MWh)

    migration = 0.0
    links = n.links[n.links["carrier"] == "VDC"] if "carrier" in n.links else pd.DataFrame()
    if not links.empty:
        migration = float((n.links_t.p0[links.index] * links["marginal_cost"]).sum().sum())

    # Custo de flexibilidade: c^flex sobre a energia retirada do cronograma de
    # referência (constraints.add_flex_cost). No ótimo s = max(0, ref − P).
    flex = 0.0
    meta = n.meta.get("vdc", {})
    if not links.empty and float(meta.get("flex_cost", 0.0)) > 0.0:
        for site, ref in meta["reference_elec_mw"].items():
            idx = links.index[links["site"] == site]
            P = n.links_t.p0[idx].sum(axis=1).values
            flex += float(meta["flex_cost"]) * float(np.clip(np.asarray(ref[:len(P)]) - P, 0.0, None).sum())

    degradation = 0.0
    su = n.storage_units
    if not su.empty:
        degradation = float((n.storage_units_t.p_dispatch[su.index] * su["marginal_cost"]).sum().sum())
    # Degradação da célula a combustível (layers.add_h2): o Link cobra sobre p0.
    h2 = n.links[n.links["carrier"] == "H2"] if "carrier" in n.links else pd.DataFrame()
    if not h2.empty:
        degradation += float((n.links_t.p0[h2.index] * h2["marginal_cost"]).sum().sum())

    # Partidas a frio do eletrolisador de três estados (constraints.add_h2_states)
    h2_meta = n.meta.get("h2", {})
    h2_startup = float(h2_meta.get("startup_cost", 0.0)) * sum(h2_meta.get("state", {}).get("z_su", []))

    other = float(n.objective) - (fuel + startup + shutdown + curtail_penalty + migration + flex
                                  + degradation + h2_startup)
    return {
        "cost_total": float(n.objective),
        "cost_fuel": fuel,
        "cost_startup": startup,
        "cost_shutdown": shutdown,
        "cost_curtail_penalty": curtail_penalty,
        "cost_migration": migration,
        "cost_flex": flex,
        "cost_degradation": degradation,
        "cost_h2_startup": h2_startup,
        "cost_other": other,
    }


def curtailment(n: pypsa.Network) -> dict:
    avail = vre_available(n)
    p = n.generators_t.p[avail.columns]
    curt = (avail - p).clip(lower=0.0)
    total_avail = float(avail.sum().sum())
    return {
        "vre_available_mwh": total_avail,
        "vre_dispatched_mwh": float(p.sum().sum()),
        "curtailment_mwh": float(curt.sum().sum()),
        "curtailment_pct": 100.0 * float(curt.sum().sum()) / total_avail if total_avail else np.nan,
    }


def startups(n: pypsa.Network) -> dict:
    """Número de partidas por combustível a partir da variável ``start_up``.

    Equivale a ``status.diff().clip(lower=0).sum()`` mas inclui a transição da
    condição inicial (``up_time_before``), que o ``diff`` não enxerga.
    """
    g = n.generators
    thermal = g.index[g["is_thermal"]]
    su = n.generators_t.start_up[thermal].sum()
    by_fuel = su.groupby(g.loc[thermal, "fuel"]).sum()
    out = {"startups_total": float(su.sum())}
    out.update({f"startups_{f.lower()}": float(v) for f, v in by_fuel.items()})
    return out


def flexibility_reserves(n: pypsa.Network) -> dict:
    """FRUS / FRDS médios (eqs. 5.1–5.2 da tese) sobre unidades ligadas e horas.

    FRUS_{g,t} = min(Pmax·u − p_t, RU − (p_t − p_{t−1}))
    FRDS_{g,t} = min(p_t − Pmin·u, RD − (p_{t−1} − p_t))
    """
    g = n.generators
    thermal = g.index[g["is_thermal"]]
    p = n.generators_t.p[thermal]
    u = n.generators_t.status[thermal]
    pmax = g.loc[thermal, "p_nom"]
    pmin = g.loc[thermal, "p_min_pu"] * pmax
    ru = g.loc[thermal, "ramp_limit_up"] * pmax
    rd = g.loc[thermal, "ramp_limit_down"] * pmax
    dp = p.diff().fillna(0.0)

    frus = np.minimum(pmax * u - p, ru - dp).clip(lower=0.0)
    frds = np.minimum(p - pmin * u, rd + dp).clip(lower=0.0)
    on = u > 0.5
    return {
        "frus_mean_mw": float(frus[on].stack().mean()),
        "frds_mean_mw": float(frds[on].stack().mean()),
        "frus_system_mean_mw": float(frus.sum(axis=1).mean()),
        "frds_system_mean_mw": float(frds.sum(axis=1).mean()),
    }


def pmin_operation(n: pypsa.Network) -> dict:
    """Energia (MWh) e horas-unidade em que térmicas ligadas operam no mínimo técnico."""
    g = n.generators
    thermal = g.index[g["is_thermal"]]
    p = n.generators_t.p[thermal]
    u = n.generators_t.status[thermal]
    pmin = g.loc[thermal, "p_min_pu"] * g.loc[thermal, "p_nom"]
    at_pmin = (u > 0.5) & (p <= pmin * (1.0 + PMIN_TOL) + 1e-6)
    return {
        "pmin_mwh": float(p[at_pmin].sum().sum()),
        "pmin_unit_hours": int(at_pmin.sum().sum()),
        "online_unit_hours": int((u > 0.5).sum().sum()),
    }


def corridor_metrics(n: pypsa.Network, cfg: dict) -> dict:
    """Horas congestionadas do corredor e spread de LMP entre 318 e 223."""
    c = cfg["corridor"]
    flow = corridor_flow(n, cfg)
    limit = corridor_limit(n, cfg)
    thr = float(c["saturation_threshold"])
    congested = flow.abs() >= thr * limit
    lmp_up = n.buses_t.marginal_price[str(c["bus_up"])]
    lmp_dn = n.buses_t.marginal_price[str(c["bus_down"])]
    spread = lmp_dn - lmp_up
    return {
        "corridor_limit_mw": limit,
        "corridor_flow_mean_mw": float(flow.mean()),
        "corridor_flow_max_mw": float(flow.abs().max()),
        "corridor_congested_hours": int(congested.sum()),
        "corridor_utilization": float(flow.abs().mean() / limit),
        "lmp_up_mean": float(lmp_up.mean()),
        "lmp_down_mean": float(lmp_dn.mean()),
        "lmp_spread_mean": float(spread.mean()),
        "lmp_spread_max": float(spread.abs().max()),
        "lmp_spread_congested_mean": float(spread[congested].mean()) if congested.any() else 0.0,
    }


def vpl_metrics(n: pypsa.Network, cfg: dict) -> dict:
    """Capacidade virtual, ciclagem e (variante explícita) horas VTL vs. arbitragem."""
    out = {"vpl_virtual_capacity_mw": 0.0, "vpl_virtual_mwh": 0.0, "vpl_cycles_up": 0.0, "vpl_cycles_down": 0.0,
           "vpl_throughput_mwh": 0.0, "vpl_hours_vtl": np.nan, "vpl_hours_arbitrage": np.nan,
           "vpl_corridor_relief_mw": np.nan, "vpl_relief_gross_mw": np.nan, "vpl_aggravation_mw": np.nan}
    # Unidades presentes (a decomposição "vpl.terminals" instala só um terminal)
    units = [u for u in (VPL_UP, VPL_DOWN) if u in n.storage_units.index]
    if not units:
        return out
    c = cfg["corridor"]
    flow = corridor_flow(n, cfg)
    congested = flow.abs() >= float(c["saturation_threshold"]) * corridor_limit(n, cfg)
    dis = n.storage_units_t.p_dispatch
    sto = n.storage_units_t.p_store
    ref = units[0]
    e_cap = n.storage_units.loc[ref, "p_nom"] * n.storage_units.loc[ref, "max_hours"]

    # Capacidade virtual (definição herdada): descarga a jusante nas horas em que
    # o corredor satura. Captura o mecanismo de um corredor de IMPORTAÇÃO — o
    # corredor traz energia ao centro de carga, satura, e a bateria local supre o
    # que não passou. Mantida por compatibilidade com os resultados anteriores.
    if VPL_DOWN in units:
        dn_sat = dis[VPL_DOWN][congested]
        out["vpl_virtual_capacity_mw"] = float(dn_sat.mean()) if congested.any() else 0.0
        out["vpl_virtual_mwh"] = float(dn_sat.sum())
        out["vpl_cycles_down"] = float(dis[VPL_DOWN].sum() / e_cap)

    # Alívio líquido do corredor (definição geral, com sinal). Num corredor de
    # EXPORTAÇÃO o alívio se faz **carregando a montante** — absorvendo o
    # excedente antes que precise escoar —, mecanismo que a métrica acima lê como
    # zero. Aqui contam os quatro movimentos, com o sinal do efeito sobre o fluxo:
    #
    #   alivia  : carga a montante, descarga a jusante
    #   agrava  : descarga a montante, carga a jusante
    #
    # O sinal se inverte se o fluxo predominante for jusante -> montante, por isso
    # a orientação é tomada do fluxo médio nas horas congestionadas.
    if congested.any() and len(units) == 2:
        orient = 1.0 if float(flow[congested].mean()) >= 0 else -1.0
        up_charge = float(sto[VPL_UP][congested].mean())
        up_disch = float(dis[VPL_UP][congested].mean())
        dn_charge = float(sto[VPL_DOWN][congested].mean())
        dn_disch = float(dis[VPL_DOWN][congested].mean())
        relief = (up_charge + dn_disch) - (up_disch + dn_charge)
        out["vpl_corridor_relief_mw"] = orient * relief
        out["vpl_relief_gross_mw"] = up_charge + dn_disch
        out["vpl_aggravation_mw"] = up_disch + dn_charge
    if VPL_UP in units:
        out["vpl_cycles_up"] = float(dis[VPL_UP].sum() / e_cap)
    out["vpl_throughput_mwh"] = float(dis[units].sum().sum())

    sigma = n.meta.get("sigma")
    if sigma is not None:
        s = pd.Series(sigma, index=n.snapshots, dtype="float")
        active = (dis[units].sum(axis=1) + sto[units].sum(axis=1)) > 1e-3
        # no modo 'dedicated' não há hora livre: o estágio intermediário também
        # é regido pela regra (w_t = z_t), então toda hora ativa é serviço VTL
        mode = str(n.meta.get("vpl", {}).get("mode", "")).lower()
        if mode == "dedicated":
            out["vpl_hours_vtl"] = int(active.sum())
            out["vpl_hours_arbitrage"] = 0
        elif mode == "hybrid":
            # só o estágio de alto uso é serviço VTL; o resto do dia é livre
            out["vpl_hours_vtl"] = int((active & (s == 1.0)).sum())
            out["vpl_hours_arbitrage"] = int((active & (s != 1.0)).sum())
        else:
            out["vpl_hours_vtl"] = int((active & s.notna()).sum())
            out["vpl_hours_arbitrage"] = int((active & s.isna()).sum())
    return out


def vdc_metrics(n: pypsa.Network) -> dict:
    """Deslocamento espacial/temporal, conclusão dos workloads e capacidade de site."""
    out = {"vdc_energy_it_mwh": 0.0, "vdc_energy_elec_mwh": 0.0, "vdc_spatial_shift_mwh": 0.0,
           "vdc_spatial_shift_pct": np.nan, "vdc_temporal_shift_mwh": 0.0, "vdc_temporal_shift_pct": np.nan,
           "vdc_mean_delay_h": np.nan, "vdc_completion_min": np.nan, "vdc_site_cap_violation_mw": 0.0}
    meta = n.meta.get("vdc", {})
    if meta.get("mode") != "flexible":
        return out
    links = n.links[n.links["carrier"] == "VDC"]
    p_elec = n.links_t.p0[links.index]                       # MW elétricos no site
    p_it = -n.links_t.p1[links.index]                        # MW de TI entregues
    hours = np.arange(len(n.snapshots))

    total_it = float(p_it.sum().sum())
    out["vdc_energy_it_mwh"] = total_it
    out["vdc_energy_elec_mwh"] = float(p_elec.sum().sum())

    # Espacial: energia servida fora do site nativo
    non_native = links.index[~links["is_native"].astype(bool)]
    spatial = float(p_it[non_native].sum().sum())
    out["vdc_spatial_shift_mwh"] = spatial
    out["vdc_spatial_shift_pct"] = 100.0 * spatial / total_it if total_it else np.nan

    # Temporal: energia consumida após a hora em que um escalonamento ASAP
    # (o mais cedo possível, na capacidade flexível do site) concluiria o workload
    temporal, delay_num = 0.0, 0.0
    completion = []
    sites = meta["sites"]
    for w in meta["workloads"]:
        wl = links.index[links["workload"] == w["id"]]
        prof = p_it[wl].sum(axis=1).values
        cap_it = sites[w["native"]]["flex_mw_elec"] / w["pue"]
        asap_end = w["arrival"] + int(np.ceil(w["E_it_mwh"] / cap_it)) - 1 if cap_it > 0 else w["deadline"]
        temporal += float(prof[hours > asap_end].sum())
        delay_num += float((prof * (hours - w["arrival"])).sum())
        e_end = float(n.stores_t.e[f"st_{w['id']}"].iloc[w["deadline"]])
        completion.append(e_end / w["E_it_mwh"] if w["E_it_mwh"] > 0 else 1.0)
    out["vdc_temporal_shift_mwh"] = temporal
    out["vdc_temporal_shift_pct"] = 100.0 * temporal / total_it if total_it else np.nan
    out["vdc_mean_delay_h"] = delay_num / total_it if total_it else np.nan
    out["vdc_completion_min"] = float(min(completion)) if completion else np.nan

    # Capacidade por site: máximo excesso (deve ser ~0)
    viol = 0.0
    for d, info in sites.items():
        idx = links.index[links["site"] == d]
        viol = max(viol, float((p_elec[idx].sum(axis=1) - info["flex_mw_elec"]).max()))
    out["vdc_site_cap_violation_mw"] = max(0.0, viol)
    return out


def h2_metrics(n: pypsa.Network) -> dict:
    """Energia da cadeia de hidrogênio e, em três estados, horas por estado e partidas."""
    out = {"h2_p2h_mwh": 0.0, "h2_h2p_mwh": 0.0, "h2_standby_mwh": 0.0,
           "h2_on_h": np.nan, "h2_standby_h": np.nan, "h2_startups": np.nan}
    if "H2_P2H" not in n.links.index:
        return out
    p0 = n.links_t.p0
    out["h2_p2h_mwh"] = float(p0["H2_P2H"].sum())
    out["h2_h2p_mwh"] = float(-n.links_t.p1["H2_H2P"].sum())       # elétrico entregue
    if "H2_P2H_sb" in n.links.index:
        out["h2_standby_mwh"] = float(p0["H2_P2H_sb"].sum())
    state = n.meta.get("h2", {}).get("state")
    if state:
        out["h2_on_h"] = float(sum(state["z_on"]))
        out["h2_standby_h"] = float(sum(state["z_sb"]))
        out["h2_startups"] = float(sum(state["z_su"]))
    return out


def emissions(n: pypsa.Network) -> dict:
    g = n.generators
    thermal = g.index[g["is_thermal"]]
    co2 = (n.generators_t.p[thermal] * g.loc[thermal, "co2_t_per_mwh"]).sum()
    out = {"co2_t": float(co2.sum())}
    out.update({f"co2_t_{f.lower()}": float(v) for f, v in co2.groupby(g.loc[thermal, "fuel"]).sum().items()})
    return out


def dispatch_by_carrier(n: pypsa.Network) -> dict:
    p = n.generators_t.p.T.groupby(n.generators["carrier"]).sum().T.sum()
    return {f"gen_mwh_{c.lower().replace(' ', '_')}": float(v) for c, v in p.items()}


# -----------------------------------------------------------------------------
# Linha completa
# -----------------------------------------------------------------------------
def compute_metrics(n: pypsa.Network, cfg: dict) -> dict:
    sc = n.meta.get("scenario", {})
    solve = n.meta.get("solve", {})
    row = {
        "scenario": sc.get("name", n.name),
        "day_key": sc.get("day_key"),
        "day": sc.get("day", n.meta.get("day")),
        "case": sc.get("case"),
        "axis": sc.get("axis", "base"),
        "variation": sc.get("variation", "base"),
        "vre_share": sc.get("vre_share"),
        "vre_scale_factor": n.meta.get("vre_scaling", {}).get("factor"),
        "vre_share_annual": n.meta.get("vre_scaling", {}).get("share_after_annual"),
        "vre_share_day": n.meta.get("vre_scaling", {}).get("share_after_day"),
        "vpl_mwh": sc.get("vpl_mwh"),
        "vpl_mode": sc.get("vpl_mode"),
        "dc_flex": sc.get("dc_flex"),
        "window_scale": sc.get("window_scale"),
        "reserve_from_flex": sc.get("reserve_from_flex"),
        "load_mwh": float(n.get_switchable_as_dense("Load", "p_set").sum().sum()),
        "solver": solve.get("solver"),
        "mip_gap": solve.get("mip_gap"),
        "time_milp_s": solve.get("time_milp_s"),
        "time_lp_s": solve.get("time_lp_s"),
        "objective_milp": solve.get("objective_milp"),
        "objective_bound": solve.get("objective_bound"),
    }
    row.update(cost_decomposition(n))
    row.update(curtailment(n))
    row.update(startups(n))
    row.update(flexibility_reserves(n))
    row.update(pmin_operation(n))
    row.update(corridor_metrics(n, cfg))
    row.update(vpl_metrics(n, cfg))
    row.update(vdc_metrics(n))
    row.update(h2_metrics(n))
    row.update(emissions(n))
    row.update(dispatch_by_carrier(n))
    return row


def append_metrics_row(row: dict, path: Path) -> None:
    """Acrescenta/atualiza a linha do cenário em ``metrics.csv`` (chave = scenario)."""
    path = Path(path)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["scenario"] != row["scenario"]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def recompute_all(results: Path, cfg: dict) -> pd.DataFrame:
    """Recalcula ``metrics.csv`` a partir de todos os NetCDF salvos (sem re-resolver)."""
    rows = []
    for nc in sorted(Path(results).glob("*.nc")):
        if nc.name.startswith("_") or nc.name == "base_validation.nc":
            continue
        n = pypsa.Network(str(nc))
        rows.append(compute_metrics(n, cfg))
    df = pd.DataFrame(rows)
    df.to_csv(Path(results) / "metrics.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Tabela central: valor marginal e complementaridade
# -----------------------------------------------------------------------------
def _applies_to(scenarios: dict | None) -> dict[str, list[str]]:
    """Mapa eixo → casos aos quais o eixo se aplica (de ``scenarios.yaml``)."""
    if not scenarios:
        return {}
    return {axis: spec.get("applies_to", ["A", "B", "C", "D"]) for axis, spec in scenarios["axes"].items()}


def marginal_value_table(df: pd.DataFrame, value_col: str = "cost_total",
                         scenarios: dict | None = None) -> pd.DataFrame:
    """Para cada (dia, variação): ΔB−A, ΔC−A, ΔD−A (custo) e complementaridade.

    ``complementarity`` = (ΔCusto_B + ΔCusto_C) − ΔCusto_D, isto é, o benefício
    conjunto menos a soma dos benefícios isolados; positivo = complementares.

    Quando uma variação não se aplica a um caso (ex.: ``vpl400`` para A e C,
    conforme ``applies_to`` em ``scenarios.yaml``), usa-se a linha ``base``
    daquele caso no mesmo dia. Casos aplicáveis ainda não resolvidos ficam
    ``NaN`` (evita tabelas parciais enganosas durante a execução da grade).
    """
    applies = _applies_to(scenarios)
    rows = []
    for day, g_day in df.groupby("day_key"):
        base = g_day[g_day["variation"] == "base"].set_index("case")[value_col]
        for var, g_var in g_day.groupby("variation"):
            vals = g_var.set_index("case")[value_col]
            if not set(vals.index) & set("ABCD"):
                continue          # famílias com rotulagem própria (ex.: escada L1–L6)
            axis = g_var["axis"].iloc[0]
            allowed = applies.get(axis, ["A", "B", "C", "D"])
            val = {c: vals.get(c, base.get(c, np.nan) if c not in allowed or var == "base" else np.nan)
                   for c in "ABCD"}
            dB, dC, dD = val["B"] - val["A"], val["C"] - val["A"], val["D"] - val["A"]
            rows.append({
                "day_key": day, "variation": var, "axis": g_var["axis"].iloc[0],
                f"{value_col}_A": val["A"],
                "delta_B_A": dB, "delta_C_A": dC, "delta_D_A": dD,
                "complementarity": (dB + dC) - dD,   # benefício conjunto − soma dos isolados
                "delta_B_A_pct": 100 * dB / val["A"], "delta_C_A_pct": 100 * dC / val["A"],
                "delta_D_A_pct": 100 * dD / val["A"],
            })
    return pd.DataFrame(rows).sort_values(["axis", "variation", "day_key"]).reset_index(drop=True)


def write_summary(df: pd.DataFrame, path: Path, scenarios: dict | None = None) -> None:
    """Gera ``results/summary.md`` com as tabelas prontas para o artigo."""
    lines = ["# Resumo dos resultados — UC day-ahead com VPL e VDC\n"]
    lines.append(f"Cenários resolvidos: **{len(df)}**. Solver: {', '.join(sorted(df['solver'].dropna().unique()))}. "
                 f"Gap MIP máximo: {df['mip_gap'].max():.2e}. Tempo médio MILP: {df['time_milp_s'].mean():.0f} s.\n")

    lines.append("## 1. Valor marginal e complementaridade (custo total, $)\n")
    mv = marginal_value_table(df, scenarios=scenarios)
    if scenarios:  # eixos extras (ex.: sensibilidade cruzada VPL × f_d) têm relatório próprio
        mv = mv[mv["axis"].isin(list(scenarios["axes"]) + ["base"])]
    lines.append(mv.round(1).to_markdown(index=False) + "\n")

    lines.append("## 2. Métricas operacionais por caso (variação base, média entre dias)\n")
    cols = ["cost_total", "curtailment_pct", "startups_total", "pmin_mwh", "frus_mean_mw", "frds_mean_mw",
            "corridor_congested_hours", "lmp_spread_mean", "vpl_virtual_capacity_mw", "vdc_spatial_shift_pct",
            "vdc_temporal_shift_pct", "co2_t"]
    base = df[(df["variation"] == "base") & (df["axis"] == "base")].groupby("case")[cols].mean()
    lines.append(base.round(2).to_markdown() + "\n")

    lines.append("## 3. Sensibilidade à penetração VRE (média entre dias)\n")
    vre = df[df["axis"].isin(["base", "vre_share"])].groupby(["variation", "case"])[
        ["cost_total", "curtailment_pct", "startups_total", "corridor_congested_hours"]].mean()
    lines.append(vre.round(2).to_markdown() + "\n")

    lines.append("## 4. Complementaridade por dia e penetração\n")
    comp = mv[mv["axis"].isin(["base", "vre_share"])].pivot(index="day_key", columns="variation", values="complementarity")
    lines.append(comp.round(1).to_markdown() + "\n")

    lines.append("## 5. Variantes com regra de chaveamento: horas em serviço VTL vs. arbitragem\n")
    exp = df[df["vpl_mode"].isin(["explicit", "dedicated"])][
        ["scenario", "vpl_mode", "vpl_hours_vtl", "vpl_hours_arbitrage", "cost_total"]]
    lines.append((exp.to_markdown(index=False) if not exp.empty else "_(não executada)_") + "\n")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    from common import load_config, results_dir

    from common import ROOT

    cfg = load_config()
    scenarios = load_config(ROOT / "config" / "scenarios.yaml")
    rdir = results_dir(cfg)
    df = recompute_all(rdir, cfg)
    write_summary(df, rdir / "summary.md", scenarios=scenarios)
    print(f"{len(df)} cenários → {rdir / 'metrics.csv'} e {rdir / 'summary.md'}")
