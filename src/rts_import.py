"""
rts_import.py — Fase 1: importação do RTS-GMLC para um ``pypsa.Network``.

Este módulo converte os arquivos-fonte do sistema teste RTS-GMLC
(``RTS_Data/SourceData/*.csv`` e ``RTS_Data/timeseries_data_files/**``) em uma
rede PyPSA pronta para o Unit Commitment (UC) day-ahead com fluxo de potência
linear (DC). A leitura do ano completo é feita uma única vez (``load_rts_data``);
a rede para um dia específico é montada por ``build_network``.

Decisões de modelagem documentadas (referências ao brief entre parênteses):

* Barras (§2.1): ``v_nom`` = BaseKV; ``area``, ``zone`` e coordenadas
  geográficas (``x`` = longitude, ``y`` = latitude) são guardadas para os mapas.
* Ramos (§2.2): o CSV traz R, X e B em p.u. na base de 100 MVA. O PyPSA espera
  ohms (base de 1 MVA na tensão nominal), logo ``Z_ohm = Z_pu * v_nom² / 100``.
  Ramos com ``Tr Ratio != 0`` ou que ligam níveis de tensão distintos são
  modelados como ``Transformer`` (reatância em p.u. na base ``s_nom``), escolha
  registrada aqui e validada com ``n.lpf()`` no relatório de F1.
  O elo CC (``dc_branch.csv``) vira um ``Link`` bidirecional. Linhas CA nunca
  são substituídas por ``Link`` (perderia-se a física do congestionamento).
* Térmicas (§2.3): ``committable=True``; a curva de heat-rate por segmentos
  (HR_avg_0, HR_incr_k) é linearizada em um único custo médio no ponto de
  80 % de PMax; o custo de partida é o de partida a frio
  (Start Heat Cold × preço do combustível + custo fixo não-combustível).
  A nuclear recebe ``min_up_time = min_down_time = 168 h`` e ``p_min_pu = 0,9``.
* Condição inicial (não existe histórico no RTS): unidades de base
  (``min_up_time >= 8 h``) iniciam ligadas com ``p_init`` no ponto médio da faixa
  operativa, para que a rampa da primeira hora não seja restritiva; unidades
  de ponta iniciam desligadas. Sem isso, toda unidade ligada na hora 0 pagaria
  custo de partida, distorcendo a métrica de ciclagem.
* VRE (§2.4): as séries DAY_AHEAD já estão em MW (máximo = PMax); a coluna
  ``Scaling Factor`` do ``timeseries_pointers.csv`` é um artefato do PLEXOS e é
  ignorada. ``p_max_pu = série / PMax``. CSP: a série ``Natural_Inflow`` é
  tratada como disponibilidade solar direta (armazenamento térmico desprezado;
  simplificação aceitável para uma única unidade de 200 MW).
* Hidro: o RTS define PMin = PMax = série no day-ahead, isto é, despacho fixo.
  Mantido por padrão (``hydro.fixed_dispatch``), com opção de flexibilizar.
* Carga (§2.5): série por área distribuída às barras proporcionalmente a
  ``MW Load`` de ``bus.csv``. Um ``Load`` por barra.
* Emissões: fator tCO₂/MWh = (lbs CO₂/MMBtu) × HR médio (MMBtu/MWh) × 0,000453592,
  guardado na coluna extra ``co2_t_per_mwh`` para pós-processamento.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

# Fatores de conversão físicos
LBS_TO_TONNE = 0.000453592          # 1 lb = 0,000453592 t
BTU_PER_KWH_TO_MMBTU_PER_MWH = 1e-3  # 1 BTU/kWh = 1e-3 MMBtu/MWh

# Combustíveis tratados como unidades térmicas committable (§2.3)
THERMAL_FUELS = ("Coal", "NG", "Oil", "Nuclear")

# Mapeamento "Unit Type" do RTS -> pasta / arquivo de série DAY_AHEAD
DA_SERIES_FILES = {
    "WIND": "WIND/DAY_AHEAD_wind.csv",
    "PV": "PV/DAY_AHEAD_pv.csv",
    "RTPV": "RTPV/DAY_AHEAD_rtpv.csv",
    "HYDRO": "Hydro/DAY_AHEAD_hydro.csv",
    "ROR": "Hydro/DAY_AHEAD_hydro.csv",
    "CSP": "CSP/DAY_AHEAD_Natural_Inflow.csv",
}


# -----------------------------------------------------------------------------
# 1. Leitura dos dados brutos (uma vez por sessão)
# -----------------------------------------------------------------------------
@dataclass
class RTSData:
    """Contêiner imutável com os dados estáticos e as séries anuais do RTS-GMLC."""

    bus: pd.DataFrame          # bus.csv indexado por Bus ID (string)
    branch: pd.DataFrame       # branch.csv indexado por UID
    dc_branch: pd.DataFrame    # dc_branch.csv indexado por UID
    gen: pd.DataFrame          # gen.csv indexado por GEN UID
    load_area: pd.DataFrame    # carga DAY_AHEAD por área (colunas "1","2","3"), MW
    gen_series: pd.DataFrame   # disponibilidade DAY_AHEAD por gerador (MW)


def _read_da_series(path: Path) -> pd.DataFrame:
    """Lê um CSV de série DAY_AHEAD e devolve DataFrame indexado por timestamp.

    O formato do RTS é ``Year, Month, Day, Period, <colunas>``, com ``Period``
    de 1 a 24. Converte-se para ``DatetimeIndex`` horário (Period 1 → 00:00).
    """
    df = pd.read_csv(path)
    idx = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"], hour=df["Period"] - 1)
    )
    out = df.drop(columns=["Year", "Month", "Day", "Period"]).set_index(idx)
    out.index.name = "snapshot"
    return out


def load_rts_data(rts_dir: str | Path) -> RTSData:
    """Carrega todos os CSVs necessários do RTS-GMLC (somente DAY_AHEAD)."""
    rts_dir = Path(rts_dir)
    src = rts_dir / "SourceData"
    ts = rts_dir / "timeseries_data_files"

    bus = pd.read_csv(src / "bus.csv")
    bus["Bus ID"] = bus["Bus ID"].astype(str)
    bus = bus.set_index("Bus ID")

    branch = pd.read_csv(src / "branch.csv").set_index("UID")
    branch["From Bus"] = branch["From Bus"].astype(str)
    branch["To Bus"] = branch["To Bus"].astype(str)

    dc_branch = pd.read_csv(src / "dc_branch.csv").set_index("UID")
    dc_branch["From Bus"] = dc_branch["From Bus"].astype(str)
    dc_branch["To Bus"] = dc_branch["To Bus"].astype(str)

    gen = pd.read_csv(src / "gen.csv").set_index("GEN UID")
    gen["Bus ID"] = gen["Bus ID"].astype(str)

    # Carga por área: as colunas vêm como "1", "2", "3"
    load_area = _read_da_series(ts / "Load" / "DAY_AHEAD_regional_Load.csv")
    load_area.columns = load_area.columns.astype(str)

    # Séries de disponibilidade: concatena todas as tecnologias em um único frame
    frames = []
    for rel in sorted(set(DA_SERIES_FILES.values())):
        frames.append(_read_da_series(ts / rel))
    gen_series = pd.concat(frames, axis=1)

    return RTSData(bus, branch, dc_branch, gen, load_area, gen_series)


# -----------------------------------------------------------------------------
# 2. Parâmetros térmicos: linearização da curva de heat-rate e custos
# -----------------------------------------------------------------------------
def average_heat_rate(row: pd.Series, load_point: float) -> float:
    """Heat-rate médio (MMBtu/MWh) no ponto ``load_point`` (fração de PMax).

    O RTS descreve o consumo de combustível por segmentos:
    ``HR_avg_0`` é o heat-rate médio no nível mínimo estável (``Output_pct_0``)
    e ``HR_incr_k`` é o heat-rate incremental entre ``Output_pct_{k-1}`` e
    ``Output_pct_k`` (BTU/kWh). O consumo F(P) é integrado segmento a segmento
    e dividido por P, o que fornece um custo linear único para o MILP
    (não se usa ``marginal_cost_quadratic``).
    """
    pmax = float(row["PMax MW"])
    if pmax <= 0 or pd.isna(row.get("HR_avg_0")):
        return np.nan
    p_target = load_point * pmax

    pct = [row.get(f"Output_pct_{k}") for k in range(5)]
    incr = [row.get(f"HR_incr_{k}") for k in range(1, 5)]

    p0 = float(pct[0]) * pmax
    fuel = float(row["HR_avg_0"]) * BTU_PER_KWH_TO_MMBTU_PER_MWH * min(p0, p_target)
    if p_target <= p0:
        return fuel / p_target

    p_prev = p0
    for k in range(1, 5):
        if pd.isna(pct[k]) or pd.isna(incr[k - 1]):
            break
        p_k = float(pct[k]) * pmax
        seg = max(0.0, min(p_target, p_k) - p_prev)
        fuel += float(incr[k - 1]) * BTU_PER_KWH_TO_MMBTU_PER_MWH * seg
        p_prev = p_k
        if p_target <= p_k:
            break
    return fuel / p_target


def thermal_parameters(row: pd.Series, cfg: dict) -> dict:
    """Traduz uma linha de ``gen.csv`` nos atributos de um gerador committable."""
    tcfg = cfg["thermal"]
    pmax = float(row["PMax MW"])
    pmin = float(row["PMin MW"])
    fuel_price = float(row["Fuel Price $/MMBTU"])
    hr = average_heat_rate(row, tcfg["marginal_cost_load_point"])
    vom = float(row["VOM"]) if not pd.isna(row["VOM"]) else 0.0

    # Rampa: MW/min × 60 / PMax, limitada a 1,0 (fração de PMax por hora)
    ramp = min(1.0, float(row["Ramp Rate MW/Min"]) * 60.0 / pmax)
    p_min_pu = pmin / pmax
    min_up = int(round(float(row["Min Up Time Hr"])))
    min_down = int(round(float(row["Min Down Time Hr"])))

    # Custo de partida a frio: calor de partida × preço do combustível + custo fixo
    start_cost = float(row["Start Heat Cold MBTU"]) * fuel_price + float(
        row["Non Fuel Start Cost $"]
    )

    co2_lbs = pd.to_numeric(row["Emissions CO2 Lbs/MMBTU"], errors="coerce")
    co2_factor = float(co2_lbs) * hr * LBS_TO_TONNE if not pd.isna(co2_lbs) else 0.0

    # Nuclear: must-run efetivo (§2.3)
    if row["Fuel"] == "Nuclear":
        min_up = min_down = int(tcfg["nuclear_min_up_down"])
        p_min_pu = max(p_min_pu, float(tcfg["nuclear_p_min_pu"]))

    # Condição inicial heurística (ver docstring do módulo)
    starts_on = min_up >= int(tcfg["initial_status_min_up_threshold_h"])

    return dict(
        p_nom=pmax,
        p_min_pu=p_min_pu,
        committable=True,
        marginal_cost=hr * fuel_price + vom,
        start_up_cost=start_cost,
        shut_down_cost=float(tcfg["shut_down_cost"]),
        min_up_time=min_up,
        min_down_time=min_down,
        ramp_limit_up=ramp,
        ramp_limit_down=ramp,
        # Na partida/parada a unidade vai de 0 a PMin (e vice-versa) em uma hora
        ramp_limit_start_up=p_min_pu,
        ramp_limit_shut_down=p_min_pu,
        up_time_before=min_up if starts_on else 0,
        down_time_before=0 if starts_on else min_down,
        p_init=0.5 * (pmin + pmax) if starts_on else 0.0,
        heat_rate_mmbtu_per_mwh=hr,
        co2_t_per_mwh=co2_factor,
    )


# -----------------------------------------------------------------------------
# 3. Construção da rede para um dia
# -----------------------------------------------------------------------------
def day_snapshots(day: str) -> pd.DatetimeIndex:
    """24 snapshots horários do dia ``day`` (formato ``YYYY-MM-DD``)."""
    return pd.date_range(day, periods=24, freq="h", name="snapshot")


def _add_buses(n: pypsa.Network, data: RTSData) -> None:
    b = data.bus
    n.add(
        "Bus",
        b.index,
        v_nom=b["BaseKV"].values,
        carrier="AC",
        area=b["Area"].astype(int).values,
        zone=b["Zone"].values,
        x=b["lng"].values,
        y=b["lat"].values,
        bus_name=b["Bus Name"].values,
    )


def _add_branches(n: pypsa.Network, data: RTSData) -> None:
    """Linhas CA, transformadores e elo CC (§2.2)."""
    br = data.branch.copy()
    v_from = br["From Bus"].map(data.bus["BaseKV"])
    v_to = br["To Bus"].map(data.bus["BaseKV"])
    is_trafo = (br["Tr Ratio"] != 0) | (v_from != v_to)

    # --- Linhas: p.u. (100 MVA) -> ohms na tensão nominal
    ln = br[~is_trafo]
    v = v_from[~is_trafo]
    n.add(
        "Line",
        ln.index,
        bus0=ln["From Bus"].values,
        bus1=ln["To Bus"].values,
        r=(ln["R"] * v**2 / 100.0).values,
        x=(ln["X"] * v**2 / 100.0).values,
        b=(ln["B"] / 100.0 / v**2).values,
        s_nom=ln["Cont Rating"].values,
        length=ln["Length"].values,
        carrier="AC",
    )

    # --- Transformadores: p.u. sobre s_nom (convenção do PyPSA)
    tr = br[is_trafo]
    n.add(
        "Transformer",
        tr.index,
        bus0=tr["From Bus"].values,
        bus1=tr["To Bus"].values,
        # r = 0 no CSV (C35) receberia aviso do PyPSA; piso numérico irrelevante no fluxo DC
        r=(tr["R"] * tr["Cont Rating"] / 100.0).clip(lower=1e-4).values,
        x=(tr["X"] * tr["Cont Rating"] / 100.0).values,
        s_nom=tr["Cont Rating"].values,
        tap_ratio=tr["Tr Ratio"].replace(0, 1.0).values,
    )

    # --- Elo CC: Link bidirecional (p_min_pu = -1), sem perdas
    dc = data.dc_branch
    n.add(
        "Link",
        dc.index,
        bus0=dc["From Bus"].values,
        bus1=dc["To Bus"].values,
        p_nom=dc["MW Load"].values,
        p_min_pu=-1.0,
        efficiency=1.0,
        carrier="DC",
    )


def _add_generators(n: pypsa.Network, data: RTSData, cfg: dict, sns: pd.DatetimeIndex) -> None:
    """Térmicas committable, VRE, hidro e armazenamento nativo (§2.3–2.4)."""
    gen = data.gen
    series = data.gen_series.reindex(sns)
    vre_cost = float(cfg["vre"]["marginal_cost"])
    vre_carriers = set(cfg["vre"]["carriers"])

    for uid, row in gen.iterrows():
        bus = row["Bus ID"]
        utype = row["Unit Type"]
        fuel = row["Fuel"]
        category = row["Category"]
        pmax = float(row["PMax MW"])
        common = dict(bus=bus, carrier=category, fuel=fuel, unit_type=utype)

        # --- Térmicas committable
        if fuel in THERMAL_FUELS:
            n.add("Generator", uid, **common, **thermal_parameters(row, cfg))
            continue

        # --- Condensadores síncronos: sem potência ativa, ignorados
        if utype == "SYNC_COND":
            continue

        # --- Armazenamento nativo do RTS (313_STORAGE_1)
        if utype == "STORAGE":
            if not cfg["rts_storage"]["include"]:
                continue
            eta_rt = float(row["Storage Roundtrip Efficiency"]) / 100.0
            n.add(
                "StorageUnit",
                uid,
                bus=bus,
                carrier="Storage_RTS",
                p_nom=pmax,
                max_hours=3.0,               # 150 MWh / 50 MW (storage.csv)
                efficiency_store=np.sqrt(eta_rt),
                efficiency_dispatch=np.sqrt(eta_rt),
                cyclic_state_of_charge=True,
            )
            continue

        # --- VRE e hidro: disponibilidade normalizada pela PMax
        if uid not in series.columns:
            raise KeyError(f"Sem série DAY_AHEAD para o gerador {uid} ({utype}).")
        avail = (series[uid] / pmax).clip(lower=0.0, upper=1.0)

        if category in vre_carriers:
            # fator de capacidade anual: usado no escalonamento VRE em base anual
            cf_annual = float((data.gen_series[uid] / pmax).clip(0.0, 1.0).mean())
            kwargs = dict(p_nom=pmax, p_max_pu=avail, marginal_cost=vre_cost, cf_annual=cf_annual)
            if category == "Solar RTPV" and cfg["vre"]["rtpv_must_take"]:
                kwargs["p_min_pu"] = avail   # não curtailável
            n.add("Generator", uid, **common, **kwargs)
        elif fuel == "Hydro":
            kwargs = dict(p_nom=pmax, p_max_pu=avail, marginal_cost=0.0)
            if cfg["hydro"]["fixed_dispatch"]:
                kwargs["p_min_pu"] = avail   # PMin = PMax = série (RTS DAY_AHEAD)
            n.add("Generator", uid, **common, **kwargs)
        else:
            raise ValueError(f"Gerador {uid} com categoria não tratada: {category}")

    # Colunas auxiliares para pós-processamento
    g = n.generators
    g["area"] = g["bus"].map(n.buses["area"]).astype(int)
    g["is_vre"] = g["carrier"].isin(vre_carriers)
    g["is_thermal"] = g["fuel"].isin(THERMAL_FUELS)
    for col in ("heat_rate_mmbtu_per_mwh", "co2_t_per_mwh"):
        if col in g.columns:
            g[col] = g[col].fillna(0.0)


def _add_loads(n: pypsa.Network, data: RTSData, sns: pd.DatetimeIndex) -> None:
    """Distribui a carga de cada área às barras proporcionalmente a ``MW Load``."""
    bus = data.bus
    load_area = data.load_area.reindex(sns)
    for area, grp in bus.groupby("Area"):
        share = grp["MW Load"] / grp["MW Load"].sum()
        profile = load_area[str(int(area))]
        for bus_id, frac in share.items():
            if frac <= 0:
                continue
            n.add(
                "Load",
                f"L_{bus_id}",
                bus=bus_id,
                p_set=profile * frac,
                carrier="electricity",
                area=int(area),
            )


def _add_carriers(n: pypsa.Network) -> None:
    """Registra os carriers usados (facilita agregação e cores nos gráficos)."""
    carriers = sorted(
        set(n.generators["carrier"])
        | set(n.storage_units["carrier"])
        | {"AC", "DC", "electricity"}
    )
    n.add("Carrier", carriers)


def build_network(data: RTSData, day: str, cfg: dict) -> pypsa.Network:
    """Monta a rede PyPSA do RTS-GMLC original para um dia (24 h).

    Parameters
    ----------
    data : RTSData
        Dados carregados por ``load_rts_data``.
    day : str
        Dia no formato ``YYYY-MM-DD`` (ano 2020 do RTS-GMLC).
    cfg : dict
        Configuração (``config/base.yaml`` já lida).
    """
    sns = day_snapshots(day)
    n = pypsa.Network(name=f"RTS-GMLC {day}")
    n.set_snapshots(sns)

    _add_buses(n, data)
    _add_branches(n, data)
    _add_generators(n, data, cfg, sns)
    _add_loads(n, data, sns)
    _add_carriers(n)

    # Picos anuais de carga por área: usados por layers.stress_system para
    # dimensionar o acréscimo de geração na Área 3.
    n.meta.update(
        {
            "day": day,
            "system": "RTS-GMLC (original)",
            "area_peak_load_mw": {a: float(data.load_area[a].max()) for a in data.load_area.columns},
            "annual_load_mwh_by_area": {a: float(data.load_area[a].sum()) for a in data.load_area.columns},
            "annual_hours": int(len(data.load_area)),
        }
    )
    n.generators["cf_annual"] = n.generators["cf_annual"].fillna(0.0)
    return n


# -----------------------------------------------------------------------------
# 4. Utilidades de diagnóstico usadas no relatório de validação (F1)
# -----------------------------------------------------------------------------
def thermal_summary(n: pypsa.Network) -> pd.DataFrame:
    """Tabela resumo por carrier térmico: capacidade, custo marginal e partida."""
    g = n.generators[n.generators["is_thermal"]]
    return (
        g.groupby("carrier")
        .agg(
            n_units=("p_nom", "size"),
            p_nom_mw=("p_nom", "sum"),
            marginal_cost_mean=("marginal_cost", "mean"),
            start_up_cost_mean=("start_up_cost", "mean"),
            min_up_time_mean=("min_up_time", "mean"),
            co2_t_per_mwh_mean=("co2_t_per_mwh", "mean"),
        )
        .round(2)
    )


def lpf_power_balance_error(n: pypsa.Network, snapshot=None) -> float:
    """Erro máximo de balanço nodal (MW) após ``n.lpf`` em um snapshot.

    Usado nos testes e na validação dos transformadores: injeções líquidas
    devem igualar o somatório dos fluxos incidentes em cada barra.
    """
    if snapshot is None:
        snapshot = n.snapshots[0]
    n.lpf(snapshot)
    # Injeção líquida por barra: geração + armazenamento - carga
    inj = pd.Series(0.0, index=n.buses.index)
    for comp, sign in (("generators", 1.0), ("storage_units", 1.0), ("loads", -1.0)):
        df = getattr(n, comp)
        if df.empty:
            continue
        p = getattr(n, f"{comp}_t").p.loc[snapshot]
        inj = inj.add(sign * p.groupby(df["bus"]).sum(), fill_value=0.0)
    flow = pd.Series(0.0, index=n.buses.index)
    for comp in ("lines", "transformers", "links"):
        df = getattr(n, comp)
        p0 = getattr(n, f"{comp}_t").p0.loc[snapshot]
        for name, b0, b1 in zip(df.index, df["bus0"], df["bus1"]):
            flow[b0] += p0[name]
            flow[b1] -= p0[name]
    return float((inj - flow).abs().max())
