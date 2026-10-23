"""
layers.py — Fases 2, 3 e 4: configuração estressada, escalonamento VRE e as
camadas de flexibilidade (Virtual Power Line e Virtual Data Center).

Todas as funções operam *in place* sobre um ``pypsa.Network`` já construído
por ``rts_import.build_network`` e registram suas decisões em ``n.meta`` para
rastreabilidade (o dicionário é exportado junto com o NetCDF).

Resumo das funções públicas
---------------------------
stress_system(n, cfg)      +40 % de carga na Área 2 e VRE equivalente na Área 3 (§3.1)
scale_vre(n, share)        escala a capacidade VRE até a penetração-alvo (§3.2)
limit_corridor(n, factor)  fallback documentado: reduz s_nom do corredor (§3.1)
add_vpl(n, cfg)            par de BESS nos terminais do corredor (§4)
compute_sigma(n, cfg)      estágio de uso da rede sigma_t para a variante explícita (§4)
add_dc_inflexible(n, cfg)  data centers como carga plana (casos sem camada VDC)
add_vdc(n, cfg)            workloads com janela, sites viáveis e custo de migração (§5)
corridor_flow(n)           série do fluxo no corredor (positivo = montante -> jusante)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa


# =============================================================================
# Utilidades do corredor
# =============================================================================
def corridor_lines(n: pypsa.Network, cfg: dict) -> pd.DataFrame:
    """Devolve as linhas do corredor com uma coluna ``sign`` de orientação.

    ``sign = +1`` quando ``bus0`` é o terminal a montante (p0 > 0 significa
    fluxo montante → jusante); ``-1`` caso contrário.
    """
    c = cfg["corridor"]
    lines = n.lines.loc[c["branches"]].copy()
    lines["sign"] = np.where(lines["bus0"] == str(c["bus_up"]), 1.0, -1.0)
    return lines


def corridor_flow(n: pypsa.Network, cfg: dict) -> pd.Series:
    """Fluxo agregado no corredor (MW), positivo no sentido montante → jusante."""
    lines = corridor_lines(n, cfg)
    return (n.lines_t.p0[lines.index] * lines["sign"]).sum(axis=1)


def corridor_limit(n: pypsa.Network, cfg: dict) -> float:
    """Capacidade agregada do corredor (soma de ``s_nom`` dos circuitos)."""
    return float(corridor_lines(n, cfg)["s_nom"].sum())


def vre_available(n: pypsa.Network) -> pd.DataFrame:
    """Potência VRE disponível por gerador e snapshot (MW) = p_max_pu × p_nom."""
    g = n.generators[n.generators["is_vre"]]
    pmax_pu = n.get_switchable_as_dense("Generator", "p_max_pu")[g.index]
    return pmax_pu * g["p_nom"]


def export_surplus(n: pypsa.Network, cfg: dict) -> pd.Series:
    """Excedente da área exportadora (MW) — proxy ex-ante do carregamento do corredor.

    ``VRE disponível + geração inflexível − carga``, tudo dentro da área do
    terminal de montante. É a grandeza física que **força** potência para o
    corredor: quanto maior o excedente local, mais precisa escoar.

    Serve onde ``corridor_flow`` não pode: ``compute_sigma`` é chamado em
    ``add_vpl``, na montagem da rede, quando ainda não há solução e portanto não
    existe ``lines_t.p0``. O excedente depende apenas de ``p_max_pu`` e ``p_set``,
    conhecidos de antemão — o que é coerente com o caráter day-ahead do problema,
    em que o estágio de uso seria definido a partir da previsão, não do despacho
    realizado.
    """
    up_area = str(cfg["corridor"]["bus_up"])[0]
    g = n.generators
    in_area = g.index[g["bus"].str[0] == up_area]

    vre = [c for c in in_area if bool(g.at[c, "is_vre"])]
    pmax = n.get_switchable_as_dense("Generator", "p_max_pu")
    disp = (pmax[vre] * g.loc[vre, "p_nom"]).sum(axis=1) if vre else pd.Series(0.0, index=n.snapshots)

    # geração inflexível da área (nuclear em p_min_pu, hidro de despacho fixo)
    pmin = n.get_switchable_as_dense("Generator", "p_min_pu")
    firm_i = [c for c in in_area if c not in vre]
    firm = (pmin[firm_i] * g.loc[firm_i, "p_nom"]).sum(axis=1) if firm_i else pd.Series(0.0, index=n.snapshots)

    loads = n.loads.index[n.loads["bus"].str[0] == up_area]
    load = n.get_switchable_as_dense("Load", "p_set")[loads].sum(axis=1) if len(loads) else 0.0
    return (disp + firm - load).rename("export_surplus")


def net_demand(n: pypsa.Network) -> pd.Series:
    """Demanda líquida do sistema (MW) = carga total − VRE disponível."""
    load = n.get_switchable_as_dense("Load", "p_set").sum(axis=1)
    return load - vre_available(n).sum(axis=1)


# =============================================================================
# Fase 2 — Configuração estressada e escalonamento VRE
# =============================================================================
def plan_added_vre(cfg: dict, gens: pd.DataFrame, mw_total: float) -> list[dict]:
    """Plano de alocação do acréscimo VRE na Área 3 (§3.1).

    Duas políticas (``stress.allocation``):

    * ``"candidates"`` (padrão): o acréscimo é distribuído pelos **nós
      candidatos da tese** (``Phd_Files/Network_RTS_Candidate_Gen_Costs.xlsx``:
      eólica em N_314/N_316/N_319, PV em N_320), com pesos iguais às
      capacidades candidatas e perfis ``p_max_pu`` de unidades existentes
      vizinhas da mesma tecnologia (``profile_from``). Estes nós ficam no anel
      de 230 kV que alimenta N_318, o que preserva o corredor N_318–N_223 como
      caminho de exportação.
    * ``"existing"``: proporcional à capacidade VRE já instalada nas barras da
      Área 3. Concentra o acréscimo em N_317 e provoca congestionamento na
      linha interna C29 (317–318) antes do corredor — documentado no F2.

    A repartição eólica/PV segue ``added_wind_share``. Devolve uma lista de
    dicionários ``{name, bus, carrier, p_nom_mw, profile_from}``.
    """
    s = cfg["stress"]
    gen_area = int(s["gen_area"])
    split = {"Wind": float(s["added_wind_share"]), "Solar PV": 1.0 - float(s["added_wind_share"])}
    base = gens[~gens.index.str.endswith("_ADD")]
    plan: list[dict] = []

    if s.get("allocation", "candidates") == "candidates":
        sites = pd.DataFrame(s["candidate_sites"])
        sites["bus"] = sites["bus"].astype(str)
        for carrier, share in split.items():
            sub = sites[sites["carrier"] == carrier]
            if sub.empty or share <= 0:
                continue
            w = sub["weight_mw"] / sub["weight_mw"].sum()
            for (_, row), wi in zip(sub.iterrows(), w):
                plan.append(dict(name=f"{row['bus']}_{carrier.replace(' ', '')}_ADD", bus=row["bus"], carrier=carrier,
                                 p_nom_mw=mw_total * share * float(wi), profile_from=str(row["profile_from"])))
    else:
        for carrier, share in split.items():
            existing = base[(base["carrier"] == carrier) & (base["area"] == gen_area)]
            if existing.empty or share <= 0:
                continue
            w = existing["p_nom"] / existing["p_nom"].sum()
            for uid, wi in w.items():
                plan.append(dict(name=f"{uid}_ADD", bus=existing.at[uid, "bus"], carrier=carrier,
                                 p_nom_mw=mw_total * share * float(wi), profile_from=uid))
    return plan


def stress_system(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """Replica a configuração da tese / Energies 2026 (§3.1).

    1. Multiplica por ``load_factor`` (1,40) todas as cargas da Área 2.
    2. Adiciona na Área 3 capacidade VRE igual ao acréscimo do pico anual de
       carga da Área 2, repartida entre eólica e PV (``added_wind_share``) e
       alocada nas barras que já possuem VRE, proporcionalmente à capacidade
       existente. Cada nova unidade herda o perfil ``p_max_pu`` da unidade
       existente correspondente.
    """
    s = cfg["stress"]
    load_area, gen_area = int(s["load_area"]), int(s["gen_area"])
    factor = float(s["load_factor"])

    # --- 1. Carga da Área 2
    loads = n.loads.index[n.loads["area"] == load_area]
    n.loads_t.p_set[loads] *= factor

    # --- 2. Acréscimo de capacidade = (fator − 1) × pico anual da Área 2
    peak = float(n.meta["area_peak_load_mw"][str(load_area)])
    mw_added_total = (factor - 1.0) * peak

    g = n.generators
    added = {}
    for item in plan_added_vre(cfg, g, mw_added_total):
        src = item["profile_from"]
        n.add(
            "Generator",
            item["name"],
            bus=item["bus"],
            carrier=item["carrier"],
            fuel=g.at[src, "fuel"],
            unit_type=g.at[src, "unit_type"],
            p_nom=item["p_nom_mw"],
            p_max_pu=n.generators_t.p_max_pu[src],
            marginal_cost=g.at[src, "marginal_cost"],
            area=gen_area,
            is_vre=True,
            is_thermal=False,
            cf_annual=g.at[src, "cf_annual"],
            added_stress=True,
        )
        added[item["name"]] = {"bus": item["bus"], "carrier": item["carrier"], "p_nom_mw": round(item["p_nom_mw"], 2),
                               "profile_from": src}

    n.generators["added_stress"] = n.generators["added_stress"].fillna(False).astype(bool)
    # Carga anual estressada (para o escalonamento VRE em base anual)
    annual = n.meta["annual_load_mwh_by_area"]
    n.meta["annual_load_mwh_stressed"] = float(
        sum(v * (factor if int(a) == load_area else 1.0) for a, v in annual.items())
    )
    n.meta["stress"] = {
        "load_area": load_area,
        "load_factor": factor,
        "area2_annual_peak_mw": peak,
        "mw_added_area3": round(mw_added_total, 2),
        "added_units": added,
    }
    n.meta["system"] = "RTS-GMLC estressado (tese / Energies 2026)"
    return n


def vre_share(n: pypsa.Network) -> float:
    """Penetração VRE diária = Σ energia VRE disponível / Σ carga, nos snapshots da rede."""
    load = n.get_switchable_as_dense("Load", "p_set").sum().sum()
    return float(vre_available(n).sum().sum() / load)


def vre_share_annual(n: pypsa.Network) -> float:
    """Penetração VRE anual = Σ_g p_nom_g · CF_g · 8784 / carga anual estressada.

    Usa o fator de capacidade anual de cada unidade (``cf_annual``, calculado no
    parser a partir da série DAY_AHEAD do ano inteiro) e a carga anual já com o
    fator de estresse da Área 2.
    """
    g = n.generators[n.generators["is_vre"]]
    energy = float((g["p_nom"] * g["cf_annual"]).sum() * n.meta["annual_hours"])
    load = float(n.meta.get("annual_load_mwh_stressed", sum(n.meta["annual_load_mwh_by_area"].values())))
    return energy / load


def scale_vre(n: pypsa.Network, target_share: float, basis: str = "annual",
              cfg: dict | None = None) -> float:
    """Escala ``p_nom`` de todas as unidades VRE por um fator único (§3.2).

    A métrica de penetração é a energia VRE *disponível* (não a despachada)
    dividida pela carga, com a configuração estressada já aplicada. Duas bases:

    * ``"annual"`` (padrão): fator único para o ano — o parque instalado é o
      mesmo em todos os dias representativos e a penetração é a anual, como na
      tese. Dias de vento forte têm share diário > alvo (curtailment máximo) e o
      dia de pico tem share < alvo. Foi a base necessária para atender ao
      critério F2 (saturação do corredor).
    * ``"day"``: fator calculado sobre os 24 snapshots do caso (leitura literal
      do brief); reduz o parque nos dias ventosos e o amplia nos dias de pico.

    Devolve o fator aplicado (também registrado em ``n.meta``).
    """
    vre = n.generators.index[n.generators["is_vre"]]

    # --- Composição do parque (opcional): pesos por tecnologia -----------------
    # O fator único preserva a proporção eólica/solar do RTS-GMLC, em que a
    # eólica da Área 3 é 69 % original e entrega 1.400-2.250 MW as 24 horas. O
    # excedente resultante é um platô eólico com uma corcova solar em cima — não
    # o regime de rampa solar ao meio-dia que caracteriza os sistemas com alta
    # penetração fotovoltaica e geração distribuída (Brasil, Chile).
    #
    # ``vre.carrier_weights`` reponderam as tecnologias ANTES do escalonamento
    # global, de modo que a **composição** passa a ser controlável e o **nível**
    # continua fixado por ``target_share``. Pesos ausentes valem 1,0.
    weights = (cfg or {}).get("carrier_weights") if isinstance(cfg, dict) else None
    if weights:
        for carrier, w in weights.items():
            idx = [g for g in vre if n.generators.at[g, "carrier"] == carrier]
            if idx:
                n.generators.loc[idx, "p_nom"] *= float(w)

    current = vre_share_annual(n) if basis == "annual" else vre_share(n)
    factor = float(target_share) / current
    n.generators.loc[vre, "p_nom"] *= factor
    n.meta["vre_scaling"] = {
        "basis": basis,
        "share_before": round(current, 4),
        "target_share": float(target_share),
        "factor": round(factor, 4),
        "carrier_weights": dict(weights) if weights else None,
        "share_after_annual": round(vre_share_annual(n), 4),
        "share_after_day": round(vre_share(n), 4),
    }
    return factor


def export_area_ties(n: pypsa.Network, cfg: dict) -> pd.DataFrame:
    """Interligações que saem da área exportadora e NÃO pertencem ao corredor.

    A área exportadora é a do terminal de montante (``corridor.bus_up``). São as
    rotas por onde o excedente pode contornar o corredor.
    """
    up_area = str(cfg["corridor"]["bus_up"])[0]
    corr = set(corridor_lines(n, cfg).index)
    rows = []
    for l in n.lines.index:
        if l in corr:
            continue
        a0, a1 = n.lines.at[l, "bus0"][0], n.lines.at[l, "bus1"][0]
        if (a0 == up_area) != (a1 == up_area):        # cruza a fronteira da área
            rows.append({"line": l, "bus0": n.lines.at[l, "bus0"], "bus1": n.lines.at[l, "bus1"],
                         "s_nom": float(n.lines.at[l, "s_nom"])})
    return pd.DataFrame(rows).set_index("line") if rows else pd.DataFrame()


def radialize_export(n: pypsa.Network, cfg: dict) -> None:
    """Torna o corredor a única rota de escoamento da área exportadora (§realismo).

    No RTS-GMLC a Área 3 exporta por **dois** caminhos — CB-1 para a Área 2 e
    CA-1 para a Área 1, que por sua vez alimenta a Área 2 por três linhas. Com a
    rota paralela disponível, restringir o corredor não cria escassez: o fluxo
    contorna. Medições no caso base mostram CB-1 saturado em 11 h enquanto CA-1
    opera a 31 % da capacidade.

    Corredores reais de exportação de geração remota — Norte–Sudeste no Brasil,
    o tronco do SIC chileno — são radiais: não existe desvio. Esta função reduz
    as demais interligações da área exportadora a uma interligação de emergência
    (``stress.emergency_tie_mw``), preservando a conectividade elétrica da rede
    mas eliminando a rota alternativa de escoamento.
    """
    st = cfg.get("stress", {})
    if not bool(st.get("radial_export", False)):
        return
    ties = export_area_ties(n, cfg)
    if ties.empty:
        return
    mw = st.get("emergency_tie_mw", 100.0)

    # Reduzir a interligação a um valor pequeno NÃO a tira do caminho: o fluxo DC
    # reparte-se entre a rota direta e a indireta segundo as impedâncias, e o
    # limite pequeno passa a travar a exportação TOTAL em vez de redirecioná-la
    # para o corredor. Medido na hora de maior excedente: CA-1 saturada em
    # 100/100 MW com o CB-1 a apenas 85 % — corte de 98 % da solar utility com o
    # tronco folgado. Radialidade de verdade exige remover a linha.
    if mw is None or float(mw) <= 0:
        removed = {l: float(ties.at[l, "s_nom"]) for l in ties.index}
        n.remove("Line", list(ties.index))
        n.meta["radial_export"] = {"ties_removed": removed}
        return

    mw = float(mw)
    n.lines.loc[ties.index, "s_nom"] = mw
    n.meta["radial_export"] = {"ties_reduced": {l: [float(ties.at[l, "s_nom"]), mw] for l in ties.index},
                               "emergency_tie_mw": mw}


def reinforce_collection(n: pypsa.Network, cfg: dict) -> None:
    """Reforça a rede interna da área exportadora (§realismo).

    Diagnóstico que motiva esta função: no caso base do RTS-GMLC a restrição
    ativa **não é o corredor**, é a linha interna C29 da Área 3 — utilização
    média 0,95 e 15 h saturadas, contra 0,94 e 17 h do CB-1. O acréscimo de VRE
    fica nas barras 314/316/319/320 e precisa atravessar o anel de 230 kV até a
    barra 318; esse anel satura antes do tronco.

    Isso invalida o teste da hipótese: o VPL instala baterias nas duas pontas do
    corredor para aliviar o corredor, mas quem limita o escoamento está a
    montante. Um sistema exportador real tem a rede de coleta dimensionada para
    entregar ao tronco — o tronco é o elemento caro e restrito, não a coleta.

    ``stress.collection_factor`` multiplica ``s_nom`` de todas as linhas internas
    à área exportadora, deixando o corredor como única restrição de escoamento.
    """
    st = cfg.get("stress", {})
    k = st.get("collection_factor")
    if not k or float(k) == 1.0:
        return
    corr = set(corridor_lines(n, cfg).index)
    scope = str(st.get("reinforce_scope", "export_area")).lower()

    if scope == "all":
        # Reforça TODA a rede exceto o corredor, isolando-o como única restrição
        # de transporte. Necessário porque o gargalo migra: com a coleta da área
        # exportadora reforçada, a restrição ativa passou para a linha B28
        # (216->219, interna à área importadora), que saturava a 100 % enquanto o
        # corredor de 1.600 MW ficava a 60 %. Um sistema exportador real tem o
        # tronco como elemento caro e restrito; a rede em torno das cargas é
        # malhada e forte. Reforçar o entorno é o que permite medir o valor da
        # capacidade do tronco sem contaminação.
        target = [l for l in n.lines.index if l not in corr]
        tag = "network_reinforced"
    else:
        up_area = str(cfg["corridor"]["bus_up"])[0]
        target = [l for l in n.lines.index
                  if l not in corr
                  and n.lines.at[l, "bus0"][0] == up_area
                  and n.lines.at[l, "bus1"][0] == up_area]
        tag = "collection_reinforced"
    if not target:
        return
    before = float(n.lines.loc[target, "s_nom"].sum())
    n.lines.loc[target, "s_nom"] *= float(k)
    n.meta[tag] = {"lines": len(target), "factor": float(k), "scope": scope,
                   "s_nom_total_mw": [before, before * float(k)]}


def shape_corridor(n: pypsa.Network, cfg: dict) -> None:
    """Redimensiona o corredor: capacidade absoluta, comprimento e impedância.

    ``corridor.s_nom_mw``  — limite térmico absoluto (MW), somado sobre os
    circuitos do corredor. Substitui o ``limit_corridor_factor``, que só escalava
    o valor herdado do RTS.

    ``corridor.x_scale``   — multiplicador da reatância. Uma linha de ~1000 km
    tem reatância muito maior que as 72 milhas do CB-1; na prática recebe
    compensação série, de modo que o multiplicador efetivo fica bem abaixo da
    razão de comprimentos. O valor entra no fluxo DC e altera a repartição de
    fluxos, então é registrado em ``n.meta`` para auditoria.

    ``corridor.length_km`` — usado apenas no pós-processamento econômico (custo
    de reforço proporcional ao comprimento); não afeta o despacho.
    """
    c = cfg["corridor"]
    lines = corridor_lines(n, cfg).index
    meta = {}
    s_abs = c.get("s_nom_mw")
    if s_abs:
        share = n.lines.loc[lines, "s_nom"] / n.lines.loc[lines, "s_nom"].sum()
        n.lines.loc[lines, "s_nom"] = float(s_abs) * share      # rateia entre circuitos
        meta["s_nom_mw"] = float(s_abs)
    xs = c.get("x_scale")
    if xs and float(xs) != 1.0:
        n.lines.loc[lines, "x"] *= float(xs)
        n.lines.loc[lines, "r"] *= float(xs)
        meta["x_scale"] = float(xs)
    if c.get("length_km"):
        meta["length_km"] = float(c["length_km"])
    if meta:
        n.meta["corridor_shape"] = meta


def limit_corridor(n: pypsa.Network, cfg: dict, factor: float) -> None:
    """Escala ``s_nom`` das linhas do corredor por ``factor``.

    ``factor < 1`` é o fallback documentado (§3.1), a ser usado só se o corredor
    não saturar mesmo após aumentar a fração VRE do acréscimo da Área 3.

    ``factor > 1`` é o **contrafactual de reforço**: em vez de instalar
    flexibilidade, constrói-se capacidade de transmissão. É o caso contra o qual
    o valor do VPL precisa ser medido — a alternativa que um planejador
    escolheria — e o que permite exprimir o par de BESS em MW de linha
    equivalente (ver ``economics.py`` §5).
    """
    lines = corridor_lines(n, cfg).index
    n.lines.loc[lines, "s_nom"] *= float(factor)
    n.meta["corridor_limit_factor"] = float(factor)


# =============================================================================
# Fase 3 — Virtual Power Line
# =============================================================================
VPL_UP, VPL_DOWN = "VPL_up", "VPL_down"


def add_vpl(n: pypsa.Network, cfg: dict) -> None:
    """Adiciona o par de BESS do VPL nos terminais do corredor (§4).

    Formulação padrão (*emergente*): nenhum acoplamento explícito entre os dois
    BESS; o comportamento de "linha virtual" emerge do spread de LMP entre os
    terminais. A variante *explícita* (regra de chaveamento da tese) é imposta
    em ``constraints.py`` quando ``cfg["vpl"]["mode"] == "explicit"``; aqui só
    se pré-calcula ``sigma_t`` para ela.

    ``cfg["vpl"]["terminals"]`` (padrão ``"both"``) permite instalar apenas o
    BESS de montante (``"up"``) ou de jusante (``"down"``). Serve à decomposição
    do benefício do VPL em *arbitragem de bateria isolada* (soma dos dois casos
    com um terminal) e *valor do pareamento* (diferença para o caso completo).
    """
    v, c = cfg["vpl"], cfg["corridor"]
    eta = np.sqrt(float(v["roundtrip_efficiency"]))
    terminals = str(v.get("terminals", "both")).lower()
    if "VPL" not in n.carriers.index:
        n.add("Carrier", "VPL")
    pairs = ((VPL_UP, str(c["bus_up"])), (VPL_DOWN, str(c["bus_down"])))
    if terminals == "up":
        pairs = pairs[:1]
    elif terminals == "down":
        pairs = pairs[1:]
    for name, bus in pairs:
        n.add(
            "StorageUnit",
            name,
            bus=bus,
            carrier="VPL",
            p_nom=float(v["p_nom_mw"]),
            max_hours=float(v["hours"]),
            efficiency_store=eta,
            efficiency_dispatch=eta,
            cyclic_state_of_charge=True,
            marginal_cost=float(v["marginal_cost"]),   # degradação, cobrado na descarga
        )
    n.meta["vpl"] = {
        "p_nom_mw": float(v["p_nom_mw"]),
        "hours": float(v["hours"]),
        "mwh_per_terminal": float(v["p_nom_mw"]) * float(v["hours"]),
        "mode": v["mode"],
        "terminals": terminals,
    }
    if str(v["mode"]).lower() in ("explicit", "dedicated", "hybrid"):
        sigma = compute_sigma(n, cfg)
        # NaN (hora intermediária) vira None para ser serializável em JSON
        n.meta["sigma"] = [None if pd.isna(x) else int(x) for x in sigma]


def add_h2(n: pypsa.Network, cfg: dict) -> None:
    """Cadeia de hidrogênio: P2H a montante, estoque, H2P a jusante (§ H2 do artigo).

    Implementa as Eqs. (11)-(12) do artigo do periódico::

        E_t = E_{t-1} + sum_e eta_e p^e_{e,t} - sum_phi p^f_{phi,t} / eta_phi
        0 <= E_t <= Emax ,  0 <= p^f <= Pmax_phi

    em componentes PyPSA: um ``Link`` de eletrólise (barra elétrica a montante →
    barra de H2, eficiência ``eta_electrolyzer``), um ``Store`` de hidrogênio e
    um ``Link`` de reconversão (barra de H2 → barra elétrica a jusante,
    eficiência ``eta_fuelcell``). O produto das duas eficiências é o round trip.

    O sítio segue Lu & Li: eletrolisador na barra a montante do corredor,
    célula a combustível na barra a jusante. A cadeia liga os dois terminais do
    corredor **sem usar o corredor** --- é o que a distingue do VPL, cujo
    caminho precisa de uma hora descongestionada na própria linha.

    Dois modos para o eletrolisador:

    * ``continuous``: carga continuamente variável entre 0 e ``p_nom``;
    * ``three_state``: produz / standby / desligado, como em Saretta et al.
      (arXiv:2306.10962, Eqs. 3b-3d). A rede ganha um segundo ``Link``,
      ``H2_P2H_sb``, de eficiência nula: é o consumo de standby
      ``P^sb z^sb``, que entra no balanço da barra a montante sem produzir
      hidrogênio. As binárias, a carga mínima ``P^min z^on <= p <= P^max z^on``
      e o custo de partida a frio ficam em ``constraints.add_h2_states``.

    ``h2.marginal_cost`` é a degradação em $/MWh **elétrico entregue pela
    célula**, a mesma base do VPL (``StorageUnit.marginal_cost``, cobrado na
    descarga). O PyPSA cobra o ``Link`` sobre ``p0``, o hidrogênio retirado do
    estoque; por isso o valor é multiplicado por ``eta_fuelcell`` antes de ir ao
    Link. Com o padrão 0 nada muda.
    """
    h = cfg["h2"]
    c = cfg["corridor"]
    modo = str(h.get("mode", "continuous")).lower()
    if modo not in ("continuous", "three_state"):
        raise ValueError(f"h2.mode={modo!r} desconhecido; use 'continuous' ou 'three_state'")

    bus_up, bus_down = str(c["bus_up"]), str(c["bus_down"])
    # ``h2.electrolyzer_bus`` / ``h2.fc_bus``: barras do eletrolisador e da
    # célula. Padrão: os terminais do corredor (Lu & Li), o que faz da cadeia um
    # caminho PARALELO ao corredor. Com ``fc_bus`` = terminal a montante a cadeia
    # vira armazenamento no terminal exportador e passa a depender do corredor,
    # como o VPL --- o teste topológico do PES GM.
    bus_up = str(h.get("electrolyzer_bus", bus_up))
    bus_down = str(h.get("fc_bus", bus_down))
    hbus = "H2_store_bus"
    for carrier in ("H2",):
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)
    n.add("Bus", hbus, carrier="H2")

    # ``h2.electrolyzer_marginal_cost``: O&M variável do eletrolisador em $/MWh
    # elétrico CONSUMIDO na produção; o PyPSA cobra sobre p0, que já é essa base.
    # O consumo de standby (H2_P2H_sb) não paga: não há produção nem desgaste de pilha.
    n.add("Link", "H2_P2H", bus0=bus_up, bus1=hbus, carrier="H2",
          p_nom=float(h["p_nom_mw"]), efficiency=float(h["eta_electrolyzer"]),
          marginal_cost=float(h.get("electrolyzer_marginal_cost", 0.0)))
    n.add("Store", "H2_store", bus=hbus, carrier="H2",
          e_nom=float(h["store_mwh"]), e_cyclic=bool(h.get("cyclic", True)))
    n.add("Link", "H2_H2P", bus0=hbus, bus1=bus_down, carrier="H2",
          p_nom=float(h["fc_p_nom_mw"]) / float(h["eta_fuelcell"]),
          efficiency=float(h["eta_fuelcell"]),
          marginal_cost=float(h.get("marginal_cost", 0.0)) * float(h["eta_fuelcell"]))

    states = {}
    if modo == "three_state":
        p_max = float(h["p_nom_mw"])
        states = {"p_min_mw": float(h["min_load"]) * p_max,
                  "p_sb_mw": float(h["standby_load"]) * p_max,
                  "startup_cost": float(h["startup_cost"])}
        if not 0.0 < states["p_sb_mw"] < states["p_min_mw"] <= p_max:
            raise ValueError(f"h2: exige 0 < standby < carga mínima <= p_nom; recebido {states}")
        # Consumo de standby: sai da barra a montante e não chega ao estoque.
        n.add("Link", "H2_P2H_sb", bus0=bus_up, bus1=hbus, carrier="H2",
              p_nom=states["p_sb_mw"], efficiency=0.0)

    n.meta["h2"] = {
        "bus_up": bus_up, "bus_down": bus_down, "mode": modo,
        "p_nom_mw": float(h["p_nom_mw"]), "fc_p_nom_mw": float(h["fc_p_nom_mw"]),
        "store_mwh": float(h["store_mwh"]),
        "eta_electrolyzer": float(h["eta_electrolyzer"]),
        "eta_fuelcell": float(h["eta_fuelcell"]),
        "roundtrip": float(h["eta_electrolyzer"]) * float(h["eta_fuelcell"]),
        "degradation_per_mwh_out": float(h.get("marginal_cost", 0.0)),
        "electrolyzer_cost_per_mwh_in": float(h.get("electrolyzer_marginal_cost", 0.0)),
        **states,
    }


def compute_sigma(n: pypsa.Network, cfg: dict) -> pd.Series:
    """Estágio de uso da rede por hora para as variantes com regra (§4).

    ``sigma_t = 1`` nas horas de **alto uso** (≥ percentil alto), ``0`` nas de
    **baixo uso** (≤ percentil baixo) e ``NaN`` nas intermediárias, em que o BESS
    fica livre para arbitragem. Os percentis são tomados sobre a curva de duração
    do horizonte resolvido (24 h), coerentes com o caráter day-ahead.

    ``vpl.sigma_basis`` escolhe a grandeza que define o estágio:

    ``"net_demand"``
        Demanda líquida do sistema — a formulação original. Descreve um corredor
        de **importação** que alimenta um centro de carga: quanto mais carregado
        o sistema, mais estressado o corredor.

    ``"corridor"``
        Excedente da área exportadora (ver ``export_surplus``), que é o que força
        potência para o corredor. Necessário quando o corredor é de **exportação**
        de geração remota, caso em que ele satura justamente quando sobra
        renovável — ou seja, quando a demanda líquida está baixa. Usa-se o
        excedente, e não o fluxo, porque ``compute_sigma`` roda na montagem da
        rede, antes de existir solução.

    Motivação da opção, medida neste sistema: a interseção entre horas de
    corredor congestionado e ``sigma_t = 1`` calculado por demanda líquida é
    **zero** em todos os dias testados, e a correlação entre |fluxo no corredor| e
    demanda líquida vai de −0,26 a −0,86. Com a base errada, a regra de
    chaveamento manda o par aliviar o corredor exatamente nas horas em que ele
    está folgado, e o libera nas horas em que satura.
    """
    v = cfg["vpl"]
    basis = str(v.get("sigma_basis", "net_demand")).lower()
    if basis == "corridor":
        x = export_surplus(n, cfg)
    elif basis == "net_demand":
        x = net_demand(n)
    else:
        raise ValueError(f"vpl.sigma_basis inválido: {basis!r} (use 'net_demand' ou 'corridor')")
    hi = np.percentile(x, float(v["sigma_high_percentile"]))
    lo = np.percentile(x, float(v["sigma_low_percentile"]))
    sigma = pd.Series(np.nan, index=n.snapshots, name="sigma")
    sigma[x >= hi] = 1.0
    sigma[x <= lo] = 0.0
    return sigma


# =============================================================================
# Fase 4 — Virtual Data Center
# =============================================================================
def _site_table(cfg: dict) -> pd.DataFrame:
    """Tabela de sites com capacidade flexível derivada de f_d e PUE_d."""
    v = cfg["vdc"]
    sites = pd.DataFrame(v["sites"]).set_index("name")
    sites["bus"] = sites["bus"].astype(str)
    sites["flex_fraction"] = float(v["flex_fraction"])
    sites["flex_mw_elec"] = sites["flex_fraction"] * sites["capacity_mw"]        # f_d · C_d
    sites["flex_mw_it"] = sites["flex_mw_elec"] / sites["pue"]                   # em MW de TI
    sites["inflex_mw_elec"] = (1.0 - sites["flex_fraction"]) * sites["capacity_mw"]
    return sites


def _edf_schedule(workloads: list[dict], cap_it_mw: float, horizon: int = 24) -> np.ndarray:
    """Cronograma EDF (MW de TI por hora) de um site: cada job executa no site
    nativo a partir da chegada, à máxima potência disponível, prioridade por
    prazo. É o *baseline* "native" (jobs inflexíveis no tempo e no espaço)."""
    remaining = {w["id"]: w["E_it_mwh"] for w in workloads}
    sched = np.zeros(horizon)
    for t in range(horizon):
        active = [w for w in workloads if w["arrival"] <= t <= w["deadline"] and remaining[w["id"]] > 1e-9]
        active.sort(key=lambda w: w["deadline"])
        cap = cap_it_mw
        for w in active:
            p_max = cap_it_mw * (0.5 if t in (w["arrival"], w["deadline"]) else 1.0)
            served = min(p_max, remaining[w["id"]], cap)
            remaining[w["id"]] -= served
            cap -= served
            sched[t] += served
            if cap <= 1e-9:
                break
    return sched


def _edf_schedule_by_job(workloads: list[dict], cap_it_mw: float, horizon: int = 24) -> dict[str, np.ndarray]:
    """O mesmo EDF de ``_edf_schedule``, separado por workload (MW de TI por hora).

    A soma sobre os workloads reproduz ``_edf_schedule`` hora a hora; é a forma
    de cada job no cronograma de referência, usada pela variante rígida.
    """
    remaining = {w["id"]: w["E_it_mwh"] for w in workloads}
    out = {w["id"]: np.zeros(horizon) for w in workloads}
    for t in range(horizon):
        active = [w for w in workloads if w["arrival"] <= t <= w["deadline"] and remaining[w["id"]] > 1e-9]
        active.sort(key=lambda w: w["deadline"])
        cap = cap_it_mw
        for w in active:
            p_max = cap_it_mw * (0.5 if t in (w["arrival"], w["deadline"]) else 1.0)
            served = min(p_max, remaining[w["id"]], cap)
            remaining[w["id"]] -= served
            cap -= served
            out[w["id"]][t] += served
            if cap <= 1e-9:
                break
    return out


def _ramp_feasible_by_job(workloads: list[dict], cap_it_mw: float, horizon: int = 24,
                          ramp_pu: float = 0.5, check_only: bool = False) -> dict[str, np.ndarray] | None:
    """Cronograma de referência que o próprio site consegue executar (MW de TI).

    O EDF de ``_edf_schedule`` aproxima a rampa por workload com meia potência
    nas horas de borda, mas ignora a rampa AGREGADA do site
    (``constraints.add_site_ramp``): na grade do periódico DC_A1 cai 37,8 MW
    numa hora e DC_A2 sobe 40 MW, contra 20 MW/h. Como o custo de flexibilidade
    incide sobre o desvio da referência, o VDC pagava c^flex por ~29 MWh/dia
    só para torná-la executável, sem ganho nenhum.

    Aqui a referência é a solução de um LP com as MESMAS restrições que o VDC
    enfrenta no site nativo, em unidades de TI:

        Σ_t x_{j,t} = E_j,  x_{j,t} = 0 fora de [a_j, b_j],
        Σ_j x_{j,t} ≤ cap,
        |x_{j,t} − x_{j,t−1}| ≤ r·cap   (rampa do Link; x_{j,−1} = 0, como o
                                          p_init nulo do PyPSA na 1ª hora),
        |X_t − X_{t−1}| ≤ r·cap, t ≥ 1   (rampa agregada; X_t = Σ_j x_{j,t}),

    e objetivo "o mais cedo possível, prazo mais curto primeiro" --- o espírito
    do EDF ---, Σ (t+1)·(1 + 0,01·(H − b_j))·x_{j,t}. A referência é então uma
    solução viável do modelo com VDC, que pode reproduzi-la a custo zero; o
    c^flex passa a remunerar só deslocamento de verdade.

    Devolve ``None`` se o LP for infactível (com ``check_only`` devolve um
    dicionário vazio quando factível, sem montar os perfis).
    """
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    jobs = [w for w in workloads if w["E_it_mwh"] > 1e-9]
    if not jobs:
        return {} if check_only else {w["id"]: np.zeros(horizon) for w in workloads}
    J, H = len(jobs), horizon
    nv = J * H
    idx = lambda j, t: j * H + t          # noqa: E731
    r = float(ramp_pu) * float(cap_it_mw)

    ub = np.zeros(nv)
    cost = np.zeros(nv)
    for j, w in enumerate(jobs):
        for t in range(H):
            if w["arrival"] <= t <= w["deadline"]:
                ub[idx(j, t)] = cap_it_mw
            cost[idx(j, t)] = (t + 1) * (1.0 + 0.01 * (H - w["deadline"]))

    A_eq = lil_matrix((J, nv))
    b_eq = np.array([w["E_it_mwh"] for w in jobs])
    for j in range(J):
        for t in range(H):
            A_eq[j, idx(j, t)] = 1.0

    rows: list[dict[int, float]] = []
    rhs: list[float] = []
    for t in range(H):                                   # capacidade do site
        rows.append({idx(j, t): 1.0 for j in range(J)}); rhs.append(cap_it_mw)
    for j in range(J):                                   # rampa por Link
        for t in range(H):
            up = {idx(j, t): 1.0}
            if t > 0:
                up[idx(j, t - 1)] = -1.0
            rows.append(up); rhs.append(r)
            if t > 0:
                rows.append({k: -v for k, v in up.items()}); rhs.append(r)
    for t in range(1, H):                                # rampa agregada do site
        up = {idx(j, t): 1.0 for j in range(J)}
        up.update({idx(j, t - 1): -1.0 for j in range(J)})
        rows.append(up); rhs.append(r)
        rows.append({k: -v for k, v in up.items()}); rhs.append(r)
    A_ub = lil_matrix((len(rows), nv))
    for i, row in enumerate(rows):
        for k, v in row.items():
            A_ub[i, k] = v

    res = linprog(cost, A_ub=A_ub.tocsr(), b_ub=np.array(rhs), A_eq=A_eq.tocsr(), b_eq=b_eq,
                  bounds=list(zip(np.zeros(nv), ub)), method="highs")
    if res.status != 0:
        return None
    if check_only:
        return {}
    x = np.clip(res.x.reshape(J, H), 0.0, None)
    out = {w["id"]: np.zeros(H) for w in workloads}
    for j, w in enumerate(jobs):
        out[w["id"]] = x[j]
    return out


def _reference_by_job(workloads: list[dict], cap_it_mw: float, cfg: dict,
                      horizon: int = 24) -> dict[str, np.ndarray]:
    """Forma de cada workload no cronograma de referência (MW de TI por hora).

    ``vdc.reference``: ``"edf"`` (padrão; GTD LA e periódico) ou
    ``"ramp_feasible"`` (``_ramp_feasible_by_job``).
    """
    v = cfg["vdc"]
    if str(v.get("reference", "edf")).lower() == "ramp_feasible":
        out = _ramp_feasible_by_job(workloads, cap_it_mw, horizon, float(v["ramp_limit_pu"]))
        if out is None:
            raise ValueError("referencia com rampa infactivel; generate_workloads deveria ter reescalado")
        return out
    return _edf_schedule_by_job(workloads, cap_it_mw, horizon)


def _reference_schedule(workloads: list[dict], cap_it_mw: float, cfg: dict,
                        horizon: int = 24) -> np.ndarray:
    """Soma de ``_reference_by_job`` sobre os workloads: a carga flexível de referência do site."""
    if str(cfg["vdc"].get("reference", "edf")).lower() != "ramp_feasible":
        return _edf_schedule(workloads, cap_it_mw, horizon)
    by_job = _reference_by_job(workloads, cap_it_mw, cfg, horizon)
    return np.sum(list(by_job.values()), axis=0) if by_job else np.zeros(horizon)


def rigid_jobs(workloads: list[dict], sites: pd.DataFrame, horizon: int = 24,
               cfg: dict | None = None) -> dict[str, dict]:
    """Variante rígida dos workloads: forma fixa, deslocável inteira.

    Cada job conserva o perfil de potência que tem no cronograma de referência
    (EDF no site nativo, a carga dos casos sem VDC) e só pode ser (i) deslocado
    inteiro no tempo, dentro da janela, e (ii) transferido inteiro para outro
    site elegível. Não há divisão entre sites, modulação de potência nem
    preempção além da forma de referência. Desloca-se ``delta`` horas, de
    ``a_j - primeira`` a ``b_j - última`` hora com carga; ``delta = 0`` no site
    nativo é a própria referência, de modo que os casos sem VDC não mudam.
    """
    out = {}
    for d, s in sites.iterrows():
        site_wl = [w for w in workloads if w["native"] == d]
        shapes = (_reference_by_job(site_wl, float(s["flex_mw_it"]), cfg, horizon) if cfg is not None
                  else _edf_schedule_by_job(site_wl, float(s["flex_mw_it"]), horizon))
        for w in site_wl:
            shape = shapes[w["id"]]
            total = float(shape.sum())
            if abs(total - w["E_it_mwh"]) > 1e-6 * max(1.0, w["E_it_mwh"]):
                raise ValueError(f"EDF não conclui {w['id']}: {total:.6f} de {w['E_it_mwh']:.6f} MWh")
            # Resíduos do EDF (DC_A2_B1 termina com 9e-7 MW) viram coeficientes
            # minúsculos na restrição do Link: o presolve do HiGHS chegou a
            # declarar ótima uma solução 191 $ pior que a referência. Abaixo de
            # 0,1 kW é zero, e o resíduo (1e-8 da energia do job) é descartado em
            # vez de redistribuído: as horas da forma em geral já estão na
            # capacidade do site, e qualquer acréscimo, por escala ou numa só
            # entrada, deixa a referência inviável por 1e-6 MW. Por isso a
            # conclusão do job rígido é garantida pela forma, não pelo Store
            # (ver add_vdc).
            shape = np.where(shape < 1e-4, 0.0, shape)
            busy = np.flatnonzero(shape > 0.0)
            first, last = int(busy[0]), int(busy[-1])
            shape = shape[first:last + 1].copy()
            out[w["id"]] = {"shape": [float(x) for x in shape],
                            "start_ref": first,
                            "starts": list(range(int(w["arrival"]), int(w["deadline"]) - (last - first) + 1))}
    return out


def add_dc_inflexible(n: pypsa.Network, cfg: dict) -> None:
    """Data centers como carga inflexível, para os casos SEM camada VDC (A e B).

    A energia diária de cada site é **exatamente** a dos workloads gerados por
    ``generate_workloads`` para a mesma configuração (f_d, janela, semente),
    incluindo os cortes de factibilidade. Assim A/B e C/D consomem a mesma
    energia e a diferença de custo isola o valor da flexibilidade.

    ``cfg["vdc"]["baseline"]``:
    * ``"flat"`` (padrão): energia flexível distribuída uniformemente nas 24 h
      (perfil plano) mais a parte inflexível ``(1 − f_d)·C_d``;
    * ``"native"``: os mesmos jobs executados no site nativo a partir da chegada
      (cronograma EDF, sem deslocamento temporal nem espacial) — contrafactual
      operacional mais estrito, usado como sensibilidade.
    """
    sites = _site_table(cfg)
    mode = str(cfg["vdc"].get("baseline", "flat")).lower()
    workloads, _ = generate_workloads(cfg)
    if "datacenter" not in n.carriers.index:
        n.add("Carrier", "datacenter")
    energy = {}
    for name, s in sites.iterrows():
        site_wl = [w for w in workloads if w["native"] == name]
        e_it = float(sum(w["E_it_mwh"] for w in site_wl))
        energy[name] = e_it * float(s["pue"])
        if mode == "native":
            prof_it = _reference_schedule(site_wl, float(s["flex_mw_it"]), cfg)
            p_set = pd.Series(s["inflex_mw_elec"] + float(s["pue"]) * prof_it, index=n.snapshots)
        else:
            p_set = s["inflex_mw_elec"] + energy[name] / 24.0
        n.add("Load", f"L_{name}_flat", bus=s["bus"], p_set=p_set, carrier="datacenter", area=int(s["area"]))
    n.meta["vdc"] = {"mode": f"inflexible-{mode}", "flex_energy_elec_mwh": energy,
                     "flat_mw": {k: float(sites.loc[k, "inflex_mw_elec"] + energy[k] / 24.0) for k in sites.index}}


def _edf_feasible(workloads: list[dict], cap_it_mw: float, horizon: int = 24,
                  ramp_pu: float | None = None) -> bool:
    """Verifica por simulação EDF (*earliest deadline first*) se um conjunto de
    workloads cabe na capacidade ``cap_it_mw`` de um único site.

    A meia potência nas horas de borda aproxima o efeito da rampa **por
    workload**. Com ``ramp_pu`` a simulação passa a impor também a rampa
    **agregada** do site — que é a restrição de fato presente no modelo
    (``constraints.add_site_ramp``): ``|Σ_j p_t − Σ_j p_(t−1)| ≤ ramp_pu·cap``.

    A distinção importa porque as duas não são equivalentes: se dois jobs
    iniciam em horas diferentes, o agregado pode saltar mais que o limite ainda
    que cada um respeite a própria borda. Sem isso o gerador certifica
    cronogramas que o MILP depois rejeita — o que se manifestou como
    infactibilidade do caso "DC local" (sem migração espacial) quando as janelas
    são estreitas e não existe escapatória para outro site.
    """
    remaining = {w["id"]: w["E_it_mwh"] for w in workloads}
    prev = 0.0
    for t in range(horizon):
        active = [w for w in workloads if w["arrival"] <= t <= w["deadline"] and remaining[w["id"]] > 1e-9]
        active.sort(key=lambda w: w["deadline"])
        cap = cap_it_mw
        if ramp_pu is not None:
            cap = min(cap, prev + float(ramp_pu) * cap_it_mw)   # rampa agregada de subida
        total = 0.0
        for w in active:
            # meia potência nas horas de borda da janela (efeito da rampa)
            p_max = cap_it_mw * (0.5 if t in (w["arrival"], w["deadline"]) else 1.0)
            served = min(p_max, remaining[w["id"]], cap - total)
            remaining[w["id"]] -= served
            total += served
            if cap - total <= 1e-9:
                break
        prev = total
    return all(r <= 1e-6 for r in remaining.values())


def generate_workloads(cfg: dict) -> tuple[list[dict], pd.DataFrame]:
    """Gera o conjunto determinístico (por *seed*) de workloads (§5.2).

    Por site *d*: energia flexível diária ``u·f_d·C_d·24`` MWh elétricos, ou
    seja ``u·f_d·C_d·24 / PUE_d`` MWh de TI, repartida entre workloads
    **batch** (janela 8–12 h, só o site nativo) e **migráveis** (janela 2–4 h,
    todos os sites). A factibilidade de cada site é checada por EDF; se
    necessário as energias do site são reduzidas por bissecção e o fator é
    registrado.

    Devolve a lista de workloads e a matriz de custos de migração
    ``c[nativo][destino]`` ($/MWh de TI), assimétrica, 0 na diagonal.
    """
    v = cfg["vdc"]
    wcfg = v["workloads"]
    rng = np.random.default_rng(int(wcfg["seed"]))
    sites = _site_table(cfg)
    u = float(v["flex_utilization"])
    scale = float(wcfg["window_scale"])
    lo_c, hi_c = wcfg["migration_cost_range"]
    spatial = bool(v.get("spatial", True))      # governa a margem de factibilidade abaixo

    # Matriz assimétrica de custo de migração (sorteada uma vez, por par de sites)
    names = list(sites.index)
    cost = pd.DataFrame(0.0, index=names, columns=names)
    for a in names:
        for b in names:
            if a != b:
                cost.loc[a, b] = float(rng.uniform(lo_c, hi_c))

    workloads: list[dict] = []
    for d, s in sites.iterrows():
        e_it_total = u * s["flex_mw_elec"] * 24.0 / s["pue"]
        e_batch = float(wcfg["batch_share"]) * e_it_total
        e_migr = e_it_total - e_batch

        def _split(total: float, k: int) -> np.ndarray:
            return total * rng.dirichlet(np.full(k, 4.0))   # partes semelhantes, não idênticas

        site_wl: list[dict] = []
        # --- batch (temporal)
        for i, e in enumerate(_split(e_batch, int(wcfg["n_batch_per_site"]))):
            L = int(np.clip(round(rng.uniform(*wcfg["batch_window_h"]) * scale), 2, 24))
            a = int(rng.integers(0, 24 - L + 1))
            site_wl.append(dict(id=f"{d}_B{i+1}", type="batch", native=d, sites=[d],
                                arrival=a, deadline=a + L - 1, E_it_mwh=float(e)))
        # --- migráveis (espacial)
        for i, e in enumerate(_split(e_migr, int(wcfg["n_migratable_per_site"]))):
            L = int(np.clip(round(rng.uniform(*wcfg["migratable_window_h"])), 2, 24))
            a = int(rng.integers(0, 24 - L + 1))
            site_wl.append(dict(id=f"{d}_M{i+1}", type="migratable", native=d, sites=names,
                                arrival=a, deadline=a + L - 1, E_it_mwh=float(e)))

        # --- limite individual: E_j ≤ (L_j − 1)·capacidade de TI do site
        for w in site_wl:
            cap_j = (w["deadline"] - w["arrival"]) * s["flex_mw_it"]
            w["E_it_mwh"] = min(w["E_it_mwh"], cap_j)

        # --- factibilidade conjunta no site nativo (conservadora: ignora migração)
        f_scale = 1.0
        # ``vdc.edf_respects_ramp`` liga a verificação da rampa AGREGADA do site.
        # Desligada (padrão) o gerador só aplica a aproximação por workload — meia
        # potência nas horas de borda —, que sub-restringe: dois jobs iniciando em
        # horas diferentes fazem o agregado saltar mais que o limite. É o
        # comportamento com que os cenários do caso base foram resolvidos, e está
        # preservado por reprodutibilidade. Ligada, a bissecção passa a devolver o
        # dimensionamento que o MILP de fato aceita — necessário sempre que não há
        # migração espacial para absorver o excesso local (``vdc.spatial: false``).
        ramp_pu = float(v.get("ramp_limit_pu", 1.0)) if v.get("edf_respects_ramp", False) else None
        if not _edf_feasible(site_wl, s["flex_mw_it"], ramp_pu=ramp_pu):
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                trial = [dict(w, E_it_mwh=w["E_it_mwh"] * mid) for w in site_wl]
                if _edf_feasible(trial, s["flex_mw_it"], ramp_pu=ramp_pu):
                    lo = mid
                else:
                    hi = mid
            # Margem de segurança quando não há migração espacial.
            #
            # A bissecção converge *em cima* da fronteira de factibilidade, e um
            # conjunto de workloads exatamente na fronteira só é factível a menos
            # do erro de ponto flutuante. Com migração espacial isso passa
            # despercebido — o excesso local escapa para outro site —, mas com
            # ``vdc.spatial = false`` o site fica sem válvula: o MILP só "resolve"
            # violando a capacidade dentro da ``FeasibilityTol`` do solver, e o LP
            # de preços depois fica infactível, porque o seu presolve propaga
            # limites com muito mais força.
            #
            # A margem fica condicionada a ``spatial = false`` de propósito. Ela
            # perturba a energia dos workloads em 1e-6 relativo, o que é ~100x a
            # violação observada e ~500x menor que o gap de MIP (5e-4) — mas essa
            # energia também define a carga plana dos DCs nos casos sem VDC (ver
            # ``add_dc_inflexible``), e perturbá-la faz o commitment degenerado
            # assentar em outro ótimo igualmente válido, deslocando o objetivo em
            # ~0,005 %. Aplicá-la só onde é necessária mantém os cenários já
            # resolvidos reproduzíveis bit a bit.
            f_scale = lo * (1.0 - 1e-6) if not spatial else lo
            for w in site_wl:
                w["E_it_mwh"] *= f_scale
        # Com ``vdc.reference: ramp_feasible`` a referência é o LP de
        # ``_ramp_feasible_by_job``, que impõe a rampa agregada nos DOIS sentidos;
        # o EDF acima só limita a subida. Se o LP não fechar, reescala-se de novo
        # pela mesma bissecção. Nos workloads do periódico ele fecha sem
        # reescalar, de modo que os jobs (e a energia) são os mesmos.
        if str(v.get("reference", "edf")).lower() == "ramp_feasible":
            rp = float(v.get("ramp_limit_pu", 1.0))
            if _ramp_feasible_by_job(site_wl, s["flex_mw_it"], ramp_pu=rp, check_only=True) is None:
                lo, hi = 0.0, 1.0
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    trial = [dict(w, E_it_mwh=w["E_it_mwh"] * mid) for w in site_wl]
                    if _ramp_feasible_by_job(trial, s["flex_mw_it"], ramp_pu=rp, check_only=True) is not None:
                        lo = mid
                    else:
                        hi = mid
                f_scale *= lo
                for w in site_wl:
                    w["E_it_mwh"] *= lo
        for w in site_wl:
            w["feasibility_scale"] = f_scale
            w["pue"] = float(s["pue"])
        workloads.extend(site_wl)

    return workloads, cost


def add_vdc(n: pypsa.Network, cfg: dict) -> None:
    """Adiciona a camada VDC com componentes nativos do PyPSA (§5.3).

    Por workload *j*: uma barra virtual ``compute_j``; um ``Link`` de cada site
    viável *d* até ``compute_j`` (eficiência 1/PUE_d, ``p_max_pu`` = 1 dentro da
    janela, custo de migração ``c_{j,d}``); e um ``Store`` em ``compute_j`` cujo
    ``e_min_pu`` salta para 1 no deadline, forçando a conclusão. A única
    restrição custom (capacidade por site) fica em ``constraints.py``.

    Com ``vdc.spatial = false`` cada workload recebe apenas o ``Link`` do próprio
    site nativo: a camada mantém a flexibilidade **temporal** (postergar dentro
    da janela, que é o que o ``Store`` representa) e perde a **espacial**
    (transferir carga entre data centers). É o caso "DC local" da escada de
    flexibilidade — um operador que só desloca trabalho no tempo, dentro de cada
    prédio. Como a factibilidade EDF de ``generate_workloads`` já é checada por
    site ignorando migração, restringir ao nativo nunca torna o caso inviável.
    """
    v = cfg["vdc"]
    spatial = bool(v.get("spatial", True))
    sites = _site_table(cfg)
    workloads, cost = generate_workloads(cfg)
    sns = n.snapshots
    hours = np.arange(len(sns))
    # Variante rígida (``vdc.workloads.rigid``): a forma de cada job é fixa e a
    # decisão é só quando começa e onde (constraints.add_vdc_rigid). A rampa por
    # Link e a rampa agregada do site deixam de ser decisão --- são da forma ---,
    # e precisam sair: a própria referência EDF as viola (DC_A2 sobe 40 MW numa
    # hora contra 20 MW de limite), e com elas nem deslocamento nulo seria viável.
    rigid = bool(v["workloads"].get("rigid", False))
    ramp = float("nan") if rigid else float(v["ramp_limit_pu"])

    # Carga inflexível plana por site
    for d, s in sites.iterrows():
        n.add("Load", f"L_{d}_inflex", bus=s["bus"], p_set=float(s["inflex_mw_elec"]),
              carrier="datacenter", area=int(s["area"]))

    if "compute" not in n.carriers.index:
        n.add("Carrier", ["compute", "VDC", "datacenter"])

    for w in workloads:
        cbus = f"compute_{w['id']}"
        n.add("Bus", cbus, carrier="compute")
        window = pd.Series(((hours >= w["arrival"]) & (hours <= w["deadline"])).astype(float), index=sns)
        for d in (w["sites"] if spatial else [w["native"]]):
            s = sites.loc[d]
            n.add(
                "Link",
                f"{w['id']}@{d}",
                bus0=s["bus"],
                bus1=cbus,
                carrier="VDC",
                efficiency=1.0 / float(s["pue"]),
                p_nom=float(s["flex_mw_elec"]),
                p_min_pu=0.0,
                p_max_pu=window,
                marginal_cost=float(cost.loc[w["native"], d]),
                ramp_limit_up=ramp,
                ramp_limit_down=ramp,
                site=d,
                workload=w["id"],
                native_site=w["native"],
                is_native=(d == w["native"]),
                arrival=int(w["arrival"]),
                deadline=int(w["deadline"]),
            )
        # Conclusão no deadline. Na variante rígida ela decorre da forma (uma única
        # opção escolhida, forma com a energia do job) e o Store fica sem piso,
        # porque a forma descarta um resíduo de ~1e-8 do EDF (rigid_jobs).
        e_min = pd.Series(0.0 if rigid else (hours >= w["deadline"]).astype(float), index=sns)
        n.add(
            "Store",
            f"st_{w['id']}",
            bus=cbus,
            carrier="compute",
            e_nom=float(w["E_it_mwh"]),
            e_initial=0.0,
            e_cyclic=False,
            standing_loss=0.0,
            e_min_pu=e_min,
            e_max_pu=1.0,
            workload=w["id"],
        )

    # Cronograma de referência por site (MW elétricos): os mesmos jobs executados
    # no site nativo a partir da chegada, por EDF --- exatamente a carga que os
    # casos sem VDC consomem com ``baseline: native``. O custo de flexibilidade
    # (``vdc.flex_cost``, em constraints.add_flex_cost) incide sobre a energia
    # retirada desse cronograma, seja ela adiada no próprio site ou transferida.
    # Com ``baseline: flat`` (config/base.yaml) a referência é o perfil plano,
    # pela mesma razão: tem de ser a carga que os casos sem VDC consomem.
    flat = str(v.get("baseline", "flat")).lower() != "native"
    reference = {}
    for d, s in sites.iterrows():
        site_wl = [w for w in workloads if w["native"] == d]
        if flat:
            e_elec = float(s["pue"]) * sum(float(w["E_it_mwh"]) for w in site_wl)
            reference[d] = [e_elec / len(sns)] * len(sns)
        else:
            prof_it = _reference_schedule(site_wl, float(s["flex_mw_it"]), cfg, horizon=len(sns))
            reference[d] = [float(s["pue"]) * float(x) for x in prof_it]

    n.meta["vdc"] = {
        "mode": "flexible",
        "spatial": spatial,
        "rigid": rigid_jobs(workloads, sites, horizon=len(sns), cfg=cfg) if rigid else None,
        "flex_cost": float(v.get("flex_cost", 0.0)),
        "reference_elec_mw": reference,
        "flex_fraction": float(v["flex_fraction"]),
        "flex_utilization": float(v["flex_utilization"]),
        "window_scale": float(v["workloads"]["window_scale"]),
        "sites": {d: {"bus": s["bus"], "capacity_mw": float(s["capacity_mw"]), "pue": float(s["pue"]),
                      "flex_mw_elec": float(s["flex_mw_elec"])} for d, s in sites.iterrows()},
        "workloads": workloads,
        "migration_cost": cost.round(3).to_dict(),
    }
