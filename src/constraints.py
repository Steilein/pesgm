"""
constraints.py — restrições adicionais ao UC nativo do PyPSA (``extra_functionality``).

O PyPSA/linopy constrói o MILP do Unit Commitment (balanço nodal, fluxo DC,
status/partida/parada, rampas, tempos mínimos, armazenamentos). Este módulo
acrescenta, via ``extra_functionality(n, snapshots)``, três blocos de
restrições descritos no brief:

1. **Reserva girante** (§6):
       Σ_g (status_{g,t} · Pmax_g − p_{g,t}) ≥ R_t,
   com R_t = a·carga_t + b·VRE_disponível_t. Opcionalmente (``reserve.from_flex``)
   a folga do VDC (carga corrente dos Links, redutível) e do VPL (folga de
   descarga do BESS) também contam como reserva.

2. **Capacidade e rampa por site do VDC** (§5.4):
       Σ_j Link-p[(j,d), t] ≤ f_d · C_d                    ∀ d, t;
       | Σ_j Link-p[(j,d),t] − Σ_j Link-p[(j,d),t−1] | ≤ Δ_d.
   A rampa agregada complementa o ``ramp_limit`` nativo do ``Link``, que só
   limita cada par tarefa-site isoladamente.

3. **Regra de chaveamento explícita do VPL** (§4, variantes ``vpl.mode`` em
   ``explicit`` / ``dedicated`` / ``hybrid``):
   binária z_t de sentido do fluxo no corredor (big-M) e binária w_t que fixa o
   modo (carga/descarga) de cada BESS conforme sigma_t (estágio de uso da rede).

4. **Eletrolisador de três estados** (``h2.mode: three_state``): produz /
   standby / desligado, carga mínima e custo de partida a frio.

Convenções do linopy/PyPSA ≥ 1.0 usadas aqui: as variáveis chamam-se
``"Generator-status"``, ``"Generator-p"``, ``"Link-p"``, ``"Line-s"``,
``"StorageUnit-p_dispatch"``, ``"StorageUnit-p_store"``; a dimensão dos
componentes chama-se ``"name"`` e a temporal ``"snapshot"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from layers import VPL_DOWN, VPL_UP, corridor_lines, vre_available


def _by_name(values: pd.Series) -> xr.DataArray:
    """Série pandas indexada por componente → DataArray alinhado na dimensão ``name``."""
    return xr.DataArray(values.values, coords={"name": values.index.values}, dims="name")


def _by_snapshot(values: pd.Series, sns: pd.Index) -> xr.DataArray:
    """Série pandas indexada por snapshot → DataArray na dimensão ``snapshot``."""
    return xr.DataArray(values.reindex(sns).values, coords={"snapshot": sns.values}, dims="snapshot")


def _mask(flags: pd.Series, sns: pd.Index) -> xr.DataArray:
    """Série booleana por snapshot → máscara linopy (linhas False são descartadas)."""
    return xr.DataArray(flags.reindex(sns).fillna(False).astype(bool).values,
                        coords={"snapshot": sns.values}, dims="snapshot")


# -----------------------------------------------------------------------------
# 1. Reserva girante
# -----------------------------------------------------------------------------
def add_spinning_reserve(n: pypsa.Network, sns: pd.Index, cfg: dict) -> None:
    r = cfg["reserve"]
    m = n.model
    g = n.generators
    com = g.index[g["committable"].astype(bool)]

    status = m["Generator-status"].sel(name=com)
    p = m["Generator-p"].sel(name=com)
    p_nom = _by_name(g.loc[com, "p_nom"])

    # Lado esquerdo: folga total das unidades térmicas ligadas
    lhs = (status * p_nom - p).sum("name")

    # Requisito R_t
    load_t = n.get_switchable_as_dense("Load", "p_set", sns).sum(axis=1)
    vre_t = vre_available(n).reindex(sns).sum(axis=1)

    # Nos casos A e B a carga flexivel do data center entra como Load plana e,
    # portanto, conta em L_t; nos casos C e D ela passa por Links e nao conta.
    # Com ``reserve.common_reference`` o requisito usa a mesma carga de
    # referencia nos quatro casos, somando de volta o equivalente plano da
    # energia flexivel. Isola o valor da flexibilidade do efeito de a carga
    # flexivel nao gerar requisito de reserva.
    if r.get("common_reference", False):
        meta = n.meta.get("vdc", {})
        if meta.get("mode") == "flexible":
            e_elec = sum(float(w["E_it_mwh"]) * float(meta["sites"][w["native"]]["pue"])
                         for w in meta["workloads"])
            load_t = load_t + e_elec / len(sns)

    req = float(r["load_fraction"]) * load_t + float(r["vre_fraction"]) * vre_t
    rhs = _by_snapshot(req, sns)

    # Provedores flexíveis (sensibilidade): VDC e VPL
    if r.get("from_flex", False):
        vdc_links = n.links.index[n.links["carrier"] == "VDC"]
        if len(vdc_links):
            # Carga corrente do VDC pode ser reduzida → headroom de reserva
            lhs = lhs + m["Link-p"].sel(name=vdc_links).sum("name")
        vpl = [u for u in (VPL_UP, VPL_DOWN) if u in n.storage_units.index]
        if vpl:
            p_dis = m["StorageUnit-p_dispatch"].sel(name=vpl)
            p_sto = m["StorageUnit-p_store"].sel(name=vpl)
            p_nom_vpl = _by_name(n.storage_units.loc[vpl, "p_nom"])
            # Folga de descarga + carga corrente interrompível (ignora limite de SOC)
            lhs = lhs + (p_nom_vpl - p_dis + p_sto).sum("name")

    m.add_constraints(lhs >= rhs, name="spinning_reserve")


# -----------------------------------------------------------------------------
# 2. Capacidade por site do VDC
# -----------------------------------------------------------------------------
def add_site_capacity(n: pypsa.Network, sns: pd.Index, cfg: dict) -> None:
    links = n.links[n.links["carrier"] == "VDC"]
    if links.empty:
        return
    m = n.model
    sites = n.meta["vdc"]["sites"]
    for site, info in sites.items():
        idx = links.index[links["site"] == site]
        if idx.empty:
            continue
        lhs = m["Link-p"].sel(name=idx).sum("name")
        m.add_constraints(lhs <= float(info["flex_mw_elec"]), name=f"site_capacity-{site}")


def add_site_ramp(n: pypsa.Network, sns: pd.Index, cfg: dict) -> None:
    """Limite de rampa da carga de TI **agregada** por site (§5.4):

        | Σ_j Link-p[(j,d),t] − Σ_j Link-p[(j,d),t−1] | ≤ Δ_d,
        Δ_d = ramp_limit_pu · f_d · C_d   (MW elétricos por hora).

    O ``ramp_limit`` nativo do ``Link`` limita cada par tarefa-site
    isoladamente; sem esta restrição a soma no site pode variar até a
    capacidade flexível inteira em uma hora.
    """
    links = n.links[n.links["carrier"] == "VDC"]
    if links.empty or len(sns) < 2 or n.meta.get("vdc", {}).get("rigid"):
        return          # na variante rígida a rampa é da forma do job (layers.add_vdc)
    m = n.model
    rl = float(cfg["vdc"]["ramp_limit_pu"])
    p = m["Link-p"]
    cur = sns[1:]
    prev = sns[:-1]
    for site, info in n.meta["vdc"]["sites"].items():
        idx = links.index[links["site"] == site]
        if idx.empty:
            continue
        delta = rl * float(info["flex_mw_elec"])
        agg = p.sel(name=idx).sum("name")
        diff = agg.sel(snapshot=cur) - agg.sel(snapshot=prev).rename(snapshot="snapshot").assign_coords(snapshot=cur)
        m.add_constraints(diff <= delta, name=f"site_ramp_up-{site}")
        m.add_constraints(diff >= -delta, name=f"site_ramp_down-{site}")


def add_flex_cost(n: pypsa.Network, sns: pd.Index, cfg: dict) -> None:
    """Custo de exercer a flexibilidade do VDC (``vdc.flex_cost``, $/MWh de site):

        s_{d,t} ≥ P^ref_{d,t} − Σ_j Link-p[(j,d),t],   s_{d,t} ≥ 0,
        objetivo += c^flex · Σ_{d,t} s_{d,t}.

    ``P^ref`` é o cronograma EDF no site nativo (``n.meta["vdc"]["reference_elec_mw"]``),
    a carga dos casos sem VDC. ``s`` mede a energia retirada da sua (site, hora)
    de referência: um MWh adiado no próprio site e um MWh transferido a outro
    pagam o mesmo, uma única vez. Sem a referência o termo seria inócuo, porque
    Σ_t x_{j,d,t} = E_j é fixo e um custo por MWh executado vira constante.
    """
    meta = n.meta.get("vdc", {})
    c = float(meta.get("flex_cost", 0.0))
    links = n.links[n.links["carrier"] == "VDC"]
    if c <= 0.0 or links.empty:
        return
    m = n.model
    p = m["Link-p"]
    total = None
    for site, ref in meta["reference_elec_mw"].items():
        idx = links.index[links["site"] == site]
        ref_da = _by_snapshot(pd.Series(ref[:len(sns)], index=sns), sns)
        s = m.add_variables(lower=0.0, coords=[sns], name=f"VDC-shift-{site}")
        agg = p.sel(name=idx).sum("name") if not idx.empty else 0.0
        m.add_constraints(s + agg >= ref_da, name=f"VDC-shift_def-{site}")
        term = (c * s).sum()
        total = term if total is None else total + term
    m.add_objective(m.objective.expression + total, overwrite=True)


def add_vpl_soc_initial(n: pypsa.Network, sns: pd.Index, cfg: dict) -> None:
    """Fixa o estado de carga inicial dos BESS do VPL (``vpl.soc_initial``).

    Com ``cyclic_state_of_charge`` o PyPSA só impõe e_final = e_inicial, deixando
    o nível livre: o otimizador escolhe o SOC de partida mais conveniente de cada
    dia. Com esta opção o ciclo continua fechado, mas ancorado numa fração fixa
    de ``e_nom`` (tipicamente 0,5), o que é o contrafactual de um operador que
    começa o dia num estado pré-definido.
    """
    frac = cfg["vpl"].get("soc_initial")
    if frac is None:
        return
    units = [u for u in (VPL_UP, VPL_DOWN) if u in n.storage_units.index]
    if not units:
        return
    m = n.model
    soc = m["StorageUnit-state_of_charge"]
    for u in units:
        e_nom = float(n.storage_units.at[u, "p_nom"] * n.storage_units.at[u, "max_hours"])
        m.add_constraints(soc.sel(name=u, snapshot=sns[-1]) == float(frac) * e_nom,
                          name=f"VPL-soc_initial-{u}")


# -----------------------------------------------------------------------------
# 3. Variante explícita do VPL (regra de chaveamento da tese)
# -----------------------------------------------------------------------------
def add_explicit_vpl(n: pypsa.Network, sns: pd.Index, cfg: dict) -> None:
    """Impõe o modo de operação dos BESS pelo sentido do fluxo e por sigma_t.

    * z_t = 1 ⇔ fluxo montante → jusante (f_t ≤ M·z_t ; f_t ≥ −M·(1 − z_t)).
    * w_t = 1 ⇔ BESS a jusante em modo descarga (e o de montante em modo carga);
      w_t = 0 ⇔ inverso. Linearização: p_dis ≤ P·w, p_ch ≤ P·(1 − w).
    * Alto uso (sigma_t = 1): w_t = z_t  → com fluxo a jusante, montante carrega
      e jusante descarrega (alívio do corredor).
    * Baixo uso (sigma_t = 0): w_t = 1 − z_t → inverso (recarga do terminal a
      jusante usando a folga do corredor).
    * Horas intermediárias: em ``explicit`` w_t fica livre (arbitragem /
      *revenue stacking*, como na versão inter-área do cap. 5 da tese); em
      ``dedicated`` vale w_t = z_t, ou seja, o par emula transferência no
      sentido do fluxo em toda hora que não seja de baixo uso. O modo
      ``dedicated`` é a leitura estrita da seção 4.2.3 da tese, em que o ESS
      alocado a uma VTL opera permanentemente em carga ou descarga e não
      presta nenhum outro serviço.
    * ``hybrid``: a regra de chaveamento **só vincula nas horas de alto uso**
      (sigma_t = 1). Fora do pico nenhuma das restrições acima é imposta — nem
      w_t = 1 − z_t, nem sequer o acoplamento p_dis ≤ P·w, que em ``explicit``
      ainda obriga os dois BESS a operarem em modos opostos mesmo nas horas
      livres. Em ``hybrid`` os dois terminais podem carregar (ou descarregar)
      simultaneamente, cada um respondendo ao próprio LMP. É o contrafactual do
      operador que contrata o serviço de VTL apenas no pico e libera os ativos
      para o mercado no resto do dia — cota superior do *revenue stacking*
      compatível com a garantia de alívio do corredor.

    A vinculação é implementada por ``mask`` do linopy: ``bind_t`` é verdadeiro
    em toda hora nos modos ``explicit``/``dedicated`` (comportamento idêntico ao
    anterior) e apenas em sigma_t = 1 no modo ``hybrid``.
    """
    if VPL_UP not in n.storage_units.index:
        return
    v = cfg["vpl"]
    m = n.model
    big_m = float(v["big_m"])
    P = float(n.storage_units.at[VPL_UP, "p_nom"])

    # Fluxo agregado no corredor como expressão linear (orientação montante→jusante)
    lines = corridor_lines(n, cfg)
    flow = None
    for name, sign in lines["sign"].items():
        term = sign * m["Line-s"].sel(name=name)
        flow = term if flow is None else flow + term

    # linopy espera um pd.Index nomeado para criar a dimensão "snapshot"
    snap_idx = pd.Index(sns, name="snapshot")
    z = m.add_variables(binary=True, coords=[snap_idx], name="VPL-z")
    w = m.add_variables(binary=True, coords=[snap_idx], name="VPL-w")

    sigma = pd.Series(n.meta["sigma"], index=n.snapshots, dtype="float").reindex(sns)
    mode = str(v.get("mode", "emergent")).lower()

    # Horas em que a regra vincula. Em 'hybrid' só o pico; nas demais horas o
    # par fica sem nenhum acoplamento (equivale localmente ao modo emergente).
    bind = (sigma == 1.0) if mode == "hybrid" else pd.Series(True, index=sns)
    mask_bind = _mask(bind, sns)
    if not bool(mask_bind.any()):
        return

    m.add_constraints(flow - big_m * z <= 0, name="VPL-flow_direction-upper", mask=mask_bind)
    m.add_constraints(flow + big_m * (1 - z) >= 0, name="VPL-flow_direction-lower", mask=mask_bind)

    dis_up = m["StorageUnit-p_dispatch"].sel(name=VPL_UP)
    sto_up = m["StorageUnit-p_store"].sel(name=VPL_UP)
    dis_dn = m["StorageUnit-p_dispatch"].sel(name=VPL_DOWN)
    sto_dn = m["StorageUnit-p_store"].sel(name=VPL_DOWN)

    m.add_constraints(dis_dn - P * w <= 0, name="VPL-down-dispatch_mode", mask=mask_bind)
    m.add_constraints(sto_dn + P * w <= P, name="VPL-down-store_mode", mask=mask_bind)
    m.add_constraints(dis_up + P * w <= P, name="VPL-up-dispatch_mode", mask=mask_bind)
    m.add_constraints(sto_up - P * w <= 0, name="VPL-up-store_mode", mask=mask_bind)

    # em 'dedicated' nao existe hora livre: tudo que nao e baixo uso segue w_t = z_t
    hi = (sigma != 0.0) if mode == "dedicated" else (sigma == 1.0)
    # em 'hybrid' o estagio de baixo uso tambem fica livre (sem a regra invertida)
    lo = pd.Series(False, index=sns) if mode == "hybrid" else (sigma == 0.0)
    mask_hi = _mask(hi, sns)
    mask_lo = _mask(lo, sns)
    if bool(mask_hi.any()):
        m.add_constraints(w - z == 0, name="VPL-sigma_high", mask=mask_hi)
    if bool(mask_lo.any()):
        m.add_constraints(w + z == 1, name="VPL-sigma_low", mask=mask_lo)


def add_vdc_rigid(n: pypsa.Network, sns: pd.Index, cfg: dict) -> None:
    """Workloads rígidos (``vdc.workloads.rigid``; forma em ``layers.rigid_jobs``).

    Para cada job j, uma binária por opção (site d elegível, hora de início s):

        sum_{d,s} y_{j,d,s} = 1
        Link-p[j@d, t] = PUE_d sum_s y_{j,d,s} r_j(t - s)     para todo d, t

    em que r_j é a forma do job no cronograma de referência (MW de TI). O job
    roda inteiro num único site, com a forma de referência, começando em s.
    """
    meta = n.meta.get("vdc", {})
    jobs = meta.get("rigid")
    if not jobs:
        return
    links = n.links[n.links["carrier"] == "VDC"]
    m = n.model
    p = m["Link-p"]
    T = len(sns)
    for j, info in jobs.items():
        lk = links[links["workload"] == j]
        shape = np.asarray(info["shape"])
        opts = [(name, int(s)) for name in lk.index for s in info["starts"]]
        dim = f"opt_{j}"
        idx = pd.Index([f"{name}|{s}" for name, s in opts], name=dim)
        y = m.add_variables(binary=True, coords=[idx], name=f"VDC-rigid-{j}")
        m.add_constraints(y.sum() == 1, name=f"VDC-rigid_one-{j}")
        for name in lk.index:
            pue = 1.0 / float(lk.at[name, "efficiency"])
            coef = np.zeros((T, len(opts)))
            for o, (lname, s) in enumerate(opts):
                if lname == name:
                    coef[s:s + len(shape), o] = pue * shape
            C = xr.DataArray(coef, coords={"snapshot": sns.values, dim: idx.values},
                             dims=("snapshot", dim))
            m.add_constraints(p.sel(name=name) - (C * y).sum(dim) == 0, name=f"VDC-rigid_link-{name}")


# -----------------------------------------------------------------------------
# 4. Eletrolisador de três estados (artigo de periódico, Sec. III-D)
# -----------------------------------------------------------------------------
H2_STATE_VARS = ("H2-z_on", "H2-z_sb", "H2-z_su")


def add_h2_states(n: pypsa.Network, sns: pd.Index, cfg: dict) -> None:
    """Produz / standby / desligado, com duas binárias (Saretta et al., Eqs. 3b-3d)::

        z^on_t + z^sb_t <= 1
        P^min z^on_t <= p_t <= P^max z^on_t          (p: Link H2_P2H, produz H2)
        p^sb_t = P^sb z^sb_t                          (Link H2_P2H_sb, eficiência 0)
        z^su_t >= (z^on_t + z^sb_t) - (z^on_{t-1} + z^sb_{t-1})
        objetivo += K^su sum_t z^su_t

    O consumo total do eletrolisador é ``p_t + p^sb_t``, a forma da Eq. (3c) de
    Saretta et al. A partida a frio é a transição de desligado para produzir ou
    para standby; standby -> produzir não paga. A condição de contorno é
    cíclica (a hora anterior à primeira é a última), coerente com o estoque de
    hidrogênio e o SOC do VPL, fechados no dia.
    """
    meta = n.meta.get("h2", {})
    if meta.get("mode") != "three_state" or "H2_P2H" not in n.links.index:
        return
    m = n.model
    snap_idx = pd.Index(sns, name="snapshot")
    z_on = m.add_variables(binary=True, coords=[snap_idx], name="H2-z_on")
    z_sb = m.add_variables(binary=True, coords=[snap_idx], name="H2-z_sb")
    z_su = m.add_variables(binary=True, coords=[snap_idx], name="H2-z_su")
    p = m["Link-p"].sel(name="H2_P2H")
    p_sb = m["Link-p"].sel(name="H2_P2H_sb")

    m.add_constraints(z_on + z_sb <= 1, name="H2-state")
    m.add_constraints(p - float(meta["p_min_mw"]) * z_on >= 0, name="H2-p_min")
    m.add_constraints(p - float(meta["p_nom_mw"]) * z_on <= 0, name="H2-p_max")
    m.add_constraints(p_sb - float(meta["p_sb_mw"]) * z_sb == 0, name="H2-standby")

    on = z_on + z_sb
    prev = on.roll(snapshot=1)          # hora anterior, cíclica no dia
    m.add_constraints(z_su - on + prev >= 0, name="H2-startup")
    k = float(meta["startup_cost"])
    if k > 0.0:
        m.add_objective(m.objective.expression + k * z_su.sum(), overwrite=True)


def h2_state_solution(n: pypsa.Network) -> dict | None:
    """Estado do eletrolisador na solução MILP, para gravar em ``n.meta`` antes do
    LP de preços (as variáveis próprias não vão para o NetCDF por conta própria)."""
    m = n.model
    if "H2-z_on" not in m.variables:
        return None
    return {v.split("-", 1)[1]: [int(round(x)) for x in m.variables[v].solution.values]
            for v in H2_STATE_VARS}


# -----------------------------------------------------------------------------
# Fábrica da extra_functionality
# -----------------------------------------------------------------------------
def make_extra_functionality(cfg: dict):
    """Devolve a função ``extra_functionality(n, snapshots)`` para ``n.optimize``."""

    def extra_functionality(n: pypsa.Network, sns: pd.Index) -> None:
        if cfg["reserve"].get("enabled", True):
            add_spinning_reserve(n, sns, cfg)
        add_site_capacity(n, sns, cfg)
        add_site_ramp(n, sns, cfg)
        add_vdc_rigid(n, sns, cfg)
        add_flex_cost(n, sns, cfg)
        add_vpl_soc_initial(n, sns, cfg)
        add_h2_states(n, sns, cfg)
        if str(cfg["vpl"]["mode"]).lower() in ("explicit", "dedicated", "hybrid"):
            add_explicit_vpl(n, sns, cfg)

    return extra_functionality
