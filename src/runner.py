"""
runner.py — Fase 5: construção, resolução e exportação dos cenários.

Fluxo de um caso (``run_case``):

1. ``rts_import.build_network``  → rede RTS-GMLC original do dia;
2. ``layers.stress_system``      → configuração estressada (tese / Energies 2026);
3. ``layers.scale_vre``          → penetração VRE alvo;
4. camadas: ``add_vpl`` e/ou ``add_vdc`` (ou ``add_dc_inflexible`` nos casos sem VDC);
5. ``n.optimize`` (MILP) com as restrições de ``constraints.py``;
6. re-solução LP com as binárias fixadas → preços nodais (LMP) e multiplicadores;
7. ``n.export_to_netcdf`` em ``results/<cenario>.nc`` e linha em ``results/metrics.csv``.

Casos: A (base), B (+VPL), C (+VDC), D (VPL+VDC). A grade de sensibilidades é
lida de ``config/scenarios.yaml`` (``run_grid``). O solver é parametrizado
(HiGHS para depuração, gap 0,5 %; Gurobi para a grade final, gap 0,05 %).

Uso pela linha de comando::

    python src/runner.py --day summer_peak --case D            # um caso
    python src/runner.py --grid --solver highs --quick         # grade reduzida
    python src/runner.py --grid --solver gurobi                # grade completa
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pypsa

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_config, load_profile, results_dir, set_by_path, solver_options  # noqa: E402
from constraints import h2_state_solution, make_extra_functionality  # noqa: E402
from layers import (add_dc_inflexible, add_h2, add_vdc, add_vpl, limit_corridor, radialize_export,  # noqa: E402
                    reinforce_collection, scale_vre, shape_corridor, stress_system)
from metrics import append_metrics_row, compute_metrics  # noqa: E402
from rts_import import RTSData, build_network, load_rts_data  # noqa: E402

_DATA_CACHE: dict[str, RTSData] = {}


def get_data(cfg: dict) -> RTSData:
    """Carrega os CSVs do RTS uma única vez por processo."""
    key = str(cfg["paths"]["rts"])
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_rts_data(ROOT / key)
    return _DATA_CACHE[key]


# -----------------------------------------------------------------------------
# Construção
# -----------------------------------------------------------------------------
def build_stressed_network(cfg: dict, day: str) -> pypsa.Network:
    """Rede estressada e escalonada para a penetração VRE alvo (Fases 1 e 2)."""
    n = build_network(get_data(cfg), day, cfg)
    stress_system(n, cfg)
    scale_vre(n, float(cfg["vre"]["target_share"]), basis=cfg["vre"].get("scaling_basis", "annual"),
              cfg=cfg["vre"])
    # Realismo estrutural (§ perfil realista): radializa o escoamento da área
    # exportadora e redimensiona o corredor. Ambos são no-op quando as chaves
    # correspondentes não estão no YAML, de modo que a configuração original do
    # brief continua reproduzindo os resultados anteriores bit a bit.
    radialize_export(n, cfg)
    reinforce_collection(n, cfg)
    shape_corridor(n, cfg)
    factor = cfg["corridor"].get("limit_corridor_factor")
    if factor:
        limit_corridor(n, cfg, float(factor))
    return n


# Ordem canônica dos recursos alternáveis. O rótulo do caso é a máscara de bits
# nesta ordem, somada a "A": com VPL=1, VDC=2, H2=4 os quatro primeiros
# subconjuntos reproduzem EXATAMENTE a rotulagem A/B/C/D da grade do artigo de
# conferência, de que dependem 581 arquivos .nc, metrics.csv e os módulos
# ablation.py, economics.py, cross_sensitivity.py e interaction_stats.py.
# Acrescentar um recurso ao final da tupla é retrocompatível; reordenar não é.
RESOURCES: tuple[str, ...] = ("VPL", "VDC", "H2")


def case_label(layers: set[str]) -> str:
    """Rótulo do caso a partir do conjunto de camadas ativas.

    >>> case_label(set()), case_label({"VPL"}), case_label({"VDC"})
    ('A', 'B', 'C')
    >>> case_label({"VPL", "VDC"}), case_label({"H2"}), case_label({"VPL", "VDC", "H2"})
    ('D', 'E', 'H')
    """
    layers = {s.upper() for s in layers}
    desconhecidas = layers - set(RESOURCES)
    if desconhecidas:
        raise ValueError(f"camada desconhecida: {sorted(desconhecidas)}; "
                         f"conhecidas: {list(RESOURCES)}")
    mask = sum(1 << i for i, r in enumerate(RESOURCES) if r in layers)
    return chr(ord("A") + mask)


def layers_from_case(case: str) -> set[str]:
    """Inverso de ``case_label``."""
    mask = ord(case.upper()) - ord("A")
    if not 0 <= mask < (1 << len(RESOURCES)):
        raise ValueError(f"caso inválido: {case!r} (esperado A..{chr(ord('A') + (1 << len(RESOURCES)) - 1)})")
    return {r for i, r in enumerate(RESOURCES) if mask >> i & 1}


def apply_layers(n: pypsa.Network, cfg: dict, layers: set[str]) -> None:
    """Aplica as camadas de flexibilidade; DCs viram carga plana quando não há VDC."""
    layers = {s.upper() for s in layers}
    case_label(layers)                      # valida os nomes antes de construir
    if "VPL" in layers:
        add_vpl(n, cfg)
    if "VDC" in layers:
        add_vdc(n, cfg)
    else:
        add_dc_inflexible(n, cfg)
    if "H2" in layers:
        add_h2(n, cfg)
    n.meta["layers"] = sorted(layers)


# -----------------------------------------------------------------------------
# Resolução: MILP + LP para preços
# -----------------------------------------------------------------------------
def _mip_gap(n: pypsa.Network) -> float | None:
    """Gap relativo reportado pelo solver após o MILP (HiGHS ou Gurobi)."""
    sm = n.model.solver_model
    try:
        return float(sm.getInfo().mip_gap)          # HiGHS
    except Exception:
        pass
    try:
        return float(sm.MIPGap)                     # Gurobi
    except Exception:
        return None


def _objective_bound(n: pypsa.Network) -> float | None:
    """Melhor cota dual (``best bound``) do MILP, tal como o solver a reporta.

    O intervalo de confiança da interação I é construído a partir dos ótimos de
    quatro MILP truncados (ver ``interaction_stats.py``). Derivar a cota do gap
    relativo obriga a assumir a convenção do solver para esse gap; gravar a cota
    diretamente elimina a suposição.
    """
    sm = n.model.solver_model
    try:
        return float(sm.getInfo().mip_dual_bound)   # HiGHS
    except Exception:
        pass
    try:
        return float(sm.ObjBound)                   # Gurobi
    except Exception:
        return None


def _fix_binaries(n: pypsa.Network) -> None:
    """Fixa todas as variáveis binárias na solução MILP e as torna contínuas.

    Com isso o modelo vira um LP cuja solução dual fornece os preços nodais
    (LMPs) e os multiplicadores dos limites de fluxo — procedimento padrão em
    mercados day-ahead (formação de preço com o commitment fixo).
    """
    m = n.model
    for name in list(m.binaries):
        var = m.variables[name]
        sol = var.solution.round()
        var.lower = sol
        var.upper = sol
        var.attrs["binary"] = False


def _export_with_retry(n: pypsa.Network, out: Path, tries: int = 3, wait_s: float = 3.0) -> None:
    """Exporta o NetCDF, tolerando um bloqueio transitório de arquivo no Windows.

    O exportador do PyPSA reabre o arquivo recém-criado em modo ``a``; se o
    antivírus (ou qualquer outro leitor) ainda estiver com ele aberto nesse
    instante, o Windows recusa com ``PermissionError`` e o cenário se perde no
    meio de uma grade de horas. Aconteceu uma vez em 118 exportações. Repetir
    depois de alguns segundos resolve; um arquivo truncado de zero byte é
    removido antes da nova tentativa, senão ele passaria a existir e seria
    tomado por um cenário já resolvido.
    """
    for k in range(tries):
        try:
            n.export_to_netcdf(out)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            if out.exists() and out.stat().st_size == 0:
                out.unlink(missing_ok=True)
            print(f"  exportação bloqueada ({out.name}); nova tentativa em {wait_s:.0f}s",
                  flush=True)
            time.sleep(wait_s)


def solve(n: pypsa.Network, cfg: dict) -> dict:
    """Resolve o UC (MILP) e a re-solução LP; devolve um dicionário de log."""
    name, mip_opts = solver_options(cfg, mip=True)
    _, lp_opts = solver_options(cfg, mip=False)
    log = bool(cfg["solver"].get("log_to_console", False))
    ef = make_extra_functionality(cfg)

    t0 = time.perf_counter()
    status, cond = n.optimize(solver_name=name, solver_options=mip_opts, extra_functionality=ef,
                              log_to_console=log)
    t_milp = time.perf_counter() - t0
    if status != "ok":
        raise RuntimeError(f"MILP não convergiu: status={status}, condição={cond}")
    info = {
        "solver": name,
        "status_milp": cond,
        "objective_milp": float(n.objective),
        "objective_bound": _objective_bound(n),
        "mip_gap": _mip_gap(n),
        "time_milp_s": round(t_milp, 1),
    }

    h2_state = h2_state_solution(n)
    if h2_state is not None:
        n.meta["h2"] = {**n.meta.get("h2", {}), "state": h2_state}

    t0 = time.perf_counter()
    _fix_binaries(n)
    status_lp, cond_lp = n.optimize.solve_model(solver_name=name, solver_options=lp_opts, log_to_console=log)
    info.update({"status_lp": cond_lp, "objective_lp": float(n.objective),
                 "time_lp_s": round(time.perf_counter() - t0, 1)})
    if status_lp != "ok":
        raise RuntimeError(f"LP de preços não convergiu: {cond_lp}")

    # O solver model não é serializável nem copiável; libera memória
    n.model.solver_model = None
    n.meta["solve"] = info
    return info


# -----------------------------------------------------------------------------
# Um caso
# -----------------------------------------------------------------------------
def run_case(
    day: str,
    vre_share: float,
    layers: set[str],
    vpl_mwh: float,
    dc_flex: float,
    solver: str,
    reserve_from_flex: bool,
    cfg: dict | None = None,
    overrides: dict | None = None,
    name: str | None = None,
    tags: dict | None = None,
    post_build=None,
) -> Path:
    """Constrói, resolve e exporta um cenário; devolve o caminho do NetCDF.

    ``post_build(n, cfg)``, se dado, é chamado depois de montar a rede e aplicar
    as camadas, antes de resolver: usado para injetar condições iniciais das
    térmicas herdadas do dia anterior (ver ``extra_runs.py``).

    Parameters
    ----------
    day : str
        Data ``YYYY-MM-DD`` ou chave de ``cfg["days"]`` (ex.: ``"summer_peak"``).
    vre_share : float
        Penetração VRE alvo (0,30 / 0,50 / 0,70).
    layers : set[str]
        Subconjunto de ``{"VPL", "VDC"}``.
    vpl_mwh : float
        Energia por terminal do VPL (MWh); potência = vpl_mwh / horas.
    dc_flex : float
        Fração flexível f_d dos data centers.
    solver : str
        ``"highs"`` ou ``"gurobi"``.
    reserve_from_flex : bool
        Se VDC/VPL contam como provedores de reserva.
    overrides : dict, opcional
        Sobrescritas adicionais ``{"caminho.pontilhado": valor}`` (eixos da grade).
    """
    cfg = copy.deepcopy(cfg or load_config())
    day_key, day_date = _resolve_day(cfg, day)
    cfg = set_by_path(cfg, "vre.target_share", float(vre_share))
    cfg = set_by_path(cfg, "vpl.p_nom_mw", float(vpl_mwh) / float(cfg["vpl"]["hours"]))
    cfg = set_by_path(cfg, "vdc.flex_fraction", float(dc_flex))
    cfg = set_by_path(cfg, "solver.name", solver)
    cfg = set_by_path(cfg, "reserve.from_flex", bool(reserve_from_flex))
    for k, v in (overrides or {}).items():
        cfg = set_by_path(cfg, k, v)

    layers = {s.upper() for s in layers}
    case = case_label(layers)
    name = name or f"{day_key}__{case}__base"

    n = build_stressed_network(cfg, day_date)
    apply_layers(n, cfg, layers)
    if post_build is not None:      # gancho para condições iniciais herdadas (extra_runs.py)
        post_build(n, cfg)
    n.meta["scenario"] = {"name": name, "day_key": day_key, "day": day_date, "case": case,
                          "vre_share": float(vre_share), "vpl_mwh": float(vpl_mwh),
                          "dc_flex": float(dc_flex), "reserve_from_flex": bool(reserve_from_flex),
                          "vpl_mode": cfg["vpl"]["mode"],
                          "window_scale": float(cfg["vdc"]["workloads"]["window_scale"]),
                          **(tags or {})}
    n.meta["config"] = json.loads(json.dumps(cfg, default=str))

    info = solve(n, cfg)
    print(f"[{name}] obj={info['objective_milp']:,.0f} $  gap={info['mip_gap']:.2e}  "
          f"t={info['time_milp_s']}s(+{info['time_lp_s']}s LP)", flush=True)

    out = results_dir(cfg) / f"{name}.nc"
    _export_with_retry(n, out)
    append_metrics_row(compute_metrics(n, cfg), results_dir(cfg) / "metrics.csv")
    return out


def _resolve_day(cfg: dict, day: str) -> tuple[str, str]:
    """Aceita chave (``summer_peak``) ou data (``2020-07-18``)."""
    if day in cfg["days"]:
        return day, cfg["days"][day]
    for k, v in cfg["days"].items():
        if v == day:
            return k, v
    return day.replace("-", ""), day


# -----------------------------------------------------------------------------
# Grade de sensibilidades
# -----------------------------------------------------------------------------
def run_grid(cfg: dict, scen: dict, solver: str, quick: bool = False, days: list[str] | None = None,
             skip_existing: bool = True, only_axes: list[str] | None = None) -> list[Path]:
    """Executa dias × casos × variações (um eixo por vez) conforme ``scenarios.yaml``.

    Variações de um eixo que não se aplicam a um caso (ex.: tamanho do VPL no
    caso A) não são re-resolvidas: a tabela de valor marginal usa a linha base.

    ``only_axes`` restringe os eixos de sensibilidade; ``[]`` roda **só a
    variação base**, que é a grade principal do artigo e a parte barata. Os
    eixos custam muito mais que ela --- com 8 casos e 16 dias, a base são 128
    instâncias e cada eixo acrescenta outro tanto ---, então vale entregar a
    tabela central antes de decidir quais varreduras pagam a pena.
    """
    day_keys = days or (scen["quick"]["days"] if quick else list(cfg["days"]))
    axes = {k: v for k, v in scen["axes"].items() if (not quick or k in scen["quick"]["axes"])}
    if only_axes is not None:
        faltantes = set(only_axes) - set(scen["axes"])
        if faltantes:
            raise ValueError(f"eixo inexistente: {sorted(faltantes)}; "
                             f"disponíveis: {sorted(scen['axes'])}")
        axes = {k: v for k, v in axes.items() if k in only_axes}
    cases = scen["cases"]
    base = {
        "vre_share": float(cfg["vre"]["target_share"]),
        "vpl_mwh": float(cfg["vpl"]["p_nom_mw"]) * float(cfg["vpl"]["hours"]),
        "dc_flex": float(cfg["vdc"]["flex_fraction"]),
        "reserve_from_flex": bool(cfg["reserve"]["from_flex"]),
    }
    hours = float(cfg["vpl"]["hours"])

    # Lista de (rótulo da variação, overrides)
    variations: list[tuple[str, str, dict]] = [("base", "base", {})]
    for axis, spec in axes.items():
        for label, value in spec["values"].items():
            if label == spec["base"]:
                continue
            variations.append((axis, label, {spec["param"]: value}))

    outputs = []
    for day_key in day_keys:
        for axis, label, over in variations:
            for case, layers in cases.items():
                if axis != "base" and case not in axes[axis].get("applies_to", list(cases)):
                    continue
                name = f"{day_key}__{case}__{label}"
                out = results_dir(cfg) / f"{name}.nc"
                if skip_existing and out.exists():
                    print(f"[{name}] já existe — pulando", flush=True)
                    outputs.append(out)
                    continue
                kw = dict(base)
                # Cópia por caso: a variação original não pode ser consumida (pop)
                # pelo primeiro caso, senão os casos seguintes rodariam no valor base.
                over_case = dict(over)
                # Eixos que coincidem com argumentos explícitos de run_case
                if "vre.target_share" in over_case:
                    kw["vre_share"] = over_case.pop("vre.target_share")
                if "vpl.p_nom_mw" in over_case:
                    kw["vpl_mwh"] = float(over_case.pop("vpl.p_nom_mw")) * hours
                if "vdc.flex_fraction" in over_case:
                    kw["dc_flex"] = over_case.pop("vdc.flex_fraction")
                if "reserve.from_flex" in over_case:
                    kw["reserve_from_flex"] = over_case.pop("reserve.from_flex")
                try:
                    outputs.append(run_case(day_key, layers=set(layers), solver=solver, cfg=cfg, overrides=over_case,
                                            name=name, tags={"axis": axis, "variation": label}, **kw))
                except Exception as exc:  # registra e segue para o próximo cenário
                    print(f"[{name}] FALHOU: {exc}", flush=True)
    return outputs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="UC day-ahead com VPL e VDC — runner de cenários")
    p.add_argument("--config", default=str(ROOT / "config" / "base.yaml"))
    p.add_argument("--scenarios", default=str(ROOT / "config" / "scenarios.yaml"))
    p.add_argument("--solver", default=None, help="highs | gurobi (padrão: config)")
    p.add_argument("--grid", action="store_true", help="executa a grade de sensibilidades")
    p.add_argument("--quick", action="store_true", help="grade reduzida (depuração)")
    p.add_argument("--days", nargs="*", default=None, help="chaves de dias a executar")
    p.add_argument("--day", default=None, help="chave ou data de um único dia")
    p.add_argument("--case", default="A",
                   choices=[chr(ord("A") + m) for m in range(1 << len(RESOURCES))],
                   help="subconjunto de recursos como mascara de bits sobre "
                        f"{list(RESOURCES)}: A={{}}, B={{VPL}}, C={{VDC}}, D={{VPL,VDC}}, "
                        "E={H2}, F={VPL,H2}, G={VDC,H2}, H={VPL,VDC,H2}")
    p.add_argument("--vre", type=float, default=None)
    p.add_argument("--force", action="store_true", help="re-resolve mesmo se o .nc existir")
    p.add_argument("--axes", nargs="*", default=None,
                   help="restringe os eixos de sensibilidade da grade; "
                        "sem valor algum (--axes) roda só a variação base")
    p.add_argument("--profile", default=None,
                   help="perfil sobreposto a base.yaml (ex.: config/realistic.yaml)")
    p.add_argument("--results", default=None,
                   help="diretório de saída; use um sandbox para não sobrescrever a grade publicada")
    args = p.parse_args(argv)

    cfg = (load_profile(args.profile, base=args.config) if args.profile
           else load_config(args.config))
    if args.results:
        # Resolver "na mão" grava por padrão em results/, com o nome canônico do
        # cenário: sem isto uma tentativa avulsa sobrescreve a rede publicada e
        # acrescenta uma linha a results/metrics.csv.
        cfg["paths"] = {**cfg["paths"], "results": args.results}
    solver = args.solver or cfg["solver"]["name"]
    if args.grid:
        scen = load_config(args.scenarios)
        run_grid(cfg, scen, solver, quick=args.quick, days=args.days, skip_existing=not args.force,
                 only_axes=args.axes)
        return
    layers = layers_from_case(args.case)
    run_case(
        day=args.day or list(cfg["days"])[0],
        vre_share=args.vre if args.vre is not None else float(cfg["vre"]["target_share"]),
        layers=layers,
        vpl_mwh=float(cfg["vpl"]["p_nom_mw"]) * float(cfg["vpl"]["hours"]),
        dc_flex=float(cfg["vdc"]["flex_fraction"]),
        solver=solver,
        reserve_from_flex=bool(cfg["reserve"]["from_flex"]),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
