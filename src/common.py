"""
common.py — utilidades compartilhadas: leitura de configuração, sobrescrita
de parâmetros por caminho pontilhado e opções de solver.

Mantido propositalmente pequeno para que os demais módulos permaneçam
legíveis e autocontidos.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Garante UTF-8 no console do Windows (cp1252 não representa "≥", "→", etc.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def load_config(path: str | Path = ROOT / "config" / "base.yaml") -> dict:
    """Lê um YAML de configuração e devolve um ``dict`` aninhado."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, over: dict) -> dict:
    """Sobrepõe ``over`` em ``base`` recursivamente (cópia; ``over`` vence).

    Dicionários aninhados são fundidos chave a chave; qualquer outro tipo
    (listas inclusive) é substituído por inteiro. É a semântica que um perfil
    de sobrescrita precisa: ``vdc.workloads.window_scale`` pode ser alterado sem
    repetir o resto do bloco ``workloads``.
    """
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_profile(profile: str | Path | Sequence[str | Path] | None,
                 base: str | Path = ROOT / "config" / "base.yaml") -> dict:
    """Lê ``base.yaml`` e sobrepõe um ou mais perfis, na ordem dada.

    Sem perfil devolve exatamente ``load_config(base)``, de modo que todo o
    trabalho anterior continua reproduzível sem alteração. Um perfil só
    (string, ``Path`` ou lista de um) se comporta como sempre.

    Encadear existe para o artigo de periódico, que parte da calibração
    realista e só precisa trocar os diretórios de saída e acrescentar o bloco
    de hidrogênio::

        load_profile(["config/realistic.yaml", "config/journal.yaml"])

    A alternativa seria copiar as 185 linhas de ``realistic.yaml`` para dentro
    de ``journal.yaml``, e aí um conserto na calibração não chegaria ao segundo
    artigo --- exatamente o tipo de deriva silenciosa que a separação
    ``src/`` × ``papers/`` existe para evitar.

    ``_profile`` guarda a cadeia inteira, separada por ``+``, para que o
    ``n.meta`` do NetCDF registre de onde veio cada número.
    """
    cfg = load_config(base)
    if not profile:
        return cfg
    if isinstance(profile, (str, Path)):
        perfis = [p for p in str(profile).split(",") if p.strip()]
    else:
        perfis = list(profile)
    for p in perfis:
        cfg = deep_merge(cfg, load_config(p))
    cfg["_profile"] = "+".join(str(p) for p in perfis)
    return cfg


def load_scenarios(path: str | Path = ROOT / "config" / "scenarios.yaml") -> dict:
    """Lê a grade de sensibilidades / escada de flexibilidade."""
    return load_config(path)


def set_by_path(cfg: dict, dotted: str, value: Any) -> dict:
    """Sobrescreve ``cfg["a"]["b"]["c"]`` a partir de ``"a.b.c"`` (cópia profunda)."""
    out = copy.deepcopy(cfg)
    node = out
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return out


def get_by_path(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Lê ``cfg["a"]["b"]["c"]`` a partir de ``"a.b.c"``; ``default`` se faltar."""
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def solver_options(cfg: dict, mip: bool = True) -> tuple[str, dict]:
    """Traduz o bloco ``solver`` do YAML nas opções específicas de cada solver.

    O gap relativo (config: 0,5 % na depuração, 0,05 % na grade final) e o
    limite de tempo são idênticos em HiGHS e Gurobi
    para garantir comparabilidade dos resultados (§7 do brief). Para a
    re-solução LP (obtenção dos LMPs) os parâmetros de MIP são omitidos.
    """
    s = cfg["solver"]
    name = str(s["name"]).lower()
    threads = int(s.get("threads", 4))
    if name == "highs":
        opts = {"threads": threads, "time_limit": float(s["time_limit"])}
        if mip:
            opts["mip_rel_gap"] = float(s["mip_rel_gap"])
    elif name == "gurobi":
        opts = {"Threads": threads, "TimeLimit": float(s["time_limit"])}
        if mip:
            opts["MIPGap"] = float(s["mip_rel_gap"])
    else:
        raise ValueError(f"Solver não suportado: {name}")
    # Repasse de parâmetros específicos do solver (ex.: ``NodefileStart`` do
    # Gurobi, para a árvore de branch-and-bound ir para disco em vez de crescer
    # em memória). Ficam fora do bloco padrão de propósito: não afetam o ótimo,
    # só o consumo de máquina, e por isso não entram na comparabilidade.
    opts.update(s.get("options") or {})
    return name, opts


def results_dir(cfg: dict) -> Path:
    p = ROOT / cfg["paths"]["results"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir(cfg: dict) -> Path:
    p = ROOT / cfg["paths"]["figures"]
    p.mkdir(parents=True, exist_ok=True)
    return p
