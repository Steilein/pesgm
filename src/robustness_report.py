"""
robustness_report.py — tabela de robustez e distribuição por instância, sistema corrigido.

Lê ``results_robust/metrics.csv`` (grade de ``robustness_tight.py``, gap 0,01 %)
e produz o que o artigo precisa:

1. a tabela de robustez --- I de cada variação, média sobre os dias, com a cota
   rigorosa implicada pelos gaps efetivamente atingidos;
2. a distribuição por instância --- cada par (dia, variação) classificado em
   substituição / complementaridade / indeterminado pela mesma cota direcional
   de ``interaction_stats.py``.

Completamento pela base. Variações que só alteram um lado (``mig05`` existe em C
e D; ``soc50`` em B e D) não têm as quatro pontas próprias: as que faltam vêm da
grade base do MESMO dia e do MESMO gap, como em ``audit_claims.I_for``. Sem
isso a instância simplesmente não fecha.

Uso::

    python src/robustness_report.py
    python src/robustness_report.py --results results_robust --out results_robust/robustness.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT  # noqa: E402

CASES = list("ABCD")
# Rótulos do artigo, na ordem da tabela
LABELS = {
    "base": "(nada: caso base)",
    "hist": "Estados térmicos herdados do dia anterior",
    "soc50": "Estado de carga inicial fixo em 50 %",
    "resref": "Mesmo requisito de reserva em todos os casos",
    "mig05": "Custo de migração x0,5",
    "mig20": "Custo de migração x2",
    "reson": "LVT e CDV autorizados a fornecer reserva",
}
# Valores publicados hoje, medidos no sistema SEM as correções (k$/dia)
PAPER = {"base": -1.3, "hist": -6.5, "soc50": -1.7, "resref": -1.4,
         "mig05": -0.9, "mig20": -1.2, "reson": -0.1}


def _err(row: pd.Series) -> float:
    """Distância entre incumbente e melhor cota dual de um MILP, em $.

    Usa ``objective_bound`` quando o solver a registrou; senão cai em g*c, que
    para a convenção do Gurobi (gap relativo ao incumbente) coincide com ela.
    """
    c = float(row["cost_total"])
    b = row.get("objective_bound")
    if pd.notna(b):
        return max(c - float(b), 0.0)
    return float(row["mip_gap"]) * c


# Variações que NÃO são uma das quatro pontas de I. O caso 3 (CDV sem migração
# espacial) existe para repartir o benefício do CDV entre tempo e espaço; entrar
# aqui produziria um "I" que não corresponde a definição nenhuma.
EXCLUDE = {"dclocal"}


def instances(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (dia, variação) com I, cota direcional e veredito."""
    base = df[df["variation"] == "base"].set_index(["day_key", "case"])
    rows = []
    for (day, var), g in df.groupby(["day_key", "variation"]):
        if var in EXCLUDE:
            continue
        own = g.set_index("case")
        use = {}
        for k in CASES:
            if k in own.index:
                use[k] = own.loc[k]
            elif (day, k) in base.index:
                use[k] = base.loc[(day, k)]          # completamento pela base
        if len(use) < 4:
            continue
        c = {k: float(use[k]["cost_total"]) for k in CASES}
        # Variação que coincide com o valor do próprio perfil não é variação: no
        # sistema corrigido f_d já é 0,2 e a janela batch já é x0,5, de modo que
        # "f02" e "w05" reproduzem a base MILP a MILP. Entrassem na distribuição,
        # a mesma instância contaria três vezes.
        if var != "base":
            cb = {k: float(base.loc[(day, k), "cost_total"]) for k in CASES
                  if (day, k) in base.index}
            if len(cb) == 4 and all(abs(c[k] - cb[k]) <= 1e-6 * abs(cb[k]) for k in CASES):
                print(f"  {day}/{var}: idêntica à base — descartada")
                continue
        e = {k: _err(use[k]) for k in CASES}
        I = c["B"] + c["C"] - c["A"] - c["D"]
        lo, hi = I - (e["B"] + e["C"]), I + (e["A"] + e["D"])
        rows.append({"day_key": day, "variation": var,
                     "I": I / 1e3, "I_lo": lo / 1e3, "I_hi": hi / 1e3,
                     "halfwidth": sum(e.values()) / 1e3,
                     "gap_max": max(float(use[k]["mip_gap"]) for k in CASES),
                     "verdict": "substituição" if hi < 0 else
                                "complementaridade" if lo > 0 else "indeterminado"})
    return pd.DataFrame(rows).sort_values(["variation", "day_key"]).reset_index(drop=True)


def table(inst: pd.DataFrame) -> pd.DataFrame:
    """Tabela de robustez: média por variação, com ruído médio e contagem conclusiva."""
    rows = []
    for var, g in inst.groupby("variation"):
        conc = (g.verdict != "indeterminado").sum()
        rows.append({"variação": LABELS.get(var, var), "chave": var,
                     "I k$/dia": round(g.I.mean(), 2),
                     "ruído k$": round(g.halfwidth.mean(), 2),
                     "conclusivas": f"{conc}/{len(g)}",
                     "artigo (sem correções)": PAPER.get(var)})
    order = {k: i for i, k in enumerate(LABELS)}
    return (pd.DataFrame(rows)
            .sort_values("chave", key=lambda s: s.map(lambda v: order.get(v, 99)))
            .reset_index(drop=True))


def report(out: Path, tab: pd.DataFrame, inst: pd.DataFrame, gap_alvo: float) -> None:
    n_sub = (inst.verdict == "substituição").sum()
    n_com = (inst.verdict == "complementaridade").sum()
    L = [
        "# Robustez no sistema corrigido, gap apertado\n",
        f"Gerado por `src/robustness_report.py` a partir de `{out.parent.name}/metrics.csv`. ",
        f"Alvo de gap: {gap_alvo:.2%}; máximo efetivamente atingido: {inst.gap_max.max():.3%}.\n",
        "\nI é a média sobre os dias; o ruído é a meia-largura média da cota direcional. ",
        "A coluna do artigo traz o valor publicado hoje, medido no sistema **sem** as correções ",
        "estruturais — os dois não precisam coincidir, e a diferença é o resultado deste teste.\n\n",
        tab.to_markdown(index=False), "\n",
        f"\n## Distribuição por instância\n\n**{n_sub}** de {len(inst)} instâncias são ",
        f"conclusivamente substitutivas e **{n_com}** conclusivamente complementares; ",
        f"{len(inst) - n_sub - n_com} ficam indeterminadas sob a cota.\n\n",
        inst.round(3).to_markdown(index=False), "\n",
    ]
    out.write_text("".join(L), encoding="utf-8")
    print(f"escrito: {out}")


def figure(inst: pd.DataFrame, fdir: Path, name: str = "fig10_interaction") -> None:
    """Distribuição por instância, no formato da Fig. 10 do artigo.

    Reaproveita o traçado de ``interaction_stats`` convertendo de volta para $ e
    para os rótulos em inglês que ele espera; a diferença é a população, que aqui
    inclui as instâncias completadas pela base.
    """
    from interaction_stats import plot_distribution

    tab = inst.rename(columns={"I": "I_k", "I_lo": "lo_k", "I_hi": "hi_k"}).copy()
    tab["I"], tab["I_lo"], tab["I_hi"] = tab.I_k * 1e3, tab.lo_k * 1e3, tab.hi_k * 1e3
    tab["verdict"] = tab["verdict"].map({"substituição": "substitute",
                                         "complementaridade": "complement",
                                         "indeterminado": "indeterminate"})
    plot_distribution(tab, fdir, name)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Relatório de robustez no sistema corrigido.")
    ap.add_argument("--results", default="results_robust")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gap", type=float, default=0.0001)
    ap.add_argument("--fig", default=None,
                    help="diretório onde gravar a figura da distribuição (ex.: figures_realistic)")
    args = ap.parse_args(argv)

    rdir = ROOT / args.results
    df = pd.read_csv(rdir / "metrics.csv")
    inst = instances(df)
    if inst.empty:
        raise SystemExit("nenhuma instância completa em " + str(rdir))
    tab = table(inst)
    print(tab.to_string(index=False))
    report(Path(args.out) if args.out else rdir / "robustness.md", tab, inst, args.gap)
    if args.fig:
        figure(inst, ROOT / args.fig)


if __name__ == "__main__":
    main()
