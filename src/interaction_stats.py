"""
interaction_stats.py — distribuição da interação I e cotas formais do MILP.

A interação entre as duas camadas é definida sobre benefícios,

    I = Δ_5 − (Δ_2 + Δ_4) = c_2 + c_4 − c_1 − c_5,

isto é, uma soma com sinais de quatro valores ótimos de problemas MILP
distintos. Cada um deles é resolvido até uma tolerância relativa ``mip_gap``,
de modo que o valor reportado é uma cota SUPERIOR do ótimo e

    c*_k ∈ [c_k (1 − g_k), c_k]      (minimização, gap relativo ao incumbente).

Como os erros de truncamento dos quatro problemas não se cancelam, a cota
rigorosa para a interação é obtida combinando os extremos no sentido que
maximiza e minimiza I:

    I ∈ [ c_2(1−g_2) + c_4(1−g_4) − c_1 − c_5 ,
          c_2 + c_4 − c_1(1−g_1) − c_5(1−g_5) ].

A meia-largura desse intervalo é Σ_k g_k c_k — a mesma quantidade usada como
"ruído do gap" nas tabelas do artigo, aqui apresentada na forma de cota que
permite classificar cada instância como substituição, complementaridade ou
INDETERMINADA, em vez de apenas comparar |I| com o ruído.

Uso::

    python src/interaction_stats.py            # tabela + figura
    python src/interaction_stats.py --no-fig
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT  # noqa: E402
from runner import RESOURCES as _RESOURCES, case_label  # noqa: E402

# Mapeamento entre os rótulos internos dos casos (A–D, herdados da primeira
# versão) e a numeração usada no artigo revisado (1, 2, 4, 5).
CASE_NUM = {"A": 1, "B": 2, "C": 4, "D": 5}


def mobius_bounds(df: pd.DataFrame, resources: Sequence[str] | None = None,
                  min_order: int = 2) -> pd.DataFrame:
    """Coeficientes de Möbius de TODA ordem, com a cota direcional rigorosa.

    Generaliza ``interaction_bounds`` para N recursos (Eqs. 19-20 do artigo de
    periódico). Para um subconjunto :math:`T` de recursos,

    .. math::
        I_T = \\sum_{S \\subseteq T} (-1)^{|T|-|S|+1} c_S = \\sum_S \\sigma_S c_S

    e, como cada ótimo verdadeiro está em :math:`[c_S - e_S,\\ c_S]` e o sinal de
    cada termo é conhecido, os extremos caem em cantos opostos da caixa::

        I_T \\in [ I - sum_{sigma=+1} e_S ,  I + sum_{sigma=-1} e_S ]

    Uma linha por (instância, subconjunto). ``subset`` é a tupla de recursos e
    ``order`` o seu tamanho; a largura da cota cresce como ``2**order``, que é o
    resultado central da Sec. III-H.

    **Qual custo entra onde.** O coeficiente usa ``cost_total`` (a grandeza
    econômica que o artigo reporta) e o erro usa ``objective_milp -
    objective_bound`` (grandeza do solver, sem supor convenção de gap). As duas
    só podem ser combinadas enquanto a diferença entre elas for desprezível
    diante de ``e``; a função verifica isso e avisa quando não for. Sem a coluna
    ``objective_bound`` cai-se em ``gap * cost_total``, que para a definição de
    MIPGap do Gurobi coincide com a cota real.
    """
    from itertools import combinations

    if resources is None:
        resources = _RESOURCES
    rows = []
    for (day, var), g in df.groupby(["day_key", "variation"]):
        cost = g.set_index("case")["cost_total"]
        presentes = set(cost.index)
        c = {k: float(cost[k]) for k in presentes}

        if "objective_bound" in g.columns and g.set_index("case")["objective_bound"].notna().all():
            inc = g.set_index("case")["objective_milp"]
            bnd = g.set_index("case")["objective_bound"]
            e = {k: max(float(inc[k]) - float(bnd[k]), 0.0) for k in presentes}
            desvio = max(abs(float(inc[k]) - c[k]) for k in presentes)
            menor_e = min((v for v in e.values() if v > 0), default=float("inf"))
            if desvio > 0.1 * menor_e:
                warnings.warn(
                    f"{day}/{var}: |cost_total - objective_milp| = {desvio:.1f} não é "
                    f"desprezível diante do menor erro de truncamento ({menor_e:.1f}); "
                    "a cota mistura duas grandezas e fica menos confiável.",
                    RuntimeWarning, stacklevel=2)
        else:
            gap = g.set_index("case")["mip_gap"]
            e = {k: float(gap[k]) * c[k] for k in presentes}

        for k in range(min_order, len(resources) + 1):
            for T in combinations(resources, k):
                subcasos = {S: case_label(set(S))
                            for m in range(len(T) + 1) for S in combinations(T, m)}
                if not set(subcasos.values()) <= presentes:
                    continue                  # instância incompleta para este subconjunto
                I = pos = neg = 0.0
                cpos = cneg = 0.0
                for S, rot in subcasos.items():
                    sigma = (-1) ** (len(T) - len(S) + 1)
                    I += sigma * c[rot]
                    if sigma > 0:
                        pos += e[rot]
                        cpos += c[rot]
                    else:
                        neg += e[rot]
                        cneg += c[rot]
                lo, hi = I - pos, I + neg
                rows.append({
                    "day_key": day, "variation": var, "axis": g["axis"].iloc[0],
                    "subset": "x".join(T), "order": len(T),
                    "I": I, "I_lo": lo, "I_hi": hi, "halfwidth": pos + neg,
                    "cost_AD": cneg, "cost_BC": cpos,
                    "verdict": "substitute" if hi < 0 else
                               "complement" if lo > 0 else "indeterminate",
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["order", "subset", "axis", "variation", "day_key"]).reset_index(drop=True)


def interaction_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada (dia, variação) com os quatro casos resolvidos, I e suas cotas.

    Retorna uma linha por instância com ``I`` (ponto), ``I_lo``/``I_hi``
    (cotas rigorosas), ``halfwidth`` (Σ g_k c_k) e ``verdict`` em
    {``substitute``, ``complement``, ``indeterminate``}.
    """
    rows = []
    for (day, var), g in df.groupby(["day_key", "variation"]):
        cost = g.set_index("case")["cost_total"]
        gap = g.set_index("case")["mip_gap"]
        if not set("ABCD") <= set(cost.index):
            continue                      # instância incompleta: não entra na estatística
        c = {k: float(cost[k]) for k in "ABCD"}
        # Distância entre o incumbente e a melhor cota dual de cada MILP. Quando a
        # coluna ``objective_bound`` existe usa-se a cota que o solver reportou,
        # sem supor convenção alguma de gap; caso contrário cai-se em g_k c_k, que
        # para a definição de MIPGap do Gurobi — relativa ao incumbente — coincide
        # com a cota real (verificado: diferença nula em precisão de máquina).
        if "objective_bound" in g.columns and g.set_index("case")["objective_bound"].notna().all():
            bound = g.set_index("case")["objective_bound"]
            e = {k: c[k] - float(bound[k]) for k in "ABCD"}
        else:
            e = {k: float(gap[k]) * c[k] for k in "ABCD"}
        I = c["B"] + c["C"] - c["A"] - c["D"]
        # Cota DIRECIONAL. I = c_2 + c_4 - c_1 - c_5 cresce com c_2, c_4 e decresce
        # com c_1, c_5, e cada otimo verdadeiro esta em [c_k - e_k, c_k]. Logo o
        # minimo de I usa os pisos de 2 e 4 contra os tetos de 1 e 5, e o maximo o
        # inverso: I em [I - (e_2 + e_4), I + (e_1 + e_5)]. Somar os quatro erros nas
        # duas direcoes, como antes, e rigoroso mas desnecessariamente conservador.
        lo = I - (e["B"] + e["C"])
        hi = I + (e["A"] + e["D"])
        rows.append({
            "day_key": day, "variation": var, "axis": g["axis"].iloc[0],
            "I": I, "I_lo": lo, "I_hi": hi, "halfwidth": sum(e.values()),
            # custos agrupados pelo sinal com que entram em I: usados para
            # reescalar a cota direcional a um gap hipotetico uniforme
            "cost_AD": c["A"] + c["D"], "cost_BC": c["B"] + c["C"],
            "verdict": "substitute" if hi < 0 else
                       "complement" if lo > 0 else "indeterminate",
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["axis", "variation", "day_key"]).reset_index(drop=True)


def _classify_at(tab: pd.DataFrame, gap: float) -> pd.Series:
    """Veredito que se obteria se todos os MILP fossem resolvidos a ``gap``.

    Mantém os mesmos valores ótimos e supõe erro ``gap * c_k`` em cada um, na
    forma direcional: o teto de I usa os casos 1 e 5, o piso usa 2 e 4. É a
    pergunta "o que este mesmo experimento teria concluído com a tolerância
    usual de 0,5 %?".
    """
    hi = tab["I"] + gap * tab["cost_AD"]
    lo = tab["I"] - gap * tab["cost_BC"]
    return np.where(hi < 0, "substitute", np.where(lo > 0, "complement", "indeterminate"))


def write_report(path: Path, tab: pd.DataFrame, gap_mean: pd.Series) -> None:
    """Grava ``results/interaction.md`` com a distribuição e as cotas."""
    n = len(tab)
    counts = tab["verdict"].value_counts()
    sub, comp, ind = (int(counts.get(k, 0)) for k in ("substitute", "complement", "indeterminate"))
    neg = int((tab["I"] < 0).sum())

    L = ["# Distribuição da interação e cotas formais do MILP\n",
         "Gerado por `src/interaction_stats.py`. Uma linha por instância com os quatro",
         "casos resolvidos; cotas rigorosas a partir do gap relativo de cada MILP.\n",
         f"Instâncias completas: **{n}**. Média de I: **{tab['I'].mean():,.0f} $/dia**; "
         f"mediana **{tab['I'].median():,.0f}**; intervalo [{tab['I'].min():,.0f}, {tab['I'].max():,.0f}].\n",
         "## 1. Classificação por cota rigorosa\n",
         "| veredito | instâncias | % |", "|:---|---:|---:|",
         f"| substituição (I_hi < 0) | {sub} | {100*sub/n:.0f} |",
         f"| complementaridade (I_lo > 0) | {comp} | {100*comp/n:.0f} |",
         f"| indeterminado | {ind} | {100*ind/n:.0f} |", "",
         f"O sinal pontual de I é negativo em **{neg}/{n}** instâncias ({100*neg/n:.0f} %), mas apenas",
         f"**{sub}** delas têm o intervalo inteiro à esquerda de zero. " +
         ("Nenhuma conclusão de complementaridade sobrevive às cotas." if comp == 0 else
          f"Em contrapartida, **{comp}** instâncias são conclusivamente COMPLEMENTARES "
          "(intervalo inteiro à direita de zero), de modo que a substituição é o "
          "comportamento dominante da amostra, não uma propriedade universal."), "",
         "## 2. O que a tolerância usual teria concluído\n",
         "Mesmos valores ótimos, meia-largura reescalada para o gap indicado:\n",
         "| gap alvo | substituição | complementaridade | indeterminado |", "|---:|---:|---:|---:|"]
    for g in (0.0005, 0.005):
        v = pd.Series(_classify_at(tab.assign(gap_mean=gap_mean), g)).value_counts()
        L.append(f"| {100*g:.2f} % | {int(v.get('substitute', 0))} | "
                 f"{int(v.get('complement', 0))} | {int(v.get('indeterminate', 0))} |")
    L += ["", "A 0,5 % praticamente toda a amostra cai em indeterminado: o experimento",
          "não distingue substituição de complementaridade, e a conclusão do artigo",
          "não poderia ser sustentada com a tolerância usual.\n",
          "## 3. Distribuição por eixo\n",
          tab.groupby("axis")["I"].agg(["count", "mean", "median", "min", "max"]).round(0).to_markdown(), "",
          "## 4. Instâncias com veredito conclusivo\n",
          tab[tab["verdict"] != "indeterminate"][
              ["day_key", "variation", "I", "I_lo", "I_hi", "verdict"]
          ].round(0).to_markdown(index=False), ""]
    path.write_text("\n".join(L), encoding="utf-8")


def width_by_order(tab: pd.DataFrame) -> pd.DataFrame:
    """Largura da cota por ordem de interação --- o teste empírico de W ~ 2^k.

    A Sec. III-H prevê que a largura dobra a cada ordem, porque a cota soma os
    erros de TODOS os 2^|T| subconjuntos. Aqui isso deixa de ser aritmética e
    vira medida: a coluna ``razao`` é a largura mediana dividida pela da ordem
    anterior, e deve rondar 2.
    """
    g = tab.groupby("order").agg(
        n=("I", "size"),
        I_abs_mediano=("I", lambda s: s.abs().median()),
        largura_mediana=("halfwidth", "median"),
        conclusivos=("verdict", lambda s: int((s != "indeterminate").sum())),
    )
    g["pct_conclusivo"] = (100 * g["conclusivos"] / g["n"]).round(0)
    g["razao"] = (g["largura_mediana"] / g["largura_mediana"].shift(1)).round(2)
    # o que se quer ver: |I| encolhe com a ordem enquanto a cota dobra
    g["I_sobre_largura"] = (g["I_abs_mediano"] / g["largura_mediana"]).round(2)
    return g.round(1)


def write_mobius_report(path: Path, tab: pd.DataFrame) -> None:
    """Relatório do fatorial de N recursos: um bloco por subconjunto.

    Separado de ``write_report`` de propósito: aquele gera
    ``results/interaction.md``, artefato do artigo submetido, e tem de continuar
    saindo byte a byte igual.
    """
    n_inst = tab.groupby(["day_key", "variation"]).ngroups
    subsets = tab["subset"].unique()

    L = ["# Interações de todas as ordens e suas cotas rigorosas\n",
         "Gerado por `src/interaction_stats.py --mobius`. Uma linha por",
         "(instância, subconjunto de recursos); a cota vem dos *best bounds* do",
         "solver, sem supor convenção de gap.\n",
         f"Instâncias: **{n_inst}**. Subconjuntos avaliados: **{len(subsets)}** "
         f"({', '.join(subsets)}).\n",
         "## 1. Resumo por subconjunto\n"]

    resumo = tab.groupby(["order", "subset"]).agg(
        n=("I", "size"), I_medio=("I", "mean"), I_mediano=("I", "median"),
        subst=("verdict", lambda s: int((s == "substitute").sum())),
        compl=("verdict", lambda s: int((s == "complement").sum())),
        indet=("verdict", lambda s: int((s == "indeterminate").sum())),
    ).round(0)
    L += [resumo.to_markdown(), ""]

    L += ["## 2. A cota dobra a cada ordem\n",
          "Previsão da Sec. III-H: `W <= 2^k g c`. A coluna `razao` é a largura",
          "mediana sobre a da ordem anterior --- deve rondar 2. A coluna",
          "`I_sobre_largura` diz se ainda há sinal acima do ruído: abaixo de 1,",
          "o coeficiente é menor que a própria incerteza.\n",
          width_by_order(tab).to_markdown(), ""]

    L += ["## 3. O que uma tolerância usual teria concluído\n",
          "Mesmos ótimos, cota reescalada para o gap indicado:\n",
          "| gap alvo | subconjunto | substituição | complementaridade | indeterminado |",
          "|---:|:---|---:|---:|---:|"]
    for g in (0.0005, 0.005):
        for s in subsets:
            t = tab[tab.subset == s]
            v = pd.Series(_classify_at(t, g)).value_counts()
            L.append(f"| {100*g:.2f} % | {s} | {int(v.get('substitute', 0))} | "
                     f"{int(v.get('complement', 0))} | {int(v.get('indeterminate', 0))} |")
    L.append("")

    L += ["## 4. Instâncias com veredito conclusivo\n"]
    concl = tab[tab["verdict"] != "indeterminate"]
    if concl.empty:
        L.append("Nenhuma: a tolerância alcançada não sustenta conclusão alguma.\n")
    else:
        L += [concl[["subset", "day_key", "variation", "I", "I_lo", "I_hi", "verdict"]]
              .round(0).to_markdown(index=False), ""]
    path.write_text("\n".join(L), encoding="utf-8")


def plot_distribution(tab: pd.DataFrame, fdir: Path, name: str) -> None:
    """Figura: I por instância com a barra de incerteza do gap, ordenado."""
    import matplotlib.pyplot as plt
    from plotstyle import GRID, INK, INK2, PALETTE, T, W1, ieee_style, save

    ieee_style()
    t = tab.sort_values("I").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(W1, 2.15))
    x = np.arange(len(t))
    color = np.where(t["verdict"] == "substitute", PALETTE[3],
                     np.where(t["verdict"] == "complement", PALETTE[0], INK2))
    ax.vlines(x, t["I_lo"] / 1e3, t["I_hi"] / 1e3, color=GRID, lw=0.8, zorder=1)
    ax.scatter(x, t["I"] / 1e3, s=7, c=color, zorder=2, linewidths=0)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlabel(T("instância (ordenada por $I$)"))
    ax.set_ylabel(T("$I$ (k\\$/dia)"))
    ax.set_xlim(-1, len(t))
    n_sub = int((t["verdict"] == "substitute").sum())
    # canto superior esquerdo: no inferior o texto colide com o primeiro ponto
    from plotstyle import LANG
    media = f"{t['I'].mean()/1e3:.1f}"
    if LANG != "en":                      # separador decimal brasileiro
        media = media.replace(".", ",")
    ax.text(0.03, 0.96, f"{T('substituição conclusiva')}: {n_sub}/{len(t)}\n"
                        f"{T('$I$ médio')} = {media} {T('k\\$/dia')}",
            transform=ax.transAxes, fontsize=6, color=INK, va="top")
    save(fig, fdir, name)


def plot_mobius(tab: pd.DataFrame, fdir: Path, name: str) -> None:
    """Um painel por subconjunto: I por instância com a cota, ordenado.

    Generalização de ``plot_distribution``. O eixo vertical é comum aos painéis
    de propósito --- é ele que mostra que o par VPL--H2 domina os demais em
    magnitude, e painéis com escala própria esconderiam justamente isso.
    """
    import matplotlib.pyplot as plt
    from plotstyle import GRID, INK, INK2, PALETTE, T, W2, ieee_style, save

    ieee_style()
    subsets = sorted(tab["subset"].unique(), key=lambda s: (s.count("x"), s))
    fig, axes = plt.subplots(1, len(subsets), figsize=(W2, 2.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, subsets):
        t = tab[tab.subset == s].sort_values("I").reset_index(drop=True)
        x = np.arange(len(t))
        cor = np.where(t["verdict"] == "substitute", PALETTE[3],
                       np.where(t["verdict"] == "complement", PALETTE[0], INK2))
        ax.vlines(x, t["I_lo"] / 1e3, t["I_hi"] / 1e3, color=GRID, lw=0.8, zorder=1)
        ax.scatter(x, t["I"] / 1e3, s=7, c=cor, zorder=2, linewidths=0)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_title(s.replace("x", "--"), fontsize=7)
        ax.set_xlabel(T("instância"))
        ax.set_xlim(-1, len(t))
        n_c = int((t["verdict"] != "indeterminate").sum())
        # canto superior esquerdo: no inferior o texto encosta no eixo quando o
        # painel tem instancias muito negativas (o par VPL--H2 tem)
        ax.text(0.04, 0.96, f"{n_c}/{len(t)}", transform=ax.transAxes,
                fontsize=6, color=INK2, va="top")
    axes[0].set_ylabel(T("$I$ (k\\$/dia)"))
    save(fig, fdir, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default="results/metrics.csv")
    ap.add_argument("--out", default="results/interaction.md")
    ap.add_argument("--no-fig", action="store_true")
    # Sem isto o modulo grava em figures/, que e do artigo submetido: rodar a
    # analise do periodico sobrescrevia fig10_interaction.pdf sem aviso.
    ap.add_argument("--figures", default="figures",
                    help="diretorio das figuras (use figures_journal para o periodico)")
    ap.add_argument("--mobius", action="store_true",
                    help="fatorial de N recursos: todos os subconjuntos de ordem >= 2, "
                         "um bloco por par e a terceira ordem (artigo de periódico)")
    a = ap.parse_args()

    df = pd.read_csv(ROOT / a.metrics)

    if a.mobius:
        tab = mobius_bounds(df)
        if tab.empty:
            sys.exit("nenhuma instância com um subconjunto completo de casos; "
                     "a grade do periódico precisa dos oito casos A–H por dia")
        out = ROOT / a.out
        write_mobius_report(out, tab)
        print(f"{tab.groupby(['day_key', 'variation']).ngroups} instâncias, "
              f"{len(tab)} coeficientes -> {a.out}")
        print(tab.groupby(["order", "subset"])["verdict"].value_counts().to_string())
        if not a.no_fig:
            plot_mobius(tab, ROOT / a.figures, "fig_mobius_map")
        return

    tab = interaction_bounds(df)
    gap_mean = df.groupby(["day_key", "variation"])["mip_gap"].mean()
    gap_mean = tab.set_index(["day_key", "variation"]).index.map(gap_mean)
    write_report(ROOT / a.out, tab, pd.Series(gap_mean, index=tab.index))
    print(f"{len(tab)} instâncias -> {a.out}")
    print(tab["verdict"].value_counts().to_string())
    if not a.no_fig:
        plot_distribution(tab, ROOT / a.figures, "fig10_interaction")


if __name__ == "__main__":
    main()
