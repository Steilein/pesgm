"""
pesgm_numbers.py — números do texto do PES GM que os relatórios do periódico não dão.

Lê só ``results_pesgm/metrics.csv`` (grade A–H, 16 dias, variação base) e grava
``results_pesgm/pesgm_numbers.md`` com:

1. dias em que o corredor separa os preços dos terminais no caso sem
   flexibilidade (spread máximo de LMP acima de 0,5 $/MWh) --- "restrição
   econômica", diferente de "fluxo no limite" (``corridor_congested_hours``);
2. alívio de corte (p.p.) do VPL e do H2 sozinhos, separado em dias com e sem
   separação de preços (doze dias mensais);
3. operação do eletrolisador: horas produzindo, em standby, partidas a frio e
   consumo de standby sobre a energia de eletrólise (casos com H2);
4. interações por componente de custo (partida, combustível, ...) de cada par,
   pela mesma fórmula de Möbius aplicada a cada coluna de custo;
5. assimetria do par VPL–VDC: I sobre o valor de cada recurso sozinho.

Uso::

    python papers/pesgm/scripts/pesgm_numbers.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / os.environ.get("RESULTS_DIR", "results_pesgm")
MONTHLY = [f"m{m:02d}" for m in range(1, 13)]
SPREAD_TOL = 0.5          # $/MWh: abaixo disso o corredor não separa preços
# rótulo do caso -> conjunto de recursos (scenarios_journal.yaml)
CASES = {"A": set(), "B": {"VPL"}, "C": {"VDC"}, "D": {"VPL", "VDC"},
         "E": {"H2"}, "F": {"VPL", "H2"}, "G": {"VDC", "H2"}, "H": {"VPL", "VDC", "H2"}}
LABEL = {frozenset(v): k for k, v in CASES.items()}
COST = ["cost_startup", "cost_fuel", "cost_flex", "cost_degradation", "cost_h2_startup",
        "cost_curtail_penalty", "cost_shutdown", "cost_other"]


def mobius(row: pd.Series, subset: set[str], col: str) -> float:
    """I_T = Σ_{S⊆T} (−1)^{|T|−|S|+1} c_S para a coluna ``col``."""
    from itertools import combinations
    tot = 0.0
    T = sorted(subset)
    for k in range(len(T) + 1):
        for S in combinations(T, k):
            sign = (-1) ** (len(T) - k + 1)
            tot += sign * row[(LABEL[frozenset(S)], col)]
    return tot


def main() -> None:
    m = pd.read_csv(RES / "metrics.csv")
    m = m[m["variation"] == "base"].drop_duplicates("scenario", keep="last")
    wide = m.set_index(["day_key", "case"])
    L: list[str] = ["# Números do texto do PES GM", "",
                    "Gerado por `papers/pesgm/scripts/pesgm_numbers.py` a partir de "
                    f"`{RES.name}/metrics.csv`.", ""]

    # 1. separação de preços no caso A
    a = wide.xs("A", level="case")
    sep = a["lmp_spread_max"] > SPREAD_TOL
    L += ["## 1. Corredor como restrição econômica (caso A)", "",
          f"Spread máximo de LMP entre terminais > {SPREAD_TOL} $/MWh em "
          f"**{int(sep.sum())} de {len(sep)}** dias: {', '.join(sorted(a.index[sep]))}.", "",
          "Sem separação: " + ", ".join(sorted(a.index[~sep])) + ".", "",
          a[["lmp_spread_max", "corridor_congested_hours"]].round(2).to_markdown(), ""]

    # 2. alívio de corte por tipo de dia (mensais)
    cur = wide["curtailment_pct"].unstack("case").loc[MONTHLY]
    rel = pd.DataFrame({"VPL_pp": cur["A"] - cur["B"], "H2_pp": cur["A"] - cur["E"],
                        "separa": sep.reindex(MONTHLY)})
    g = rel.groupby("separa")[["VPL_pp", "H2_pp"]].mean().round(2)
    L += ["## 2. Alívio de corte (p.p.), doze dias mensais", "",
          f"Média geral: VPL {rel.VPL_pp.mean():.2f}, H2 {rel.H2_pp.mean():.2f}.", "",
          g.rename(index={True: "corredor separa preços", False: "sem separação"}).to_markdown(),
          "", f"Dias mensais com separação: {int(rel.separa.sum())}; sem: {int((~rel.separa).sum())}.", ""]

    # 3. eletrolisador
    h2 = m[m["case"].isin(["E", "F", "G", "H"]) & m["day_key"].isin(MONTHLY)].copy()
    h2["sb_frac"] = h2["h2_standby_mwh"] / h2["h2_p2h_mwh"]
    L += ["## 3. Eletrolisador (casos com H2, doze dias mensais)", "",
          # Seleciona antes de agrupar: a coluna ``axis`` do metrics colide com o
          # atributo que o pandas consulta em ``agg`` com lista de funções.
          h2[["case", "h2_on_h", "h2_standby_h", "h2_startups", "h2_standby_mwh", "sb_frac"]]
            .groupby("case").agg(["min", "max"]).round(3).to_markdown(), "",
          f"Partidas a frio no total: {int(h2['h2_startups'].sum())}.", ""]

    # 4. interação por componente de custo (mensais, média)
    rows = []
    for name, T in (("VPL–VDC", {"VPL", "VDC"}), ("VPL–H2", {"VPL", "H2"}),
                    ("VDC–H2", {"VDC", "H2"}), ("VPL–VDC–H2", {"VPL", "VDC", "H2"})):
        rec = {"subset": name}
        for col in COST + ["cost_total", "startups_total"]:
            vals = [mobius(pd.Series({(c, col): wide.loc[(d, c), col] for c in CASES}), T, col)
                    for d in MONTHLY]
            rec[col] = sum(vals) / len(vals)
        rows.append(rec)
    comp = pd.DataFrame(rows).set_index("subset")
    comp[[c for c in comp if c.startswith("cost")]] /= 1e3
    L += ["## 4. Interação por componente (média mensal; custos em k$/dia, partidas em nº/dia)", "",
          "Sinal: negativo = substituição (custo conjunto maior que a soma prevista).", "",
          comp.round(3).to_markdown(), ""]

    # 5. assimetria VPL–VDC
    cost = wide["cost_total"].unstack("case").loc[MONTHLY].mean()
    dB, dC = cost["A"] - cost["B"], cost["A"] - cost["C"]
    i = comp.loc["VPL–VDC", "cost_total"]
    L += ["## 5. Assimetria do par VPL–VDC (doze dias mensais)", "",
          f"Valor do VPL sozinho {dB/1e3:.2f} k$/dia; do VDC {dC/1e3:.2f} k$/dia; I = {i:.3f} k$/dia.", "",
          f"Perda do valor do VPL com o VDC: {100*abs(i)*1e3/dB:.2f} %; "
          f"perda do valor do VDC com o VPL: {100*abs(i)*1e3/dC:.1f} %.", ""]

    # 6. estabilidade do sinal entre dias e fração do benefício conjunto
    #    A cota (8) só cobre o erro do solver; a variação entre os doze dias é a
    #    outra incerteza, e o teste de postos sinalizados diz se a média difere de zero.
    from scipy import stats
    SUBSETS = (("VPL–VDC", {"VPL", "VDC"}), ("VPL–H2", {"VPL", "H2"}),
               ("VDC–H2", {"VDC", "H2"}), ("VPL–VDC–H2", {"VPL", "VDC", "H2"}))
    alone = {r: (cost["A"] - cost[LABEL[frozenset({r})]]) / 1e3 for r in ("VPL", "VDC", "H2")}
    rows = []
    for name, T in SUBSETS:
        per_day = pd.Series({d: mobius(pd.Series({(c, "cost_total"): wide.loc[(d, c), "cost_total"]
                                                   for c in CASES}), T, "cost_total") / 1e3
                             for d in MONTHLY})
        joint = (cost["A"] - cost[LABEL[frozenset(T)]]) / 1e3
        rows.append({"subset": name, "media": per_day.mean(), "negativos": int((per_day < 0).sum()),
                     "p_wilcoxon": stats.wilcoxon(per_day).pvalue,
                     "p_t": stats.ttest_1samp(per_day, 0).pvalue,
                     "media_abs_sobre_conjunto_%": 100 * per_day.abs().mean() / joint,
                     "sobre_menor_recurso_%": 100 * abs(per_day.mean()) / min(alone[r] for r in T)})
    L += ["## 6. Sinal entre dias e fração do benefício conjunto (doze dias mensais)", "",
          "p: teste de Wilcoxon (postos sinalizados) e t de uma amostra contra zero. "
          "Fração: média de |I| sobre a redução conjunta do subconjunto (coluna da Tabela II). "
          "Último: |média de I| sobre o valor sozinho do menor recurso do subconjunto.", "",
          pd.DataFrame(rows).set_index("subset").round(3).to_markdown(), ""]

    # 7. sensibilidade à tolerância: pior caso δ_S = g·ĉ_S, incumbentes fixos
    def conclusive(g: float | None) -> int:
        n = 0
        for d in sorted(m["day_key"].unique()):
            for _, T in SUBSETS:
                Ts = sorted(T)
                I = lo = hi = 0.0
                for k in range(len(Ts) + 1):
                    from itertools import combinations
                    for S in combinations(Ts, k):
                        r = wide.loc[(d, LABEL[frozenset(S)])]
                        sign = (-1) ** (len(Ts) - k + 1)
                        delta = (r["objective_milp"] - r["objective_bound"]) if g is None \
                            else g * r["objective_milp"]
                        I += sign * r["objective_milp"]
                        lo, hi = (lo + delta, hi) if sign > 0 else (lo, hi + delta)
                n += (I - lo > 0) or (I + hi < 0)
        return n
    L += ["## 7. Vereditos conclusivos contra a tolerância (16 dias, 64 coeficientes)", "",
          "Pior caso: cada δ_S no limite g·ĉ_S, com os incumbentes atuais.", "",
          f"- gaps obtidos: {conclusive(None)}"]
    L += [f"- g = {100*g:g} %: {conclusive(g)}" for g in (1e-4, 5e-4, 1e-3, 5e-3)] + [""]

    # 8. corte com as duas baterias livres (results_pesgm_robust, vpl_emergent)
    rob = ROOT / os.environ.get("ROBUST_DIR", "results_pesgm_robust") / "metrics.csv"
    if rob.exists():
        e = pd.read_csv(rob)
        e = e[(e["vpl_mode"] == "emergent") & e["day_key"].isin(MONTHLY)].set_index("day_key")
        free = (cur["A"] - e[e["case"] == "B"]["curtailment_pct"]).reindex(MONTHLY)
        L += ["## 8. Alívio de corte do VPL com as duas baterias livres (doze dias mensais)", "",
              f"Corte médio com o VPL livre: {e[e['case'] == 'B']['curtailment_pct'].mean():.2f} % "
              f"(híbrido {cur['B'].mean():.2f} %, H2 {cur['E'].mean():.2f} %, sem flexibilidade "
              f"{cur['A'].mean():.2f} %).", "",
              pd.DataFrame({"VPL_hibrido_pp": rel.VPL_pp, "VPL_livre_pp": free, "H2_pp": rel.H2_pp,
                            "separa": rel.separa}).groupby("separa").mean().round(2)
                .rename(index={True: "corredor separa preços", False: "sem separação"}).to_markdown(), ""]

    out = RES / "pesgm_numbers.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
