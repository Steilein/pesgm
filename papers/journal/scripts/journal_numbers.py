"""
journal_numbers.py — os números do artigo de periódico que saem direto da grade.

Tabelas e frases da Sec. IV/V que não têm gerador próprio:

1. **Oito casos** (Tab. ``tab:cases8``): média dos doze dias mensais de custo,
   benefício sobre o caso A, corte e partidas térmicas.
2. **Desconto por par** (parágrafo de prática da conclusão): coeficiente de par
   sobre o benefício conjunto do par, ``I_T / Delta_T``.
3. **Tempo de solução por caso** (Tab. ``tab:cost``): mediana, média e máximo
   sobre os dezesseis dias, tempo total e instâncias que pararam no limite.
4. **Conferência com a grade da conferência**: os casos A–D do periódico
   contra a grade de c^flex = 10 $/MWh do GTD LA (``results_flexcost/``),
   dia a dia. A nota da Tab. ``tab:cases8`` só pode dizer que reproduz a
   conferência se esta conferência fechar.

Uso::

    python papers/journal/scripts/journal_numbers.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / os.environ.get("RESULTS_DIR", "results_journal")
MONTHLY = [f"m{m:02d}" for m in range(1, 13)]
CASES = list("ABCDEFGH")
RESOURCES = {"A": "---", "B": "V", "C": "D", "D": "VD", "E": "H", "F": "VH", "G": "DH", "H": "VDH"}
# Par -> (caso conjunto, caso do primeiro, caso do segundo)
PAIRS = {"VPL--VDC": ("D", "B", "C"), "VPL--H2": ("F", "B", "E"), "VDC--H2": ("G", "C", "E")}
# Grade da conferência (GTD LA, c^flex = 10): VDC = caso C e ambos = caso D na variação fc10.
CONF = ROOT / "results_flexcost" / "metrics.csv"
TIME_LIMIT = 1800.0


def base_grid() -> pd.DataFrame:
    df = pd.read_csv(RES / "metrics.csv")
    df = df[df.variation == "base"]
    dup = df.duplicated(["day_key", "case"], keep=False)
    if dup.any():
        raise SystemExit(f"linhas duplicadas em metrics.csv: {df[dup].scenario.tolist()}")
    return df


def cases8(df: pd.DataFrame) -> pd.DataFrame:
    m = df[df.day_key.isin(MONTHLY)]
    n = m.groupby("case").size()
    if (n.reindex(CASES) != len(MONTHLY)).any():
        raise SystemExit(f"grade mensal incompleta: {n.to_dict()}")
    g = m.groupby("case")
    t = pd.DataFrame({
        "cost_k": g.cost_total.mean() / 1e3,
        "curt_pct": g.curtailment_pct.mean(),
        "startups": g.startups_total.mean(),
    }).reindex(CASES)
    t["delta_k"] = t.cost_k["A"] - t.cost_k
    t["delta_pct"] = 100 * t.delta_k / t.cost_k["A"]
    return t


def pair_haircut(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (j, a, b) in PAIRS.items():
        I = t.delta_k[j] - t.delta_k[a] - t.delta_k[b]
        rows.append({"par": name, "I_k": I, "joint_k": t.delta_k[j], "haircut_pct": 100 * -I / t.delta_k[j]})
    hc = t.delta_k["H"] - (t.delta_k["D"] + t.delta_k["F"] + t.delta_k["G"]) \
        + (t.delta_k["B"] + t.delta_k["C"] + t.delta_k["E"])
    rows.append({"par": "VPL--VDC--H2", "I_k": hc, "joint_k": t.delta_k["H"],
                 "haircut_pct": 100 * -hc / t.delta_k["H"]})
    return pd.DataFrame(rows).set_index("par")


def solve_times(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("case").time_milp_s
    t = pd.DataFrame({"n": g.size(), "median_s": g.median(), "mean_s": g.mean(), "max_s": g.max(),
                      "no_limite": df.assign(lim=df.time_milp_s >= TIME_LIMIT).groupby("case").lim.sum(),
                      "gap_max": df.groupby("case").mip_gap.max()}).reindex(CASES)
    return t


def conference_check(df: pd.DataFrame) -> pd.DataFrame | None:
    if not CONF.exists():
        return None
    c = pd.read_csv(CONF)
    c = c[c.variation == "fc10"].set_index(["day_key", "case"]).cost_total
    rows = []
    for jc, cc in (("C", "C"), ("D", "D")):
        j = df[df.case == jc].set_index("day_key").cost_total
        for d in j.index.intersection(c.xs(cc, level="case").index):
            rows.append({"day": d, "case": jc, "journal": j[d], "conference": c[(d, cc)],
                         "rel_diff": (j[d] - c[(d, cc)]) / c[(d, cc)]})
    return pd.DataFrame(rows)


def main() -> None:
    df = base_grid()
    t = cases8(df)
    print("Oito casos, media dos doze dias mensais:")
    print(t.round(2).to_string())
    hc = pair_haircut(t)
    print("\nDesconto por subconjunto (I sobre o beneficio conjunto, media mensal):")
    print(hc.round(2).to_string())
    st = solve_times(df)
    print(f"\nTempo de solucao por caso ({len(df)} instancias):")
    print(st.round(1).to_string())
    print(f"tempo total de solver: {df.time_milp_s.sum() / 3600:.2f} h; "
          f"LP de precos: {df.time_lp_s.sum() / 60:.1f} min")
    chk = conference_check(df)
    if chk is not None and not chk.empty:
        print(f"\nCasos C e D contra a grade da conferencia (c^flex=10): {len(chk)} pares dia-caso, "
              f"maior |diferenca relativa| = {chk.rel_diff.abs().max():.1e}")

    out = RES / "journal_numbers.md"
    out.write_text(
        "# Números da grade do periódico\n\nGerado por `papers/journal/scripts/journal_numbers.py`.\n\n"
        "## Oito casos (doze dias mensais)\n\n" + t.round(2).to_markdown() + "\n\n"
        "## Desconto por subconjunto\n\n" + hc.round(2).to_markdown() + "\n\n"
        "## Tempo de solução por caso\n\n" + st.round(1).to_markdown() + "\n\n"
        f"Tempo total de solver: {df.time_milp_s.sum() / 3600:.2f} h.\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
