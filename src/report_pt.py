"""
report_pt.py — gera as tabelas LaTeX (em português) do relatório para o orientador.

Lê ``results/metrics.csv`` e escreve ``docs/relatorio/tabelas.tex`` com um comando
``\\tab<nome>`` por tabela, além de ``docs/relatorio/numeros.tex`` com macros
``\\num<nome>`` para os valores citados no texto corrido. Assim nenhum número do
relatório é digitado à mão.

Uso::

    python src/report_pt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_config, results_dir  # noqa: E402

DAYS = ["spring_wind", "summer_peak", "solar_ramp", "median"]
DAY_PT = {"spring_wind": "primavera (vento alto)", "summer_peak": "pico de verão",
          "solar_ramp": "rampa solar", "median": "mediano"}
CASE_PT = {"A": "A (base)", "B": "B (+VPL)", "C": "C (+VDC)", "D": "D (ambos)"}


def _fmt(x, dec=1):
    if pd.isna(x):
        return "---"
    return f"{x:,.{dec}f}".replace(",", "\\,").replace(".", ",")


def _tabular(df: pd.DataFrame, colfmt: str, dec=1, index_name="") -> str:
    head = " & ".join([index_name] + [str(c) for c in df.columns]) + r" \\"
    body = []
    for idx, row in df.iterrows():
        cells = [_fmt(v, dec) if isinstance(v, (int, float, np.floating)) else str(v) for v in row]
        body.append(" & ".join([str(idx)] + cells) + r" \\")
    return ("\\begin{tabular}{@{}" + colfmt + "@{}}\n\\toprule\n" + head
            + "\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}")


def build(df_all: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    from common import results_dir
    cfg_results_dir = results_dir(cfg)
    d = df_all.set_index("scenario")
    T, N = {}, {}
    # Agregações usam só os quatro dias representativos e os eixos da grade
    # principal: dias mensais, condições herdadas, baseline alternativo e a
    # decomposição têm tabelas próprias e contaminariam as médias.
    MAIN_AXES = ["base", "vre_share", "vpl_size", "dc_flex", "window_scale",
                 "reserve_from_flex", "vpl_mode", "migration_cost"]
    df = df_all[df_all["day_key"].isin(DAYS) & df_all["axis"].isin(MAIN_AXES)]

    def C0(day):
        return cost(day, "C")

    def cost(day, case, var="base"):
        return d.loc[f"{day}__{case}__{var}", "cost_total"]

    # ---------------------------------------------------------------- T1: caso base por dia
    rows = []
    for day in DAYS:
        r = {"dia": DAY_PT[day]}
        for c in "ABCD":
            r[CASE_PT[c]] = cost(day, c) / 1e3
        rows.append(r)
    t1 = pd.DataFrame(rows).set_index("dia")
    t1["$\Delta_B$"] = t1[CASE_PT["A"]] - t1[CASE_PT["B"]]
    t1["$\Delta_C$"] = t1[CASE_PT["A"]] - t1[CASE_PT["C"]]
    t1["$\Delta_D$"] = t1[CASE_PT["A"]] - t1[CASE_PT["D"]]
    t1["$I$"] = t1["$\Delta_D$"] - (t1["$\Delta_B$"] + t1["$\Delta_C$"])
    t1.loc["média"] = t1.mean(numeric_only=True)
    T["custodia"] = _tabular(t1, "lrrrrrrrr", 1, "dia")

    # ---------------------------------------------------------------- T2: métricas operacionais
    cols = [("cost_total", "custo (k\\$)", 1e-3), ("curtailment_mwh", "curtail. (MWh)", 1),
            ("curtailment_pct", "curtail. (\\%)", 1), ("startups_total", "partidas", 1),
            ("pmin_unit_hours", "h em Pmin", 1), ("corridor_congested_hours", "h congest.", 1),
            ("lmp_spread_mean", "spread (\\$/MWh)", 1), ("vpl_virtual_mwh", "cap. virtual (MWh)", 1),
            ("co2_t", "CO$_2$ (t)", 1)]
    base = df[(df["variation"] == "base") & (df["axis"] == "base")]
    rows = {}
    for col, lbl, k in cols:
        rows[lbl] = {CASE_PT[c]: base[base["case"] == c][col].mean() * k for c in "ABCD"}
    t2 = pd.DataFrame(rows).T
    T["operacional"] = _tabular(t2, "lrrrr", 1, "métrica")

    # ---------------------------------------------------------------- T3: interação por penetração
    rows = []
    for var, lbl in (("vre30", "30\\%"), ("base", "50\\%"), ("vre70", "70\\%")):
        sub = df[df["variation"] == var].pivot_table(index="day_key", columns="case", values="cost_total")
        gp = df[df["variation"] == var].pivot_table(index="day_key", columns="case", values="mip_gap")
        noise = (gp * sub).sum(axis=1)
        dB, dC, dD = sub["A"] - sub["B"], sub["A"] - sub["C"], sub["A"] - sub["D"]
        I = dD - (dB + dC)
        rows.append({"VRE": lbl, "A (k\\$)": sub["A"].mean() / 1e3, "$\Delta_B$": dB.mean() / 1e3,
                     "$\Delta_C$": dC.mean() / 1e3, "$\Delta_D$": dD.mean() / 1e3, "$I$": I.mean() / 1e3,
                     "ruído": noise.mean() / 1e3, "$I<0$": f"{int((I < 0).sum())}/4"})
    t3 = pd.DataFrame(rows).set_index("VRE")
    T["penetracao"] = _tabular(t3, "lrrrrrrc", 1, "VRE")

    # ---------------------------------------------------------------- T4: sensibilidades (custo)
    piv = df.pivot_table(index="variation", columns="case", values="cost_total", aggfunc="mean") / 1e3
    order = [("base", "base (200 MW, $f_d$=0,5)"), ("vre30", "VRE 30\\%"), ("vre70", "VRE 70\\%"),
             ("vpl100", "VPL 100 MW"), ("vpl400", "VPL 400 MW"), ("vpl650", "VPL 650 MW"),
             ("f02", "$f_d$ = 0,2"), ("f08", "$f_d$ = 0,8"), ("w05", "janela $\\times$0,5"),
             ("w20", "janela $\\times$2"), ("mig05", "migração $\\times$0,5"), ("mig20", "migração $\\times$2"),
             ("reson", "reserva flexível"), ("explicit", "VPL explícito"),
             ("dedicated", "VPL dedicado")]
    rows = []
    for var, lbl in order:
        if var not in piv.index:
            continue
        r = {"variação": lbl}
        A = piv.loc[var, "A"] if not pd.isna(piv.loc[var].get("A", np.nan)) else piv.loc["base", "A"]
        for c in "BCD":
            v = piv.loc[var].get(c, np.nan)
            r["$\Delta_" + c + "$"] = (A - v) if not pd.isna(v) else np.nan
        base_A = piv.loc["base", "A"]
        dB = r["$\Delta_B$"] if not pd.isna(r["$\Delta_B$"]) else base_A - piv.loc["base", "B"]
        dC = r["$\Delta_C$"] if not pd.isna(r["$\Delta_C$"]) else base_A - piv.loc["base", "C"]
        r["$I$"] = r["$\Delta_D$"] - (dB + dC) if not pd.isna(r["$\Delta_D$"]) else np.nan
        rows.append(r)
    t4 = pd.DataFrame(rows).set_index("variação")
    T["sensibilidades"] = _tabular(t4, "lrrrr", 1, "variação")

    # ---------------------------------------------------------------- T5: decomposição do VPL
    rows = []
    for day in DAYS:
        A, B = cost(day, "A"), cost(day, "B")
        up, dn = cost(day, "B", "vplup"), cost(day, "B", "vpldown")
        rows.append({"dia": DAY_PT[day], "$\Delta_B$ (par)": (A - B) / 1e3, "só montante": (A - up) / 1e3,
                     "só jusante": (A - dn) / 1e3, "pareamento": ((A - B) - (2 * A - up - dn)) / 1e3,
                     "\\%": 100 * ((A - B) - (2 * A - up - dn)) / (A - B)})
    t5 = pd.DataFrame(rows).set_index("dia")
    t5.loc["média"] = t5.mean(numeric_only=True)
    T["decomposicao"] = _tabular(t5, "lrrrrr", 1, "dia")
    N["pareamentopct"] = _fmt(t5.loc["média", "\\%"], 0)
    N["parup"] = _fmt(t5.loc["média", "só montante"], 1)
    N["pardn"] = _fmt(t5.loc["média", "só jusante"], 1)
    N["parpar"] = _fmt(t5.loc["média", "$\Delta_B$ (par)"], 1)

    # ---------------------------------------------------------------- T6: superfície VPL x f_d
    FD, MW = [0.2, 0.5, 0.8], [100, 200, 400]
    lab = {0.2: "f02", 0.5: "base", 0.8: "f08"}
    labB = {100: "vpl100", 200: "base", 400: "vpl400"}

    def nameD(p, f):
        if p == 200 and f == 0.5:
            return "base"
        if p == 200:
            return lab[f]
        if f == 0.5:
            return labB[p]
        return f"vpl{p}_{ {0.2:'f02', 0.8:'f08'}[f] }"
    def nameB(p, f):
        if f == 0.5:
            return labB[p]
        if p == 200:
            return lab[f]
        return f"vpl{p}_{ {0.2:'f02', 0.8:'f08'}[f] }"
    grid = pd.DataFrame(index=[f"{p} MW" for p in MW], columns=[f"$f_d$={f}".replace(".", ",") for f in FD], dtype=float)
    for p in MW:
        for f in FD:
            vals = []
            for day in DAYS:
                A, C = cost(day, "A", lab[f]), cost(day, "C", lab[f])
                B, D = cost(day, "B", nameB(p, f)), cost(day, "D", nameD(p, f))
                vals.append(((A - D) - ((A - B) + (A - C))) / 1e3)
            grid.loc[f"{p} MW", f"$f_d$={f}".replace(".", ",")] = float(np.mean(vals))
    T["superficie"] = _tabular(grid, "lrrr", 1, "VPL/terminal")

    # ---------------------------------------------------------------- T7: dias mensais
    mon = df_all[df_all["axis"] == "monthly"]
    piv = mon.pivot_table(index="day_key", columns="case", values=["cost_total", "curtailment_mwh", "startups_total"])
    c = piv["cost_total"]
    t7 = pd.DataFrame({"A (k\\$)": c["A"] / 1e3, "$\Delta_B$": (c["A"] - c["B"]) / 1e3, "$\Delta_C$": (c["A"] - c["C"]) / 1e3,
                       "$\Delta_D$": (c["A"] - c["D"]) / 1e3})
    t7["$I$"] = t7["$\Delta_D$"] - (t7["$\Delta_B$"] + t7["$\Delta_C$"])
    t7["partidas A"] = piv["startups_total"]["A"]
    t7["partidas D"] = piv["startups_total"]["D"]
    t7.index = [f"mês {i[1:]}" for i in t7.index]
    t7.loc["média"] = t7.mean(numeric_only=True)
    T["mensais"] = _tabular(t7, "lrrrrrrr", 1, "dia")
    N["mensalcusto"] = _fmt(100 * (1 - c["D"].sum() / c["A"].sum()), 1)
    N["mensalcurt"] = _fmt(100 * (1 - piv["curtailment_mwh"]["D"].sum() / piv["curtailment_mwh"]["A"].sum()), 1)
    N["mensalsu"] = _fmt(100 * (1 - piv["startups_total"]["D"].sum() / piv["startups_total"]["A"].sum()), 1)
    N["mensalIneg"] = str(int((t7["$I$"].iloc[:-1] < 0).sum()))

    # ---------------------------------------------------------------- T8: condições iniciais
    rows = []
    for day in DAYS:
        b = {c: d.loc[f"{day}__{c}__base"] for c in "ABCD"}
        h = {c: d.loc[f"{day}__{c}__hist"] for c in "ABCD"}
        Ib = (b["A"].cost_total - b["D"].cost_total) - ((b["A"].cost_total - b["B"].cost_total) + (b["A"].cost_total - b["C"].cost_total))
        Ih = (h["A"].cost_total - h["D"].cost_total) - ((h["A"].cost_total - h["B"].cost_total) + (h["A"].cost_total - h["C"].cost_total))
        rows.append({"dia": DAY_PT[day], "part. A": b["A"].startups_total, "part. A herd.": h["A"].startups_total,
                     "part. D": b["D"].startups_total, "part. D herd.": h["D"].startups_total,
                     "$I$": Ib / 1e3, "$I$ herd.": Ih / 1e3})
    t8 = pd.DataFrame(rows).set_index("dia")
    t8.loc["média"] = t8.mean(numeric_only=True)
    T["condicoes"] = _tabular(t8, "lrrrrrr", 1, "dia")
    m8 = t8.loc["média"]
    N["redsubase"] = _fmt(100 * (1 - m8["part. D"] / m8["part. A"]), 0)
    N["redsuhist"] = _fmt(100 * (1 - m8["part. D herd."] / m8["part. A herd."]), 0)
    N["Ibase"] = _fmt(m8["$I$"], 1)
    N["Ihist"] = _fmt(m8["$I$ herd."], 1)

    # ---------------------------------------------------------------- T9: baseline plano vs nativo
    rows = []
    for day in DAYS:
        A, C, D = cost(day, "A"), cost(day, "C"), cost(day, "D")
        An, Bn = cost(day, "A", "native"), cost(day, "B", "native")
        B = cost(day, "B")
        rows.append({"dia": DAY_PT[day], "$\Delta_C$ plano": (A - C) / 1e3, "$\Delta_C$ nativo": (An - C) / 1e3,
                     "$\Delta_D$ plano": (A - D) / 1e3, "$\Delta_D$ nativo": (An - D) / 1e3,
                     "$I$ plano": ((A - D) - ((A - B) + (A - C))) / 1e3,
                     "$I$ nativo": ((An - D) - ((An - Bn) + (An - C))) / 1e3})
    t9 = pd.DataFrame(rows).set_index("dia")
    t9.loc["média"] = t9.mean(numeric_only=True)
    T["baseline"] = _tabular(t9, "lrrrrrr", 1, "dia")
    N["dCplano"] = _fmt(t9.loc["média", "$\Delta_C$ plano"], 1)
    N["dCnativo"] = _fmt(t9.loc["média", "$\Delta_C$ nativo"], 1)

    # ---------------------------------------------------------------- T10: custo de migração
    rows = []
    for day in DAYS:
        A, B = cost(day, "A"), cost(day, "B")
        r = {"dia": DAY_PT[day]}
        for lbl, tag in (("base", "base"), ("$\\times$0,5", "mig05"), ("$\\times$2", "mig20")):
            C, D = cost(day, "C", tag), cost(day, "D", tag)
            r[f"$\Delta_C$ {lbl}"] = (A - C) / 1e3
            r[f"$I$ {lbl}"] = ((A - D) - ((A - B) + (A - C))) / 1e3
        rows.append(r)
    t10 = pd.DataFrame(rows).set_index("dia")
    t10.loc["média"] = t10.mean(numeric_only=True)
    T["migracao"] = _tabular(t10, "lrrrrrr", 1, "dia")

    # ---------------------------------------------------------------- T11: SOC inicial da VPL
    rows = []
    for day in DAYS:
        A, B, D = cost(day, "A"), cost(day, "B"), cost(day, "D")
        Bs, Ds = cost(day, "B", "soc50"), cost(day, "D", "soc50")
        rows.append({"dia": DAY_PT[day],
                     r"$\Delta_B$ livre": (A - B) / 1e3, r"$\Delta_B$ 50\%": (A - Bs) / 1e3,
                     r"$\Delta_D$ livre": (A - D) / 1e3, r"$\Delta_D$ 50\%": (A - Ds) / 1e3,
                     r"$I$ livre": ((A - D) - ((A - B) + (A - C0(day)))) / 1e3,
                     r"$I$ 50\%": ((A - Ds) - ((A - Bs) + (A - C0(day)))) / 1e3})
    t11 = pd.DataFrame(rows).set_index("dia")
    t11.loc["média"] = t11.mean(numeric_only=True)
    T["soc"] = _tabular(t11, "lrrrrrr", 1, "dia")
    m11 = t11.loc["média"]
    N["socdB"] = _fmt(m11[r"$\Delta_B$ 50\%"], 1)
    N["socdD"] = _fmt(m11[r"$\Delta_D$ 50\%"], 1)
    N["socI"] = _fmt(m11[r"$I$ 50\%"], 1)

    # ---------------------------------------------------------------- T12: as tres leituras da VPL
    rows = []
    for day in DAYS:
        A, C = cost(day, "A"), cost(day, "C")
        r = {"dia": DAY_PT[day]}
        for lbl, var in (("emerg.", "base"), ("explíc.", "explicit"), ("dedic.", "dedicated")):
            B, D = cost(day, "B", var), cost(day, "D", var)
            r[f"$\\Delta_B$ {lbl}"] = (A - B) / 1e3
            r[f"$I$ {lbl}"] = ((A - D) - ((A - B) + (A - C))) / 1e3
        rows.append(r)
    t12 = pd.DataFrame(rows).set_index("dia")
    t12.loc["média"] = t12.mean(numeric_only=True)
    T["leituras"] = _tabular(t12, "lrrrrrr", 1, "dia")
    m12 = t12.loc["média"]
    N["dBemerg"] = _fmt(m12["$\\Delta_B$ emerg."], 1)
    N["dBexplic"] = _fmt(m12["$\\Delta_B$ explíc."], 1)
    N["dBdedic"] = _fmt(m12["$\\Delta_B$ dedic."], 1)
    N["Idedic"] = _fmt(m12["$I$ dedic."], 1)

    # ------------------------------------------------------- T13: as tres leituras com VPL de 400 MW
    rows = []
    for day in DAYS:
        A, C = cost(day, "A"), cost(day, "C")
        r = {"dia": DAY_PT[day]}
        for lbl, tag in (("emerg.", "vpl400"), ("explíc.", "vpl400explic"), ("dedic.", "vpl400dedic")):
            B, D = cost(day, "B", tag), cost(day, "D", tag)
            r[f"$\\Delta_B$ {lbl}"] = (A - B) / 1e3
            r[f"$I$ {lbl}"] = ((A - D) - ((A - B) + (A - C))) / 1e3
        rows.append(r)
    t13 = pd.DataFrame(rows).set_index("dia")
    t13.loc["média"] = t13.mean(numeric_only=True)
    T["leiturasbig"] = _tabular(t13, "lrrrrrr", 1, "dia")
    m13 = t13.loc["média"]
    N["dBbigemerg"] = _fmt(m13["$\\Delta_B$ emerg."], 1)
    N["dBbigexplic"] = _fmt(m13["$\\Delta_B$ explíc."], 1)
    N["dBbigdedic"] = _fmt(m13["$\\Delta_B$ dedic."], 1)
    N["Ibigemerg"] = _fmt(m13["$I$ emerg."], 1)
    N["Ibigexplic"] = _fmt(m13["$I$ explíc."], 1)
    N["Ibigdedic"] = _fmt(m13["$I$ dedic."], 1)

    # ------------------------------------------------- T14: requisito de reserva comum aos 4 casos
    rows = []
    for day in DAYS:
        A, B = cost(day, "A"), cost(day, "B")
        r = {"dia": DAY_PT[day]}
        for lbl, tag in (("base", "base"), ("$R$ comum", "resref")):
            C, D = cost(day, "C", tag), cost(day, "D", tag)
            r[f"$\\Delta_C$ {lbl}"] = (A - C) / 1e3
            r[f"$\\Delta_D$ {lbl}"] = (A - D) / 1e3
            r[f"$I$ {lbl}"] = ((A - D) - ((A - B) + (A - C))) / 1e3
        rows.append(r)
    t14 = pd.DataFrame(rows).set_index("dia")
    t14.loc["média"] = t14.mean(numeric_only=True)
    T["reserva"] = _tabular(t14, "lrrrrrr", 1, "dia")
    m14 = t14.loc["média"]
    N["dCresref"] = _fmt(m14["$\\Delta_C$ $R$ comum"], 1)
    N["dDresref"] = _fmt(m14["$\\Delta_D$ $R$ comum"], 1)
    N["Iresref"] = _fmt(m14["$I$ $R$ comum"], 1)
    suA = df_all.set_index("scenario")
    su = lambda s: float(suA.loc[s, "startups_total"])
    N["suDresref"] = _fmt(sum(su(f"{d}__D__resref") for d in DAYS) / len(DAYS), 1)
    N["redsuDresref"] = _fmt(100 * (1 - sum(su(f"{d}__D__resref") for d in DAYS)
                                    / sum(su(f"{d}__A__base") for d in DAYS)), 0)

    # ------------------------------------------------- T15/T16: composicao dos data centers
    import pypsa  # local: so este bloco precisa abrir uma rede

    nC = pypsa.Network(cfg_results_dir / f"{DAYS[0]}__C__base.nc")
    meta = nC.meta["vdc"]
    sites_m, wls = meta["sites"], meta["workloads"]

    rows = []
    for k, s in sites_m.items():
        C, pue, fe = float(s["capacity_mw"]), float(s["pue"]), float(s["flex_mw_elec"])
        rows.append({"site": k.replace("_", "\\_"), "barra": s["bus"],
                     "$C_d$ (MW el.)": C, "PUE": pue,
                     "flex. (MW el.)": fe, "flex. (MW TI)": fe / pue,
                     "inflex. (MW el.)": C - fe})
    t15 = pd.DataFrame(rows).set_index("site")
    T["dcsites"] = _tabular(t15, "lcrrrrr", 1, "site")

    w = pd.DataFrame(wls)
    w["janela"] = w["deadline"] - w["arrival"] + 1
    w["tipo"] = np.where(w["janela"] >= 8, "batch (temporal)", "migrável (espacial)")
    w["nsites"] = w["sites"].apply(len)
    t16 = w.groupby("tipo").agg(**{
        "tarefas": ("id", "size"),
        "energia (MWh TI)": ("E_it_mwh", "sum"),
        "janela média (h)": ("janela", "mean"),
        "sites elegíveis": ("nsites", "mean")})
    # contagens como inteiros e celulas sem sentido na linha de total ficam vazias
    t16b = pd.DataFrame({
        "tarefas": t16["tarefas"].astype(int).astype(str),
        "energia (MWh TI)": [_fmt(v, 1) for v in t16["energia (MWh TI)"]],
        "janela média (h)": [_fmt(v, 1) for v in t16["janela média (h)"]],
        "sites elegíveis": [_fmt(v, 0) for v in t16["sites elegíveis"]]},
        index=t16.index)
    t16b.loc["total"] = [str(int(t16["tarefas"].sum())),
                         _fmt(float(t16["energia (MWh TI)"].sum()), 1), "---", "---"]
    T["dctarefas"] = _tabular(t16b, "lrrrr", 1, "tipo")

    s0 = list(sites_m.values())[0]
    N["dcnsites"] = str(len(sites_m))
    N["dccap"] = _fmt(sum(float(s["capacity_mw"]) for s in sites_m.values()), 0)
    N["dcflexit"] = _fmt(float(s0["flex_mw_elec"]) / float(s0["pue"]), 1)
    N["dcrampa"] = _fmt(0.5 * float(s0["flex_mw_elec"]) / float(s0["pue"]), 1)
    N["dcinflexdia"] = _fmt(sum(float(s["capacity_mw"]) - float(s["flex_mw_elec"])
                                for s in sites_m.values()) * 24 / 1e3, 1)
    N["dcelet"] = _fmt(float(w["E_it_mwh"].sum()) * float(s0["pue"]) / 1e3, 1)
    N["dctotaldia"] = _fmt((sum(float(s["capacity_mw"]) - float(s["flex_mw_elec"])
                                for s in sites_m.values()) * 24
                            + float(w["E_it_mwh"].sum()) * float(s0["pue"])) / 1e3, 1)
    N["dcutil"] = _fmt(float(meta.get("flex_utilization", cfg["vdc"]["flex_utilization"])), 2)
    sw = d.loc[f"{DAYS[0]}__C__base"]
    N["dcatraso"] = _fmt(sw.vdc_mean_delay_h, 1)
    N["dcmigcusto"] = _fmt(sw.cost_migration / 1e3, 1)

    # ---------------------------------------------------------------- números soltos
    g = df_all[df_all["axis"] == "base"]
    for c in "ABCD":
        sub = g[g["case"] == c]
        N[f"custo{c}"] = _fmt(sub["cost_total"].mean() / 1e3, 1)
        N[f"part{c}"] = _fmt(sub["startups_total"].mean(), 1)
        N[f"curt{c}"] = _fmt(sub["curtailment_pct"].mean(), 1)
        N[f"emis{c}"] = _fmt(sub["co2_t"].mean() / 1e3, 1)
    A = g[g["case"] == "A"]
    for c in "BCD":
        N[f"red{c}"] = _fmt(100 * (1 - g[g["case"] == c]["cost_total"].mean() / A["cost_total"].mean()), 1)
    N["redcurtD"] = _fmt(100 * (1 - g[g["case"] == "D"]["curtailment_mwh"].mean() / A["curtailment_mwh"].mean()), 1)
    N["redsuD"] = _fmt(100 * (1 - g[g["case"] == "D"]["startups_total"].mean() / A["startups_total"].mean()), 0)
    N["ncen"] = str(len(df_all))
    N["tmed"] = _fmt(df_all["time_milp_s"].median(), 0)
    N["tmax"] = _fmt(df_all["time_milp_s"].max(), 0)
    N["gapmax"] = f"{df_all['mip_gap'].max() * 100:.3f}".replace(".", ",")
    sw = d.loc["spring_wind__C__base"]
    N["vdcespacial"] = _fmt(sw.vdc_spatial_shift_pct, 1)
    N["vdctemporal"] = _fmt(sw.vdc_temporal_shift_pct, 1)
    N["vdcit"] = _fmt(sw.vdc_energy_it_mwh, 0)
    return T, N


def main():
    cfg = load_config()
    df = pd.read_csv(results_dir(cfg) / "metrics.csv")
    T, N = build(df, cfg)
    out = ROOT / "docs" / "relatorio"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tabelas.tex").write_text(
        "\n".join(f"\\newcommand{{\\tab{k}}}{{%\n{v}}}\n" for k, v in T.items()), encoding="utf-8")
    (out / "numeros.tex").write_text(
        "\n".join(f"\\newcommand{{\\num{k}}}{{{v}}}" for k, v in N.items()), encoding="utf-8")
    print(f"{len(T)} tabelas e {len(N)} números em {out}")


if __name__ == "__main__":
    main()
