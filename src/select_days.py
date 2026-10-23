"""
select_days.py — seleção automática dos dias representativos (§8.1 do brief).

Segue a caracterização de rampa da demanda líquida da tese (Seção 5.2.1.1):

1. Constrói a série anual de **demanda líquida** do sistema já na configuração
   estressada: carga (Área 2 × 1,40) menos a disponibilidade VRE (parque
   existente + acréscimo na Área 3), com o parque VRE escalado por um fator
   anual único que leva a penetração alvo (energia disponível / carga) ao valor
   do caso base.
2. Aplica a **Transformada Discreta de Fourier (DFT)** à série anual para
   caracterizar as periodicidades dominantes (diária, semanal, sazonal) —
   reportadas como contexto no relatório.
3. Calcula, por dia, as **rampas máximas** da demanda líquida (1 h e 3 h),
   o excedente VRE, a carga de pico e a energia diária.
4. Seleciona automaticamente quatro dias, um por critério:
   * *spring_wind*  — eólica alta + carga baixa (primavera, MAM): maior
     excedente VRE diário (curtailment potencial máximo);
   * *summer_peak*  — pico de carga de verão (JJA);
   * *solar_ramp*   — rampa de 3 h mais acentuada da demanda líquida no
     outono/inverno (SON + DJF), típica do pôr do sol;
   * *median*       — dia cuja energia de demanda líquida é a mais próxima da
     mediana anual, excluídos fins de semana.

A escolha é gravada em ``results/days_report.md`` para confirmação manual;
a decisão final entra em ``config/base.yaml`` (bloco ``days``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_config, results_dir  # noqa: E402
from rts_import import load_rts_data  # noqa: E402

SEASONS = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn_winter": [9, 10, 11, 12, 1, 2],
}


def annual_net_demand(cfg: dict) -> pd.DataFrame:
    """Séries anuais (MW) de carga estressada, VRE disponível e demanda líquida."""
    data = load_rts_data(ROOT / cfg["paths"]["rts"])
    s = cfg["stress"]
    load = data.load_area.copy()
    load[str(s["load_area"])] *= float(s["load_factor"])
    load_total = load.sum(axis=1)

    gen = data.gen
    vre_cats = set(cfg["vre"]["carriers"])
    vre = gen[gen["Category"].isin(vre_cats)]
    series = data.gen_series[vre.index].clip(lower=0.0)
    series = series.where(series.le(vre["PMax MW"], axis=1), vre["PMax MW"], axis=1)

    # Acréscimo da Área 3 (mesmo plano de layers.plan_added_vre), com perfis existentes
    from layers import plan_added_vre

    peak2 = float(data.load_area[str(s["load_area"])].max())
    mw_added = (float(s["load_factor"]) - 1.0) * peak2
    gens = pd.DataFrame({"p_nom": vre["PMax MW"], "bus": vre["Bus ID"], "carrier": vre["Category"],
                         "area": vre["Bus ID"].map(data.bus["Area"]).astype(int)})
    added = pd.Series(0.0, index=series.index)
    for item in plan_added_vre(cfg, gens, mw_added):
        src = item["profile_from"]
        added += series[src] / float(vre.at[src, "PMax MW"]) * item["p_nom_mw"]

    vre_avail = series.sum(axis=1) + added
    factor = float(cfg["vre"]["target_share"]) * load_total.sum() / vre_avail.sum()
    vre_avail *= factor

    return pd.DataFrame({"load": load_total, "vre_avail": vre_avail, "net": load_total - vre_avail,
                         "vre_scale_annual": factor})


def dft_dominant_periods(x: pd.Series, k: int = 6) -> pd.DataFrame:
    """Períodos (h) dos ``k`` maiores picos do espectro de amplitude (DFT)."""
    y = x.values - x.values.mean()
    amp = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1.0)   # ciclos por hora
    order = np.argsort(amp)[::-1]
    rows = []
    for i in order:
        if freq[i] == 0:
            continue
        rows.append({"period_h": 1.0 / freq[i], "amplitude_mw": 2 * amp[i] / len(y)})
        if len(rows) >= k:
            break
    return pd.DataFrame(rows).round(1)


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Métricas diárias usadas nos critérios de seleção."""
    g = df.groupby(df.index.date)
    out = pd.DataFrame({
        "peak_load_mw": g["load"].max(),
        "energy_load_mwh": g["load"].sum(),
        "energy_net_mwh": g["net"].sum(),
        "min_net_mw": g["net"].min(),
        "vre_surplus_mwh": g.apply(lambda d: (d["vre_avail"] - d["load"]).clip(lower=0).sum()),
        "vre_share_day": g["vre_avail"].sum() / g["load"].sum(),
        "max_ramp_1h_mw": g["net"].apply(lambda s: s.diff().abs().max()),
        "max_ramp_3h_mw": g["net"].apply(lambda s: s.diff(3).abs().max()),
    })
    out.index = pd.to_datetime(out.index)
    out["month"] = out.index.month
    out["weekday"] = out.index.weekday
    return out


def select_days(feat: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Aplica os quatro critérios e devolve {chave: (data, justificativa)}."""
    spring = feat[feat["month"].isin(SEASONS["spring"])]
    summer = feat[feat["month"].isin(SEASONS["summer"])]
    aw = feat[feat["month"].isin(SEASONS["autumn_winter"])]
    weekdays = feat[feat["weekday"] < 5]

    d1 = spring["vre_surplus_mwh"].idxmax()
    d2 = summer["peak_load_mw"].idxmax()
    d3 = aw["max_ramp_3h_mw"].idxmax()
    med = weekdays["energy_net_mwh"].median()
    d4 = (weekdays["energy_net_mwh"] - med).abs().idxmin()
    fmt = lambda d: d.strftime("%Y-%m-%d")
    return {
        "spring_wind": (fmt(d1), f"maior excedente VRE diário na primavera: {feat.at[d1, 'vre_surplus_mwh']:.0f} MWh"),
        "summer_peak": (fmt(d2), f"pico anual de carga no verão: {feat.at[d2, 'peak_load_mw']:.0f} MW"),
        "solar_ramp": (fmt(d3), f"maior rampa de 3 h da demanda líquida (out–fev): {feat.at[d3, 'max_ramp_3h_mw']:.0f} MW"),
        "median": (fmt(d4), f"energia de demanda líquida mais próxima da mediana anual ({med:.0f} MWh)"),
    }


def main():
    cfg = load_config()
    df = annual_net_demand(cfg)
    feat = daily_features(df)
    chosen = select_days(feat)
    dft = dft_dominant_periods(df["net"])

    lines = ["# Seleção de dias representativos\n",
             f"Série anual de demanda líquida (configuração estressada, VRE escalado por "
             f"{df['vre_scale_annual'].iloc[0]:.3f} para {100*cfg['vre']['target_share']:.0f} % de penetração anual).\n",
             "## Periodicidades dominantes (DFT da demanda líquida)\n", dft.to_markdown(index=False), "\n",
             "## Estatísticas diárias (resumo)\n",
             feat[["peak_load_mw", "energy_net_mwh", "vre_surplus_mwh", "max_ramp_1h_mw", "max_ramp_3h_mw"]]
             .describe().round(0).to_markdown(), "\n",
             "## Dias selecionados (confirmar manualmente em config/base.yaml)\n",
             "| chave | data | critério | pico (MW) | excedente VRE (MWh) | rampa 3 h (MW) | share VRE dia |",
             "|---|---|---|---|---|---|---|"]
    for k, (d, why) in chosen.items():
        r = feat.loc[pd.Timestamp(d)]
        lines.append(f"| {k} | {d} | {why} | {r['peak_load_mw']:.0f} | {r['vre_surplus_mwh']:.0f} | "
                     f"{r['max_ramp_3h_mw']:.0f} | {100*r['vre_share_day']:.0f} % |")
    out = results_dir(cfg) / "days_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    feat.to_csv(results_dir(cfg) / "daily_features.csv")
    print("\n".join(lines[-6:]))
    print(f"\nRelatório: {out}")


if __name__ == "__main__":
    main()
