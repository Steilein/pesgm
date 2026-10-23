"""
economics.py — comparação normalizada VPL x VDC (pós-processamento).

Responde à pergunta que a tabela de custo operacional sozinha não responde: o
VPL entrega mais benefício que o VDC em termos absolutos, mas exige CAPEX que o
VDC não exige. Comparar ΔB com ΔC lado a lado é comparar um ativo com um
software. Este módulo normaliza os dois em três denominadores:

1. **Por MW de flexibilidade** — quanto cada camada rende por MW controlável.
2. **Por MWh instalado** (só VPL) — base do cálculo de CAPEX suportado.
3. **Por CAPEX anualizado** — o benefício operacional anual, dividido pelo fator
   de recuperação de capital, dá o CAPEX máximo que o ativo pagaria só com a
   economia de operação. Confrontado com a faixa de referência de mercado
   (``economics.bess_capex_usd_per_kwh``), diz se o VPL se paga sozinho ou se
   depende do valor de adiamento de reforço da transmissão.

Também consolida:

* a **decomposição ΔD = ΔB + ΔC + I**, com o sinal de I (complementaridade ou
  substituição entre as camadas);
* a **elasticidade do VDC ao custo de migração** e o custo em que ΔC se anula
  (``axis = migration_cost``, sete pontos com ``extra_runs.py --migsweep``);
* a **escada de restrição do VPL** (emergent → hybrid → explicit → dedicated):
  quanto custa cada grau de vinculação da regra de chaveamento.

Nada aqui altera o modelo — lê ``results/metrics.csv`` e escreve
``results/economics.md``.

Uso::

    python src/economics.py                 # usa os dias mensais (estimativa anual)
    python src/economics.py --basis days    # usa os 4 dias representativos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_config, load_profile, results_dir  # noqa: E402

CASES = ("A", "B", "C", "D")


# -----------------------------------------------------------------------------
# Dimensionamento das camadas no caso base
# -----------------------------------------------------------------------------
def layer_sizing(cfg: dict) -> dict:
    """MW e MWh de cada camada na variação base, para normalizar os benefícios.

    VPL: potência somada dos dois terminais e energia instalada total.
    VDC: potência flexível = Σ_d f_d · C_d (a energia diária flexível depende de
    ``flex_utilization`` e é reportada só como referência).
    """
    v = cfg["vpl"]
    p_vpl = 2.0 * float(v["p_nom_mw"])                     # dois terminais
    e_vpl = p_vpl * float(v["hours"])

    f = float(cfg["vdc"]["flex_fraction"])
    u = float(cfg["vdc"]["flex_utilization"])
    cap = sum(float(s["capacity_mw"]) for s in cfg["vdc"]["sites"])
    p_vdc = f * cap
    return {"vpl_mw": p_vpl, "vpl_mwh": e_vpl, "vdc_mw": p_vdc, "vdc_mwh_day": u * p_vdc * 24.0}


def deferral_value(df: pd.DataFrame, cfg: dict) -> dict | None:
    """Valor de adiamento de transmissão do VPL, a partir da varredura ``defer_monthly``.

    O benefício operacional é apenas parte do que um ativo de linha virtual
    entrega: a outra parte é a capacidade de transmissão que ele **dispensa**.
    Creditar só a economia de despacho e comparar com o custo total do ativo é
    meia conta — e foi o erro que produziu as razões B/C de 0,2 a 0,3 reportadas
    antes desta função existir.

    A medição é operacional, não conceitual: varre-se a capacidade do corredor
    **com** o VPL e compara-se com o custo de operar **sem** ele no corredor
    original. O maior corte que não piora NENHUM dos períodos é o adiamento
    *firme* — o único que um planejador pode contratar, já que a capacidade
    precisa estar disponível justamente no período crítico.

    O critério "nenhum período piora" é essencial: na média dos 12 meses um corte
    de 400 MW parece viável, mas julho e agosto ficam positivos. Adiamento médio
    não é adiamento contratável.
    """
    sub = df[df["axis"] == "defer_monthly"]
    if sub.empty:
        return None
    piv = sub.set_index(["day_key", "variation", "case"])["cost_total"]
    ref = {d: v for (d, var, c), v in piv.items() if c == "L1"}
    if not ref:
        return None
    s_ref = max(int(var[1:]) for (_, var, c) in piv.index if c == "L1")

    out = []
    for s_test in sorted({int(var[1:]) for (_, var, c) in piv.index if c == "L2"}, reverse=True):
        diffs = {}
        for (d, var, c), v in piv.items():
            if c == "L2" and int(var[1:]) == s_test and d in ref:
                diffs[d] = v - ref[d]
        if not diffs:
            continue
        out.append({"corredor_MW": s_test, "adiado_MW": s_ref - s_test,
                    "periodos": len(diffs), "piores": sum(1 for v in diffs.values() if v > 0),
                    "media_$dia": sum(diffs.values()) / len(diffs),
                    "pior_$dia": max(diffs.values()),
                    "firme": all(v <= 0 for v in diffs.values())})
    if not out:
        return None
    firm = [r for r in out if r["firme"]]
    return {"tabela": pd.DataFrame(out).sort_values("adiado_MW"),
            "adiado_firme_MW": max((r["adiado_MW"] for r in firm), default=0.0),
            "s_nom_ref_MW": s_ref}


def corridor_cost_per_mw_year(cfg: dict) -> float | None:
    """Custo anualizado de reforço do corredor, em $/MW-ano, escalado pelo comprimento.

    O valor de referência em ``economics.corridor_reinforcement_usd_per_mw_year``
    foi derivado para as 72 milhas do CB-1 do RTS-GMLC. O custo de recondutorar é
    proporcional ao comprimento, de modo que aplicá-lo sem escalar a um corredor
    de 1.000 km subestima o reforço em quase uma ordem de grandeza — e a
    comparação com o armazenamento fica enviesada a favor do fio.

    ``corridor.length_km`` (quando presente) reescala o valor a partir do
    comprimento de referência ``economics.corridor_length_miles``.
    """
    econ = cfg["economics"]
    base = econ.get("corridor_reinforcement_usd_per_mw_year")
    if not base:
        return None
    ref_mi = float(econ.get("corridor_length_miles", 72.0))
    km = cfg.get("corridor", {}).get("length_km")
    if not km:
        return float(base)
    return float(base) * (float(km) / 1.609344) / ref_mi


def crf(rate: float, life: int) -> float:
    """Fator de recuperação de capital."""
    if rate == 0:
        return 1.0 / life
    return rate * (1 + rate) ** life / ((1 + rate) ** life - 1)


# -----------------------------------------------------------------------------
# Benefícios por dia
# -----------------------------------------------------------------------------
def benefits(df: pd.DataFrame, axis: str, variation: str = "base") -> pd.DataFrame:
    """ΔB, ΔC, ΔD e I = ΔD − (ΔB + ΔC) por dia, a partir do custo total.

    I > 0: as camadas são complementares (juntas rendem mais que a soma).
    I < 0: são substitutas parciais — ambas aliviam a mesma restrição.
    """
    sub = df[(df["axis"] == axis) & (df["variation"] == variation)]
    if sub.empty:
        return pd.DataFrame()
    piv = sub.set_index(["day_key", "case"])["cost_total"].unstack()
    # Sistemas com rotulagem própria (a escada L1–L6 do perfil realista) não têm
    # os casos A–D: devolve vazio para que o relatório omita as seções que deles
    # dependem, em vez de falhar.
    if not set(CASES) <= set(piv.columns):
        return pd.DataFrame()
    piv = piv.dropna(subset=list(CASES))
    out = pd.DataFrame({
        "dB": piv["A"] - piv["B"],
        "dC": piv["A"] - piv["C"],
        "dD": piv["A"] - piv["D"],
    })
    out["I"] = out["dD"] - out["dB"] - out["dC"]
    out["I_pct_dD"] = 100.0 * out["I"] / out["dD"]
    return out


# -----------------------------------------------------------------------------
# Seções do relatório
# -----------------------------------------------------------------------------
def section_normalized(ben: pd.DataFrame, size: dict, econ: dict, basis: str) -> list[str]:
    yr = float(econ["days_per_year"])
    b, c, d, i = (ben[k].mean() for k in ("dB", "dC", "dD", "I"))

    rows = [
        dict(camada="VPL (B−A)", benef_dia=b, benef_ano_M=b * yr / 1e6, mw=size["vpl_mw"],
             por_MW_ano_k=b * yr / size["vpl_mw"] / 1e3, por_MWh_ano=b * yr / size["vpl_mwh"]),
        dict(camada="VDC (C−A)", benef_dia=c, benef_ano_M=c * yr / 1e6, mw=size["vdc_mw"],
             por_MW_ano_k=c * yr / size["vdc_mw"] / 1e3, por_MWh_ano=np.nan),
        dict(camada="VPL+VDC (D−A)", benef_dia=d, benef_ano_M=d * yr / 1e6,
             mw=size["vpl_mw"] + size["vdc_mw"],
             por_MW_ano_k=d * yr / (size["vpl_mw"] + size["vdc_mw"]) / 1e3, por_MWh_ano=np.nan),
    ]
    tab = pd.DataFrame(rows)

    lines = [f"## 1. Benefício operacional normalizado (base: {basis}, n = {len(ben)} dias)\n",
             "`benef_dia` em $/dia; `benef_ano_M` em M$/ano; `por_MW_ano_k` em k$/MW-ano. "
             "A potência do VPL é a soma dos dois terminais; a do VDC é Σ_d f_d·C_d.\n",
             tab.round(2).to_markdown(index=False) + "\n"]

    ratio_abs = b / c if c else float("nan")
    ratio_mw = (b / size["vpl_mw"]) / (c / size["vdc_mw"]) if c else float("nan")
    lines.append(f"O VPL rende **{ratio_abs:.1f}x** o VDC em termos absolutos e "
                 f"**{ratio_mw:.2f}x** por MW de flexibilidade controlável. "
                 "Os dois números favorecem o VPL, mas nenhum dos dois desconta o CAPEX — ver §2.\n")

    sign = "complementares" if i > 0 else "substitutas parciais"
    lines.append(f"\n### 1b. Decomposição ΔD = ΔB + ΔC + I\n")
    lines.append(f"I = **{i:,.0f} $/dia** ({ben['I_pct_dD'].mean():.1f} % de ΔD): as camadas são "
                 f"**{sign}**. " +
                 ("O valor conjunto excede a soma das partes.\n" if i > 0 else
                  "Ambas aliviam o mesmo congestionamento do corredor, então parte do benefício "
                  "de uma já foi capturado pela outra: dimensionar as duas pela soma dos ganhos "
                  "isolados superestima o resultado.\n"))
    lines.append(ben.round(0).to_markdown() + "\n")
    return lines


def section_capex(ben: pd.DataFrame, size: dict, econ: dict) -> list[str]:
    """CAPEX que o VPL sustenta, líquido de O&M fixo.

    O passo que decide o veredito é descontar o FOM antes de capitalizar: o
    benefício operacional bruto paga primeiro a operação e manutenção do próprio
    ativo, e só o que sobra amortiza o investimento. Como o FOM é proporcional à
    potência instalada e o benefício não cresce na mesma proporção, ignorá-lo
    superestima a viabilidade de forma grosseira.
    """
    yr = float(econ["days_per_year"])
    rate = float(econ["discount_rate"])
    life = int(econ["lifetime_years"])
    b_year = ben["dB"].mean() * yr
    p_kw = size["vpl_mw"] * 1e3
    e_kwh = size["vpl_mwh"] * 1e3
    k = crf(rate, life)

    def supported(fom_per_kw_year: float) -> float:
        """CAPEX suportado em $/kWh, dado um FOM em $/kW-ano."""
        net = b_year - p_kw * float(fom_per_kw_year)
        return max(net, 0.0) / k / e_kwh

    lo_capex, hi_capex = econ["bess_capex_usd_per_kwh"]
    fom = float(econ["bess_fom_usd_per_kw_year"])
    fom_lo, fom_hi = econ.get("bess_fom_range_usd_per_kw_year", [fom, fom])

    lines = ["## 2. CAPEX que o VPL sustenta, líquido de O&M fixo",
             "",
             f"Benefício operacional bruto: **{b_year / 1e6:.2f} M$/ano** para "
             f"{size['vpl_mw']:.0f} MW / {size['vpl_mwh']:.0f} MWh instalados. Desse valor sai "
             "primeiro o O&M fixo do próprio ativo; o resto amortiza o CAPEX a "
             f"{rate:.0%} em {life} anos.",
             ""]

    rows = []
    for f in sorted({float(fom_lo), 28.0, 40.0, float(fom), 0.0}):
        cost = p_kw * f
        rows.append({"FOM $/kW-ano": f, "FOM M$/ano": cost / 1e6,
                     "sobra M$/ano": (b_year - cost) / 1e6,
                     "CAPEX suportado $/kWh": supported(f)})
    lines += [pd.DataFrame(rows).round(1).to_markdown(index=False), ""]

    base = supported(fom)
    lines += ["",
              f"Com o FOM de referência do NREL ({fom:.0f} $/kW-ano): **{base:,.0f} $/kWh** "
              f"suportados, contra {lo_capex}–{hi_capex} $/kWh de custo instalado. "
              f"Mesmo no extremo otimista da literatura ({fom_lo:.0f} $/kW-ano): "
              f"**{supported(fom_lo):,.0f} $/kWh**.",
              ""]

    if supported(fom_lo) >= hi_capex:
        verdict = ("O benefício operacional **basta sozinho** para justificar o VPL em toda a faixa "
                   "de custo de referência.")
    elif supported(fom_lo) >= lo_capex:
        verdict = ("O VPL fica **na fronteira**: fecha a conta apenas na combinação mais favorável "
                   "(FOM baixo e CAPEX na projeção otimista). Reportar viabilidade sem explicitar "
                   "essa combinação seria seletivo.")
    else:
        verdict = ("O benefício operacional **não basta sob nenhuma combinação** de FOM e CAPEX da "
                   "faixa de referência. O VPL, neste dimensionamento e com os serviços que este "
                   "modelo remunera (energia day-ahead e reserva girante), não se paga. Reportar "
                   "ΔB isolado como evidência de viabilidade seria incorreto.")
    lines += [f"> {verdict}", ""]

    lines += ["",
              "**O que este número não inclui.** O modelo remunera energia e reserva girante. "
              "Ficam de fora pagamentos por capacidade, regulação de frequência e demais serviços "
              "ancilares, e o adiamento de reforços de rede fora do corredor estudado. O valor de "
              "adiamento do próprio corredor está quantificado em §5 e é pequeno. O dimensionamento "
              "também não é otimizado: é um cenário. O eixo `vpl_size` mostra que o benefício por "
              "MWh é praticamente plano entre 100 e 200 MW por terminal e decresce acima disso, "
              "de modo que reduzir o VPL não melhora a relação.",
              ""]

    lines += ["",
              "O VDC não entra nesta tabela porque não tem CAPEX comparável: a capacidade elétrica "
              "dos data centers já está instalada e o custo de exercer a flexibilidade (migração) já "
              "está descontado dentro de ΔC. Formalmente sua razão benefício/custo é ilimitada — "
              "motivo pelo qual a comparação honesta entre as camadas é a de §1 (por MW) somada ao "
              "serviço que cada uma presta (§4), e não uma razão B/C.",
              ""]
    return lines


def section_migration(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Elasticidade de ΔC ao custo de migração e custo em que o benefício se anula."""
    from extra_runs import MIGSWEEP

    base = df[(df["axis"] == "base") & (df["variation"] == "base")].set_index(["day_key", "case"])
    mig = df[df["axis"] == "migration_cost"].set_index(["day_key", "variation", "case"])
    lo, hi = cfg["vdc"]["workloads"]["migration_cost_range"]

    rows = []
    for label, k in sorted(MIGSWEEP.items(), key=lambda kv: kv[1]):
        rec = {"variação": label, "multiplicador": k, "custo $/MWh(TI)": f"{lo * k:.1f}–{hi * k:.1f}"}
        dC, dD = [], []
        for day in cfg["days"]:
            try:
                A = base.loc[(day, "A"), "cost_total"]
            except KeyError:
                continue
            try:
                C = base.loc[(day, "C"), "cost_total"] if k == 1.0 else mig.loc[(day, label, "C"), "cost_total"]
                D = base.loc[(day, "D"), "cost_total"] if k == 1.0 else mig.loc[(day, label, "D"), "cost_total"]
            except KeyError:
                continue
            dC.append(A - C)
            dD.append(A - D)
        if not dC:
            continue
        rec["dC"] = float(np.mean(dC))
        rec["dD"] = float(np.mean(dD))
        rows.append(rec)

    if not rows:
        return []
    tab = pd.DataFrame(rows)
    tab["dC_rel_x0"] = 100.0 * tab["dC"] / tab["dC"].iloc[0]

    lines = ["## 3. Sensibilidade do VDC ao custo de migração\n",
             "Média entre os dias representativos. `dC_rel_x0` = ΔC como % do valor com migração "
             "gratuita (x0), que é a cota superior do benefício da camada: o que sobra abaixo de "
             "100 % é o custo de exercer a flexibilidade; o que falta para 100 % de ΔC ser maior é "
             "limite físico (janela, capacidade de site, rampa agregada), não de preço.\n",
             tab.round(1).to_markdown(index=False) + "\n"]

    fit = tab[tab["multiplicador"] > 0]
    if len(fit) >= 3:
        el = np.polyfit(np.log(fit["multiplicador"]), np.log(fit["dC"].clip(lower=1.0)), 1)[0]
        lines.append(f"\nElasticidade de ΔC ao custo de migração: **{el:.3f}** (dlnΔC/dln c) no "
                     "intervalo resolvido.\n")

    # ΔC NÃO tende a zero: a flexibilidade temporal (postergar dentro do próprio
    # site) é gratuita, então nenhum custo de migração a elimina. O último ponto
    # da varredura é a melhor estimativa empírica desse piso; extrapolar uma reta
    # até ΔC = 0 seria ajustar uma curva côncava com assíntota não nula.
    ceiling, floor = tab["dC"].iloc[0], tab["dC"].iloc[-1]
    k_max = tab["multiplicador"].iloc[-1]
    lines.append(f"\n**ΔC não se anula.** Entre migração gratuita e custo x{k_max:.0f} o benefício cai "
                 f"de {ceiling:,.0f} para {floor:,.0f} $/dia ({100 * floor / ceiling:.0f} % do teto) e "
                 "a queda já está achatando. A razão é estrutural: o custo de migração só incide "
                 "sobre o deslocamento **espacial**; postergar carga dentro do próprio data center "
                 "não custa nada. O piso da curva é, portanto, o valor puro da flexibilidade "
                 "temporal, inatingível por preço — e deve coincidir com o degrau L3 da escada "
                 "(§6), que remove a migração por construção em vez de encarecê-la.\n")
    lines.append(f"\nDecomposição implícita, ao custo base: temporal ≈ {floor:,.0f} $/dia (piso); "
                 f"espacial ≈ {tab['dC'][tab['multiplicador'] == 1.0].iloc[0] - floor:,.0f} $/dia "
                 f"(o que o caso base captura acima do piso); espacial com migração gratuita ≈ "
                 f"{ceiling - floor:,.0f} $/dia (teto do que a migração poderia render).\n")
    lines.append("\nA leitura para o artigo: o benefício do VDC é **pouco elástico** ao custo de "
                 "migração. Isso é bom para a robustez do resultado (não depende de calibrar bem um "
                 "parâmetro difícil de observar) e ruim para a magnitude — mostra que o que limita a "
                 "camada não é o preço de mover carga, e sim quanto de carga há para mover.\n")
    return lines


def section_vpl_modes(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Escada de restrição do VPL: quanto custa cada grau de vinculação da regra."""
    order = [("emergent", "base", "nenhuma restrição — VTL emerge do spread de LMP"),
             ("hybrid", "hybrid", "regra só no pico (sigma_t = 1); livre no resto do dia"),
             ("explicit", "explicit", "regra no pico e no vale; modos opostos nas horas livres"),
             ("dedicated", "dedicated", "regra em toda hora que não seja de baixo uso")]
    idx = df.set_index(["day_key", "case", "variation"])

    rows = []
    for mode, var, desc in order:
        rec = {"modo": mode, "regra": desc}
        for case in ("B", "D"):
            vals, vtl, virt = [], [], []
            for day in cfg["days"]:
                try:
                    A = idx.loc[(day, "A", "base"), "cost_total"]
                    r = idx.loc[(day, case, var)]
                except KeyError:
                    continue
                vals.append(A - float(r["cost_total"]))
                vtl.append(float(r.get("vpl_hours_vtl", np.nan)))
                virt.append(float(r.get("vpl_virtual_capacity_mw", np.nan)))
            if vals:
                rec[f"d{case}"] = float(np.mean(vals))
                if case == "B":
                    rec["h_VTL"] = float(np.nanmean(vtl)) if len(vtl) else np.nan
                    rec["cap_virtual_MW"] = float(np.nanmean(virt)) if len(virt) else np.nan
        rows.append(rec)

    tab = pd.DataFrame(rows)
    if "dB" not in tab.columns:          # sem casos B/D nesta base (ex.: escada L1–L6)
        return []
    tab = tab.dropna(subset=["dB"])
    if tab.empty or len(tab) < 2:
        return []
    ref = tab.loc[tab["modo"] == "emergent", "dB"]
    if not ref.empty and ref.iloc[0]:
        tab["custo_da_regra_pct"] = 100.0 * (ref.iloc[0] - tab["dB"]) / ref.iloc[0]

    return ["## 4. Escada de restrição do VPL\n",
            "Média entre os dias representativos, $/dia. `custo_da_regra_pct` = quanto do benefício "
            "do modo emergente se perde ao impor a regra de chaveamento. `h_VTL` = horas em que o "
            "par opera sob a regra; `cap_virtual_MW` = descarga média a jusante nas horas de "
            "corredor saturado.\n",
            tab.round(1).to_markdown(index=False) + "\n",
            "\nA escada mede o preço da **garantia**: o modo emergente rende mais porque o par pode "
            "abandonar o corredor sempre que a arbitragem pagar melhor, mas nada garante ao operador "
            "que a capacidade virtual estará lá na hora do congestionamento. O modo `hybrid` é a "
            "posição intermediária — compromisso de alívio no pico, liberdade no resto — e a "
            "diferença entre ele e `emergent` é o preço dessa garantia.\n"]



def section_deferral(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Deslocamento de capacidade de corredor: resultado operacional e o que ele NÃO prova.

    O modelo é day-ahead — 24 h, previsão perfeita, sem encadeamento entre dias.
    Ele mede **substituição operacional**: com o VPL, operar um corredor menor
    custa menos que operar um maior sem ele. Isso é condição *necessária* para
    adiamento de investimento em transmissão, e não é suficiente.

    O que falta para uma afirmação de adiamento em sentido de planejamento:

    * **Critério.** Planejamento usa adequação na condição crítica, não
      equivalência de custo de despacho. Um UC com curtailment permitido sempre
      resolve — corta mais — de modo que custos podem empatar enquanto o
      atendimento degrada.
    * **Horizonte.** Doze dias amostrais não sustentam firmeza sobre 8760 h com
      contingências. O critério "nenhum período piora" usado aqui é o mais
      exigente que a amostra permite, não o do planejador.
    * **Duração.** Um BESS de 4 h não substitui transporte num evento sustentado
      de vários dias — e o modelo, que reinicia o estado a cada dia, sequer
      enxerga esse tipo de evento.

    Por isso a valoração do deslocamento entra como **cenário superior
    explicitamente condicionado**, separada da razão B/C operacional, que é o
    resultado que este modelo sustenta sem ressalva.
    """
    dv = deferral_value(df, cfg)
    if not dv:
        return []
    econ = cfg["economics"]
    yr = float(econ["days_per_year"])
    size = layer_sizing(cfg)
    c_mw = corridor_cost_per_mw_year(cfg)

    lines = ["## 7. Deslocamento de capacidade de corredor",
             "",
             f"Referência: operar SEM o VPL com o corredor em {dv['s_nom_ref_MW']:.0f} MW. "
             "Teste: operar COM o VPL num corredor menor. A coluna `firme` marca os cortes "
             "que não pioram **nenhum** período da amostra.",
             "",
             dv["tabela"].round(0).to_markdown(index=False),
             ""]

    mw = float(dv["adiado_firme_MW"])
    if mw <= 0:
        lines += ["> Nenhum corte sobrevive a todos os períodos.", ""]
        return lines

    # --- Resultado operacional (sem ressalva) --------------------------------
    ref_mw = dv["s_nom_ref_MW"]
    dm = df[(df["axis"] == "defer_monthly") & (df["variation"] == f"s{ref_mw:.0f}")]
    if not dm.empty and {"L1", "L2"} <= set(dm["case"].unique()):
        piv = dm.set_index(["day_key", "case"])["cost_total"].unstack()
        basis_op = f"12 dias mensais, corredor de {ref_mw:.0f} MW"
    else:
        mo = df[df["axis"] == "ladder_monthly"]
        piv = (mo if not mo.empty else df[df["axis"] == "ladder"]).set_index(
            ["day_key", "case"])["cost_total"].unstack()
        basis_op = "escada (base disponível)"
    if not ({"L1", "L2"} <= set(piv.columns)):
        return lines

    ben_op = float((piv["L1"] - piv["L2"]).dropna().mean()) * yr
    p_kw, e_kwh = size["vpl_mw"] * 1e3, size["vpl_mwh"] * 1e3
    k = crf(float(econ["discount_rate"]), int(econ["lifetime_years"]))
    fom = float(econ["bess_fom_usd_per_kw_year"])
    ben_def = mw * float(c_mw) if c_mw else 0.0

    rows = []
    for capex in econ["bess_capex_usd_per_kwh"]:
        cost = p_kw * fom + e_kwh * float(capex) * k
        rows.append({"CAPEX $/kWh": capex, "custo M$/ano": cost / 1e6,
                     "B/C operacional": ben_op / cost,
                     "B/C c/ deslocamento": (ben_op + ben_def) / cost})

    lines += ["### 7a. Resultado que o modelo sustenta", "",
              f"**Deslocamento operacional de {mw:.0f} MW** de capacidade de corredor: com o VPL, "
              f"operar {ref_mw - mw:.0f} MW custa menos que operar {ref_mw:.0f} MW sem ele, em "
              f"todos os {dv['tabela'].iloc[0]['periodos']:.0f} períodos da amostra. É um resultado "
              "**físico e operacional**, medido, e é condição necessária para adiamento.",
              "",
              f"Benefício operacional: **{ben_op / 1e6:.2f} M$/ano** ({basis_op}).",
              "",
              pd.DataFrame(rows).round(2).to_markdown(index=False),
              ""]

    lines += ["", "### 7b. O que o modelo NÃO sustenta", "",
              f"A coluna `B/C c/ deslocamento` valora os {mw:.0f} MW a "
              f"{c_mw / 1e3:,.0f} k$/MW-ano (= {ben_def / 1e6:.2f} M$/ano) **como se fossem "
              "adiamento contratável de investimento**. Não são, e três razões separam uma coisa "
              "da outra:",
              "",
              "1. **Critério.** Planejamento decide por adequação na condição crítica, não por "
              "equivalência de custo de despacho. Um UC com curtailment permitido sempre resolve — "
              "corta mais —, de modo que o custo pode empatar enquanto o atendimento degrada.",
              "2. **Horizonte.** Doze dias amostrais não sustentam firmeza sobre 8760 h com "
              "contingências N-1.",
              "3. **Duração.** Um BESS de 4 h não substitui transporte num evento sustentado de "
              "vários dias, e o modelo day-ahead — que reinicia o estado a cada dia — não enxerga "
              "esse tipo de evento.",
              "",
              "> Use a coluna operacional como resultado. A coluna com deslocamento é **cota "
              "superior**, e converter deslocamento operacional em adiamento de investimento exige "
              "estudo de planejamento com ano completo, encadeamento entre dias e critério de "
              "adequação — fora do escopo deste modelo.",
              ""]
    return lines



def section_ladder(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Escada de flexibilidade: valor isolado de cada grau de liberdade.

    Cada degrau de ``scenarios.yaml:ladder`` acrescenta um grau de liberdade ao
    anterior, então a diferença entre degraus consecutivos isola o valor daquele
    grau — o que o desenho A/B/C/D não separa (lá o VDC entra inteiro, com as
    flexibilidades temporal e espacial somadas).
    """
    from common import load_scenarios

    try:
        ladder = load_scenarios()["ladder"]
    except (KeyError, FileNotFoundError):
        return []
    # Prefere a base mensal (12 dias, amostra anual não enviesada). Os 4 dias
    # representativos são extremos escolhidos a dedo e inflam o valor anual da
    # flexibilidade em ~28 %: o bastante para inverter uma razão B/C próxima de 1.
    monthly = df[df["axis"] == "ladder_monthly"]
    reuse = {"L1": "A", "L4": "C"}
    if not monthly.empty:
        extra = df[(df["axis"] == "monthly") & (df["case"].isin(reuse.values()))].copy()
        inv = {v: k for k, v in reuse.items()}
        extra["case"] = extra["case"].map(inv)
        sub = pd.concat([monthly, extra], ignore_index=True)
        basis_note = f"12 dias mensais (amostra anual não enviesada)"
    else:
        sub = df[df["axis"] == "ladder"]
        basis_note = "4 dias representativos (extremos — superestima o anual)"
    if sub.empty:
        return []
    piv = sub.set_index(["day_key", "case"])["cost_total"].unstack()
    steps = [k for k in ladder if k in piv.columns]
    piv = piv.dropna(subset=steps)
    if piv.empty or "L1" not in steps:
        return []

    mean = pd.Series({k: (piv["L1"] - piv[k]).mean() for k in steps if k != "L1"})
    rows = [{"degrau": k, "camada": ladder[k]["label"], "benefício $/dia": mean[k],
             "% de L5": 100.0 * mean[k] / mean["L5"] if "L5" in mean.index else np.nan}
            for k in steps if k != "L1"]

    lines = ["## 6. Escada de flexibilidade (6 degraus)",
             "",
             f"Benefício de cada degrau sobre o caso base L1, média entre {len(piv)} dias. "
             f"Base: {basis_note}.",
             "",
             pd.DataFrame(rows).round(1).to_markdown(index=False),
             ""]

    s_nom = float(sub[sub["case"] == "L1"]["corridor_limit_mw"].mean())
    inc = []
    if {"L3", "L4"} <= set(steps):
        inc.append(("deslocamento temporal (dentro do DC)", "L3 − L1", mean["L3"]))
        inc.append(("deslocamento espacial (entre DCs)", "L4 − L3", mean["L4"] - mean["L3"]))
    if "L2" in steps:
        inc.append(("transmissão virtual (VPL híbrido)", "L2 − L1", mean["L2"]))
    if {"L4", "L5"} <= set(steps):
        inc.append(("somar o VPL ao VDC completo", "L5 − L4", mean["L5"] - mean["L4"]))
    if "L6" in steps:
        f = float(ladder["L6"]["over"]["corridor.limit_corridor_factor"])
        inc.append((f"reforço do corredor (+{(f - 1) * 100:.0f} % = +{(f - 1) * s_nom:.0f} MW)",
                    "L6 − L1", mean["L6"]))
    if inc:
        lines += ["", "### 6b. Valor isolado de cada grau de liberdade", "",
                  pd.DataFrame(inc, columns=["grau de liberdade", "diferença", "$/dia"]
                               ).round(1).to_markdown(index=False), ""]

    if {"L2", "L4", "L5"} <= set(steps):
        i_lad = mean["L5"] - mean["L2"] - mean["L4"]
        sign = ("se reforçam" if i_lad > 0 else
                "disputam o mesmo congestionamento: somar os ganhos isolados superestima")
        lines += ["", f"Complementaridade da escada (L5 − L2 − L4): **{i_lad:,.0f} $/dia** — "
                      f"o VPL híbrido e o VDC {sign}.", ""]

    if {"L2", "L6"} <= set(steps):
        f = float(ladder["L6"]["over"]["corridor.limit_corridor_factor"])
        mw_wire = (f - 1) * s_nom

        # Teto do alívio de corredor: reforço grande o bastante para zerar as horas
        # congestionadas (``--wirecurve`` / +400 MW). É o máximo que QUALQUER medida
        # que atue só sobre o congestionamento pode render. Sem ele, interpolar
        # linearmente L2 contra L6 produz um "equivalente em MW de linha" sem
        # sentido, porque extrapola muito além do ponto em que a curva satura.
        ceil_rows = df[df["axis"].isin(["wire_ceiling", "wire_calib_monthly", "wire_calib"])]
        ceiling = None
        if not ceil_rows.empty:
            # referência do teto: o caso sem camadas na mesma base temporal
            if "wire_ceiling" in set(ceil_rows["axis"]):
                base_a = df[(df["axis"] == "ladder_monthly") & (df["case"] == "L1")]
                big = ceil_rows
            else:
                base_a = df[(df["axis"] == ("monthly" if "wire_calib_monthly" in
                                            set(ceil_rows["axis"]) else "base")) &
                            (df["case"] == "A") & (df["variation"] == "base")]
                big = ceil_rows[ceil_rows["variation"] == "f1.80"]
            common = sorted(set(base_a["day_key"]) & set(big["day_key"]))
            if common:
                a = base_a.set_index("day_key")["cost_total"]
                w = big.set_index("day_key")["cost_total"]
                ceiling = float((a[common] - w[common]).mean())

        if ceiling and ceiling > 0:
            ratio = mean["L2"] / ceiling
            share = 100.0 * (1.0 - ceiling / mean["L2"])
            lines += ["", "### 6d. O VPL substitui transmissão?", "",
                      f"Teto do alívio de corredor (reforço que zera as horas congestionadas): "
                      f"**{ceiling:,.0f} $/dia**. É tudo que existe para ganhar eliminando por "
                      f"completo a restrição do corredor. O VPL híbrido rende "
                      f"**{mean['L2']:,.0f} $/dia** — **{ratio:.1f}x** o teto.", "",
                      f"> **{share:.0f} % do benefício do VPL não é atribuível a alívio de "
                      "transmissão.** Nenhum reforço de fio, de qualquer tamanho, reproduz o valor "
                      "do par de BESS: o que ele entrega vem de arbitragem de energia, partidas "
                      "térmicas evitadas e reserva. O arranjo de fato alivia o corredor no pico — a "
                      "capacidade virtual medida é real — mas chamar isso de linha virtual descreve "
                      "o mecanismo e atribui mal a origem do ganho.", ""]
        else:
            lines += ["", f"> Reforço de {mw_wire:.0f} MW: {mean['L6']:,.0f} $/dia, contra "
                          f"{mean['L2']:,.0f} $/dia do VPL híbrido. Para exprimir o VPL em MW de "
                          "linha é preciso o teto do alívio (`extra_runs.py --wirecurve`): sem ele, "
                          "interpolar linearmente extrapola além da saturação da curva e produz um "
                          "equivalente sem sentido físico.", ""]

        # --- razão benefício/custo das duas alternativas, na MESMA base -------
        econ = cfg["economics"]
        yr = float(econ["days_per_year"])
        c_wire_mw = corridor_cost_per_mw_year(cfg)
        if c_wire_mw and mw_wire > 0:
            size = layer_sizing(cfg)
            fom = float(econ["bess_fom_usd_per_kw_year"])
            capex_hi = float(econ["bess_capex_usd_per_kwh"][1])
            k = crf(float(econ["discount_rate"]), int(econ["lifetime_years"]))

            cost_wire = mw_wire * float(c_wire_mw)
            ben_wire = mean["L6"] * yr
            cost_vpl_om = size["vpl_mw"] * 1e3 * fom
            cost_vpl_cap = size["vpl_mwh"] * 1e3 * capex_hi * k
            ben_vpl = mean["L2"] * yr

            bc = pd.DataFrame([
                {"alternativa": f"Reforço do corredor (+{mw_wire:.0f} MW)",
                 "benefício M$/ano": ben_wire / 1e6, "custo M$/ano": cost_wire / 1e6,
                 "B/C": ben_wire / cost_wire},
                {"alternativa": "VPL híbrido — só O&M fixo",
                 "benefício M$/ano": ben_vpl / 1e6, "custo M$/ano": cost_vpl_om / 1e6,
                 "B/C": ben_vpl / cost_vpl_om},
                {"alternativa": "VPL híbrido — O&M + CAPEX",
                 "benefício M$/ano": ben_vpl / 1e6,
                 "custo M$/ano": (cost_vpl_om + cost_vpl_cap) / 1e6,
                 "B/C": ben_vpl / (cost_vpl_om + cost_vpl_cap)},
            ])
            lines += ["", "### 6c. Razão benefício/custo: fio contra bateria", "",
                      "Mesma base temporal nos dois lados. O reforço é anualizado por CRF a "
                      f"{econ.get('transmission_lifetime_years', 40)} anos, para um corredor de "
                      f"{cfg.get('corridor', {}).get('length_km', econ.get('corridor_length_miles', 72) * 1.609):.0f} km "
                      f"({c_wire_mw / 1e3:,.0f} k$/MW-ano); o VPL a "
                      f"{econ['lifetime_years']} anos, com CAPEX de {capex_hi:.0f} $/kWh e FOM de "
                      f"{fom:.0f} $/kW-ano.", "",
                      bc.round(2).to_markdown(index=False), ""]
            if bc["B/C"].iloc[0] > 1 and bc["B/C"].iloc[2] < 1:
                lines += ["", "> **O reforço convencional se paga e o VPL não.** É a comparação que "
                              "o degrau L6 existe para permitir: as duas alternativas entregam "
                              "alívio do mesmo corredor, medidas no mesmo modelo e na mesma base "
                              "temporal, e só uma cobre o próprio custo.", ""]
    return lines


def section_corridor(ben: pd.DataFrame, df: pd.DataFrame, cfg: dict, econ: dict) -> list[str]:
    """Valoração da capacidade virtual contra o custo de reforço do corredor."""
    ref = econ.get("corridor_reinforcement_usd_per_mw_year")
    sub = df[(df["axis"] == "base") & (df["variation"] == "base") & (df["case"] == "B")]
    if sub.empty:
        return []
    cap = float(sub["vpl_virtual_capacity_mw"].mean())
    lines = ["## 5. Capacidade virtual de transmissão\n",
             f"Capacidade virtual média do VPL nas horas de corredor saturado: **{cap:.1f} MW** "
             f"(limite físico do corredor: {float(sub['corridor_limit_mw'].mean()):.0f} MW).\n"]
    if ref:
        yr = float(econ["days_per_year"])
        value = cap * float(ref)
        b = ben["dB"].mean() * yr
        lines.append(f"\nValorada ao custo de reforço de referência ({float(ref):,.0f} $/MW-ano), "
                     f"essa capacidade vale **{value / 1e6:.2f} M$/ano**, contra {b / 1e6:.2f} M$/ano "
                     f"de benefício operacional — total **{(value + b) / 1e6:.2f} M$/ano**.\n")
    else:
        lines.append("\n> `economics.corridor_reinforcement_usd_per_mw_year` não está definido em "
                     "`config/base.yaml`. Sem essa referência, a capacidade virtual fica reportada "
                     "em MW mas não entra na comparação econômica — e é exatamente a parcela que "
                     "falta para o VPL fechar a conta em §2.\n")
    return lines


# -----------------------------------------------------------------------------
def build_report(cfg: dict, basis: str) -> str:
    """Monta o relatório, omitindo as seções para as quais não há dados.

    O caso base e o perfil realista têm estruturas diferentes: o primeiro tem a
    grade A/B/C/D com todos os eixos de sensibilidade, o segundo só a escada
    L1–L6. Em vez de exigir a estrutura completa, cada seção se omite quando a
    sua fonte não existe, de modo que o mesmo módulo serve aos dois sistemas.
    """
    rdir = results_dir(cfg)
    df = pd.read_csv(rdir / "metrics.csv")
    econ = cfg["economics"]
    size = layer_sizing(cfg)
    profile = cfg.get("_profile")

    label = ("12 dias mensais (estimativa anual não viesada)" if basis == "monthly"
             else "4 dias representativos (extremos)")
    lines = ["# Comparação econômica normalizada",
             "",
             f"Gerado por `src/economics.py` a partir de `{rdir.name}/metrics.csv`"
             + (f", perfil `{profile}`" if profile else "") + ".",
             f"Base temporal: {label}; anualização por {econ['days_per_year']} dias.",
             "",
             f"Dimensionamento: VPL {size['vpl_mw']:.0f} MW / {size['vpl_mwh']:.0f} MWh "
             f"(dois terminais); VDC {size['vdc_mw']:.0f} MW flexíveis.",
             ""]

    # Seções que dependem da grade A/B/C/D — ausentes no perfil realista.
    ben = benefits(df, "monthly" if basis == "monthly" else "base")
    if not ben.empty:
        lines += section_normalized(ben, size, econ, label)
        lines += section_capex(ben, size, econ)
        lines += section_corridor(ben, df, cfg, econ)
    else:
        lines += ["> Sem cenários A/B/C/D nesta base: as seções de benefício "
                  "normalizado e de CAPEX são omitidas. Este sistema é avaliado "
                  "pela escada de flexibilidade.", ""]
    lines += section_migration(df, cfg)
    lines += section_vpl_modes(df, cfg)
    lines += section_ladder(df, cfg)
    lines += section_deferral(df, cfg)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Comparação econômica normalizada VPL x VDC.")
    ap.add_argument("--basis", choices=["monthly", "days"], default="monthly",
                    help="monthly = 12 dias mensais (padrão); days = 4 dias representativos")
    ap.add_argument("--profile", default=None,
                    help="perfil a sobrepor ao base.yaml (ex.: config/realistic.yaml). "
                         "Define também o diretório de resultados lido.")
    ap.add_argument("--out", default=None, help="caminho do markdown de saída")
    args = ap.parse_args(argv)

    cfg = load_profile(args.profile)
    text = build_report(cfg, args.basis)
    out = Path(args.out) if args.out else results_dir(cfg) / "economics.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[escrito em {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}]")


if __name__ == "__main__":
    main()
