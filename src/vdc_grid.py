"""
vdc_grid.py — sensibilidade cruzada do VDC: tamanho × custo da flexibilidade.

A grade principal varia um eixo por vez e, por isso, não enxerga a interação
entre **quanta** flexibilidade existe e **quanto custa** exercê-la. As duas
dimensões são as que um operador de data center de fato negocia:

    f_d ∈ {0,2; 0,4; 0,6; 0,8}          → 120, 240, 360, 480 MW flexíveis
    custo de migração × {0; 1; 2; 4}    → grátis, 5–20, 10–40, 20–80 $/MWh (TI)

O baseline (sem camada VDC) depende de ``f_d`` — a carga plana dos data centers
é dimensionada pela mesma energia dos workloads — mas não do custo de migração,
que só existe quando há transferência. Por isso são 4 baselines e 16 casos com
VDC por dia.

A leitura que a grade permite, e que nenhum corte unidimensional dá: os dois eixos
são **aproximadamente separáveis**. A fração do benefício que sobrevive ao custo de
migração mais alto fica em torno de 60 % em toda a faixa de capacidade (67, 54, 62,
61 %), e o retorno por MW varia pouco entre 120 e 480 MW. Dimensionar a
flexibilidade e negociar o custo de exercê-la podem, portanto, ser decisões
independentes.

Convém registrar a hipótese que os dados NÃO sustentaram, porque é intuitiva: como
o custo de migração incide apenas sobre o deslocamento espacial — postergar dentro
do próprio site é grátis —, seria de esperar que ampliar a capacidade deslocasse o
ganho marginal para o mecanismo pago e, com isso, aumentasse a exposição ao preço.
Com dois pontos a série sugere exatamente isso (67 % → 54 %); com os quatro, a
tendência desaparece. A proporção entre os dois mecanismos se mantém estável
conforme a camada cresce.

Uso::

    python src/vdc_grid.py                      # tabela + figura do que já existe
    python src/vdc_grid.py --out grade.md       # escreve em outro caminho
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_profile, results_dir  # noqa: E402

FD = [20, 40, 60, 80]                       # f_d × 100
KMIG = ["000", "100", "200", "400"]         # multiplicador × 100
KLABEL = {"000": "grátis", "100": "5–20", "200": "10–40", "400": "20–80"}
SITES_MW = 3 * 200                          # capacidade elétrica total dos três sites


def read_grid(results: Path) -> pd.DataFrame:
    """Lê os custos da grade dos logs de execução.

    Os cenários entram em ``metrics.csv`` com ``axis = vdc_grid``, mas os logs
    são a fonte mais direta enquanto a grade está sendo construída (evita
    reconstruir o CSV inteiro a cada bloco).
    """
    rows = []
    for f in glob.glob(str(results / "grid_f*.log")):
        for ln in open(f, encoding="utf-8", errors="ignore"):
            m = re.match(r"\[(\w+)__G(\d)__f(\d+)_?k?(\d+)?\S*\] obj=([\d,]+)", ln)
            if not m:
                continue
            rows.append({"dia": m.group(1), "tipo": m.group(2), "fd": int(m.group(3)),
                         "k": m.group(4), "custo": float(m.group(5).replace(",", ""))})
    return pd.DataFrame(rows)


def benefit_table(g: pd.DataFrame) -> pd.DataFrame:
    """Benefício do VDC (baseline − com VDC), média entre os dias, por (f_d, custo)."""
    base = {(r.dia, r.fd): r.custo for r in g[g.tipo == "0"].itertuples()}
    out = {}
    for fd in FD:
        row = {}
        for k in KMIG:
            vals = [base[(r.dia, fd)] - r.custo
                    for r in g[(g.tipo == "1") & (g.fd == fd) & (g.k == k)].itertuples()
                    if (r.dia, fd) in base]
            row[KLABEL[k]] = float(np.mean(vals)) if vals else np.nan
        out[fd] = row
    t = pd.DataFrame(out).T
    t.index = [f"{fd / 100:.1f} ({SITES_MW * fd // 100} MW)" for fd in t.index]
    t.index.name = "f_d (MW flexíveis)"
    return t


def figure(t: pd.DataFrame, fdir: Path) -> None:
    """Duas vistas da superfície: nível por tamanho e sensibilidade ao custo."""
    import matplotlib.pyplot as plt
    from plotstyle import INK2, PALETTE, W2, ieee_style, save

    ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 2.4))

    mw = [SITES_MW * fd // 100 for fd in FD]
    for i, col in enumerate(t.columns):
        ax1.plot(mw, t[col].to_numpy() / 1e3, "o-", color=PALETTE[i], ms=3, label=col)
    ax1.set_xlabel("capacidade flexível (MW)")
    ax1.set_ylabel("benefício do VDC (k$/dia)")
    ax1.legend(title="custo de migração\n($/MWh de TI)", loc="upper left")

    # Perda relativa ao caso de migração gratuita: mostra a interação entre eixos
    rel = t.div(t[KLABEL["000"]], axis=0) * 100.0
    for i, col in enumerate(t.columns[1:], start=1):
        ax2.plot(mw, rel[col].to_numpy(), "o-", color=PALETTE[i], ms=3, label=col)
    ax2.axhline(100, color=INK2, lw=0.6, ls=":")
    ax2.set_xlabel("capacidade flexível (MW)")
    ax2.set_ylabel("% do benefício com migração grátis")
    ax2.set_ylim(top=105)

    fig.tight_layout()
    save(fig, fdir, "fig9_vdc_grid")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Grade cruzada do VDC: tamanho × custo.")
    ap.add_argument("--profile", default="config/realistic.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_profile(args.profile)
    rdir = results_dir(cfg)
    g = read_grid(rdir)
    if g.empty:
        raise SystemExit(f"Nenhum log de grade em {rdir}. Rode os blocos primeiro.")
    t = benefit_table(g)

    lines = ["# Sensibilidade cruzada do VDC — tamanho × custo da flexibilidade",
             "",
             f"Perfil `{args.profile}`; média dos dias representativos disponíveis. "
             "Benefício em $/dia sobre o caso sem camada VDC, com o baseline redimensionado "
             "para cada `f_d`.",
             "",
             t.round(0).to_markdown(),
             ""]

    # Interação entre os eixos: quanto do benefício sobrevive ao custo mais alto
    surv = (t[KLABEL["400"]] / t[KLABEL["000"]] * 100.0).round(0)
    lines += ["## Interação entre os eixos",
              "",
              "Fração do benefício que sobrevive ao custo de migração mais alto (20–80 $/MWh), "
              "relativa ao caso de migração gratuita:",
              "",
              surv.to_frame("% do benefício com migração grátis").to_markdown(),
              "",
              "A sobrevivência é aproximadamente **invariante ao tamanho**: cerca de 60 % do "
              "benefício resiste ao custo mais alto, em toda a faixa de capacidade. O custo de "
              "migração incide apenas sobre o deslocamento espacial — postergar dentro do "
              "próprio site é grátis —, e a proporção entre os dois mecanismos se mantém "
              "estável conforme a camada cresce.",
              "",
              "A implicação prática é que **os dois eixos são aproximadamente separáveis**: "
              "dimensionar a capacidade flexível e negociar o custo de migração podem ser "
              "decisões independentes, sem que uma altere materialmente o retorno da outra.",
              ""]

    mw = np.array([SITES_MW * fd // 100 for fd in FD], dtype=float)
    per_mw = t.div(mw, axis=0)
    lines += ["## Retorno por MW instalado",
              "",
              "Benefício por MW de capacidade flexível ($/MW-dia). Retornos aproximadamente "
              "constantes indicam que a camada escala bem na faixa estudada.",
              "",
              per_mw.round(1).to_markdown(),
              ""]

    out = Path(args.out) if args.out else rdir / "vdc_grid.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[escrito em {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}]")

    if not args.no_fig:
        from common import figures_dir
        figure(t, figures_dir(cfg))


if __name__ == "__main__":
    main()
