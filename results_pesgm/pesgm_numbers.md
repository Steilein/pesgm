# Números do texto do PES GM

Gerado por `papers/pesgm/scripts/pesgm_numbers.py` a partir de `results_pesgm/metrics.csv`.

## 1. Corredor como restrição econômica (caso A)

Spread máximo de LMP entre terminais > 0.5 $/MWh em **7 de 16** dias: m05, m06, m07, m08, m09, solar_ramp, summer_peak.

Sem separação: m01, m02, m03, m04, m10, m11, m12, median, spring_wind.

| day_key     |   lmp_spread_max |   corridor_congested_hours |
|:------------|-----------------:|---------------------------:|
| m01         |             0    |                          7 |
| m02         |             0    |                          7 |
| m03         |             0    |                          9 |
| m04         |             0    |                          0 |
| m05         |            22.15 |                          7 |
| m06         |            26    |                          8 |
| m07         |            28.65 |                         10 |
| m08         |            35.13 |                          9 |
| m09         |            22.45 |                          7 |
| m10         |             0    |                          6 |
| m11         |             0    |                          2 |
| m12         |             0    |                          6 |
| spring_wind |             0    |                          9 |
| summer_peak |            23.3  |                         11 |
| solar_ramp  |            21.26 |                          7 |
| median      |             0    |                          8 |

## 2. Alívio de corte (p.p.), doze dias mensais

Média geral: VPL 1.79, H2 2.13.

| separa                 |   VPL_pp |   H2_pp |
|:-----------------------|---------:|--------:|
| sem separação          |     1.45 |    2.31 |
| corredor separa preços |     2.27 |    1.89 |

Dias mensais com separação: 5; sem: 7.

## 3. Eletrolisador (casos com H2, doze dias mensais)

| case   |   ('h2_on_h', 'min') |   ('h2_on_h', 'max') |   ('h2_standby_h', 'min') |   ('h2_standby_h', 'max') |   ('h2_startups', 'min') |   ('h2_startups', 'max') |   ('h2_standby_mwh', 'min') |   ('h2_standby_mwh', 'max') |   ('sb_frac', 'min') |   ('sb_frac', 'max') |
|:-------|---------------------:|---------------------:|--------------------------:|--------------------------:|-------------------------:|-------------------------:|----------------------------:|----------------------------:|---------------------:|---------------------:|
| E      |                    7 |                   20 |                         4 |                        17 |                        0 |                        0 |                       45    |                      191.25 |                0.027 |                0.146 |
| F      |                    6 |                   19 |                         5 |                        18 |                        0 |                        0 |                       56.25 |                      202.5  |                0.031 |                0.15  |
| G      |                    7 |                   20 |                         4 |                        17 |                        0 |                        0 |                       45    |                      191.25 |                0.027 |                0.147 |
| H      |                    6 |                   19 |                         5 |                        18 |                        0 |                        0 |                       56.25 |                      202.5  |                0.029 |                0.15  |

Partidas a frio no total: 0.

## 4. Interação por componente (média mensal; custos em k$/dia, partidas em nº/dia)

Sinal: negativo = substituição (custo conjunto maior que a soma prevista).

| subset     |   cost_startup |   cost_fuel |   cost_flex |   cost_degradation |   cost_h2_startup |   cost_curtail_penalty |   cost_shutdown |   cost_other |   cost_total |   startups_total |
|:-----------|---------------:|------------:|------------:|-------------------:|------------------:|-----------------------:|----------------:|-------------:|-------------:|-----------------:|
| VPL–VDC    |         -0.959 |       0.225 |       0.032 |             -0.058 |                 0 |                 -0.001 |               0 |           -0 |       -0.761 |           -0.333 |
| VPL–H2     |         -3.232 |      -1.687 |       0     |             -0.196 |                 0 |                 -0.012 |               0 |           -0 |       -5.128 |            1.75  |
| VDC–H2     |         -0.541 |      -0.165 |       0.06  |             -0.03  |                 0 |                 -0.002 |               0 |           -0 |       -0.677 |           -0.25  |
| VPL–VDC–H2 |          1.891 |      -2.011 |       0.031 |             -0.093 |                 0 |                  0.002 |               0 |            0 |       -0.179 |           -0.667 |

## 5. Assimetria do par VPL–VDC (doze dias mensais)

Valor do VPL sozinho 62.92 k$/dia; do VDC 4.99 k$/dia; I = -0.761 k$/dia.

Perda do valor do VPL com o VDC: 1.21 %; perda do valor do VDC com o VPL: 15.3 %.

## 6. Sinal entre dias e fração do benefício conjunto (doze dias mensais)

p: teste de Wilcoxon (postos sinalizados) e t de uma amostra contra zero. Fração: média de |I| sobre a redução conjunta do subconjunto (coluna da Tabela II). Último: |média de I| sobre o valor sozinho do menor recurso do subconjunto.

| subset     |   media |   negativos |   p_wilcoxon |   p_t |   media_abs_sobre_conjunto_% |   sobre_menor_recurso_% |
|:-----------|--------:|------------:|-------------:|------:|-----------------------------:|------------------------:|
| VPL–VDC    |  -0.761 |           9 |        0.176 | 0.165 |                        2.213 |                  15.264 |
| VPL–H2     |  -5.128 |          12 |        0     | 0     |                        5.709 |                  16.006 |
| VDC–H2     |  -0.677 |           8 |        0.176 | 0.14  |                        3.371 |                  13.585 |
| VPL–VDC–H2 |  -0.179 |           7 |        0.622 | 0.794 |                        1.949 |                   3.599 |

## 7. Vereditos conclusivos contra a tolerância (16 dias, 64 coeficientes)

Pior caso: cada δ_S no limite g·ĉ_S, com os incumbentes atuais.

- gaps obtidos: 57
- g = 0.01 %: 54
- g = 0.05 %: 23
- g = 0.1 %: 9
- g = 0.5 %: 0

## 8. Alívio de corte do VPL com as duas baterias livres (doze dias mensais)

Corte médio com o VPL livre: 5.68 % (híbrido 7.58 %, H2 7.24 %, sem flexibilidade 9.37 %).

| separa                 |   VPL_hibrido_pp |   VPL_livre_pp |   H2_pp |
|:-----------------------|-----------------:|---------------:|--------:|
| sem separação          |             1.45 |           4.32 |    2.31 |
| corredor separa preços |             2.27 |           2.81 |    1.89 |
