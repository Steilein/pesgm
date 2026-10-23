# Interações de todas as ordens e suas cotas rigorosas

Gerado por `src/interaction_stats.py --mobius`. Uma linha por
(instância, subconjunto de recursos); a cota vem dos *best bounds* do
solver, sem supor convenção de gap.

Instâncias: **16**. Subconjuntos avaliados: **4** (VDCxH2, VPLxH2, VPLxVDC, VPLxVDCxH2).

## 1. Resumo por subconjunto

|                   |   n |   I_medio |   I_mediano |   subst |   compl |   indet |
|:------------------|----:|----------:|------------:|--------:|--------:|--------:|
| (2, 'VDCxH2')     |  16 |      -598 |        -538 |       9 |       5 |       2 |
| (2, 'VPLxH2')     |  16 |     -4980 |       -4542 |      15 |       1 |       0 |
| (2, 'VPLxVDC')    |  16 |      -961 |        -990 |      11 |       2 |       3 |
| (3, 'VPLxVDCxH2') |  16 |      -102 |        -185 |       8 |       6 |       2 |

## 2. A cota dobra a cada ordem

Previsão da Sec. III-H: `W <= 2^k g c`. A coluna `razao` é a largura
mediana sobre a da ordem anterior --- deve rondar 2. A coluna
`I_sobre_largura` diz se ainda há sinal acima do ruído: abaixo de 1,
o coeficiente é menor que a própria incerteza.

|   order |   n |   I_abs_mediano |   largura_mediana |   conclusivos |   pct_conclusivo |   razao |   I_sobre_largura |
|--------:|----:|----------------:|------------------:|--------------:|-----------------:|--------:|------------------:|
|       2 |  48 |          1699.6 |             578.3 |            43 |               90 |   nan   |               2.9 |
|       3 |  16 |          1281.6 |            1069   |            14 |               88 |     1.8 |               1.2 |

## 3. O que uma tolerância usual teria concluído

Mesmos ótimos, cota reescalada para o gap indicado:

| gap alvo | subconjunto | substituição | complementaridade | indeterminado |
|---:|:---|---:|---:|---:|
| 0.05 % | VDCxH2 | 3 | 1 | 12 |
| 0.05 % | VPLxH2 | 12 | 0 | 4 |
| 0.05 % | VPLxVDC | 4 | 2 | 10 |
| 0.05 % | VPLxVDCxH2 | 0 | 1 | 15 |
| 0.50 % | VDCxH2 | 0 | 0 | 16 |
| 0.50 % | VPLxH2 | 0 | 0 | 16 |
| 0.50 % | VPLxVDC | 0 | 0 | 16 |
| 0.50 % | VPLxVDCxH2 | 0 | 0 | 16 |

## 4. Instâncias com veredito conclusivo

| subset     | day_key     | variation   |      I |   I_lo |   I_hi | verdict    |
|:-----------|:------------|:------------|-------:|-------:|-------:|:-----------|
| VDCxH2     | m02         | base        |   -688 |   -956 |   -387 | substitute |
| VDCxH2     | m03         | base        |    337 |     79 |    491 | complement |
| VDCxH2     | m04         | base        |    907 |    746 |   1191 | complement |
| VDCxH2     | m05         | base        |  -3747 |  -4084 |  -3400 | substitute |
| VDCxH2     | m06         | base        |   -389 |   -840 |    -96 | substitute |
| VDCxH2     | m07         | base        |   -929 |  -1450 |   -329 | substitute |
| VDCxH2     | m08         | base        |    816 |    346 |   1310 | complement |
| VDCxH2     | m09         | base        |  -1665 |  -1889 |  -1191 | substitute |
| VDCxH2     | m10         | base        |  -2178 |  -2449 |  -1867 | substitute |
| VDCxH2     | m11         | base        |   1226 |    968 |   1276 | complement |
| VDCxH2     | m12         | base        |  -1810 |  -2145 |  -1541 | substitute |
| VDCxH2     | median      | base        |   3426 |   3094 |   3742 | complement |
| VDCxH2     | solar_ramp  | base        |  -1074 |  -1451 |   -912 | substitute |
| VDCxH2     | summer_peak | base        |  -3561 |  -4026 |  -3138 | substitute |
| VPLxH2     | m01         | base        |  -1401 |  -1729 |  -1073 | substitute |
| VPLxH2     | m02         | base        |  -1649 |  -1912 |  -1497 | substitute |
| VPLxH2     | m03         | base        |  -2930 |  -3171 |  -2800 | substitute |
| VPLxH2     | m04         | base        | -12101 | -12298 | -11827 | substitute |
| VPLxH2     | m05         | base        |  -7088 |  -7425 |  -6762 | substitute |
| VPLxH2     | m06         | base        |  -8433 |  -8655 |  -8084 | substitute |
| VPLxH2     | m07         | base        |  -2407 |  -2920 |  -1805 | substitute |
| VPLxH2     | m08         | base        |  -1669 |  -2059 |  -1227 | substitute |
| VPLxH2     | m09         | base        |  -6730 |  -6953 |  -6272 | substitute |
| VPLxH2     | m10         | base        |  -7827 |  -8096 |  -7526 | substitute |
| VPLxH2     | m11         | base        |  -6577 |  -6857 |  -6505 | substitute |
| VPLxH2     | m12         | base        |  -2721 |  -3024 |  -2464 | substitute |
| VPLxH2     | median      | base        |   1731 |   1422 |   2048 | complement |
| VPLxH2     | solar_ramp  | base        |  -6153 |  -6522 |  -5803 | substitute |
| VPLxH2     | spring_wind | base        |  -2203 |  -2406 |  -1939 | substitute |
| VPLxH2     | summer_peak | base        | -11525 | -11969 | -11026 | substitute |
| VPLxVDC    | m01         | base        |   2333 |   2165 |   2672 | complement |
| VPLxVDC    | m02         | base        |  -1401 |  -1718 |  -1120 | substitute |
| VPLxVDC    | m04         | base        |  -1060 |  -1096 |   -906 | substitute |
| VPLxVDC    | m05         | base        |  -1804 |  -2135 |  -1461 | substitute |
| VPLxVDC    | m06         | base        |   -587 |   -816 |   -242 | substitute |
| VPLxVDC    | m07         | base        |  -1154 |  -1753 |   -757 | substitute |
| VPLxVDC    | m09         | base        |  -4173 |  -4174 |  -3933 | substitute |
| VPLxVDC    | m10         | base        |  -2585 |  -2942 |  -2331 | substitute |
| VPLxVDC    | m11         | base        |   1867 |   1548 |   1867 | complement |
| VPLxVDC    | m12         | base        |   -663 |  -1005 |   -469 | substitute |
| VPLxVDC    | median      | base        |  -1484 |  -1773 |  -1326 | substitute |
| VPLxVDC    | spring_wind | base        |   -920 |  -1195 |   -733 | substitute |
| VPLxVDC    | summer_peak | base        |  -3737 |  -4349 |  -3280 | substitute |
| VPLxVDCxH2 | m01         | base        |  -3303 |  -3910 |  -2863 | substitute |
| VPLxVDCxH2 | m02         | base        |  -1959 |  -2389 |  -1534 | substitute |
| VPLxVDCxH2 | m03         | base        |   -686 |  -1062 |   -140 | substitute |
| VPLxVDCxH2 | m04         | base        |   -619 |  -1074 |   -271 | substitute |
| VPLxVDCxH2 | m05         | base        |   3822 |   3157 |   4486 | complement |
| VPLxVDCxH2 | m06         | base        |   -999 |  -1725 |   -359 | substitute |
| VPLxVDCxH2 | m08         | base        |  -1446 |  -2411 |   -530 | substitute |
| VPLxVDCxH2 | m09         | base        |   3969 |   3278 |   4379 | complement |
| VPLxVDCxH2 | m10         | base        |   1118 |    620 |   1566 | complement |
| VPLxVDCxH2 | m11         | base        |  -2965 |  -3087 |  -2537 | substitute |
| VPLxVDCxH2 | m12         | base        |    507 |    177 |    997 | complement |
| VPLxVDCxH2 | median      | base        |  -2585 |  -3061 |  -1969 | substitute |
| VPLxVDCxH2 | spring_wind | base        |    567 |    120 |   1047 | complement |
| VPLxVDCxH2 | summer_peak | base        |   2290 |   1271 |   3316 | complement |
