# Testes de robustez do PES GM (parecer simulado, 23/10/2026)

Gerado por `papers/pesgm/scripts/robust_report.py`.

## Referência (grade principal, doze dias mensais)

| subset     |   mean_k |   median_k |   complement |   indeterminate |   substitute |
|:-----------|---------:|-----------:|-------------:|----------------:|-------------:|
| VPLxVDC    |   -0.761 |     -0.862 |            2 |               2 |            8 |
| VPLxH2     |   -5.128 |     -4.754 |            0 |               0 |           12 |
| VDCxH2     |   -0.677 |     -0.538 |            4 |               1 |            7 |
| VPLxVDCxH2 |   -0.179 |     -0.652 |            4 |               1 |            7 |

## vpl_emergent (12 dias completos)

| subset     |   mean_k |   median_k |   complement |   indeterminate |   substitute |
|:-----------|---------:|-----------:|-------------:|----------------:|-------------:|
| VPLxVDC    |   -0.816 |     -0.994 |            2 |               2 |            8 |
| VPLxH2     |   -6.413 |     -5.428 |            0 |               0 |           12 |
| VDCxH2     |   -0.677 |     -0.538 |            4 |               1 |            7 |
| VPLxVDCxH2 |    0.077 |     -0.027 |            4 |               3 |            5 |

Referência nos mesmos dias:

| subset     |   mean_k |   median_k |   complement |   indeterminate |   substitute |
|:-----------|---------:|-----------:|-------------:|----------------:|-------------:|
| VPLxVDC    |   -0.761 |     -0.862 |            2 |               2 |            8 |
| VPLxH2     |   -5.128 |     -4.754 |            0 |               0 |           12 |
| VDCxH2     |   -0.677 |     -0.538 |            4 |               1 |            7 |
| VPLxVDCxH2 |   -0.179 |     -0.652 |            4 |               1 |            7 |

Valor do VPL sozinho: 78.68 k$/dia (grade principal 62.92); do H2 sozinho: 32.04 (grade principal 32.04).

## h2_su10 (12 dias completos)

| subset     |   mean_k |   median_k |   complement |   indeterminate |   substitute |
|:-----------|---------:|-----------:|-------------:|----------------:|-------------:|
| VPLxVDC    |   -0.761 |     -0.862 |            2 |               2 |            8 |
| VPLxH2     |   -5.311 |     -4.974 |            0 |               0 |           12 |
| VDCxH2     |   -0.679 |     -0.424 |            4 |               2 |            6 |
| VPLxVDCxH2 |   -0.108 |     -0.478 |            4 |               2 |            6 |

Referência nos mesmos dias:

| subset     |   mean_k |   median_k |   complement |   indeterminate |   substitute |
|:-----------|---------:|-----------:|-------------:|----------------:|-------------:|
| VPLxVDC    |   -0.761 |     -0.862 |            2 |               2 |            8 |
| VPLxH2     |   -5.128 |     -4.754 |            0 |               0 |           12 |
| VDCxH2     |   -0.677 |     -0.538 |            4 |               1 |            7 |
| VPLxVDCxH2 |   -0.179 |     -0.652 |            4 |               1 |            7 |

Valor do VPL sozinho: 62.92 k$/dia (grade principal 62.92); do H2 sozinho: 35.16 (grade principal 32.04).

Eletrolisador (caso E): partidas a frio 12, horas produzindo 7.8, em standby 0.0, standby/eletrólise 0.0 %.

## h2_sb1 (12 dias completos)

| subset     |   mean_k |   median_k |   complement |   indeterminate |   substitute |
|:-----------|---------:|-----------:|-------------:|----------------:|-------------:|
| VPLxVDC    |   -0.761 |     -0.862 |            2 |               2 |            8 |
| VPLxH2     |   -5.231 |     -4.756 |            0 |               0 |           12 |
| VDCxH2     |   -0.647 |     -0.454 |            4 |               2 |            6 |
| VPLxVDCxH2 |   -0.276 |     -0.603 |            4 |               1 |            7 |

Referência nos mesmos dias:

| subset     |   mean_k |   median_k |   complement |   indeterminate |   substitute |
|:-----------|---------:|-----------:|-------------:|----------------:|-------------:|
| VPLxVDC    |   -0.761 |     -0.862 |            2 |               2 |            8 |
| VPLxH2     |   -5.128 |     -4.754 |            0 |               0 |           12 |
| VDCxH2     |   -0.677 |     -0.538 |            4 |               1 |            7 |
| VPLxVDCxH2 |   -0.179 |     -0.652 |            4 |               1 |            7 |

Valor do VPL sozinho: 62.92 k$/dia (grade principal 62.92); do H2 sozinho: 36.35 (grade principal 32.04).

Eletrolisador (caso E): partidas a frio 0, horas produzindo 8.0, em standby 16.0, standby/eletrólise 2.4 %.

## fc_up: célula a combustível na barra 318 (VPL–H2 contra o corredor)

| dia         |   MW |    I_k |   lo_k |   hi_k | veredito   |   dH2_k |   dVPL_k |
|:------------|-----:|-------:|-------:|-------:|:-----------|--------:|---------:|
| spring_wind |  600 |  -0.97 |  -1.12 |  -0.85 | substitute |   37.07 |    56.47 |
| spring_wind |  900 |  -2.2  |  -2.34 |  -1.93 | substitute |   36.13 |    57.6  |
| spring_wind | 1200 |  -2.2  |  -2.33 |  -2.07 | substitute |   36.13 |    57.6  |
| spring_wind | 1500 |  -2.2  |  -2.36 |  -2.06 | substitute |   36.13 |    57.6  |
| summer_peak |  600 | -10.25 | -10.91 |  -9.68 | substitute |   42.79 |    94.42 |
| summer_peak |  900 | -11.53 | -11.9  | -11.03 | substitute |   39.05 |    92.52 |
| summer_peak | 1200 | -22.05 | -22.68 | -21.41 | substitute |   32.2  |    85.66 |
| summer_peak | 1500 | -12.46 | -13.09 | -11.82 | substitute |   22.48 |    75.92 |
| solar_ramp  |  600 |  -6.01 |  -6.36 |  -5.64 | substitute |   32.52 |    63.45 |
| solar_ramp  |  900 |  -6.15 |  -6.53 |  -5.8  | substitute |   31.43 |    62.93 |
| solar_ramp  | 1200 |  -7.03 |  -7.35 |  -6.66 | substitute |   29.34 |    62.93 |
| solar_ramp  | 1500 |  -7.03 |  -7.23 |  -6.65 | substitute |   29.34 |    62.93 |
| median      |  600 |   3.34 |   3    |   3.49 | complement |   34.4  |    57.73 |
| median      |  900 |   1.73 |   1.42 |   1.89 | complement |   32.71 |    59.1  |
| median      | 1200 |   1.73 |   1.45 |   2.05 | complement |   32.71 |    59.1  |
| median      | 1500 |   1.73 |   1.44 |   1.98 | complement |   32.71 |    59.11 |
