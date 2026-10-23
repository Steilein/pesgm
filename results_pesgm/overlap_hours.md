# Sobreposição das horas de atuação de cada par contra I

Gerado por `papers/journal/scripts/overlap_hours.py` (variação base; caso D, F ou G).

| pair    | day_key     |   h1 |   h2 |   jaccard |          I | verdict       |
|:--------|:------------|-----:|-----:|----------:|-----------:|:--------------|
| VPLxVDC | m01         |   16 |   16 |     0.684 |   2333.32  | complement    |
| VPLxVDC | m02         |   17 |   17 |     0.7   |  -1400.72  | substitute    |
| VPLxVDC | m03         |   14 |   19 |     0.435 |    147.385 | indeterminate |
| VPLxVDC | m04         |   14 |   19 |     0.5   |  -1059.62  | substitute    |
| VPLxVDC | m05         |   15 |   19 |     0.545 |  -1804.24  | substitute    |
| VPLxVDC | m06         |   13 |   22 |     0.591 |   -586.531 | substitute    |
| VPLxVDC | m07         |   19 |   21 |     0.739 |  -1153.53  | substitute    |
| VPLxVDC | m08         |   19 |   21 |     0.739 |    -55.815 | indeterminate |
| VPLxVDC | m09         |   13 |   21 |     0.478 |  -4172.66  | substitute    |
| VPLxVDC | m10         |   17 |   19 |     0.565 |  -2585.08  | substitute    |
| VPLxVDC | m11         |   15 |   13 |     0.4   |   1866.73  | complement    |
| VPLxVDC | m12         |   16 |   18 |     0.478 |   -663.455 | substitute    |
| VPLxVDC | median      |   14 |   17 |     0.55  |  -1483.75  | substitute    |
| VPLxVDC | solar_ramp  |   15 |   15 |     0.429 |   -104.887 | indeterminate |
| VPLxVDC | spring_wind |   17 |   18 |     0.591 |   -919.644 | substitute    |
| VPLxVDC | summer_peak |   17 |   18 |     0.522 |  -3736.91  | substitute    |
| VPLxH2  | m01         |   16 |   10 |     0.444 |  -1401.47  | substitute    |
| VPLxH2  | m02         |   15 |   13 |     0.867 |  -1648.5   | substitute    |
| VPLxH2  | m03         |   16 |   14 |     0.5   |  -2930.46  | substitute    |
| VPLxH2  | m04         |   16 |   22 |     0.583 | -12101.2   | substitute    |
| VPLxH2  | m05         |   12 |   13 |     0.667 |  -7087.66  | substitute    |
| VPLxH2  | m06         |   16 |   21 |     0.762 |  -8432.63  | substitute    |
| VPLxH2  | m07         |   17 |   15 |     0.455 |  -2407.38  | substitute    |
| VPLxH2  | m08         |   17 |   17 |     0.619 |  -1668.56  | substitute    |
| VPLxH2  | m09         |   14 |   23 |     0.609 |  -6729.85  | substitute    |
| VPLxH2  | m10         |   17 |    9 |     0.529 |  -7827.09  | substitute    |
| VPLxH2  | m11         |   15 |   23 |     0.583 |  -6577.37  | substitute    |
| VPLxH2  | m12         |   15 |   13 |     0.75  |  -2720.83  | substitute    |
| VPLxH2  | median      |   14 |   13 |     0.5   |   1730.65  | complement    |
| VPLxH2  | solar_ramp  |   16 |   18 |     0.7   |  -6152.75  | substitute    |
| VPLxH2  | spring_wind |   16 |   14 |     0.5   |  -2202.64  | substitute    |
| VPLxH2  | summer_peak |   18 |   22 |     0.667 | -11524.7   | substitute    |
| VDCxH2  | m01         |   18 |   11 |     0.381 |    -11.672 | indeterminate |
| VDCxH2  | m02         |   16 |   12 |     0.474 |   -688.21  | substitute    |
| VDCxH2  | m03         |   17 |   17 |     0.478 |    337.245 | complement    |
| VDCxH2  | m04         |   21 |   21 |     0.75  |    906.84  | complement    |
| VDCxH2  | m05         |   20 |   13 |     0.435 |  -3746.76  | substitute    |
| VDCxH2  | m06         |   21 |   16 |     0.609 |   -388.718 | substitute    |
| VDCxH2  | m07         |   19 |   17 |     0.565 |   -928.765 | substitute    |
| VDCxH2  | m08         |   21 |   14 |     0.591 |    816.391 | complement    |
| VDCxH2  | m09         |   20 |   21 |     0.783 |  -1664.63  | substitute    |
| VDCxH2  | m10         |   18 |   11 |     0.45  |  -2177.64  | substitute    |
| VDCxH2  | m11         |   14 |   24 |     0.583 |   1225.91  | complement    |
| VDCxH2  | m12         |   15 |   12 |     0.286 |  -1809.51  | substitute    |
| VDCxH2  | median      |   18 |   15 |     0.435 |   3425.79  | complement    |
| VDCxH2  | solar_ramp  |   17 |   13 |     0.429 |  -1074.09  | substitute    |
| VDCxH2  | spring_wind |   17 |   16 |     0.571 |   -235.112 | indeterminate |
| VDCxH2  | summer_peak |   18 |   21 |     0.696 |  -3560.72  | substitute    |

## Correlações

|         |   n |   spearman |   pearson |   n_concl |   spearman_concl |
|:--------|----:|-----------:|----------:|----------:|-----------------:|
| VDCxH2  |  16 |      0.157 |     0.008 |        14 |            0.202 |
| VPLxH2  |  16 |     -0.323 |    -0.243 |        16 |           -0.323 |
| VPLxVDC |  16 |     -0.072 |     0.051 |        13 |            0.129 |
| pooled  |  48 |     -0.214 |    -0.231 |        43 |           -0.173 |
