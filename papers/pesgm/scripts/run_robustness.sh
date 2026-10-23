#!/usr/bin/env bash
# Testes do parecer simulado (23/10/2026): um processo por variante e dia.
#   bash papers/pesgm/scripts/run_robustness.sh >> results_pesgm_robust.log 2>&1
set -u
cd "$(dirname "$0")/../../.."
# venv do Windows, do Linux/macOS, ou o python do PATH (sobrescreva com PY=...)
if [ -n "${PY:-}" ]; then :
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python; fi
echo "=== inicio $(date)"
for d in spring_wind summer_peak solar_ramp median; do
  $PY src/robustness_pesgm.py --variant fc_up --days "$d"
done
for v in vpl_emergent h2_su10 h2_sb1; do
  for d in m01 m02 m03 m04 m05 m06 m07 m08 m09 m10 m11 m12; do
    $PY src/robustness_pesgm.py --variant "$v" --days "$d"
  done
done
echo "=== fim $(date)"
