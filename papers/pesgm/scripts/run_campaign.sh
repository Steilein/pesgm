#!/usr/bin/env bash
# Campanha do PES GM (perfil config/pesgm.yaml), sequencial: um processo por
# dia na grade principal, depois a varredura do corredor e a variante rígida.
# Um processo de cada vez porque a máquina tem ~3 GB livres.
#
#   bash papers/pesgm/scripts/run_campaign.sh  > results_pesgm_campaign.log 2>&1
set -u
cd "$(dirname "$0")/../../.."
# venv do Windows, do Linux/macOS, ou o python do PATH (sobrescreva com PY=...)
if [ -n "${PY:-}" ]; then :
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python; fi
PROF=config/realistic.yaml,config/journal.yaml,config/pesgm.yaml
DAYS="m01 m02 m03 m04 m05 m06 m07 m08 m09 m10 m11 m12 spring_wind summer_peak solar_ramp median"

echo "=== grade principal $(date)"
for d in $DAYS; do
  $PY src/runner.py --grid --axes --profile "$PROF" \
      --scenarios config/scenarios_journal.yaml --solver gurobi --days "$d"
done

echo "=== corredor $(date)"
$PY src/corridor_sweep.py --extra-profile config/pesgm.yaml --results results_pesgm_corridor --threads 4

echo "=== rigida $(date)"
for d in m01 m02 m03 m04 m05 m06 m07 m08 m09 m10 m11 m12; do
  $PY src/indivisible.py --extra-profile config/pesgm.yaml --results results_pesgm_rigid --threads 4 --days "$d"
done
echo "=== fim $(date)"
