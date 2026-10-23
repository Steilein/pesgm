# Interactions Among Flexibility Providers in Day-Ahead Unit Commitment

Code, configuration and per-instance results for

> G. Steilein, F. A. L. Ferreira and C. Unsihuay-Vila, "Interactions Among
> Flexibility Providers in Day-Ahead Unit Commitment: Data Centers, Battery
> Storage and Hydrogen," submitted to the 2027 IEEE PES General Meeting.

A network-constrained day-ahead unit commitment (PyPSA + Gurobi) co-optimizes a
virtual power line (a battery at each terminal of an export corridor), a
virtual data center (deferrable and migratable workloads paid at an offer
price) and a hydrogen chain (three-state electrolyzer, store, fuel cell) on a
solar-dominated variant of the RTS-GMLC. Interactions of every order among the
three resources are computed as Möbius coefficients of the operating cost, each
with a deterministic bound derived from the solvers' best bounds.

## What is here

| Path | Content |
|---|---|
| `src/` | model, constraints, metrics, campaign drivers (shared core) |
| `config/` | profiles, applied in order `base.yaml` ← `realistic.yaml` ← `journal.yaml` ← `pesgm.yaml` |
| `papers/pesgm/scripts/` | paper-specific reports, Fig. 2, campaign shell scripts |
| `papers/journal/scripts/` | reports shared with a longer manuscript in preparation |
| `tests/` | tests of the data-center reference schedule, rigid workloads and electrolyzer states |
| `results_*/metrics.csv` | **one row per MILP**: costs, incumbent, best bound, gap, curtailment, start-ups, corridor and resource metrics |
| `results_*/*.md` | the reports from which every number in the paper is taken |
| `figures_pesgm/en/` | figures of the paper |

The solved networks (`.nc`, about 280 MB) are not included; the campaigns
below regenerate them. Source comments are in Portuguese.

### Per-instance results

In every `metrics.csv`, `scenario` is `day__case__variation`. Cases A–H are the
2³ subsets of resources: A none, B VPL, C VDC, D VPL+VDC, E H2, F VPL+H2, G
VDC+H2, H all three. `cost_total` is the reported cost, `objective_milp` the
incumbent and `objective_bound` the solver's best bound, so the interval of
every interaction coefficient can be recomputed from these files alone:

```bash
python src/interaction_stats.py --mobius --metrics results_pesgm/metrics.csv \
       --out results_pesgm/interaction_mobius.md --figures figures_pesgm
```

| Directory | Campaign | MILPs |
|---|---|---|
| `results_pesgm/` | main grid: 16 days × 8 cases, 0.01 % gap | 128 |
| `results_pesgm_corridor/` | corridor rated 600, 1200, 1500 MW (cases A, B, E, F; four extreme days) | 48 |
| `results_pesgm_rigid/` | indivisible (rigid) workloads | 72 |
| `results_pesgm_h2cont/` | continuously variable electrolyzer without operating cost (case E) | 12 |
| `results_pesgm_robust/` | fuel cell at the upstream bus; VPL without the direction rule; cheaper electrolyzer cold start; 1 % standby | 176 |

### Paper ↔ files

| In the paper | Source |
|---|---|
| Table I, Sec. V-A | `results_pesgm/journal_numbers.md`, `results_pesgm/pesgm_numbers.md`, `results_pesgm_h2cont/metrics.csv` |
| Table II, Fig. 2, Sec. V-B, V-D, V-E | `results_pesgm/interaction_mobius.md`, `third_order.md`, `pesgm_numbers.md` |
| Table III, Sec. V-C | `results_pesgm_corridor/corridor_report.md`, `results_pesgm_robust/robust_report.md` (control with the fuel cell moved) |
| Sec. V-B (direction rule), V-A (electrolyzer sensitivity) | `results_pesgm_robust/robust_report.md` |
| Sec. V-F | `results_pesgm_rigid/indivisible.md` |

## Reproducing

Tested with Python 3.13, PyPSA 1.3.0 and Gurobi 13.0.3 (a Gurobi license is
required; academic licenses are free).

```bash
pip install -r requirements.txt
git clone https://github.com/GridMod/RTS-GMLC data/RTS-GMLC   # path set in config/base.yaml

# main grid, corridor sweep and rigid workloads (a few hours on a laptop)
bash papers/pesgm/scripts/run_campaign.sh
# robustness tests
bash papers/pesgm/scripts/run_robustness.sh

# reports
export PLOTS_LANG=en RESULTS_DIR=results_pesgm \
       CORRIDOR_DIR=results_pesgm_corridor RIGID_DIR=results_pesgm_rigid
python papers/journal/scripts/journal_numbers.py
python papers/journal/scripts/corridor_report.py
python papers/journal/scripts/indivisible_report.py
python papers/journal/scripts/third_order.py      # needs the .nc networks
python papers/journal/scripts/overlap_hours.py    # needs the .nc networks
python papers/pesgm/scripts/pesgm_numbers.py
python papers/pesgm/scripts/robust_report.py
python papers/pesgm/scripts/plot_mobius_col.py
pytest tests
```

With the `metrics.csv` files in this repository, every report except
`third_order.py` and `overlap_hours.py` runs without solving anything and
reproduces the `.md` files here exactly.

## License

MIT (see `LICENSE`). The RTS-GMLC data are not redistributed here; see
<https://github.com/GridMod/RTS-GMLC> for their terms.
