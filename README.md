# CAR — Constraint-satisfying Architectural design using pRobabilistic graphical models

A hybrid machine-learning framework that generates probabilistic architectural designs by balancing an architect's creative intent with regulatory, environmental, and structural constraints — using Bayesian Networks.

---

## Overview

CAR encodes architectural knowledge as a **16-node Bayesian Network** with 23 directed edges across three layers:

```
Layer 0 — Design Intent (latent)      aesthetic_feel · view_priority · sketch_topology
               ↓  (weight 0.7–0.9, primary drivers)
Layer 1 — Constraints (observed)      FAR · height · setback · solar · wind · seismic · material · wall thickness
               ↓  (weight 0.4–0.8, filters)
Layer 2 — Design Decisions            structural_system · num_floors · window_size · wall_type · roof_type
```

**Design-first philosophy** — the architect's rough sketch carries the strongest priors. Constraints layer on afterward to filter, not override.

---

## Key Features

| Feature | Detail |
|---|---|
| **Bayesian Network** | 16 nodes, 23 edges via `pgmpy` (`DiscreteBayesianNetwork`) |
| **Inference modes** | MAP (fastest), MCMC (alternatives), Variational |
| **Confidence scoring** | `C = 0.5·D + 0.3·P + 0.2·M` (deterministic + probabilistic + model) |
| **Compliance checking** | FAR, height, setback, seismic, wall thickness, floor height |
| **Web interface** | Flask form with 3 presets, floor-plan and section visualizations |
| **CLI** | `car run`, `car example`, `car show-network`, `car web` |
| **Visualizations** | Network DAG, floor plan, building section, compliance table |
| **Pydantic v2 models** | Fully validated inputs and outputs |

---

## Project Structure

```
CAR/
├── car/
│   ├── network/
│   │   ├── nodes.py          # 16-node registry (latent / observed / decision)
│   │   ├── edges.py          # 23 directed edges with per-edge weights
│   │   ├── cpd_factory.py    # Parametric CPD construction (rule-based weights)
│   │   └── builder.py        # Assembles & validates the BayesianNetwork
│   ├── inference/
│   │   ├── pipeline.py       # Main orchestrator (design-first approach)
│   │   ├── map_inference.py  # MAP via VariableElimination
│   │   ├── mcmc_sampler.py   # MCMC sampling (PyMC) with fallback
│   │   └── variational.py    # Variational / direct marginal sampling
│   ├── scoring/
│   │   ├── compliance.py     # 6 hard-constraint checks
│   │   └── confidence.py     # Composite C = 0.5D + 0.3P + 0.2M scorer
│   ├── models/
│   │   ├── constraints.py    # Pydantic: SiteConstraints (regulatory/env/geo/tech)
│   │   ├── design.py         # Pydantic: DesignIntent, BuildingDesign
│   │   ├── results.py        # Pydantic: InferenceResult, ComplianceResult
│   │   └── network_spec.py   # Pydantic: NodeSpec, EdgeSpec, NetworkTopology
│   ├── visualization/
│   │   ├── network_plot.py   # Bayesian network DAG (category-colored)
│   │   ├── design_plot.py    # Floor plan + building section plots
│   │   ├── compliance_report.py  # Violations table
│   │   └── trace_plot.py     # MCMC trace diagnostics
│   ├── web/
│   │   ├── app.py            # Flask routes (/  /run  /preset/<name>  /download)
│   │   └── templates/        # index.html (form) · results.html
│   ├── examples/
│   │   ├── residential_low_rise.py   # Los Angeles single-family home
│   │   ├── commercial_high_rise.py   # Tokyo office tower
│   │   └── mixed_use_urban.py        # Copenhagen mixed-use
│   ├── cli.py                # Click CLI entry-point
│   ├── config.py             # Constants (weights, floor heights, MCMC defaults)
│   └── torch_mock.py         # Lightweight torch+pyro mock (avoids 100 MB install)
├── tests/                    # pytest suite (models, network, inference, scoring, viz)
├── examples/
│   ├── residential.json      # JSON config for residential preset
│   └── commercial.json       # JSON config for commercial preset
├── run_web.py                # Standalone web-server launcher
└── pyproject.toml
```

---

## Installation

> **Windows note:** If `python` and `pip` point to different versions, use explicit paths.  
> This project was built with **Python 3.14** (`C:\Users\USER\AppData\Local\Programs\Python\Python314\`).

### 1 — Install dependencies

```bash
cd C:\Users\USER\Desktop\Projects\CAR

# Lightweight packages first
pip install flask pydantic click numpy networkx matplotlib scipy pandas scikit-learn

# pgmpy without its optional torch/pyro GPU dependencies (saves ~1 GB)
pip install --no-deps pgmpy statsmodels joblib tqdm patsy opt-einsum
```

> `torch` and `pyro` are **not required** — CAR ships a lightweight mock
> (`car/torch_mock.py`) that satisfies pgmpy's import chain while keeping
> pgmpy running entirely on its numpy backend.

---

## Running the Project

### Web Interface (recommended)

```bash
cd C:\Users\USER\Desktop\Projects\CAR
C:\Users\USER\AppData\Local\Programs\Python\Python314\python.exe run_web.py
```

Open **http://127.0.0.1:5000** in your browser.

Options:
```bash
python run_web.py --host 0.0.0.0 --port 8080
```

**Presets available in the UI:**

| Preset | Location | Aesthetic | Key feature |
|---|---|---|---|
| Residential (LA) | Los Angeles | Minimalist | Low seismic zone, timber frame |
| Commercial (Tokyo) | Tokyo | Industrial | High seismic zone, steel frame |
| Mixed-Use (Copenhagen) | Copenhagen | Organic | High sustainability, green roof |

---

### CLI

```bash
# Set PYTHONPATH so the package resolves without pip install -e
set PYTHONPATH=C:\Users\USER\Desktop\Projects\CAR

# Run a built-in example
python -m car.cli example residential
python -m car.cli example commercial
python -m car.cli example mixed_use

# Run inference from a JSON config file
python -m car.cli run examples/residential.json

# Change inference method
python -m car.cli run examples/residential.json --method mcmc
python -m car.cli run examples/commercial.json --method variational

# Visualize the Bayesian network topology
python -m car.cli show-network

# Start the web interface via CLI
python -m car.cli web --port 5000
```

> **Note:** The CLI entry-point needs the torch mock installed before pgmpy loads.
> Prepend this one-liner if running without `run_web.py`:
> ```bash
> python -c "from car.torch_mock import install; install(); from car.cli import cli; cli()" example residential
> ```

---

## Inference Methods

| Method | Flag | Description |
|---|---|---|
| **MAP** | `--method map` | Maximum A Posteriori — fastest, single best design (default) |
| **MCMC** | `--method mcmc` | Markov Chain Monte Carlo via PyMC — returns design alternatives |
| **Variational** | `--method variational` | Direct marginal sampling from BN — lightweight alternative sampling |

---

## Confidence Score

Each design is scored with:

```
C = 0.5 × D  +  0.3 × P  +  0.2 × M
```

| Term | Name | Meaning |
|---|---|---|
| **D** | Deterministic compliance | Fraction of hard constraints passed (FAR, height, setback, …) |
| **P** | Probabilistic margin | How far the design sits from constraint boundaries |
| **M** | Model confidence | Average marginal probability of the MAP-chosen states |

---

## Bayesian Network — Node & Edge Summary

### Nodes (16 total)

| Layer | Node | States |
|---|---|---|
| Intent | `aesthetic_feel` | minimalist · industrial · organic · classical |
| Intent | `view_priority` | low · medium · high |
| Intent | `sketch_topology` | compact · linear · courtyard |
| Constraint | `far_class` | low_far · medium_far · high_far |
| Constraint | `height_restriction` | unrestricted · moderate · strict |
| Constraint | `setback_class` | standard · generous |
| Constraint | `solar_orientation` | north · east · south · west |
| Constraint | `wind_exposure` | sheltered · moderate · exposed |
| Constraint | `seismic_zone_class` | low_risk · moderate_risk · high_risk |
| Constraint | `material_class` | steel · concrete · timber · masonry |
| Constraint | `wall_thickness_class` | thin · standard · thick |
| Decision | `structural_system` | steel_frame · reinforced_concrete · timber_frame · masonry · hybrid |
| Decision | `num_floors` | 1 · 2-3 · 4-7 · 8+ floors |
| Decision | `window_size` | small · medium · large · full_glass |
| Decision | `wall_type` | load_bearing · curtain_wall · partition |
| Decision | `roof_type` | flat · pitched · green_roof |

### Edge Weights

- **Intent → Decision**: 0.7–0.9 (architect's vision is the primary driver)
- **Constraint → Decision**: 0.4–0.8 (constraints filter, not override)

---

## Tests

```bash
cd C:\Users\USER\Desktop\Projects\CAR
set PYTHONPATH=.
python -m pytest tests/ -v
```

Test modules:
- `tests/test_models.py` — Pydantic model validation
- `tests/test_network_builder.py` — BN construction & CPD validation
- `tests/test_inference.py` — MAP, MCMC, and variational inference
- `tests/test_scoring.py` — Compliance checking and confidence scoring
- `tests/test_visualization.py` — Plot generation (smoke tests)

---

## Sample Output

```
Residential (LA) — MAP Inference
─────────────────────────────────────────────
Structural System : Timber Frame
Floors            : 1  (3.2 m floor-to-floor)
Floor Area        : 300 sqm
Wall Type         : Partition
Wall Thickness    : 250 mm
Windows           : Large
Roof Type         : Flat
Footprint         : 16.4 m × 11.4 m

Compliance        : COMPLIANT  (5/6 constraints passed)
Confidence        : 59%
Aesthetic Score   : 0.80
View Score        : 0.80
Overall Score     : 0.74
Elapsed           : 0.04 s
```

---

## Architecture Decisions

- **Design-first CPDs** — intent edges carry weights 0.7–0.9 vs. constraint edges at 0.4–0.8, so the architect's vision dominates when constraints allow.
- **No torch required** — pgmpy's numpy backend is used exclusively. A mock (`car/torch_mock.py`) satisfies pgmpy's import chain at startup.
- **Lazy imports** — heavy pgmpy imports are deferred until the first `/run` request, keeping the web server startup instant.
- **Pydantic v2** — all inputs and outputs are fully validated with descriptive error messages.

---

## License

MIT
