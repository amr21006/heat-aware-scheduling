# Heat-Aware Project Scheduling Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Optimization: OR-Tools](https://img.shields.io/badge/Solver-OR--Tools%20CP--SAT-brightgreen.svg)](https://developers.google.com/optimization)

An open-source Python framework for **integrating meteorological thermal hazard modeling and trade vulnerability into resource-constrained project scheduling optimization (RCPSP)**.

---

## 📌 Framework Overview

Standard project scheduling algorithms minimize makespan and resource leveling without accounting for calendar-level environmental and climatic hazards. This repository provides a modular, reproducible computational framework to incorporate empirical thermal risk penalties directly into activity sequencing.

### Key Pipeline Modules:
1. **Data Ingestion & Filtering:** Processes occupational injury records and verifies checksums.
2. **Outcome Validation:** Evaluates narrative NLP classifiers against standard event codings.
3. **Meteorological Linkage:** Matches geocoded event coordinates to nearest weather station observations.
4. **Epidemiological Risk Modeling:** Implements time-stratified case-crossover designs with conditional logistic regression.
5. **Mechanism Attribution:** Applies natural language processing rules to stratify incidents into primary and secondary exposure mechanisms.
6. **Day-Ahead Machine Learning:** Trains operational predictors using strictly pre-workday observable features with probability calibration.
7. **Craft Vulnerability Weighting:** Calibrates activity-level exposure weights across industry trade classifications.
8. **Network Parsing:** Ingests activity networks with precedence relationships and renewable resource demands.
9. **Risk Calendars:** Builds date-indexed continuous thermal penalty series across geographic climate regions.
10. **Paired Schedule Optimization:** Solves the heat-aware RCPSP via Google OR-Tools CP-SAT with element constraint encodings.
11. **Automated Verification:** Runs comprehensive scientific verification tests across all pipeline stages.

---

## 🗂️ Repository Structure

```text
heat-aware-scheduling-code/
├── data_raw/             # Raw incident records & checksum files
├── data_intermediate/    # Processed case-crossover strata & linked meteorological series
├── data_external/        # Station caches & project schedule networks
├── scripts/              # Sequential execution pipeline (01 to 11)
│   ├── 01_filter_osha.py            # Data ingestion & checksum verification
│   ├── 02_validate_heat_labels.py   # Narrative NLP outcome validation
│   ├── 03_match_noaa_weather.py     # Meteorological station matching
│   ├── 04_model_case_crossover.py   # Conditional logistic regression & controls
│   ├── 05_heat_attribution.py       # NLP mechanism stratification & refits
│   ├── 06_dayahead_prediction.py    # Day-ahead machine learning & calibration
│   ├── 07_exposure_weights.py       # Industry craft vulnerability weighting
│   ├── 08_parse_dslib.py            # Construction project network parser
│   ├── 09_build_risk_calendars.py   # Multi-location thermal risk calendars
│   ├── 10_schedule_optimization.py  # Google OR-Tools CP-SAT RCPSP optimization
│   ├── 11_validate_verify.py        # Automated multi-gate scientific verification
│   ├── common.py                    # Shared helper functions & logging
│   ├── config.py                    # Global constants, paths, and fixed seeds
│   └── run_all.py                   # Master end-to-end pipeline runner
├── data_dictionary.md    # Variable definitions and schema
├── requirements.txt      # Python dependencies
└── README.md             # Repository documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/anonymous/heat-aware-scheduling.git
cd heat-aware-scheduling
```

### 2. Set up a virtual environment (Python 3.10+)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage

### Run the entire pipeline end-to-end:
```bash
python scripts/run_all.py
```

### Run specific pipeline steps:
```bash
# Resume execution from a specific step (e.g., Step 08)
python scripts/run_all.py --from 08

# Execute a single pipeline step (e.g., Step 10)
python scripts/run_all.py --only 10
```

### Run automated verification tests:
```bash
python scripts/11_validate_verify.py
```

---

## 📊 Optimization Formulation

The heat-aware Resource-Constrained Project Scheduling Problem (RCPSP) is formulated as:

$$\min H = \sum_{i \in A} \sum_{t=s_i}^{s_i + d_i - 1} w_i \cdot r_t$$

Subject to:
1. **Precedence Constraints:** $s_j \ge s_i + d_i + \ell_{ij} \quad \forall (i,j) \in E$
2. **Renewable Resource Limits:** $\sum_{i \in A_t} r_{ik} \le R_k \quad \forall k, \forall t$
3. **Makespan Preservation:** $C_{max} \le C_{max}^{base}$

Where:
* $s_i$: Integer start day of activity $i$.
* $d_i$: Duration of activity $i$.
* $w_i \in (0, 1]$: Craft vulnerability exposure weight.
* $r_t \in [0, 1]$: Bounded day-ahead thermal risk index on calendar day $t$.
* $C_{max}^{base}$: Optimal baseline project makespan.

---

## 📄 License

This software is released under the **MIT License**.
