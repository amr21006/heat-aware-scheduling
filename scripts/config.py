"""
config.py — Central configuration for the heat-risk-aware scheduling study.

All paths are resolved relative to the project root (the parent of /scripts),
so the package is portable. No analytic values are hardcoded here; only
filtering rules, thresholds, seeds, and source definitions taken directly from
the research plan.
"""
from __future__ import annotations
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_INTERMEDIATE = PROJECT_ROOT / "data_intermediate"
DATA_EXTERNAL = PROJECT_ROOT / "data_external"
OUTPUTS = PROJECT_ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
LOGS = OUTPUTS / "logs"
WEATHER_CACHE = DATA_EXTERNAL / "weather_cache"

for _d in (DATA_INTERMEDIATE, DATA_EXTERNAL, TABLES, FIGURES, MODELS, LOGS, WEATHER_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_RAW / "January2015toAugust2025.csv"

# Expected integrity values (Verification 1). Computed from the local file.
EXPECTED_SHA256 = "73C02E93635585E9C0D9B4816E864739C3900B78484DF356D5F7BD02E25D09C6"
EXPECTED_ROWS = 103750
EXPECTED_CONSTRUCTION = 18617
EXPECTED_HEAT = 605

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_SEED = 20250604

# ----------------------------------------------------------------------------
# OSHA filtering rules (frozen; from research plan Case Definition)
# ----------------------------------------------------------------------------
CONSTRUCTION_NAICS_PREFIX = "23"

# Primary environmental-heat outcome: any of these OSHA-coded signals.
HEAT_EVENTTITLE_PREFIX = "exposure to environmental heat"
HEAT_SOURCETITLE_SET = {"heat-environmental", "heat environmental"}
HEAT_NATURE_TERMS = [
    "effects of heat and light",
    "heat exhaustion",
    "heat fatigue",
    "heat syncope",
    "heat stroke",
]

# Secondary (EXCLUDED from primary): burns / hot-object thermal contact.
BURN_NATURE_TERMS = [
    "heat (thermal) burns",
    "first degree heat",
    "second degree heat",
    "third or fourth degree heat",
]

# ----------------------------------------------------------------------------
# Weather matching protocol (research plan §Weather Matching)
# ----------------------------------------------------------------------------
PRIMARY_DISTANCE_KM = 50.0           # preferred max NOAA station distance
SENSITIVITY_DISTANCE_KM = [25.0, 75.0, 100.0]
MAX_SEARCH_DISTANCE_KM = 150.0       # hard cap before gridded fallback
N_NEAREST_STATIONS = 12              # candidate stations to probe per location
HEAT_HISTORY_BUFFER_DAYS = 10        # extra days fetched before month start for lag features

# NOAA source providers (meteostat enum names) — documented provenance
PRIMARY_DAILY_PROVIDER = "GHCND"     # NOAA Global Historical Climatology Network-Daily
HOURLY_HI_PROVIDER = "ISD_LITE"      # NOAA Integrated Surface Database (Lite)
GRIDDED_FALLBACK = "OPEN_METEO_ERA5" # ECMWF ERA5 reanalysis (flagged, non-NOAA, fallback only)

# ----------------------------------------------------------------------------
# Case-crossover design (time-stratified; research plan Design A)
# ----------------------------------------------------------------------------
# Referent (control) days = same calendar month & year & same day-of-week as the
# case, excluding the case day itself. This is the standard time-stratified
# case-crossover referent scheme.
CASECROSS_MATCH_DOW = True

# ----------------------------------------------------------------------------
# Predictive model (research plan Design C / Validation 4)
# ----------------------------------------------------------------------------
TRAIN_YEARS = (2015, 2022)   # inclusive
TEST_YEARS = (2023, 2025)    # inclusive (through Aug 2025)

# ----------------------------------------------------------------------------
# Label validation sampling (research plan §Manual Label Verification)
# ----------------------------------------------------------------------------
LABEL_SAMPLE = {
    "primary_heat": 150,
    "excluded_burn": 100,
    "nonheat_summer": 150,
    "nonheat_nonsummer": 100,
}
SUMMER_MONTHS = [6, 7, 8]

# ----------------------------------------------------------------------------
# Scheduling experiment (research plan §Scheduling Automation)
# ----------------------------------------------------------------------------
# Outdoor exposure scenario weights (scenario parameters, NOT observed values).
OUTDOOR_WEIGHTS = {"low": 0.25, "medium": 0.50, "high": 1.00}
# Default project start months and locations for scheduling scenarios.
SCHEDULE_START_MONTHS = [5, 6, 7, 8]   # May-Aug (heat season)
# Representative high-heat-burden states (top by case count) for scheduling calendars.
SCHEDULE_LOCATIONS = {
    "TX_Austin": (30.27, -97.74),
    "FL_Orlando": (28.54, -81.38),
    "GA_Atlanta": (33.75, -84.39),
    "AZ_Phoenix": (33.45, -112.07),
    "IL_Chicago": (41.88, -87.63),     # moderate-climate location
}
HIGH_RISK_QUANTILE = 0.80   # day is "high-risk" if risk score >= this quantile

# PSPLIB benchmark sets J30..J120 (research plan: "PSPLIB J30 to J120").
# n_instances and time_limit are bounded so the full multi-set, multi-location
# CP-SAT experiment runs in reasonable wall-clock; larger sets get more time.
PSPLIB_SETS = {
    "j30":  {"jobs": 32,  "n_instances": 12, "param_max": 40, "time_limit": 8.0,  "pareto": True},
    "j60":  {"jobs": 62,  "n_instances": 10, "param_max": 40, "time_limit": 12.0, "pareto": True},
    "j90":  {"jobs": 92,  "n_instances": 8,  "param_max": 40, "time_limit": 15.0, "pareto": False},
    "j120": {"jobs": 122, "n_instances": 6,  "param_max": 60, "time_limit": 25.0, "pareto": False},
}
# Locations used per set (hotter/limited set for the largest instances to bound runtime)
SCHEDULE_LOCATIONS_BY_SET = {
    "j30":  ["TX_Austin", "FL_Orlando", "GA_Atlanta", "AZ_Phoenix"],
    "j60":  ["TX_Austin", "FL_Orlando", "GA_Atlanta", "AZ_Phoenix"],
    "j90":  ["TX_Austin", "AZ_Phoenix"],
    "j120": ["TX_Austin", "AZ_Phoenix"],
}

# Plotting
FIG_DPI = 300
