# Data Dictionary

## `data_intermediate/environmental_heat_cases.csv`
Primary incident subset filtered from occupational safety records.
All original registry fields are retained, plus derived variables:

| Column | Description |
|---|---|
| `EventDate_parsed` | Parsed incident date |
| `year`, `month`, `dow` | Event year, month, day-of-week (0=Mon) |
| `lat`, `lon` | Numeric latitude/longitude coordinates |
| `naics3` | 3-digit Primary NAICS industry code |
| `geo_valid` | Boolean indicating coordinates fall within national weather frame |
| `heat_via_event`, `heat_via_source`, `heat_via_nature` | Registry code trigger flags |

## `data_intermediate/weather_matches.csv` (and `_negcontrol.csv`)
Matched case-day and referent-day observations for time-stratified case-crossover modeling.

| Column | Description | Source |
|---|---|---|
| `case_id`, `stratum` | Unique incident ID; case-crossover stratum identifier | Registry |
| `date` | Calendar date | Derived |
| `is_case` | 1 = incident day, 0 = referent (control) day | Derived |
| `lat`, `lon`, `state`, `naics3` | Spatial and industry attributes | Registry |
| `year`, `month`, `doy`, `dow` | Calendar features | Derived |
| `tmax`, `tmin`, `tavg` | Daily maximum, minimum, and mean temperature (°C) | Meteorological station |
| `prcp` | Daily precipitation (mm) | Meteorological station |
| `heat_index` | Daily maximum apparent heat index (°C) | Hourly station observations |
| `station_id`, `distance_km` | Station identifier and great-circle distance | Station metadata |
| `hi_station_km` | Station distance for heat-index calculations | Station metadata |
| `source` | Station observation vs. flagged fallback flag | Provenance flag |
| `tmax_lag1` | Previous-day maximum temperature (°C) | Meteorological station |
| `hot_days_prior3`, `hot_days_prior7` | Cumulative hot-day counts in prior 3/7 days | Derived |
| `prcp_flag` | Binary indicator for precipitation presence | Derived |

## `data_external/`
- `risk_calendars/<loc>_<year>.csv` — Date-indexed thermal risk calendars (date, tmax, heat_index, risk_score).
- `weather_cache/` — Cached station daily observations.
