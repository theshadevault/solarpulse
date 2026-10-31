# SolarPulse: short-term solar forecasting & dispatch prototype

Working prototype of the 5-layer "SolarPulse AI" blueprint: physics-guided hybrid
forecasting for 96 × 15-min DSM blocks, calibrated uncertainty, cost-aware scheduling,
and a cockpit for plant operators and grid managers.

```bash
pip install -r requirements.txt
python server.py          # first run downloads data + trains (~3 min) -> http://127.0.0.1:8010
python pipeline.py        # force a rebuild: fetch, train, backtest
python kaggle_india.py    # re-run the Indian plant analysis only
```

## Data

| Plants | Telemetry | Forecasts | Used for |
|---|---|---|---|
| **SR_CO**, Kersey CO (4.7 MWdc tracker) | **Real**: NREL PVDAQ 9068, 8 inverter modules, POA pyranometers, module temps, 2024-01 → 2025-04 | Real archived ECMWF/GFS/ICON | Full forecasting backtest |
| **Farm Solar Array**, Arbuckle CA (0.9 MWdc fixed) | **Real**: NREL PVDAQ 2107, revenue meter, POA, weather, 2024-02 → 2025-12 | Real archived NWP | Full forecasting backtest |
| **Kaggle Plants 1 & 2**, India (~28 MW each) | **Real**: 22 inverters each, irradiance/temperature sensor, May–Jun 2020 | none (no 2020 archive) | Data QC, inverter health, twin calibration, locating the plants |
| Bhadla, Charanka, Pavagada, Kamuthi (India) | **Simulated** from ERA5 with hidden soiling, outages, curtailment, sensor faults | Real archived + live NWP | Indian DSM demo, grid desk, de-curtailment benchmark |

All data is public and downloaded automatically without login (OEDI S3, Kaggle public download, Open-Meteo).
Real and simulated plants use separate models. Live mode for the US plants is forecast-only, because no live telemetry is public.

### Sources
- NREL PVDAQ 2023 Solar Data Prize (OEDI): https://data.openei.org/submissions/4568 · files: https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/ (systems `9068_OEDI`, `2107_OEDI`)
- Kaggle, Solar Power Generation Data: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
- Open-Meteo (CC BY 4.0): [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) (archived ECMWF IFS, GFS, ICON) · [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5) · [Forecast API](https://open-meteo.com/en/docs) (live forecasts, cloud grid)

## Data problems found by the QC layer
- SR_CO revenue meter reads up to 16 MW on a 3.8 MW plant from Aug 2024 (r = 0.49 vs inverters) → plant output uses the inverter sum instead.
- SR_CO thermocouple pad 2 #1 stuck at 3276.7 °C (disconnected sensor) → rejected automatically.
- Farm Solar Array logs one file in UTC and others in DST wall-clock time; the meter is 5 min off the pyranometer → aligned by correlation.
- Kaggle Plant 1 DC_POWER is 10× too large (unit error) → rescaled. Plant 2 has inverters at 0 kW for 14% of sunny inverter-hours → treated as outages, and the lost energy is reported.

## Backtest (held-out days)

| | Persistence | Physics twin | Hybrid P50 | 80% band coverage |
|---|---|---|---|---|
| **Real US plants** (174 plant-days) | 27.7% | 20.3% | **14.9%** nRMSE (+46% skill) | 80.1% |
| Simulated India fleet (108 plant-days) | 15.8% | 11.1% | **10.2%** nRMSE | 80.4% |

On the real plants, the S* schedule cuts DSM charges 57% vs scheduling the physics model alone and 5.6% vs submitting P50. The DSM charges use simplified CERC-style slabs applied as if the plants were in India.
On the Kaggle Indian plants, the digital twin calibrated on half the days scores R² 0.99 (Plant 1) and 0.65 (Plant 2) on the other half.

## Layers → code

| Layer | File |
|---|---|
| L1 Ingest & sanitize | `data_sources.py`, `real_data.py`, `scada.py`, `kaggle_india.py` |
| L2 Physics twin & nowcast | `twin.py`, `nowcast.py` |
| L3 Hybrid ML | `features.py`, `model.py` |
| L4 Risk & dispatch | `model.py`, `dsm.py` |
| L5 Cockpit & XAI | `server.py`, `static/index.html` |

Settings (DSM slabs, PPA rate, under-injection multiplier, revision trigger, plant specs) are in `config.py`.

## Hosting (Vercel)

Live at **https://solarpulse-dusky.vercel.app** (Vercel project `solarpulse`, FastAPI preset, entrypoint `server.py`).

```bash
python pipeline.py          # rebuild models + data/dataset_serve.pkl locally first
vercel deploy --prod --yes  # upload; .vercelignore keeps raw data and the full dataset out
```

- The deployment ships only held-out replay days (`dataset_serve.pkl`, ~15 MB) and the trained models. It never trains.
- `requirements.txt` uses `xgboost-cpu` and `opencv-python-headless` to fit Vercel's 500 MB Python limit. Models are served via the core XGBoost Booster API, so scikit-learn isn't needed.
- On Vercel the API cache and approved revisions live in `/tmp`. Revisions are per-instance and reset on cold starts.
