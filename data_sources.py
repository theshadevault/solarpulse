"""Layer 1 (external feeds): Open-Meteo NWP, ERA5 reanalysis and gridded cloud fields.

Everything is cached on disk so the pipeline is reproducible and works offline after
the first run.
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd
import requests

from config import CACHE, NWP_MODELS, NWP_VARS

HIST_FC_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TRUTH_VARS = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
              "temperature_2m", "wind_speed_10m", "cloud_cover", "precipitation"]


def _get(url, params, max_age_s=None):
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:16]
    path = CACHE / f"{key}.json"
    if path.exists() and (max_age_s is None or time.time() - path.stat().st_mtime < max_age_s):
        return json.loads(path.read_text())
    last = None
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                data = r.json()
                CACHE.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
                return data
            last = f"{r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 * (attempt + 1))
    if path.exists():  # stale cache beats no data
        return json.loads(path.read_text())
    raise RuntimeError(f"Open-Meteo request failed: {last}")


def _frame(block, tz):
    df = pd.DataFrame(block)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(tz)
    return df.set_index("time").astype(float)


def _multi_model(data, tz):
    """Split '<var>_<model>' columns into per-variable ensemble mean and spread."""
    df = _frame(data["hourly"], tz)
    out = pd.DataFrame(index=df.index)
    for v in NWP_VARS:
        cols = [f"{v}_{m}" for m in NWP_MODELS if f"{v}_{m}" in df]
        if not cols:
            continue
        out[v] = df[cols].mean(axis=1)
        out[v + "_spread"] = df[cols].std(axis=1)
    out["n_models"] = df[[f"shortwave_radiation_{m}" for m in NWP_MODELS
                          if f"shortwave_radiation_{m}" in df]].notna().sum(axis=1)
    return out


def historical_nwp(plant, start, end):
    """Archived multi-model NWP forecasts (what the forecaster would have seen)."""
    data = _get(HIST_FC_URL, {
        "latitude": plant.lat, "longitude": plant.lon, "start_date": start, "end_date": end,
        "hourly": ",".join(NWP_VARS), "models": ",".join(NWP_MODELS), "timezone": plant.tz})
    return _multi_model(data, plant.tz)


def live_nwp(plant):
    """Latest multi-model forecast: yesterday, today and the next two days."""
    data = _get(FORECAST_URL, {
        "latitude": plant.lat, "longitude": plant.lon, "past_days": 1, "forecast_days": 3,
        "hourly": ",".join(NWP_VARS), "models": ",".join(NWP_MODELS), "timezone": plant.tz},
        max_age_s=3600)
    return _multi_model(data, plant.tz)


def reanalysis(plant, start, end):
    """ERA5-based reanalysis weather: drives the simulated 'true' plant."""
    data = _get(ARCHIVE_URL, {
        "latitude": plant.lat, "longitude": plant.lon, "start_date": start, "end_date": end,
        "hourly": ",".join(TRUTH_VARS), "timezone": plant.tz})
    return _frame(data["hourly"], plant.tz)


def live_truth_proxy(plant):
    """Recent best-match analysis/short-range weather (stand-in for live sensors today)."""
    data = _get(FORECAST_URL, {
        "latitude": plant.lat, "longitude": plant.lon, "past_days": 1, "forecast_days": 2,
        "hourly": ",".join(TRUTH_VARS), "timezone": plant.tz}, max_age_s=1800)
    return _frame(data["hourly"], plant.tz)


def cloud_grid(lat, lon, n=9, step=0.25, past_hours=2, forecast_hours=4, tz="Asia/Kolkata"):
    """Cloud-cover field on an n x n grid around a point (Layer 2 nowcast input).

    Returns (times, lats, lons, cube[t, y, x]) with y running south -> north.
    """
    half = (n - 1) / 2
    lats = lat + (np.arange(n) - half) * step
    lons = lon + (np.arange(n) - half) * step
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    data = _get(FORECAST_URL, {
        "latitude": ",".join(f"{v:.3f}" for v in la.ravel()),
        "longitude": ",".join(f"{v:.3f}" for v in lo.ravel()),
        "minutely_15": "cloud_cover,shortwave_radiation",
        "past_minutely_15": past_hours * 4, "forecast_minutely_15": forecast_hours * 4,
        "timezone": tz}, max_age_s=900)
    if isinstance(data, dict):
        data = [data]
    times = pd.to_datetime(data[0]["minutely_15"]["time"])
    cube = np.array([d["minutely_15"]["cloud_cover"] for d in data], dtype=float)
    cube = np.nan_to_num(cube.T.reshape(len(times), n, n), nan=0.0)
    return times, lats, lons, cube
