"""Layer 1 for real plants: NREL PVDAQ telemetry -> the same SCADA frame the simulator makes.

Sources (public, no login): NREL PVDAQ 2023 Solar Data Prize sets on the OEDI data lake.
  9068  SR_CO, Kersey CO      4.74 MWdc single-axis tracker, CdTe, 8 inverter modules
  2107  Farm Solar Array, CA  0.89 MWdc fixed 25 deg, mono-Si, revenue meter
Each feed is aligned to the plant's fixed local standard time, aggregated to 15-min
blocks, and lag-corrected against the plane-of-array pyranometer.
"""
import numpy as np
import pandas as pd
import requests

import twin
from config import DATA

RAW = DATA / "raw"
OEDI = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize"
FILES = {
    "srco": ["9068_OEDI/data/9068_ac_power_data_20240101_20250430.csv",
             "9068_OEDI/data/9068_irradiance_data_20240101_20250430.csv",
             "9068_OEDI/data/9068_environment_data_20240101_20250430.csv"],
    "fsa": ["2107_OEDI/data/2107_meter_15m_data_2025.csv",
            "2107_OEDI/data/2107_irradiance_15m_data_2025.csv",
            "2107_OEDI/data/2107_environment_15m_data_2025.csv"],
}


def fetch(pid):
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES[pid]:
        path = RAW / f.split("/")[-1]
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"    downloading {path.name} from NREL PVDAQ ...")
        with requests.get(f"{OEDI}/{f}", stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(path, "wb") as out:
                for chunk in r.iter_content(1 << 20):
                    out.write(chunk)
    return [RAW / f.split("/")[-1] for f in FILES[pid]]


def _read(path, cols=None):
    df = pd.read_csv(path, index_col=0, usecols=cols)
    df.index = pd.to_datetime(df.index)
    return df[~df.index.duplicated()].sort_index().apply(pd.to_numeric, errors="coerce")


def _localize(df, src_tz, dst_tz):
    if src_tz == "America/Los_Angeles":   # wall clock with DST
        idx = df.index.tz_localize(src_tz, ambiguous="NaT", nonexistent="NaT")
    else:
        idx = df.index.tz_localize(src_tz)
    df = df[~idx.isna()]
    df.index = idx[~idx.isna()].tz_convert(dst_tz)
    return df


def _qc_channels(df, lo, hi, label, log):
    """Per-sensor range/stuck QC: drop channels that are mostly out of range or frozen."""
    keep = []
    for c in df.columns:
        x = df[c]
        bad = ((x < lo) | (x > hi)).mean()
        stuck = (x.diff().abs() < 1e-9).mean()
        if bad > 0.2 or stuck > 0.9:
            log.append(f"{label} '{c.split('_o_')[0]}' rejected: "
                       f"{bad:.0%} out of range (median {x.median():.1f}), {stuck:.0%} frozen")
        else:
            keep.append(c)
    out = df[keep].where((df[keep] >= lo) & (df[keep] <= hi))
    return out


def _best_lag(power, poa):
    """Shift (minutes) that best aligns a power series with the POA sensor (5-min grid)."""
    p = power.resample("5min").mean().ffill(limit=2)
    g = poa.resample("5min").mean()
    best, score = 0, -1
    for lag in range(-30, 31, 5):
        c = p.shift(freq=f"{lag}min").reindex(g.index).corr(g)
        if c > score:
            best, score = lag, c
    return best, score


def _blocks(df, geo):
    """5-min (or 15-min) samples -> 15-min block means labelled by block start."""
    out = df.resample("15min").mean()
    return out.reindex(geo.index)


def load(plant, geo, truth_hourly):
    """Real SCADA/WMS for `geo` blocks, in the column layout of scada.simulate_scada."""
    files = fetch(plant.id)
    tz = plant.tz
    log = []
    if plant.id == "srco":
        ac = _localize(_read(files[0]), "Etc/GMT+7", tz)
        mods = ac[[c for c in ac.columns if c.startswith("inverter_module")]].clip(lower=0)
        irr = _localize(_read(files[1]), "Etc/GMT+7", tz)
        poa = _qc_channels(irr[[c for c in irr.columns if "temp_compensated" in c]], -20, 1600,
                           "Pyranometer", log).median(axis=1)
        env = _localize(_read(files[2], cols=[0, 1, 2, 3, 4, 5]), "Etc/GMT+7", tz)
        t_amb = _qc_channels(env.iloc[:, [0]], -45, 50, "Ambient sensor", log).iloc[:, 0]
        t_mod = _qc_channels(env.iloc[:, 1:5], -45, 95, "Thermocouple", log).median(axis=1)
        power = mods.sum(axis=1, min_count=6) / 1000                  # kW -> MW
        lag, corr = _best_lag(power, poa)
        power = power.shift(freq=f"{lag}min")
        mods = mods.shift(freq=f"{lag}min")
        mod_b = _blocks(mods, geo)
        units = mods.shape[1]
        producing = mod_b.max(axis=1)
        online = (mod_b > 0.15 * producing.values[:, None]).sum(axis=1).astype(float)
        night = (geo["cs_ghi"] < 80).values | (producing.fillna(0).values < 0.03 * plant.ac_mw * 1000 / units)
        online = np.where(night, units, online)
        wind = None
    else:
        meter = _localize(_read(files[0]), "America/Los_Angeles", tz).iloc[:, 0].clip(lower=0)
        irr = _localize(_read(files[1]), "UTC", tz)
        poa = _qc_channels(irr.iloc[:, [0]], -20, 1600, "Pyranometer", log).iloc[:, 0]
        env = _localize(_read(files[2]), "UTC", tz)
        t_amb = (env.iloc[:, 0] - 32) * 5 / 9                         # F -> C
        wind = env.iloc[:, 1] * 0.44704                                # mph -> m/s
        power = meter / 1000
        lag, corr = _best_lag(power, poa)
        power = power.shift(freq=f"{lag}min")
        units, online = 1, np.ones(len(geo))
        t_mod = None

    b = pd.DataFrame(index=geo.index)
    b["p_scada"] = _blocks(power.to_frame(), geo).iloc[:, 0]
    b["poa_obs"] = _blocks(poa.to_frame(), geo).iloc[:, 0].clip(lower=0)
    b["t_amb"] = _blocks(t_amb.to_frame(), geo).iloc[:, 0]
    if t_mod is not None:
        b["t_mod"] = _blocks(t_mod.to_frame(), geo).iloc[:, 0]
    b["wind_obs"] = _blocks(wind.to_frame(), geo).iloc[:, 0] if wind is not None else np.nan

    w = twin.hourly_to_blocks(truth_hourly, geo)
    sat = pd.DataFrame({"ghi": w["shortwave_radiation"], "dni": w["direct_normal_irradiance"],
                        "dhi": w["diffuse_radiation"], "temp": w["temperature_2m"],
                        "wind": w["wind_speed_10m"] / 3.6}, index=geo.index)
    # short gaps are interpolated; longer ones stay NaN (those days are dropped later)
    for c in ("p_scada", "poa_obs", "t_amb"):
        b[c] = b[c].interpolate(limit=2, limit_area="inside")
    b["t_amb"] = b["t_amb"].fillna(sat["temp"])
    wind_ms = b["wind_obs"].fillna(sat["wind"])
    if t_mod is None:     # no back-of-module sensor: Faiman from measured POA and weather
        import pvlib
        t_cell = pvlib.temperature.faiman(b["poa_obs"].fillna(0), b["t_amb"], wind_ms)
        b["t_mod"] = t_cell - b["poa_obs"].fillna(0) / 1000 * 3
    b["t_mod"] = b["t_mod"].interpolate(limit=4, limit_area="inside").fillna(b["t_amb"])
    night = (geo["cs_ghi"] <= 0).values
    b.loc[night, "p_scada"] = b.loc[night, "p_scada"].fillna(0)
    b.loc[night, "poa_obs"] = b.loc[night, "poa_obs"].fillna(0)

    n = len(geo)
    out = pd.DataFrame({
        "p_scada": b["p_scada"].values, "setpoint": plant.ac_mw, "curtail": 0.0,
        "online": np.asarray(online, float), "units": float(units),
        "avc": plant.ac_mw * np.asarray(online, float) / units,
        "poa_obs": b["poa_obs"].values, "ghi_obs": np.full(n, np.nan),
        "t_mod": b["t_mod"].values, "t_amb": b["t_amb"].values,
        "p_avail_true": np.full(n, np.nan), "p_full_true": np.full(n, np.nan),
        "soiling_true": np.full(n, np.nan), "clipped_true": np.full(n, np.nan),
        "sat_ghi": sat["ghi"].values, "sat_dni": sat["dni"].values, "sat_dhi": sat["dhi"].values,
        "wind": wind_ms.values,
    }, index=geo.index)
    out.attrs.update(lag_min=lag, lag_corr=round(float(corr), 3), sensor_log=log)
    return out


def live_stub(plant, geo, truth_hourly):
    """No live telemetry is public for these plants: forecast-only frame."""
    w = twin.hourly_to_blocks(truth_hourly, geo)
    n = len(geo)
    nan = np.full(n, np.nan)
    return pd.DataFrame({
        "p_scada": nan, "setpoint": plant.ac_mw, "curtail": 0.0, "online": 1.0, "units": 1.0,
        "avc": plant.ac_mw, "poa_obs": nan, "ghi_obs": nan, "t_mod": w["temperature_2m"].values,
        "t_amb": w["temperature_2m"].values, "p_avail_true": nan, "p_full_true": nan,
        "soiling_true": nan, "clipped_true": nan, "sat_ghi": w["shortwave_radiation"].values,
        "sat_dni": w["direct_normal_irradiance"].values, "sat_dhi": w["diffuse_radiation"].values,
        "wind": w["wind_speed_10m"].values / 3.6}, index=geo.index)
