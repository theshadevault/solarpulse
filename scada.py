"""Layer 1: simulated plant SCADA + the sanitization engine.

No real plant telemetry is available for a prototype, so each plant is simulated from
reanalysis weather with physics the forecaster does NOT know (true soiling drift, a
different temperature coefficient, inverter outages, grid curtailment, sensor faults).
The sanitization engine then has to recover the clean, unconstrained signal from the
messy SCADA, exactly as it would on a real plant.
"""
import numpy as np
import pandas as pd

import twin

N_UNITS = 20          # inverter blocks per plant


def _day_rng(plant, day):
    # stable across runs (python hash is salted) -> derive from the date ordinal
    return np.random.default_rng(plant.seed * 100003 + pd.Timestamp(day).toordinal())


def truth_weather(plant, geo, hourly):
    """Reanalysis weather on 15-min blocks plus sub-hourly cloud variability."""
    w = twin.hourly_to_blocks(hourly, geo)
    days = sorted(set(geo.index.date))
    noise = np.zeros(len(geo))
    for i, d in enumerate(days):
        rng = _day_rng(plant, d)
        e = rng.normal(0, 1, 96)
        for k in range(1, 96):
            e[k] = 0.75 * e[k - 1] + 0.66 * e[k]
        noise[i * 96:(i + 1) * 96] = e
    cc = (w["cloud_cover"].values / 100).clip(0, 1)
    amp = 0.30 * 4 * cc * (1 - cc) + 0.02
    cs = geo["cs_ghi"].values
    ghi_s, dni_s = w["shortwave_radiation"].values, w["direct_normal_irradiance"].values
    ghi = np.clip(ghi_s * (1 + amp * noise), 0, 1.25 * cs + 1)
    dni = np.clip(dni_s * (1 + 1.6 * amp * noise), 0, geo["cs_dni"].values * 1.1 + 1)
    cosz = np.cos(np.radians(geo["zenith"].clip(upper=90))).values
    dni = np.minimum(dni, ghi / np.maximum(cosz, 0.05))
    return pd.DataFrame({
        "ghi": ghi, "dni": dni, "dhi": np.clip(ghi - dni * cosz, 0, None),
        "ghi_smooth": ghi_s, "dni_smooth": dni_s,
        "dhi_smooth": np.clip(ghi_s - np.minimum(dni_s, ghi_s / np.maximum(cosz, .05)) * cosz, 0, None),
        "temp": w["temperature_2m"].values, "wind": w["wind_speed_10m"].values / 3.6,
        "cloud": w["cloud_cover"].values,
        "precip": w["precipitation"].values if "precipitation" in w else 0.0,
    }, index=geo.index)


def simulate_scada(plant, geo, wx, soiling_start=0.985):
    """Produce what the plant's SCADA/WMS would log, plus hidden ground truth."""
    days = sorted(set(geo.index.date))
    n = len(geo)
    soiling = np.ones(n)
    online = np.full(n, N_UNITS, float)
    setpoint = np.full(n, plant.ac_mw)
    curtail = np.zeros(n)
    s = soiling_start
    daily_rain = wx["precip"].groupby(wx.index.date).sum() * 0.25 if np.ndim(wx["precip"]) else None
    for i, d in enumerate(days):
        rng = _day_rng(plant, d)
        sl = slice(i * 96, (i + 1) * 96)
        rain = float(daily_rain.get(d, 0)) if daily_rain is not None else 0.0
        if rain > 4:
            s = min(0.995, s + 0.03 * min(rain / 4, 2))
        elif (pd.Timestamp(d).toordinal() + plant.seed) % 24 == 0:
            s = 0.995                                    # scheduled module washing
        else:
            s = max(0.90, s - rng.uniform(0.0012, 0.0030))
        soiling[sl] = s
        if rng.random() < 0.12:                          # inverter trip / maintenance
            start = rng.integers(28, 64)
            online[i * 96 + start: i * 96 + min(96, start + rng.integers(4, 28))] -= rng.integers(1, 4)
        kt_noon = wx["ghi_smooth"].values[i * 96 + 48] / max(geo["cs_ghi"].values[i * 96 + 48], 1)
        p_curt = 0.30 if plant.state in ("Rajasthan", "Gujarat") else 0.15
        if kt_noon > 0.75 and rng.random() < p_curt:     # SLDC midday congestion
            a, b = rng.integers(42, 48), rng.integers(54, 62)
            setpoint[i * 96 + a: i * 96 + b] = plant.ac_mw * rng.uniform(0.5, 0.75)
            curtail[i * 96 + a: i * 96 + b] = 1

    sim = twin.simulate(plant, geo, wx["ghi"], wx["dni"], wx["dhi"], wx["temp"], wx["wind"],
                        soiling=soiling, gamma=-0.0042, u0=23.5, dc_scale=0.985)
    frac = online / N_UNITS
    p_avail = sim["p_ac"].values * frac
    rng = np.random.default_rng(plant.seed)
    p_scada = np.minimum(p_avail, setpoint) * (1 + rng.normal(0, 0.003, n))
    curtail = curtail * (p_avail > setpoint)              # only binding curtailment counts
    poa_obs = sim["poa"].values * (1 + rng.normal(0, 0.01, n))
    ghi_obs = wx["ghi"].values * (1 + rng.normal(0, 0.01, n))
    t_mod = sim["t_cell"].values - sim["poa"].values / 1000 * 3 + rng.normal(0, 0.4, n)

    # WMS sensor faults: frozen Modbus registers and dew/dirt on the pyranometer
    for i, d in enumerate(days):
        r2 = _day_rng(plant, d)
        if r2.random() < 0.05:
            a = i * 96 + r2.integers(36, 60)
            b = min(a + r2.integers(6, 20), (i + 1) * 96)
            poa_obs[a:b] = poa_obs[a]
            ghi_obs[a:b] = ghi_obs[a]
        if r2.random() < 0.04:
            a = i * 96 + 26
            poa_obs[a:a + 10] *= 0.55
            ghi_obs[a:a + 10] *= 0.55

    return pd.DataFrame({
        "p_scada": np.clip(p_scada, 0, None) * geo["daylight"].values,
        "setpoint": setpoint, "curtail": curtail, "online": online, "units": N_UNITS,
        "avc": plant.ac_mw * frac, "poa_obs": poa_obs, "ghi_obs": ghi_obs, "t_mod": t_mod,
        "t_amb": wx["temp"].values,
        # hidden truth, used only for evaluation
        "p_avail_true": p_avail, "p_full_true": sim["p_ac"].values,
        "soiling_true": soiling, "clipped_true": sim["clipped"].values,
        # coarse satellite-derived irradiance for imputation
        "sat_ghi": wx["ghi_smooth"].values, "sat_dni": wx["dni_smooth"].values,
        "sat_dhi": wx["dhi_smooth"].values, "wind": wx["wind"].values,
    }, index=geo.index)


# ---------------------------------------------------------------- sanitization engine

def qc_flags(sc, geo):
    """Clear-sky envelope + frozen-register + cross-sensor checks on the WMS."""
    cs = geo["cs_ghi"].values
    day = cs > 80
    ghi = sc["ghi_obs"].values
    over = day & (ghi > 1.35 * cs + 20)
    frozen = (pd.Series(sc["poa_obs"].values).rolling(3).std().fillna(1).values < 1e-6) & day
    frozen = pd.Series(frozen).rolling(3, min_periods=1).max().shift(-2).fillna(0).astype(bool).values | frozen
    # pyranometer reading far below the satellite estimate in clear-ish sky -> dew/soiled
    sat = sc["sat_ghi"].values
    low = day & (sat > 300) & (ghi < 0.7 * sat)
    return pd.DataFrame({"qc_over": over, "qc_frozen": frozen, "qc_low": low,
                         "qc_bad": over | frozen | low}, index=sc.index)


def clean_poa(plant, sc, geo, flags):
    """Replace flagged POA readings with satellite irradiance transposed by the twin."""
    sat = twin.simulate(plant, geo, sc["sat_ghi"], sc["sat_dni"], sc["sat_dhi"],
                        sc["t_amb"], sc["wind"])
    poa = np.where(flags["qc_bad"].values, sat["poa"].values, sc["poa_obs"].values)
    return np.clip(poa, 0, None)


def soiling_estimate(plant, sc, geo, poa, fill=0.985):
    """14-day EWMA of (clear-sky noon SCADA output) / (twin output at measured POA)."""
    ref = twin.from_poa(plant, poa, sc["t_mod"].values, gamma=plant.gamma) * sc["online"].values / sc["units"].values
    hour = sc.index.hour + sc.index.minute / 60
    good = ((hour >= 10.5) & (hour <= 14) & (sc["curtail"].values == 0) & (ref > 0.3 * plant.ac_mw)
            & (ref < 0.97 * plant.ac_mw * sc["online"].values / sc["units"].values))
    ratio = pd.Series(np.where(good, sc["p_scada"].values / np.maximum(ref, 1e-6), np.nan),
                      index=sc.index)
    daily = ratio.groupby(ratio.index.date).median()
    est = daily.ewm(span=14, min_periods=1).mean().clip(0.70, 1.05).ffill().fillna(fill)
    # use yesterday's estimate for today (no look-ahead)
    est = est.shift(1).fillna(est.iloc[0] if len(est) else fill)
    return pd.Series(sc.index.date, index=sc.index).map(est).values


def decurtail(plant, sc, geo, soiling_fill=0.985):
    """Reconstruct unconstrained, full-availability generation (normalized 0..1)."""
    flags = qc_flags(sc, geo)
    poa = clean_poa(plant, sc, geo, flags)
    soil = soiling_estimate(plant, sc, geo, poa, fill=soiling_fill)
    frac = sc["online"].values / sc["units"].values
    scaled = sc["p_scada"].values / np.maximum(frac, 0.05)
    twin_p = twin.from_poa(plant, poa, sc["t_mod"].values, soiling=soil, gamma=plant.gamma)
    p_unc = np.where(sc["curtail"].values > 0, twin_p, scaled)
    out = flags.copy()
    out["poa_clean"] = poa
    out["soiling_est"] = soil
    out["p_unc"] = np.clip(p_unc, 0, plant.ac_mw)
    out["y"] = out["p_unc"] / plant.ac_mw
    return out
