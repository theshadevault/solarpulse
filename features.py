"""Feature tensor X_t per 15-min block: physics twin output on NWP + NWP covariates."""
import numpy as np
import pandas as pd

import real_data
import scada
import twin

FEATURES = [
    "phys", "phys_cs", "nwp_kt", "ghi_spread_n", "dni_kb", "cloud", "cloud_low",
    "cloud_spread", "temp", "rh", "wind", "cos_zen", "sin_az", "cos_az", "hour",
    "doy_sin", "doy_cos", "clip_phys", "soiling_est", "lat", "tracker", "dc_ac",
    "phys_lag_d", "kt_trend",
]

# Feature families for the explainability card
GROUPS = {
    "Cloud cover (NWP)": ["cloud", "cloud_low", "nwp_kt", "dni_kb", "kt_trend"],
    "NWP model disagreement": ["ghi_spread_n", "cloud_spread"],
    "Module temperature & humidity": ["temp", "rh", "wind"],
    "Sun geometry & time": ["cos_zen", "sin_az", "cos_az", "hour", "doy_sin", "doy_cos"],
    "Physics twin level": ["phys", "phys_cs", "clip_phys", "phys_lag_d"],
    "Soiling & plant config": ["soiling_est", "lat", "tracker", "dc_ac"],
}


def build(plant, geo, nwp_hourly, soiling_est):
    nwp = twin.hourly_to_blocks(nwp_hourly, geo)
    ghi = nwp["shortwave_radiation"].values
    dni = nwp["direct_normal_irradiance"].values
    dhi = nwp["diffuse_radiation"].values
    temp = nwp["temperature_2m"].values
    wind = nwp["wind_speed_10m"].values / 3.6
    phys = twin.simulate(plant, geo, ghi, dni, dhi, temp, wind, soiling=soiling_est,
                         gamma=plant.gamma)
    cs = twin.simulate(plant, geo, geo["cs_ghi"], geo["cs_dni"], geo["cs_dhi"], temp, wind,
                       soiling=soiling_est, gamma=plant.gamma)
    f = pd.DataFrame(index=geo.index)
    f["phys"] = phys["p_ac"].values / plant.ac_mw
    f["phys_cs"] = cs["p_ac"].values / plant.ac_mw
    csg = np.maximum(geo["cs_ghi"].values, 1)
    f["nwp_kt"] = np.where(geo["cs_ghi"] > 20, ghi / csg, 0).clip(0, 1.3)
    f["ghi_spread_n"] = np.where(geo["cs_ghi"] > 20, nwp["shortwave_radiation_spread"].values / csg, 0)
    f["dni_kb"] = np.where(geo["cs_dni"] > 20, dni / np.maximum(geo["cs_dni"].values, 1), 0).clip(0, 1.3)
    f["cloud"] = nwp["cloud_cover"].values
    f["cloud_low"] = nwp["cloud_cover_low"].values
    f["cloud_spread"] = nwp["cloud_cover_spread"].values
    f["temp"] = temp
    f["rh"] = nwp["relative_humidity_2m"].values
    f["wind"] = wind
    f["cos_zen"] = np.cos(np.radians(geo["zenith"].values))
    f["sin_az"] = np.sin(np.radians(geo["azimuth"].values))
    f["cos_az"] = np.cos(np.radians(geo["azimuth"].values))
    f["hour"] = geo.index.hour + geo.index.minute / 60
    doy = geo.index.dayofyear.values
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    f["clip_phys"] = phys["clipped"].values
    f["soiling_est"] = soiling_est
    f["lat"], f["tracker"], f["dc_ac"] = plant.lat, float(plant.tracker), plant.dc_ac
    f["phys_lag_d"] = pd.Series(f["phys"].values).shift(96).bfill().values
    f["kt_trend"] = pd.Series(f["nwp_kt"].values).diff(4).fillna(0).values
    f["daylight"] = (geo["cs_ghi"].values > 10).astype(float)
    f["cs_ghi"] = geo["cs_ghi"].values
    f["t_cell_phys"] = phys["t_cell"].values
    f["poa_phys"] = phys["poa"].values
    f["nwp_ghi"] = ghi
    return f


def plant_frame(plant, geo, nwp_hourly, truth_hourly, soiling_start=0.985, live=False):
    """Full per-block frame: SCADA (simulated or real), sanitization outputs and features."""
    if plant.kind == "pvdaq":
        sc = (real_data.live_stub if live else real_data.load)(plant, geo, truth_hourly)
    else:
        wx = scada.truth_weather(plant, geo, truth_hourly)
        sc = scada.simulate_scada(plant, geo, wx, soiling_start=soiling_start)
    clean = scada.decurtail(plant, sc, geo, soiling_fill=soiling_start)
    feats = build(plant, geo, nwp_hourly, clean["soiling_est"].values)
    df = pd.concat([feats, sc.rename(columns={"wind": "wind_obs"}), clean.drop(columns=[c for c in clean if c in feats])], axis=1)
    df["plant"] = plant.id
    df["date"] = df.index.date
    df["block"] = np.tile(np.arange(96), len(df) // 96)
    df.attrs = dict(sc.attrs)
    return df
