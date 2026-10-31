"""Layer 2: pvlib physics-informed PV digital twin ("white-box engine").

SPA solar position -> single-axis tracker with backtracking (or fixed tilt)
-> Perez-Driesse transposition -> ASHRAE IAM + soiling -> Faiman cell temperature
-> PVWatts DC with temperature coefficient -> inverter efficiency + AC clipping.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import pvlib


HALF_BLOCK = pd.Timedelta(minutes=7.5)


def blocks(start, days, tz="Asia/Kolkata"):
    """15-minute block start times (96 per day) in the plant's local standard time."""
    return pd.date_range(pd.Timestamp(start, tz=tz), periods=96 * days, freq="15min")


@lru_cache(maxsize=64)
def _geometry_cached(plant, start, n):
    idx = pd.date_range(start, periods=n, freq="15min")
    centers = idx + HALF_BLOCK
    loc = pvlib.location.Location(plant.lat, plant.lon, tz=plant.tz, altitude=250)
    sp = loc.get_solarposition(centers)
    cs = loc.get_clearsky(centers, model="ineichen", solar_position=sp)
    g = pd.DataFrame({
        "zenith": sp["zenith"].values, "app_zenith": sp["apparent_zenith"].values,
        "azimuth": sp["azimuth"].values, "cs_ghi": cs["ghi"].values,
        "cs_dni": cs["dni"].values, "cs_dhi": cs["dhi"].values}, index=idx)
    if plant.tracker:
        tr = pvlib.tracking.singleaxis(sp["apparent_zenith"], sp["azimuth"], axis_tilt=0,
                                       axis_azimuth=180, max_angle=60, backtrack=True,
                                       gcr=plant.gcr)
        g["surf_tilt"] = tr["surface_tilt"].fillna(0).values
        g["surf_az"] = tr["surface_azimuth"].fillna(180).values
    else:
        g["surf_tilt"] = plant.tilt
        g["surf_az"] = 180.0
    g["aoi"] = pvlib.irradiance.aoi(g["surf_tilt"], g["surf_az"], g["app_zenith"], g["azimuth"])
    g["dni_extra"] = pvlib.irradiance.get_extra_radiation(centers).values
    g["airmass"] = pvlib.atmosphere.get_relative_airmass(g["app_zenith"]).values
    g["daylight"] = (g["zenith"] < 88).astype(float)
    return g


def geometry(plant, idx):
    return _geometry_cached(plant, idx[0], len(idx)).copy()


def hourly_to_blocks(hourly, geo, prefix_cols=None):
    """Interpolate hourly Open-Meteo data onto 15-min blocks.

    Open-Meteo radiation is the mean of the preceding hour, so each hourly value is placed
    at the hour's midpoint. Irradiance is interpolated as clear-sky index (not raw W/m2) so
    sunrise/sunset ramps keep the right shape.
    """
    h = hourly.copy()
    h.index = h.index - pd.Timedelta(minutes=30)
    centers = geo.index + HALF_BLOCK
    full = h.reindex(h.index.union(centers)).sort_index()

    # clear-sky reference at the hourly midpoints, to form indices
    cs_h = np.interp(h.index.asi8, centers.asi8, geo["cs_ghi"].values)
    csdni_h = np.interp(h.index.asi8, centers.asi8, geo["cs_dni"].values)
    out = pd.DataFrame(index=geo.index)
    for col in h.columns:
        if col in ("shortwave_radiation", "direct_normal_irradiance"):
            ref = cs_h if col == "shortwave_radiation" else csdni_h
            k = pd.Series(np.where(ref > 30, h[col].values / np.maximum(ref, 1), np.nan),
                          index=h.index).clip(0, 1.3)
            k = k.reindex(full.index).interpolate(limit_direction="both").reindex(centers)
            refb = geo["cs_ghi"].values if col == "shortwave_radiation" else geo["cs_dni"].values
            out[col] = np.nan_to_num(k.values, nan=0.0) * refb
        elif col == "diffuse_radiation":
            continue  # rebuilt from closure below
        else:
            s = full[col].interpolate(limit_direction="both").reindex(centers)
            out[col] = s.values
    cosz = np.cos(np.radians(geo["zenith"].clip(upper=90))).values
    if "shortwave_radiation" in out and "direct_normal_irradiance" in out:
        dni = np.minimum(out["direct_normal_irradiance"].values,
                         out["shortwave_radiation"].values / np.maximum(cosz, 0.05))
        out["direct_normal_irradiance"] = dni
        out["diffuse_radiation"] = np.clip(out["shortwave_radiation"].values - dni * cosz, 0, None)
    return out


def from_poa(plant, poa, t_mod, soiling=1.0, gamma=-0.0038):
    """Twin driven by measured plane-of-array irradiance and back-of-module temperature
    (used for de-curtailment: what the plant *could* have produced)."""
    poa = np.clip(np.asarray(poa, float), 0, None)
    e_eff = poa * 0.965 * soiling
    t_cell = np.asarray(t_mod, float) + poa / 1000 * 3.0   # Sandia back-of-module dT
    p_dc = pvlib.pvsystem.pvwatts_dc(e_eff, t_cell, plant.dc_mw, gamma)
    p_ac = pvlib.inverter.pvwatts(p_dc, pdc0=plant.ac_mw / 0.97, eta_inv_nom=0.97)
    return np.clip(np.nan_to_num(np.asarray(p_ac)), 0, plant.ac_mw)


def simulate(plant, geo, ghi, dni, dhi, temp_air, wind_ms, soiling=1.0,
             gamma=-0.0038, u0=25.0, u1=6.84, dc_scale=1.0):
    """Run the physical chain. Returns a DataFrame with POA, cell temp, DC and AC (MW)."""
    ghi = np.asarray(ghi, float)
    poa = pvlib.irradiance.get_total_irradiance(
        geo["surf_tilt"], geo["surf_az"], geo["app_zenith"], geo["azimuth"],
        dni=np.asarray(dni, float), ghi=ghi, dhi=np.asarray(dhi, float),
        dni_extra=geo["dni_extra"], airmass=geo["airmass"], albedo=0.25,
        model="perez-driesse")
    poa = poa.fillna(0)
    iam = pvlib.iam.ashrae(geo["aoi"], b=0.05).fillna(0)
    e_eff = (poa["poa_direct"] * iam + poa["poa_sky_diffuse"] * 0.95
             + poa["poa_ground_diffuse"] * 0.9) * soiling
    e_eff = e_eff.clip(lower=0) * geo["daylight"]
    t_cell = pvlib.temperature.faiman(poa["poa_global"], temp_air, wind_ms, u0=u0, u1=u1)
    pdc0 = plant.dc_mw * dc_scale
    p_dc = pvlib.pvsystem.pvwatts_dc(e_eff, t_cell, pdc0, gamma)
    pac0 = plant.ac_mw
    p_ac = pvlib.inverter.pvwatts(p_dc, pdc0=pac0 / 0.97, eta_inv_nom=0.97)
    p_ac = np.clip(np.nan_to_num(p_ac), 0, pac0)
    return pd.DataFrame({
        "poa": poa["poa_global"].values * geo["daylight"].values,
        "e_eff": e_eff.values, "t_cell": np.asarray(t_cell),
        "p_dc": np.clip(np.nan_to_num(np.asarray(p_dc)), 0, None),
        "p_ac": p_ac, "clipped": (np.asarray(p_dc) * 0.97 > pac0).astype(float),
    }, index=geo.index)
