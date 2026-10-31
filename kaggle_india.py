"""Real Indian plant data: Kaggle "Solar Power Generation Data" (2 plants, 15 May - 17 Jun 2020).

22 inverters per plant, 15-min DC/AC power, one weather sensor (irradiation, ambient and
module temperature). There is no archived NWP for 2020 and the dataset does not say
where the plants are, so this module exercises the layers that do not need a forecast:
  * data QC (unit errors, gaps, frozen/implausible sensors)
  * inverter health (underperformers, trips while the sun is up)
  * digital-twin calibration and validation on real Indian telemetry
  * locating the plant from its own sensor (solar noon -> longitude, day length -> latitude)
"""
import json
import zipfile

import numpy as np
import pandas as pd
import pvlib
import requests

import data_sources as ds
from config import DATA

ZIP = DATA / "raw" / "kaggle_india_solar.zip"
URL = "https://www.kaggle.com/api/v1/datasets/download/anikannal/solar-power-generation-data"
TZ = "Asia/Kolkata"
GAMMA = -0.0040


def fetch():
    if not ZIP.exists():
        ZIP.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        ZIP.write_bytes(r.content)
    return zipfile.ZipFile(ZIP)


def _read(z, name):
    d = pd.read_csv(z.open(name))
    fmt = "%d-%m-%Y %H:%M" if d["DATE_TIME"].iloc[0][2] == "-" else "%Y-%m-%d %H:%M:%S"
    d["t"] = pd.to_datetime(d["DATE_TIME"], format=fmt)
    return d


def locate(w):
    """Longitude from solar noon, latitude from day length, both from the irradiance sensor."""
    s = w["E"]
    days = []
    for d, g in s.groupby(s.index.date):
        g = g[g > 5]
        if len(g) < 30:
            continue
        mins = np.asarray(g.index.hour * 60 + g.index.minute, float) + 7.5
        noon = float((mins * g.values).sum() / g.values.sum())
        days.append({"date": d, "noon": noon, "rise": float(mins.min()), "set": float(mins.max()),
                     "smooth": float(np.abs(np.diff(g.values)).sum() / max(g.max(), 1))})
    dd = pd.DataFrame(days)
    clear = dd[dd["smooth"] <= dd["smooth"].quantile(0.4)]
    doy = pd.to_datetime(clear["date"]).dt.dayofyear.values
    eot = pvlib.solarposition.equation_of_time_spencer71(doy)
    lon = float(np.median(82.5 - (clear["noon"].values - 720 + eot) / 4))
    # latitude: day length where sensor > 5 W/m2 ~ sun elevation above ~3 degrees
    length = float(np.median(clear["set"] - clear["rise"]))
    mid = pd.Timestamp(clear["date"].iloc[len(clear) // 2])
    best, err = None, 1e9
    for lat in np.arange(6, 36, 0.25):
        t = pd.date_range(mid, periods=24 * 60, freq="1min", tz=TZ)
        el = pvlib.solarposition.get_solarposition(t, lat, lon)["apparent_elevation"].values
        dl = (el > 3).sum()
        if abs(dl - length) < err:
            best, err = float(lat), abs(dl - length)
    return {"lat": round(best, 2), "lon": round(lon, 2), "clear_days_used": int(len(clear)),
            "day_length_min": round(length), "lon_spread_deg": round(float(np.std(82.5 - (clear["noon"].values - 720 + eot) / 4)), 2)}


def validate_location(loc, w):
    """Correlate the plant's daily irradiation with ERA5 at the estimated location."""
    class P:
        lat, lon, tz = loc["lat"], loc["lon"], TZ
    try:
        era = ds.reanalysis(P, "2020-05-15", "2020-06-17")["shortwave_radiation"]
    except Exception:
        return None
    sensor_daily = w["E"].resample("1D").sum() * 0.25 / 1000
    era_daily = era.tz_localize(None).resample("1D").sum() / 1000
    j = pd.concat([sensor_daily.rename("s"), era_daily.rename("e")], axis=1).dropna()
    return {"daily_corr": round(float(j.corr().iloc[0, 1]), 3),
            "sensor_kwh_m2_day": round(float(j["s"].mean()), 2),
            "era5_kwh_m2_day": round(float(j["e"].mean()), 2)}


def analyse_plant(z, n):
    gen = _read(z, f"Plant_{n}_Generation_Data.csv")
    wx = _read(z, f"Plant_{n}_Weather_Sensor_Data.csv")
    issues = []
    idx = pd.date_range(gen["t"].min().normalize(), gen["t"].max().normalize() + pd.Timedelta("23:45:00"), freq="15min")
    expected = len(idx)

    # unit check: DC should be ~1.0-1.1x AC for a string inverter
    day = gen[gen["AC_POWER"] > 50]
    ratio = float((day["DC_POWER"] / day["AC_POWER"]).median())
    if ratio > 5:
        gen["DC_POWER"] = gen["DC_POWER"] / 10
        issues.append({"sev": "high", "text": f"DC_POWER logged 10x too high (median DC/AC = {ratio:.1f}); rescaled /10"})
    inv_ac = gen.pivot_table(index="t", columns="SOURCE_KEY", values="AC_POWER").reindex(idx)
    n_inv = inv_ac.shape[1]

    w = wx.set_index("t")[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]].reindex(idx)
    w.columns = ["E", "t_amb", "t_mod"]
    w["E"] = w["E"] * 1000                                        # kW/m2 -> W/m2
    miss_w = int(w["E"].isna().sum())
    if miss_w:
        issues.append({"sev": "med", "text": f"Weather sensor missing {miss_w} of {expected} blocks ({miss_w / expected:.1%})"})
    frozen = int(((w["E"].diff() == 0) & (w["E"] > 50)).rolling(4).sum().ge(4).sum())
    if frozen:
        issues.append({"sev": "med", "text": f"Irradiance sensor frozen for {frozen} daytime blocks"})
    cold = int(((w["t_mod"] < w["t_amb"] - 2) & (w["E"] > 400)).sum())
    if cold:
        issues.append({"sev": "low", "text": f"Module colder than ambient under >400 W/m2 in {cold} blocks (sensor placement?)"})
    gaps = inv_ac.isna().mean()
    worst_gap = gaps.sort_values(ascending=False)
    if worst_gap.iloc[0] > 0.02:
        issues.append({"sev": "med", "text": f"Inverter telemetry gaps: {(gaps > 0.02).sum()} inverters missing >2% of blocks "
                                              f"(worst {worst_gap.iloc[0]:.1%})"})

    # inverter is offline when it logs 0 kW while its peers produce in daylight
    sunny = w["E"] > 300
    med = inv_ac.median(axis=1)
    offline = (inv_ac == 0) & (med > 50).values[:, None]
    trips = (offline & sunny.values[:, None]).sum()
    n_sunny = int(sunny.sum())
    # plant output, availability-corrected for missing records and offline inverters
    online = (inv_ac.notna() & ~offline).sum(axis=1)
    reporting = inv_ac.notna().sum(axis=1)
    p_ac = inv_ac.sum(axis=1, min_count=1) / 1000                  # MW
    p_full = p_ac * n_inv / online.replace(0, np.nan)
    rel = inv_ac.where(~offline).div(med, axis=0)[sunny]
    off_frac = float((offline & sunny.values[:, None]).sum().sum() / max(n_sunny * n_inv, 1))
    if off_frac > 0.01:
        issues.append({"sev": "high", "text": f"Inverters offline (0 kW while peers produce) for {off_frac:.1%} of "
                                              f"inverter-hours in sun; AvC derated accordingly"})
    inv_rows = []
    for k in inv_ac.columns:
        r = float(rel[k].median())
        daily = rel[k].groupby(rel.index.date).median()
        inv_rows.append({"id": k[-6:], "rel_output": round(r, 3), "trip_blocks": int(trips[k]),
                         "gap_pct": round(float(gaps[k]) * 100, 1),
                         "bad_days": int((daily < 0.9).sum()),
                         "trip_pct": round(float(trips[k]) / max(n_sunny, 1) * 100, 1),
                         "status": ("fault" if r < 0.85 else "trips" if trips[k] > 0.05 * n_sunny
                                    else "under" if r < 0.95 or (daily < 0.9).sum() >= 3 else "ok")})
    inv_rows.sort(key=lambda x: x["rel_output"])
    n_bad = sum(r["status"] != "ok" for r in inv_rows)
    if n_bad:
        kinds = {s: sum(r["status"] == s for r in inv_rows) for s in ("fault", "trips", "under")}
        issues.append({"sev": "high", "text": f"{n_bad} of {n_inv} inverters need attention: {kinds['fault']} faulty, "
                                              f"{kinds['trips']} tripping >5% of sunny time, {kinds['under']} underperforming "
                                              f"(worst at {inv_rows[0]['rel_output']:.0%} of fleet median)"})

    # digital twin calibration: fit DC capacity on clean blocks, then validate
    t_cell = w["t_mod"] + w["E"] / 1000 * 3
    shape = 0.965 * w["E"] / 1000 * (1 + GAMMA * (t_cell - 25)) * 0.97
    fit = (w["E"] > 200) & p_full.notna() & shape.notna() & (online >= n_inv * 0.8)
    dates = np.array(sorted(set(idx.date)))
    cal_days = set(dates[: len(dates) // 2])
    cal = fit & pd.Series(idx.date, index=idx).isin(cal_days)
    pdc0 = float((p_full[cal] * shape[cal]).sum() / (shape[cal] ** 2).sum())
    ac_cap = float(np.nanpercentile(p_full, 99.8))
    twin = np.minimum(shape * pdc0, ac_cap).clip(lower=0)
    val = ((w["E"] > 20) & p_full.notna() & shape.notna() & (online >= n_inv * 0.8)
           & ~pd.Series(idx.date, index=idx).isin(cal_days))
    # morning vs afternoon yield per unit irradiance: sensor misalignment / shading check
    hr = (p_full / w["E"] * 1000)[fit]
    by_hr = hr.groupby(hr.index.hour).median()
    am, pm = float(by_hr.loc[8:10].mean()), float(by_hr.loc[14:16].mean())
    drift = pm / am - 1
    if abs(drift) > 0.08:
        issues.append({"sev": "med", "text": f"Yield per unit irradiance is {abs(drift):.0%} {'higher' if drift > 0 else 'lower'} "
                                             f"in the afternoon than the morning: pyranometer tilt/azimuth differs from the "
                                             f"array, or {'east' if drift > 0 else 'west'}-side shading"})
    err = (twin - p_full)[val]
    nmae = float(err.abs().mean() / ac_cap * 100)
    r2 = float(1 - (err ** 2).sum() / ((p_full[val] - p_full[val].mean()) ** 2).sum())

    loc = locate(w)
    loc["era5_check"] = validate_location(loc, w)

    daily = pd.DataFrame({"actual": p_ac.resample("1D").sum() * 0.25,
                          "full": p_full.resample("1D").sum() * 0.25,
                          "twin": twin.resample("1D").sum() * 0.25,
                          "insol": w["E"].resample("1D").sum() * 0.25 / 1000})
    # showcase day: the most variable day with complete data
    var = w["E"].groupby(w.index.date).apply(lambda s: np.abs(s.diff()).sum())
    show = str(var.idxmax())
    sd = slice(show, show)
    return {
        "plant": f"Plant {n}", "plant_id": int(gen["PLANT_ID"].iloc[0]),
        "period": [str(idx[0].date()), str(idx[-1].date())], "inverters": n_inv,
        "ac_capacity_mw_est": round(ac_cap, 2), "dc_capacity_mw_fit": round(pdc0, 2),
        "twin": {"nMAE_pct": round(nmae, 2), "R2": round(r2, 3), "cal_days": len(cal_days),
                 "val_days": len(dates) - len(cal_days)},
        "energy_mwh": round(float(p_ac.sum() * 0.25), 1),
        "lost_mwh_outages": round(float((p_full - p_ac).clip(lower=0).sum() * 0.25), 1),
        "lost_mwh_inverters": round(float(sum(max(0, 1 - r["rel_output"]) for r in inv_rows)
                                          * float((p_full.sum() * 0.25)) / n_inv), 1),
        "issues": issues, "inverters_table": inv_rows, "location": loc,
        "daily": {"date": [str(d.date()) for d in daily.index], **{c: daily[c].round(2).tolist() for c in daily}},
        "showcase": {"date": show, "times": [t.strftime("%H:%M") for t in idx[idx.normalize() == pd.Timestamp(show)]],
                     "actual": p_full[sd].round(3).tolist(), "twin": twin[sd].round(3).tolist(),
                     "irr": w["E"][sd].round(0).tolist()},
    }


def run():
    z = fetch()
    out = {"source": "Kaggle: Solar Power Generation Data (anikannal), two Indian plants, May-June 2020",
           "plants": [analyse_plant(z, 1), analyse_plant(z, 2)]}
    (DATA / "india.json").write_text(json.dumps(out, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o)))
    return out


if __name__ == "__main__":
    r = run()
    for p in r["plants"]:
        print(p["plant"], p["ac_capacity_mw_est"], "MW", p["twin"], p["location"])
        for i in p["issues"]:
            print("  -", i["sev"], i["text"])
        print("  inverters:", [(x["id"], x["rel_output"], x["status"]) for x in p["inverters_table"][:5]])
