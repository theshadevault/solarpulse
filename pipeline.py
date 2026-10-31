"""Build the dataset, train the hybrid model, backtest it, and serve live frames.

    python pipeline.py          # fetch history, train, backtest -> data/
"""
import json
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

import data_sources as ds
import dsm
import features
import twin
from config import ALL_PLANTS, DATA, PLANT_BY_ID, PLANTS, QUANTILES, REAL_PLANTS
from model import (PAIRS, REGIMES, GroupForecaster, aci_scale, fit_blend, kalman_states,
                   regime, scale_quantiles)

QCOLS = [f"q{int(a * 100):02d}" for a in QUANTILES]
GROUP_OF = {p.id: ("real" if p.kind == "pvdaq" else "sim") for p in ALL_PLANTS}


def history_window():
    end = date.today() - timedelta(days=4)            # ERA5 archive lag
    start = end - timedelta(days=179)
    return start, end


def plant_window(p):
    start, end = history_window()
    if p.kind == "pvdaq":
        start = date.fromisoformat(p.start)
        end = min(date.fromisoformat(p.end), end)
    return start, end


SENSOR_LOG = {}


def build_history():
    frames = []
    for p in ALL_PLANTS:
        start, end = plant_window(p)
        days = (end - start).days + 1
        print(f"  {p.name}: NWP + reanalysis + {'NREL telemetry' if p.kind == 'pvdaq' else 'simulated SCADA'}"
              f" {start} -> {end}")
        idx = twin.blocks(str(start), days, p.tz)
        geo = twin.geometry(p, idx)
        nwp = ds.historical_nwp(p, str(start), str(end))
        truth = ds.reanalysis(p, str(start), str(end))
        f = features.plant_frame(p, geo, nwp, truth)
        attrs = dict(f.attrs)
        # drop days where a feed has gaps in daylight (real telemetry has outages)
        gap = (f[["nwp_ghi", "temp"]].isna().any(axis=1)
               | (f["y"].isna() & (f["daylight"] > 0)) | f["p_scada"].isna())
        bad = gap.groupby(f["date"]).transform("sum") > 2
        f = f[~bad.values].copy()
        SENSOR_LOG[p.id] = {"days_kept": int(f["date"].nunique()), "days_total": days, **attrs}
        frames.append(f)
    return pd.concat(frames)


def split(df):
    """Chronological 70/15/15 split per plant (plants cover different periods)."""
    parts = {"train": [], "calib": [], "test": []}
    for _, g in df.groupby("plant", sort=False):
        dates = np.array(sorted(g.date.unique()))
        a, b = int(len(dates) * 0.70), int(len(dates) * 0.85)
        parts["train"].append(g[g.date.isin(dates[:a])])
        parts["calib"].append(g[g.date.isin(dates[a:b])])
        parts["test"].append(g[g.date.isin(dates[b:])])
    return tuple(pd.concat(parts[k]) for k in ("train", "calib", "test"))


def add_predictions(fc, df, lam0=None):
    """Static CQR quantiles -> online ACI scaling -> Kalman bias state, per plant in time order."""
    q = fc.predict(df)
    df["cqr10"], df["cqr90"] = q[:, QUANTILES.index(0.10)], q[:, QUANTILES.index(0.90)]
    df["regime"] = regime(df)
    lam, bias = np.ones(len(df)), np.zeros(len(df))
    for pid, g in df.groupby("plant", sort=False):
        m = (df.plant == pid).values
        lam[m], _ = aci_scale(q[m], g["y"].values, g["daylight"].values,
                              lam0=(lam0 or {}).get(pid, 1.0))
    q = np.minimum(scale_quantiles(q, lam), 1.0)
    for i, c in enumerate(QCOLS):
        df[c] = q[:, i]
    df["aci_lam"] = lam
    for pid, g in df.groupby("plant", sort=False):
        m = (df.plant == pid).values
        bias[m] = kalman_states(g["y"].values, g["q50"].values, g["daylight"].values)
    df["kf_bias"] = bias
    return df


def _err(y, f, m):
    e = (f - y)[m]
    return {"nMAE": float(np.mean(np.abs(e)) * 100), "nRMSE": float(np.sqrt(np.mean(e ** 2)) * 100),
            "bias": float(np.mean(e) * 100)}


def backtest(fc, test, blend):
    test = test.copy()
    plants = [PLANT_BY_ID[p] for p in test.plant.unique()]
    out = {"plants": [p.id for p in plants],
           "test_days": int(test.groupby("plant").date.nunique().sum()),
           "windows": {p.id: [str(test[test.plant == p.id].date.min()),
                              str(test[test.plant == p.id].date.max())] for p in plants},
           "deterministic": {}, "intraday": {},
           "probabilistic": {}, "dsm": {}, "sanitization": {}, "per_plant": {}}
    y = test["y"].values
    m = (test.daylight > 0).values & ~np.isnan(y)
    # day-ahead persistence: same block on the previous calendar day (if that day was kept)
    prev = test[["plant", "date", "block", "y"]].copy()
    prev["date"] = prev["date"] + pd.Timedelta(days=1)
    lag = test[["plant", "date", "block"]].merge(prev, on=["plant", "date", "block"],
                                                 how="left")["y"].values
    mm = m & ~np.isnan(lag)
    det = out["deterministic"]
    det["Day-ahead persistence"] = _err(y, np.nan_to_num(lag), mm)
    det["Physics twin on NWP"] = _err(y, test["phys"].values, mm)
    det["Hybrid P50 (twin + ML residual)"] = _err(y, test["q50"].values, mm)
    base = det["Day-ahead persistence"]["nRMSE"]
    for k in det:
        det[k]["skill_vs_persistence"] = float((1 - det[k]["nRMSE"] / base) * 100)

    # intraday: smart persistence vs Kalman-corrected hybrid at several leads
    for lead in (1, 4, 8, 16):
        g = test.groupby("plant")
        cs_prev = g["phys_cs"].shift(lead).values
        y_prev = g["y"].shift(lead).values
        b_prev = g["kf_bias"].shift(lead).values
        day_prev = g["daylight"].shift(lead).values
        mk = m & ~np.isnan(y_prev) & (day_prev > 0) & (cs_prev > 0.05)
        pers = np.clip(y_prev / np.maximum(cs_prev, 0.05), 0, 1.2) * test["phys_cs"].values
        scale = np.clip(test["phys_cs"].values / np.maximum(cs_prev, 0.05), 0, 1.5)
        corr = np.nan_to_num(b_prev) * np.exp(-lead / 10) * scale
        hyb = np.clip(test["q50"].values + corr, 0, 1)
        w = blend.get(lead, 0.0)
        bl = w * pers + (1 - w) * hyb
        sp = _err(y, pers, mk)
        p0 = _err(y, test["q50"].values, mk)
        hk = _err(y, hyb, mk)
        hb = _err(y, bl, mk)
        out["intraday"][f"{lead * 15} min"] = {
            "Smart persistence": sp, "Hybrid (no SCADA feedback)": p0, "Hybrid + Kalman": hk,
            "Horizon-adaptive blend": hb, "persistence_weight": w,
            "skill_vs_persistence": float((1 - hb["nRMSE"] / sp["nRMSE"]) * 100)}

    # probabilistic
    prob = out["probabilistic"]
    for lo, hi in PAIRS:
        a, b = f"q{int(lo * 100):02d}", f"q{int(hi * 100):02d}"
        inside = (y >= test[a].values - 1e-9) & (y <= test[b].values + 1e-9)
        nominal = round((hi - lo) * 100)
        prob[f"PICP {nominal}%"] = {
            "nominal": nominal, "empirical": float(inside[m].mean() * 100),
            "mean_width": float((test[b] - test[a]).values[m].mean() * 100),
            "by_regime": {REGIMES[r]: float(inside[m & (test.regime.values == r)].mean() * 100)
                          for r in range(3) if (m & (test.regime.values == r)).any()}}
    pin = []
    for i, a in enumerate(QUANTILES):
        d = y - test[QCOLS[i]].values
        pin.append(np.maximum(a * d, (a - 1) * d)[m].mean())
    prob["mean_pinball"] = float(np.mean(pin) * 100)
    static = (y >= test["cqr10"].values - 1e-9) & (y <= test["cqr90"].values + 1e-9)
    prob["Static CQR 80% (no online adaptation)"] = float(static[m].mean() * 100)

    # DSM settlement over the test window
    tot = {"Schedule = physics twin": 0.0, "Schedule = P50": 0.0, "Schedule = S* (optimized)": 0.0}
    comp = {k: [] for k in tot}
    for p in plants:
        g = test[test.plant == p.id]
        if g.empty:
            continue
        avc = g["avc"].values
        qmw = g[QCOLS].values * avc[:, None]
        scheds = {"Schedule = physics twin": g["phys"].values * avc,
                  "Schedule = P50": g["q50"].values * avc,
                  "Schedule = S* (optimized)": dsm.optimize(qmw, avc)}
        pp = {}
        for k, s in scheds.items():
            pen = np.nan_to_num(dsm.penalty(s, g["p_scada"].values, avc, exempt=g["curtail"].values))
            tot[k] += float(pen.sum())
            pp[k] = float(pen.sum())
            ok = g["p_scada"].notna().values
            comp[k].append(dsm.compliance(s[ok], g["p_scada"].values[ok], avc[ok],
                                          g["curtail"].values[ok], g["daylight"].values[ok]))
        mp = (g.daylight > 0).values & g["y"].notna().values
        out["per_plant"][p.id] = {
            "nMAE_P50": float(np.abs(g["q50"] - g["y"]).values[mp].mean() * 100),
            "nMAE_twin": float(np.abs(g["phys"] - g["y"]).values[mp].mean() * 100),
            "dsm_inr": pp}
    out["dsm"] = {k: {"penalty_inr": v, "band_compliance_pct": float(np.mean(comp[k]))}
                  for k, v in tot.items()}
    base_pen = tot["Schedule = physics twin"]
    out["dsm_saving_pct"] = float((1 - tot["Schedule = S* (optimized)"] / max(base_pen, 1)) * 100)

    # sanitization quality (needs hidden truth)
    c = (test.curtail > 0).values
    cap = np.mean([p.ac_mw for p in plants])
    truth_unc = test["p_avail_true"].values * test["units"].values / np.maximum(test["online"].values, 1)
    derated = (test["online"] < test["units"]).values & (test.daylight > 0).values
    out["sanitization"] = {
        "curtailed_blocks": int(c.sum()),
        "decurtail_error_pct": (float(np.abs(test["p_unc"].values - truth_unc)[c].mean() / cap * 100)
                                if c.any() else None),
        "raw_scada_error_pct": (float(np.abs(test["p_scada"].values - truth_unc)[c].mean() / cap * 100)
                                if c.any() else None),
        "soiling_est_mae_pct": (float(np.nanmean(np.abs(test["soiling_est"] - test["soiling_true"])) * 100)
                                if test["soiling_true"].notna().any() else None),
        "sensor_blocks_flagged": int(test["qc_bad"].sum()),
        "derated_blocks": int(derated.sum()),
    }
    return out


def daily_stats(df):
    rows = []
    for (pid, d), g in df.groupby(["plant", "date"]):
        m = (g.daylight > 0) & g.y.notna()
        rows.append({"plant": pid, "date": str(d),
                     "nmae": float(np.abs(g.q50 - g.y)[m].mean() * 100),
                     "variability": float(np.nansum(np.abs(np.diff(g.y[m])))),
                     "energy": float(g.p_scada.sum() * 0.25)})
    return rows


def run():
    t0 = time.time()
    DATA.mkdir(exist_ok=True)
    print("1/4 building history dataset")
    df = build_history()
    train, calib, test = split(df)
    df = pd.concat([train.assign(split="train"), calib.assign(split="calib"),
                    test.assign(split="test")])
    print(f"2/4 training on {len(train) // 96} plant-days, calibrating on {len(calib) // 96}")
    fc = GroupForecaster(GROUP_OF).fit(train, calib)
    fc.save()
    print("3/4 backtesting")
    df = add_predictions(fc, df)
    blend = fit_blend(df[df.split == "calib"])
    (DATA / "blend.json").write_text(json.dumps(blend))
    test = df[df.split == "test"]
    real_ids = {p.id for p in REAL_PLANTS}
    metrics = {"real": backtest(fc, test[test.plant.isin(real_ids)], blend),
               "sim": backtest(fc, test[~test.plant.isin(real_ids)], blend)}
    metrics["split"] = {pid: {s: [str(g[g.split == s].date.min()), str(g[g.split == s].date.max()),
                                  int(g[g.split == s].date.nunique())]
                              for s in ("train", "calib", "test")}
                        for pid, g in df.groupby("plant", sort=False)}
    metrics["sensor_log"] = SENSOR_LOG
    metrics["feature_gain"] = {}
    for k, f in fc.models.items():
        imp = f.med.get_booster().get_score(importance_type="gain")
        metrics["feature_gain"][k] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:12])
    metrics["built_at"] = pd.Timestamp.now(tz="Asia/Kolkata").isoformat(timespec="minutes")
    print("4/4 saving")
    df.to_pickle(DATA / "dataset.pkl")
    # the server only needs held-out days (replay) and each plant's latest state
    df[df.split == "test"].to_pickle(DATA / "dataset_serve.pkl")
    (DATA / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (DATA / "days.json").write_text(json.dumps(daily_stats(df[df.split == "test"])))
    try:
        import kaggle_india
        kaggle_india.run()
        print("  Kaggle India plant analysis -> data/india.json")
    except Exception as e:
        print("  Kaggle India analysis skipped:", e)
    for k in ("real", "sim"):
        print(k, json.dumps({n: round(v["nRMSE"], 2) for n, v in metrics[k]["deterministic"].items()}),
              "PICP80", round(metrics[k]["probabilistic"]["PICP 80%"]["empirical"], 1))
    print(f"done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- live frames

def live_frame(fc, plant, soiling_start, lam0=None):
    """Yesterday/today/tomorrow for one plant from the latest forecast run."""
    today = pd.Timestamp.now(tz=plant.tz).normalize()
    start = (today - pd.Timedelta(days=1)).date()
    idx = twin.blocks(str(start), 3, plant.tz)
    geo = twin.geometry(plant, idx)
    nwp = ds.live_nwp(plant)
    truth = ds.live_truth_proxy(plant)
    df = features.plant_frame(plant, geo, nwp, truth, soiling_start=soiling_start, live=True)
    df = add_predictions(fc, df, lam0)
    return df


if __name__ == "__main__":
    run()
