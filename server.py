"""Layer 5: FastAPI backend for the SolarPulse operator & grid cockpit.

    python server.py   ->  http://127.0.0.1:8010
"""
import io
import json
import threading
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import dsm
import nowcast
import pipeline
from config import (ALL_PLANTS, DATA, GATE_LEAD_BLOCKS, ON_VERCEL, PLANT_BY_ID, PLANTS,
                    QUANTILES, REVISION_TRIGGER_PCT, ROOT, SCRATCH)
from model import BLEND_LEADS, GroupForecaster, intraday_inputs

QCOLS = pipeline.QCOLS
I10, I50, I90 = QUANTILES.index(0.10), QUANTILES.index(0.50), QUANTILES.index(0.90)

app = FastAPI(title="SolarPulse")
_lock = threading.Lock()
STATE = {}
REVISIONS_PATH = SCRATCH / "revisions.json"


def boot():
    serve = DATA / "dataset_serve.pkl"
    if not serve.exists() and not ON_VERCEL:
        print("No trained model found - running the pipeline (a few minutes)...")
        pipeline.run()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    STATE["hist"] = pd.read_pickle(serve)
    STATE["fc"] = GroupForecaster.load(pipeline.GROUP_OF)
    STATE["blend"] = {int(k): v for k, v in json.loads((DATA / "blend.json").read_text()).items()}
    STATE["metrics"] = json.loads((DATA / "metrics.json").read_text())
    STATE["days"] = json.loads((DATA / "days.json").read_text())
    STATE["live"] = {}
    STATE["revisions"] = json.loads(REVISIONS_PATH.read_text()) if REVISIONS_PATH.exists() else {}
    india = DATA / "india.json"
    if not india.exists() and not ON_VERCEL:
        try:
            import kaggle_india
            kaggle_india.run()
        except Exception as e:
            print("Kaggle India analysis unavailable:", e)
    STATE["india"] = json.loads(india.read_text()) if india.exists() else None


def today(tz="Asia/Kolkata"):
    return pd.Timestamp.now(tz=tz)


def num(x, d=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else round(x, d)


def live_df(pid):
    """Yesterday/today/tomorrow frame from the latest forecast, refreshed hourly."""
    with _lock:
        hit = STATE["live"].get(pid)
        if hit and time.time() - hit[0] < 3600:
            return hit[1]
        hist = STATE["hist"]
        last = hist[hist.plant == pid].iloc[-1]
        p = PLANT_BY_ID[pid]
        soil = last["soiling_true"] if p.kind == "sim" else last["soiling_est"]
        df = pipeline.live_frame(STATE["fc"], p, float(soil), {pid: float(last["aci_lam"])})
        STATE["live"][pid] = (time.time(), df)
        return df


def day_frame(pid, date):
    hist = STATE["hist"]
    d = pd.Timestamp(date).date()
    g = hist[(hist.plant == pid) & (hist.date == d)]
    if len(g) == 96:
        return g, "replay"
    lv = live_df(pid)
    g = lv[lv.date == d]
    if len(g) == 96:
        return g, "live"
    raise HTTPException(404, f"No data for {pid} on {date}")


@lru_cache(maxsize=256)
def day_ahead_schedule(pid, date):
    g, _ = day_frame(pid, date)
    avc = g["avc"].values
    return dsm.optimize(g[QCOLS].values * avc[:, None], avc)


def rev_key(pid, date):
    return f"{pid}|{date}"


def current_schedule(pid, date):
    s = day_ahead_schedule(pid, date).copy()
    for b, mw in STATE["revisions"].get(rev_key(pid, date), {}).get("blocks", {}).items():
        s[int(b)] = mw
    return s


def default_now(mode, tz="Asia/Kolkata"):
    if mode == "live":
        t = today(tz)
        return int((t.hour * 60 + t.minute) // 15)
    return 54            # replay default: 13:30


def intraday(g, now):
    """Horizon-adaptive intraday forecast from the last completed block (normalized)."""
    y, cs, b, q50 = (g[c].values for c in ("y", "phys_cs", "kf_bias", "q50"))
    day = g["daylight"].values
    out = q50.copy()
    s = now - 1
    if s < 0 or day[s] <= 0 or cs[s] < 0.05 or np.isnan(y[s]):
        return out, 0.0
    for t in range(now, 96):
        lead = t - s
        p, h = intraday_inputs(y[s], cs[s], b[s], cs[t], q50[t], lead)
        w = STATE["blend"].get(lead, 0.0) if lead <= BLEND_LEADS else 0.0
        out[t] = w * p + (1 - w) * h if day[t] > 0 else 0.0
    return out, float(b[s])


def hhmm(b):
    return f"{(b * 15) // 60:02d}:{(b * 15) % 60:02d}"


def r1(a, d=3):
    return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), d)
            for x in a]


# ------------------------------------------------------------------------- API

class Revision(BaseModel):
    plant: str
    date: str
    blocks: dict[int, float]
    note: str = ""


@app.get("/api/meta")
def meta():
    m = STATE["metrics"]
    plants = []
    for p in ALL_PLANTS:
        rows = [d for d in STATE["days"] if d["plant"] == p.id]
        days = sorted(d["date"] for d in rows)
        cloudy = [d["date"] for d in sorted(rows, key=lambda d: -d["variability"])[:6]]
        t = today(p.tz)
        plants.append({"id": p.id, "name": p.name, "state": p.state, "lat": p.lat, "lon": p.lon,
                       "ac_mw": p.ac_mw, "dc_mw": round(p.dc_mw, 2), "tracker": p.tracker,
                       "kind": p.kind, "tz": p.tz, "source": p.source,
                       "replay_dates": days, "cloudy_dates": cloudy,
                       "today": str(t.date()), "tomorrow": str((t + pd.Timedelta(days=1)).date())})
    sim = [d for d in STATE["days"] if d["plant"] in {p.id for p in PLANTS}]
    by_date = {}
    for d in sim:
        by_date.setdefault(d["date"], []).append(d["variability"])
    fleet_dates = sorted(by_date, key=lambda d: -np.mean(by_date[d]))
    t = today()
    return {
        "plants": plants, "fleet_dates": sorted(by_date), "fleet_cloudy": fleet_dates[:6],
        "today": str(t.date()), "tomorrow": str((t + pd.Timedelta(days=1)).date()),
        "split": m.get("split"), "built_at": m.get("built_at"),
    }


@app.get("/api/day")
def day(plant: str, date: str, now: int | None = None):
    if plant not in PLANT_BY_ID:
        raise HTTPException(404, "unknown plant")
    p = PLANT_BY_ID[plant]
    g, mode = day_frame(plant, date)
    if mode == "live" and pd.Timestamp(date).date() > today(p.tz).date():
        now = 0 if now is None else now
    now = default_now(mode, p.tz) if now is None else int(np.clip(now, 0, 96))
    avc = g["avc"].values
    q = g[QCOLS].values
    qmw = q * avc[:, None]
    s_da = day_ahead_schedule(plant, date)
    s_cur = current_schedule(plant, date)
    intra, bias = intraday(g, now)
    intra_mw = intra * avc
    # intraday band: shift day-ahead quantiles onto the updated median, shrink by lead
    shift = (intra - q[:, I50])[:, None]
    q_intra = np.clip(q + shift, 0, 1) * avc[:, None]
    s_new = dsm.optimize(q_intra, avc)
    s_new[: now + GATE_LEAD_BLOCKS] = s_cur[: now + GATE_LEAD_BLOCKS]
    first, sugg = dsm.gate_closure(now, s_cur, s_new, avc)
    for x in sugg:
        x["time"] = hhmm(x["block"])

    actual = g["p_scada"].values.astype(float).copy()
    actual[now:] = np.nan
    past = np.arange(96) < now
    exempt = g["curtail"].values
    day = g["daylight"].values
    led = {}
    seen = past & ~np.isnan(g["p_scada"].values)
    for name, sched in (("Current schedule", s_cur), ("Day-ahead S*", s_da),
                        ("Submit P50", qmw[:, I50]), ("Physics twin only", g["phys"].values * avc)):
        pen = dsm.penalty(sched[seen], g["p_scada"].values[seen], avc[seen], exempt[seen])
        led[name] = {"inr": round(float(pen.sum())),
                     "compliance": round(dsm.compliance(sched[seen], g["p_scada"].values[seen],
                                                        avc[seen], exempt[seen], day[seen]), 1)
                     if seen.any() else None}
    exp_rest = dsm.expected_penalty(s_cur[now:], q_intra[now:], avc[now:]) if now < 96 else []
    pen_blocks = dsm.penalty(s_cur, g["p_scada"].values, avc, exempt)
    pen_blocks[now:] = np.nan

    shap = STATE["fc"].explain(g)
    contrib = {k: np.asarray(v) * avc for k, v in shap.items()}
    total = sum(contrib.values())
    contrib["Capacity / night limits"] = qmw[:, I50] - total
    nb = max(now - 1, 0)
    flags = g["qc_bad"].values.astype(bool)
    return {
        "plant": plant, "date": date, "mode": mode, "now": now, "now_time": hhmm(now),
        "ac_mw": p.ac_mw, "times": [hhmm(b) for b in range(96)], "kind": p.kind, "tz": p.tz,
        "has_actuals": bool(np.isfinite(g["p_scada"].values).any()),
        "q": {f"p{int(a * 100):02d}": r1(qmw[:, i]) for i, a in enumerate(QUANTILES)},
        "q_intra": {"p10": r1(q_intra[:, I10]), "p90": r1(q_intra[:, I90])},
        "schedule_da": r1(s_da), "schedule": r1(s_cur), "intraday": r1(intra_mw),
        "proposed": r1(s_new), "actual": r1(actual), "avc": r1(avc),
        "curtail": [int(x) for x in g["curtail"].values * past],
        "setpoint": r1(np.where(g["curtail"].values > 0, g["setpoint"].values, np.nan)),
        "unconstrained": r1(np.where(past, g["p_unc"].values * avc / p.ac_mw, np.nan)),
        "sensor_log": pipeline_sensor_log(plant),
        "penalty_blocks": r1(pen_blocks),
        "regime": [int(x) for x in g["regime"].values],
        "gate": {"first_block": first, "first_time": hhmm(min(first, 95)),
                 "trigger_pct": REVISION_TRIGGER_PCT, "suggestions": sugg,
                 "minutes_left": (15 - today(p.tz).minute % 15) if mode == "live" else 9},
        "ledger": led, "expected_rest_inr": round(float(np.sum(exp_rest))),
        "health": {
            "soiling_est": num(g["soiling_est"].iloc[0]),
            "soiling_true": num(g["soiling_true"].iloc[0]),
            "online": int(g["online"].values[nb]), "units": int(g["units"].values[nb]),
            "curtailed_now": bool(g["curtail"].values[nb] > 0),
            "curtailed_blocks": int((g["curtail"].values * past).sum()),
            "clipping_blocks": int(((g["clip_phys"].values > 0) & (day > 0)).sum()),
            "sensor_flags": [hhmm(b) for b in np.where(flags & past)[0]],
            "aci_scale": round(float(g["aci_lam"].values[nb]), 2),
            "kalman_bias_mw": round(bias * avc[nb], 3),
            "t_cell_peak": round(float(g["t_cell_phys"].max()), 1),
        },
        "explain": {k: r1(v) for k, v in contrib.items()},
        "kpi": {
            "energy_p50_mwh": round(float(qmw[:, I50].sum() * 0.25), 3),
            "energy_p10_mwh": round(float(qmw[:, I10].sum() * 0.25), 3),
            "energy_p90_mwh": round(float(qmw[:, I90].sum() * 0.25), 3),
            "energy_actual_mwh": round(float(np.nansum(actual) * 0.25), 3),
            "peak_p50_mw": round(float(qmw[:, I50].max()), 3),
        },
        "revisions": STATE["revisions"].get(rev_key(plant, date), {}).get("log", []),
    }


@app.post("/api/revise")
def revise(r: Revision):
    if r.plant not in PLANT_BY_ID:
        raise HTTPException(404, "unknown plant")
    entry = STATE["revisions"].setdefault(rev_key(r.plant, r.date), {"blocks": {}, "log": []})
    for b, mw in r.blocks.items():
        entry["blocks"][str(int(b))] = round(float(mw), 2)
    entry["log"].append({"at": today().isoformat(timespec="seconds"), "blocks": len(r.blocks),
                         "first": hhmm(min(r.blocks)), "last": hhmm(max(r.blocks)), "note": r.note})
    REVISIONS_PATH.write_text(json.dumps(STATE["revisions"], indent=1))
    return {"ok": True, "revision_no": len(entry["log"])}


@app.post("/api/revise/reset")
def revise_reset(plant: str, date: str):
    STATE["revisions"].pop(rev_key(plant, date), None)
    REVISIONS_PATH.write_text(json.dumps(STATE["revisions"], indent=1))
    return {"ok": True}


@app.get("/api/schedule.csv")
def schedule_csv(plant: str, date: str):
    g, _ = day_frame(plant, date)
    s_da, s_cur = day_ahead_schedule(plant, date), current_schedule(plant, date)
    buf = io.StringIO()
    buf.write("Block,From,To,AvC_MW,DayAhead_MW,Current_MW\n")
    for b in range(96):
        buf.write(f"{b + 1},{hhmm(b)},{hhmm(b + 1) if b < 95 else '24:00'},"
                  f"{g['avc'].values[b]:.2f},{s_da[b]:.2f},{s_cur[b]:.2f}\n")
    buf.seek(0)
    name = f"SolarPulse_{plant}_{date}_schedule.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={name}"})


@app.get("/api/fleet")
def fleet(date: str, now: int | None = None):
    plants, agg = [], None
    mode = None
    for p in PLANTS:
        g, mode = day_frame(p.id, date)
        n = default_now(mode) if now is None else now
        if mode == "live" and pd.Timestamp(date).date() > today().date():
            n = 0
        intra, _ = intraday(g, n)
        avc = g["avc"].values
        q = g[QCOLS].values * avc[:, None]
        p50 = np.where(np.arange(96) >= n, intra * avc, q[:, I50])
        lo = np.clip(q[:, I10] + (p50 - q[:, I50]), 0, None)
        hi = q[:, I90] + (p50 - q[:, I50])
        act = g["p_scada"].values.astype(float).copy()
        act[n:] = np.nan
        cs = g["phys_cs"].values * avc
        # cloud-driven ramps: drop in forecast relative to the clear-sky profile
        ramps = []
        k = np.where(cs > 0.15 * p.ac_mw, p50 / np.maximum(cs, 1), np.nan)
        for t in range(max(n, 1), 92):
            d4 = (p50[t + 4] - p50[t]) - (cs[t + 4] - cs[t])
            if not np.isnan(k[t]) and d4 <= -0.15 * p.ac_mw:
                ramps.append({"block": t, "time": hhmm(t), "drop_mw": round(float(-d4), 1)})
        hourly = []
        for h in range(24):
            sl = slice(h * 4, h * 4 + 4)
            width = float(np.nanmean((hi[sl] - lo[sl]) / max(p.ac_mw, 1)))
            drop = max([r["drop_mw"] for r in ramps if h * 4 <= r["block"] < h * 4 + 4] or [0])
            level = 0 if cs[sl].max() < 0.05 * p.ac_mw else (
                2 if drop >= 0.25 * p.ac_mw or width > 0.45 else 1 if drop > 0 or width > 0.25 else 0)
            hourly.append(level if cs[sl].max() >= 0.05 * p.ac_mw else -1)
        # merge consecutive ramp blocks into events
        events = []
        for r in ramps:
            if events and r["block"] - events[-1]["end"] <= 1:
                events[-1]["end"] = r["block"]
                events[-1]["drop_mw"] = max(events[-1]["drop_mw"], r["drop_mw"])
            else:
                events.append({"start": r["block"], "end": r["block"], "time": r["time"],
                               "drop_mw": r["drop_mw"]})
        plants.append({"id": p.id, "name": p.name, "state": p.state, "ac_mw": p.ac_mw,
                       "lat": p.lat, "lon": p.lon,
                       "p50": r1(p50), "p10": r1(lo), "p90": r1(hi), "actual": r1(act),
                       "ramps": events, "hourly_risk": hourly,
                       "energy_mwh": round(float(np.nansum(p50) * 0.25), 1),
                       "now_mw": None if n == 0 else round(float(g["p_scada"].values[n - 1]), 1)})
        a = np.vstack([p50, lo, hi, np.nan_to_num(act)])
        agg = a if agg is None else agg + a
    n = default_now(mode) if now is None else now
    total_ac = sum(p.ac_mw for p in PLANTS)
    p50, lo, hi, act = agg
    # independent-error assumption is optimistic; use a mild diversification factor instead
    reserve = np.clip((p50 - lo) * 0.8, 0, None)
    act = np.where(np.arange(96) < n, act, np.nan)
    ramp_rate = np.diff(p50, prepend=p50[0]) / 15
    return {"date": date, "mode": mode, "now": n, "times": [hhmm(b) for b in range(96)],
            "plants": plants, "total_ac": total_ac,
            "aggregate": {"p50": r1(p50), "p10": r1(np.clip(p50 - (p50 - lo) * 0.8, 0, None)),
                          "p90": r1(p50 + (hi - p50) * 0.8), "actual": r1(act),
                          "reserve_mw": r1(reserve), "ramp_mw_per_min": r1(ramp_rate)},
            "max_reserve_mw": round(float(reserve.max()), 1),
            "evening_ramp_mw_per_min": round(float(-ramp_rate[60:80].min()), 2)}


@app.get("/api/nowcast")
def get_nowcast(plant: str):
    try:
        return nowcast.run(PLANT_BY_ID[plant])
    except Exception as e:  # network trouble shouldn't break the cockpit
        raise HTTPException(503, f"nowcast unavailable: {e}")


@app.get("/api/metrics")
def metrics():
    return STATE["metrics"]


@app.get("/api/india")
def india():
    if not STATE.get("india"):
        raise HTTPException(503, "Kaggle India data not available")
    return STATE["india"]


def pipeline_sensor_log(pid):
    return STATE["metrics"].get("sensor_log", {}).get(pid, {}).get("sensor_log", [])


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


boot()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
