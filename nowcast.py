"""Layer 2: cloud-motion-vector nowcast with dense optical flow (Farneback).

Input is a gridded cloud-cover field around the plant. In production this is the
INSAT-3DS / Meteosat cloud index; the prototype uses the latest NWP cloud field at
15-min cadence so the pipeline (flow -> advection -> plant shadow) is identical.
"""
import cv2
import numpy as np
import pandas as pd

import data_sources as ds

N, STEP, UP = 11, 0.2, 8          # grid points, degrees between them, upsampling factor
KM_PER_DEG = 111.0


def _flow(a, b):
    a8 = cv2.resize(a.astype(np.float32), None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
    b8 = cv2.resize(b.astype(np.float32), None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
    a8 = np.clip(a8 * 2.55, 0, 255).astype(np.uint8)
    b8 = np.clip(b8 * 2.55, 0, 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 3, 15, 3, 5, 1.2, 0), a8, b8


def _advect(field_up, flow, steps):
    h, w = field_up.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = (gx - flow[..., 0] * steps).astype(np.float32)
    my = (gy - flow[..., 1] * steps).astype(np.float32)
    return cv2.remap(field_up, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def run(plant, now=None):
    times, lats, lons, cube = ds.cloud_grid(plant.lat, plant.lon, n=N, step=STEP,
                                            past_hours=2, forecast_hours=4, tz=plant.tz)
    now = now or pd.Timestamp.now(tz=plant.tz).tz_localize(None)
    t_idx = int(np.searchsorted(times, now, side="right") - 1)
    t_idx = max(4, min(t_idx, len(times) - 2))
    # flow from the last hour of observed frames (smoothed over 4 pairs)
    flows = [_flow(cube[t_idx - k - 1], cube[t_idx - k])[0] for k in range(4)]
    flow = np.mean(flows, axis=0)
    cur_up = cv2.resize(cube[t_idx].astype(np.float32), None, fx=UP, fy=UP,
                        interpolation=cv2.INTER_CUBIC)
    c = (N * UP) // 2
    leads = list(range(0, 13))      # 0..180 min
    adv_frames, nwp_frames, plant_cmv, plant_nwp = [], [], [], []
    for k in leads:
        f = _advect(cur_up, flow, k)
        plant_cmv.append(float(np.clip(f[c, c], 0, 100)))
        small = cv2.resize(f, (N * 3, N * 3), interpolation=cv2.INTER_AREA)
        adv_frames.append(np.clip(small, 0, 100).round().astype(int).tolist())
        ti = min(t_idx + k, len(times) - 1)
        nwp_up = cv2.resize(cube[ti].astype(np.float32), (N * 3, N * 3), interpolation=cv2.INTER_CUBIC)
        nwp_frames.append(np.clip(nwp_up, 0, 100).round().astype(int).tolist())
        plant_nwp.append(float(cube[ti, N // 2, N // 2]))
    # coarse vectors for arrows: grid cells per 15 min -> km/h
    step_px = UP
    vec = flow[step_px // 2::step_px, step_px // 2::step_px]
    km_per_px = STEP * KM_PER_DEG / UP
    u, v = vec[..., 0] * km_per_px * 4, vec[..., 1] * km_per_px * 4   # km/h (east, north)
    mean_u, mean_v = float(np.mean(u)), float(np.mean(v))
    speed = float(np.hypot(mean_u, mean_v))
    heading = (np.degrees(np.arctan2(mean_u, mean_v)) + 360) % 360  # direction clouds move toward
    # ramp warning: cloud cover over the plant rising sharply within 2h
    rise = max(plant_cmv[1:9]) - plant_cmv[0]
    alert = None
    if rise >= 35:
        k = int(np.argmax(np.array(plant_cmv[1:9]) - plant_cmv[0] >= 35)) + 1
        alert = {"minutes": k * 15, "rise": round(rise), "text":
                 f"Cloud bank arriving in ~{k * 15} min (+{round(rise)}% cover)"}
    return {
        "plant": plant.id, "issued": str(times[t_idx]), "leads_min": [k * 15 for k in leads],
        "lats": lats.round(3).tolist(), "lons": lons.round(3).tolist(),
        "cmv_frames": adv_frames, "nwp_frames": nwp_frames,
        "vectors": {"u": np.round(u, 1).tolist(), "v": np.round(v, 1).tolist()},
        "speed_kmh": round(speed, 1), "heading_deg": round(float(heading)),
        "plant_cmv": plant_cmv, "plant_nwp": plant_nwp, "alert": alert,
        "source": "NWP cloud field at 15-min cadence (stand-in for INSAT-3DS cloud index)",
    }
