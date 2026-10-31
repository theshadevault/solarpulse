"""Layers 3-4: hybrid residual ML, conformal uncertainty, Kalman SCADA assimilation, SHAP.

The ML model never learns solar geometry: it learns the residual between the de-curtailed
plant output and the physics twin run on NWP (y - P_phys). P50 = P_phys + r50.
"""
import json

import numpy as np
import xgboost as xgb

from config import DATA, QUANTILES
from features import FEATURES, GROUPS

PAIRS = [(0.05, 0.95), (0.10, 0.90), (0.25, 0.75)]
REGIMES = ["Clear", "Broken cloud", "Overcast"]

XGB_PARAMS = dict(n_estimators=500, learning_rate=0.04, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=20, tree_method="hist")


def regime(df):
    """Sky regime from NWP clear-sky index and model disagreement."""
    kt, spread = df["nwp_kt"].values, df["ghi_spread_n"].values
    r = np.where(kt > 0.75, 0, np.where(kt > 0.35, 1, 2))
    r = np.where((r == 0) & (spread > 0.12), 1, r)
    return r


class _Booster:
    """Saved model served with the core Booster API (no scikit-learn needed at runtime)."""

    def __init__(self, path):
        self.booster = xgb.Booster()
        self.booster.load_model(path)

    def predict(self, X):
        return self.booster.predict(xgb.DMatrix(X))

    def get_booster(self):
        return self.booster


class Forecaster:
    def __init__(self):
        self.med = None
        self.qm = None
        self.conf = {}          # regime -> pair -> widening
        self.cal_scores = {}    # (regime, pair) -> scores for ACI

    # ------------------------------------------------------------ training
    def fit(self, train, calib):
        tr = train[(train.daylight > 0) & train["y"].notna()]
        X, r = tr[FEATURES], (tr["y"] - tr["phys"]).values
        self.med = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **XGB_PARAMS)
        self.med.fit(X, r)
        self.qm = xgb.XGBRegressor(objective="reg:quantileerror",
                                   quantile_alpha=np.array(QUANTILES), **XGB_PARAMS)
        self.qm.fit(X, r)
        self.calibrate(calib)
        return self

    def raw_quantiles(self, df):
        """Uncalibrated quantiles, normalized to plant capacity, shape (n, len(QUANTILES))."""
        X = df[FEATURES]
        phys = df["phys"].values
        p50 = phys + self.med.predict(X)
        q = self.qm.predict(X)
        i50 = QUANTILES.index(0.5)
        q = np.sort(q - q[:, [i50]], axis=1) + p50[:, None]
        q[:, i50] = p50
        return q

    def calibrate(self, calib):
        cal = calib[(calib.daylight > 0) & calib["y"].notna()]
        q = self.raw_quantiles(cal)
        y = cal["y"].values
        reg = regime(cal)
        for g in range(3):
            m = reg == g
            self.conf[g] = {}
            for lo, hi in PAIRS:
                il, ih = QUANTILES.index(lo), QUANTILES.index(hi)
                e = np.maximum(q[m, il] - y[m], y[m] - q[m, ih])
                self.cal_scores[(g, lo)] = np.sort(e)
                n = max(len(e), 1)
                level = min(1.0, (1 - (lo + 1 - hi)) * (1 + 1 / n))
                self.conf[g][lo] = float(np.quantile(e, level)) if len(e) else 0.0

    # ------------------------------------------------------------ inference
    def predict(self, df, avc_frac=None, widen=None):
        """Calibrated quantiles (CQR) clipped to [0, available capacity]."""
        q = self.raw_quantiles(df)
        reg = regime(df)
        for lo, hi in PAIRS:
            il, ih = QUANTILES.index(lo), QUANTILES.index(hi)
            w = np.array([self.conf[g][lo] for g in reg])
            if widen is not None and lo in widen:
                w = w + widen[lo]
            q[:, il] -= w
            q[:, ih] += w
        q = np.maximum.accumulate(np.sort(q, axis=1), axis=1)
        cap = 1.0 if avc_frac is None else np.asarray(avc_frac)[:, None]
        q = np.clip(q, 0, cap)
        q *= df["daylight"].values[:, None]
        return q

    def explain(self, df):
        """Additive explanation of P50 per block, in normalized units.

        clear-sky physics -> cloud attenuation (physics on NWP) -> ML residual by family.
        """
        X = df[FEATURES]
        contrib = self.med.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        out = {"Clear-sky physical twin": df["phys_cs"].values,
               "NWP irradiance through twin": (df["phys"] - df["phys_cs"]).values}
        idx = {f: i for i, f in enumerate(FEATURES)}
        for name, cols in GROUPS.items():
            out["ML: " + name] = contrib[:, [idx[c] for c in cols]].sum(axis=1)
        out["ML: site bias"] = contrib[:, -1]
        return out

    # ------------------------------------------------------------ persistence
    def save(self, tag="all"):
        self.med.save_model(DATA / f"model_p50_{tag}.json")
        self.qm.save_model(DATA / f"model_quantiles_{tag}.json")
        (DATA / f"conformal_{tag}.json").write_text(json.dumps({
            "conf": {str(g): {str(k): v for k, v in d.items()} for g, d in self.conf.items()},
            "scores": {f"{g}|{lo}": s.tolist()[::max(1, len(s) // 2000)]
                       for (g, lo), s in self.cal_scores.items()}}))

    @classmethod
    def load(cls, tag="all"):
        f = cls()
        f.med = _Booster(DATA / f"model_p50_{tag}.json")
        f.qm = _Booster(DATA / f"model_quantiles_{tag}.json")
        c = json.loads((DATA / f"conformal_{tag}.json").read_text())
        f.conf = {int(g): {float(k): v for k, v in d.items()} for g, d in c["conf"].items()}
        f.cal_scores = {(int(k.split("|")[0]), float(k.split("|")[1])): np.array(v)
                        for k, v in c["scores"].items()}
        return f


# ---------------------------------------------------------------- Kalman assimilation

class BiasKalman:
    """Scalar random-walk Kalman filter on the live SCADA residual (normalized units)."""

    def __init__(self, q=4e-4, r=2.5e-3):
        self.b, self.p, self.q, self.r = 0.0, 1e-2, q, r

    def update(self, e):
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.b += k * (e - self.b)
        self.p *= 1 - k
        return self.b


def kalman_states(y, p50, daylight):
    """Bias state after each block (resets each night). Returns array aligned with input."""
    kf, out = BiasKalman(), np.zeros(len(y))
    for t in range(len(y)):
        if daylight[t] <= 0:
            kf = BiasKalman()
        elif not np.isnan(y[t]):
            kf.update(y[t] - p50[t])
        out[t] = kf.b
    return out


def kalman_correction(bias_now, phys_cs_now, phys_cs_future, lead, tau=10.0):
    """Correction applied to a block `lead` blocks ahead, decaying back to the forecast."""
    scale = np.clip(phys_cs_future / max(phys_cs_now, 0.05), 0, 1.5)
    return bias_now * np.exp(-lead / tau) * scale


def scale_quantiles(q, lam):
    """Stretch/shrink every quantile around P50 by lam (per block)."""
    i50 = QUANTILES.index(0.5)
    p50 = q[:, [i50]]
    return np.clip(p50 + np.asarray(lam)[:, None] * (q - p50), 0, None)


def aci_scale(q, y, daylight, gamma=0.03, target=0.20, lam0=1.0):
    """Adaptive Conformal Inference on an interval-scale factor.

    lam[t] only uses observations before t. After each daylight block the 80% band is
    widened on a miss and tightened on a hit, so long-run coverage converges to 1-target
    even when the season shifts away from the calibration period.
    Returns (lam per block, hit per block or nan at night).
    """
    il, ih = QUANTILES.index(0.10), QUANTILES.index(0.90)
    lam, out, hits = lam0, np.zeros(len(y)), np.full(len(y), np.nan)
    for t in range(len(y)):
        out[t] = lam
        if daylight[t] <= 0 or np.isnan(y[t]):
            continue
        p50 = q[t, QUANTILES.index(0.5)]
        lo = max(0.0, p50 + lam * (q[t, il] - p50))
        hi = p50 + lam * (q[t, ih] - p50)
        miss = float(not (lo - 1e-9 <= y[t] <= hi + 1e-9))
        hits[t] = 1 - miss
        lam = float(np.clip(lam * np.exp(gamma * (miss - target) / target * 0.25), 0.3, 3.0))
    return out, hits


BLEND_LEADS = 16


def intraday_inputs(y_now, cs_now, bias_now, phys_cs_f, p50_f, lead):
    """Smart-persistence and Kalman-corrected hybrid forecasts for a block `lead` ahead."""
    pers = np.clip(y_now / max(cs_now, 0.05), 0, 1.2) * phys_cs_f
    hyb = np.clip(p50_f + kalman_correction(bias_now, cs_now, phys_cs_f, lead), 0, 1)
    return pers, hyb


def fit_blend(df):
    """Least-squares weight on smart persistence per lead (Horizon-adaptive blending)."""
    w = {}
    for lead in range(1, BLEND_LEADS + 1):
        num = den = 0.0
        for _, g in df.groupby("plant", sort=False):
            y, cs, b = g["y"].values, g["phys_cs"].values, g["kf_bias"].values
            day = g["daylight"].values
            for t in range(lead, len(g)):
                s = t - lead
                if day[t] <= 0 or day[s] <= 0 or cs[s] < 0.05 or np.isnan(y[t]) or np.isnan(y[s]):
                    continue
                p, h = intraday_inputs(y[s], cs[s], b[s], cs[t], g["q50"].values[t], lead)
                num += (y[t] - h) * (p - h)
                den += (p - h) ** 2
        w[lead] = float(np.clip(num / den, 0, 1)) if den > 0 else 0.0
    return w


class GroupForecaster:
    """One Forecaster per data group: real telemetry plants and simulated plants.

    Trained jointly, the simulator's error structure and real-site quirks (shading,
    tracker stow, sensor offsets) contaminated each other; separate models were better
    for both groups on the held-out test.
    """

    def __init__(self, group_of):
        self.group_of = group_of          # plant id -> "real" | "sim"
        self.models = {}

    def _groups(self, df):
        g = df["plant"].map(self.group_of).values
        return {k: g == k for k in np.unique(g)}

    def fit(self, train, calib):
        for k in sorted(set(self.group_of.values())):
            tr = train[train["plant"].map(self.group_of) == k]
            ca = calib[calib["plant"].map(self.group_of) == k]
            if len(tr):
                self.models[k] = Forecaster().fit(tr, ca)
        return self

    def _apply(self, fn, df, *args):
        out = None
        for k, m in self._groups(df).items():
            r = getattr(self.models[k], fn)(df[m], *args)
            if out is None:
                out = np.zeros((len(df),) + r.shape[1:])
            out[m] = r
        return out

    def predict(self, df):
        return self._apply("predict", df)

    def raw_quantiles(self, df):
        return self._apply("raw_quantiles", df)

    def explain(self, df):
        k = self.group_of[df["plant"].iloc[0]]
        return self.models[k].explain(df)

    def model_for(self, pid):
        return self.models[self.group_of[pid]]

    def save(self):
        for k, f in self.models.items():
            f.save(tag=k)

    @classmethod
    def load(cls, group_of):
        g = cls(group_of)
        for k in sorted(set(group_of.values())):
            if (DATA / f"model_p50_{k}.json").exists():
                g.models[k] = Forecaster.load(tag=k)
        return g
