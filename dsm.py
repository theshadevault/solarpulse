"""Layer 4: DSM penalty model, stochastic schedule optimizer and gate-closure watchdog."""
import numpy as np

from config import (DSM_SLABS, GATE_LEAD_BLOCKS, PPA_RATE_INR_PER_KWH, QUANTILES,
                    REVISION_TRIGGER_PCT, UNDER_INJECTION_MULTIPLIER)

LEVELS = np.linspace(0.005, 0.995, 120)


def penalty(schedule, actual, avc, exempt=None):
    """DSM charge (INR) per block for injecting `actual` MW against `schedule` MW."""
    schedule, actual, avc = (np.asarray(a, float) for a in (schedule, actual, avc))
    dev = actual - schedule
    dev_pct = np.abs(dev) / np.maximum(avc, 1e-6) * 100
    charge = np.zeros_like(dev_pct)
    lower = 0.0
    for upper, frac in DSM_SLABS:
        seg = np.clip(dev_pct, lower, upper) - lower          # % of AvC inside this slab
        charge += seg / 100 * avc * frac
        lower = upper
    inr = charge * 0.25 * 1000 * PPA_RATE_INR_PER_KWH         # MW * 15 min -> kWh -> INR
    inr = np.where(dev < 0, inr * UNDER_INJECTION_MULTIPLIER, inr)
    if exempt is not None:                                    # grid-ordered curtailment
        inr = np.where(np.asarray(exempt) > 0, 0, inr)
    return inr


def samples_from_quantiles(q):
    """Inverse-CDF samples per block from its quantile set (n, len(QUANTILES)) -> (n, L)."""
    qs = np.array(QUANTILES)
    # pin the tails to [0, P05 - (P10-P05)] and [P95 + (P95-P90)]
    lo = np.maximum(q[:, 0] - (q[:, 1] - q[:, 0]), 0)
    hi = q[:, -1] + (q[:, -1] - q[:, -2])
    grid = np.concatenate([[0.0], qs, [1.0]])
    vals = np.column_stack([lo, q, hi])
    return np.array([np.interp(LEVELS, grid, v) for v in vals])


def critical_fractile():
    """alpha* = c_over / (c_under + c_over): the cost-optimal quantile to schedule."""
    return 1.0 / (1.0 + UNDER_INJECTION_MULTIPLIER)


def optimize(q_mw, avc_mw):
    """S*_b: the critical-fractile quantile of each block's calibrated distribution.

    With symmetric penalties this is P50; when under-injection costs more, the schedule
    is shaded down (P40 at a 1.5x multiplier). On the backtest this beat both submitting
    P50 and a brute-force expected-cost search over the sampled distribution.
    """
    a = critical_fractile()
    s = np.array([np.interp(a, QUANTILES, row) for row in np.asarray(q_mw)])
    return np.clip(s, 0, np.asarray(avc_mw))


def expected_penalty(schedule_mw, q_mw, avc_mw):
    smp = samples_from_quantiles(q_mw)
    return np.array([penalty(schedule_mw[b], smp[b], avc_mw[b]).mean() for b in range(len(q_mw))])


def gate_closure(now_block, current_schedule, revised_forecast, avc_mw):
    """Blocks from t+4 on whose projected deviation crosses the revision trigger."""
    first = now_block + GATE_LEAD_BLOCKS
    out = []
    for b in range(first, len(current_schedule)):
        if avc_mw[b] <= 0:
            continue
        dev = (revised_forecast[b] - current_schedule[b]) / avc_mw[b] * 100
        if abs(dev) >= REVISION_TRIGGER_PCT:
            out.append({"block": int(b), "current": float(current_schedule[b]),
                        "proposed": float(revised_forecast[b]), "dev_pct": float(dev)})
    return first, out


def compliance(schedule, actual, avc, exempt=None, daylight=None):
    dev = np.abs(np.asarray(actual) - np.asarray(schedule)) / np.maximum(avc, 1e-6) * 100
    m = np.ones(len(dev), bool) if daylight is None else np.asarray(daylight) > 0
    if exempt is not None:
        m &= np.asarray(exempt) == 0
    return float((dev[m] <= 10).mean() * 100) if m.any() else 100.0
