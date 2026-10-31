"""Plant fleet, DSM rules and pipeline settings for the SolarPulse prototype."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ON_VERCEL = bool(os.environ.get("VERCEL"))
# Vercel's filesystem is read-only outside /tmp
SCRATCH = Path("/tmp/solarpulse") if ON_VERCEL else DATA
CACHE = SCRATCH / "cache"
TZ = "Asia/Kolkata"

# Training history window (archived NWP + ERA5 reanalysis).
HISTORY_START = "2026-05-01"
HISTORY_DAYS = 175

NWP_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
NWP_VARS = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
            "temperature_2m", "wind_speed_10m", "cloud_cover", "cloud_cover_low",
            "relative_humidity_2m"]
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


@dataclass(frozen=True)
class Plant:
    id: str
    name: str
    state: str
    lat: float
    lon: float
    ac_mw: float          # inverter (AC) capacity = contracted capacity
    dc_ac: float          # DC/AC ratio -> midday clipping
    tracker: bool         # single-axis tracker vs fixed tilt
    tilt: float = 0.0
    gcr: float = 0.35
    seed: int = 0
    tz: str = "Asia/Kolkata"
    kind: str = "sim"            # "sim" = simulated SCADA, "pvdaq" = real NREL telemetry
    gamma: float = -0.0038       # module power temperature coefficient (1/C)
    start: str = ""              # data window for real plants
    end: str = ""
    source: str = ""

    @property
    def dc_mw(self):
        return self.ac_mw * self.dc_ac


# Locations are real Indian solar regions; plant specs and telemetry are simulated.
PLANTS = [
    Plant("bhadla", "Bhadla-A", "Rajasthan", 27.53, 71.91, 300, 1.30, True, seed=11),
    Plant("charanka", "Charanka-B", "Gujarat", 23.90, 71.20, 150, 1.25, False, tilt=23, seed=22),
    Plant("pavagada", "Pavagada-C", "Karnataka", 14.25, 77.28, 250, 1.35, True, seed=33),
    Plant("kamuthi", "Kamuthi-D", "Tamil Nadu", 9.35, 78.39, 100, 1.20, False, tilt=10, seed=44),
]
# Real plants: public NREL PVDAQ telemetry (2023 Solar Data Prize), real archived NWP.
REAL_PLANTS = [
    Plant("srco", "SR_CO", "Kersey, Colorado (USA)", 40.3864, -104.5512, 3.82, 1.24, True,
          seed=55, tz="Etc/GMT+7", kind="pvdaq", gamma=-0.0028, start="2024-01-01",
          end="2025-04-30", source="NREL PVDAQ system 9068 (8 inverter modules, POA pyranometers)"),
    Plant("fsa", "Farm Solar Array", "Arbuckle, California (USA)", 38.9963, -122.1341, 0.72,
          1.24, False, tilt=25, seed=66, tz="Etc/GMT+8", kind="pvdaq", gamma=-0.0041,
          start="2024-02-01", end="2025-12-31",
          source="NREL PVDAQ system 2107 (revenue meter, POA pyranometer)"),
]
ALL_PLANTS = PLANTS + REAL_PLANTS
PLANT_BY_ID = {p.id: p for p in ALL_PLANTS}

# Simplified CERC-style DSM slabs for solar sellers: deviation as % of AvC
# -> charge as a fraction of the PPA rate applied to the deviated energy.
PPA_RATE_INR_PER_KWH = 2.60
DSM_SLABS = [(10, 0.0), (20, 0.10), (30, 0.20), (999, 0.30)]
UNDER_INJECTION_MULTIPLIER = 1.5   # under-injection hurts more (frequency support)

GATE_LEAD_BLOCKS = 4              # intraday revision takes effect from block t+4
REVISION_TRIGGER_PCT = 6.5        # suggest a revision when |Δ| / AvC exceeds this
