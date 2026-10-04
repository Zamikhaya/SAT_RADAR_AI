
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

"""
Test satellite-radar collocation
"""
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import sys


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\SAT_RADAR_AI"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# PROJECT IMPORTS
# ==========================================================

from data.dataset import SatelliteRadarDataset




from pathlib import Path
from datetime import datetime

from data.dataset import SatelliteRadarDataset


SATELLITE = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY--DIS-NC4E_C_EUMT_20260504105500_IDPFI_OPE_20260504105024_20260504105100_N_JLS_O_0066_0004.nc"
)

RADAR = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\DUR\cfrad.20260504_084856.000_to_20260504_085345.000_Durban_SUR.nc"
)


# ==========================================================
# Satellite observation time
# ==========================================================

SATELLITE_TIME = datetime(
    2026,
    5,
    4,
    10,
    55,
    0,
)


# ==========================================================
# CREATE DATASET
# ==========================================================

dataset = SatelliteRadarDataset(

    satellite_files=[
        SATELLITE
    ],

    radar_files=[
        RADAR
    ],

    radar_elevation=2.3,

    tolerance_seconds=600,

    satellite_times=[
        SATELLITE_TIME
    ],
)


# ==========================================================
# DATASET INFORMATION
# ==========================================================

print()
print("=" * 70)
print("SATELLITE-RADAR DATASET")
print("=" * 70)

print()

print("Number of samples:")
print(len(dataset))


# ==========================================================
# LOAD FIRST SAMPLE
# ==========================================================

sample = dataset[0]


print()
print("=" * 70)
print("SAMPLE")
print("=" * 70)

print()

print("Sample type:")
print(type(sample))

print()

print("Sample:")
print(sample)