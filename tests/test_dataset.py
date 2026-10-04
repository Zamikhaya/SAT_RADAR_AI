# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:15:57 2026

@author: Zamikhaya.Magogotya

Test paired Satellite-Radar dataset.

Author: Zamikhaya Magogotya
"""

from pathlib import Path
from datetime import datetime

from data.dataset import SatelliteRadarDataset


# ==========================================================
# Files
# ==========================================================

SATELLITE = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY--DIS-NC4E_C_EUMT_20260504105500_IDPFI_OPE_20260504105024_20260504105100_N_JLS_O_0066_0004.nc"
)


RADAR_FILES = [

    Path(
        r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\DUR\cfrad.20260504_084856.000_to_20260504_085345.000_Durban_SUR.nc"
    ),

]


SATELLITE_TIME = datetime(
    2026,
    5,
    4,
    10,
    55,
    0,
)


# ==========================================================
# Dataset
# ==========================================================

dataset = SatelliteRadarDataset()


# ==========================================================
# Load sample
# ==========================================================

sample = dataset.load_sample(

    satellite_file=SATELLITE,

    radar_files=RADAR_FILES,

    satellite_time=SATELLITE_TIME,

)


# ==========================================================
# Summary
# ==========================================================

dataset.summary(sample)
