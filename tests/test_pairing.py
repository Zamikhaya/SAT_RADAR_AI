# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 12:28:41 2026

@author: Zamikhaya.Magogotya
"""

"""
Test satellite-radar temporal pairing.
"""

from pathlib import Path
from datetime import datetime

from data.pairing import SatelliteRadarPairer

# ==========================================================

# Satellite

# ==========================================================

SATELLITE = Path(
r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY--DIS-NC4E_C_EUMT_20260504105500_IDPFI_OPE_20260504105024_20260504105100_N_JLS_O_0066_0004.nc"
)

# ==========================================================

# Radar directory

# ==========================================================

RADAR_DIR = Path(
r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\DUR"
)

RADAR_FILES = sorted(
RADAR_DIR.glob(
"cfrad.*_Durban_SUR.nc"
)
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

# Pairer

# ==========================================================

pairer = SatelliteRadarPairer(

radar_elevation=2.3,

tolerance_seconds=600,


)

# ==========================================================

# Pair

# ==========================================================

result = pairer.pair(

satellite_file=SATELLITE,

radar_files=RADAR_FILES,

satellite_time=SATELLITE_TIME,


)

# ==========================================================

# Summary

# ==========================================================

pairer.summary(result)

# ==========================================================

# Additional diagnostics

# ==========================================================

if result.valid:

    print()
    print("=" * 60)
    print("PAIRING DETAILS")
    print("=" * 60)

    print("Satellite file:")
    print(result.satellite_file)

    print()
    print("Radar file:")
    print(result.radar_file)

    print()
    print("Satellite time:")
    print(result.satellite_time)

    print()
    print("Radar time:")
    print(result.radar_time)

    print()
    print("Time difference:")
    print(f"{result.time_difference_seconds:.1f} seconds")

    print()
    print("Radar elevation:")
    print(f"{result.radar_elevation:.2f} degrees")

    print()
    print("Radar sweep:")
    print(result.radar_sweep_index)

    print()
    print("Valid:")
    print(result.valid)

    print("=" * 60)