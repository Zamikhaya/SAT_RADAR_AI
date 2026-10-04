# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 01:26:46 2026

@author: Zamikhaya.Magogotya
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.satellite_loader import SatelliteLoader

SAT_DIR = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT"
)

files = sorted(SAT_DIR.glob("*.nc"))

loader = SatelliteLoader()

img = loader.load(files[14])

stats = loader.statistics(img)

print(stats)
