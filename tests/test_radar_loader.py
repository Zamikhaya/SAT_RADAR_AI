# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 00:01:16 2026

@author: Zamikhaya.Magogotya
"""

from pathlib import Path
from data.radar_loader import RadarLoader
import xarray as xr

RADAR = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\DUR\cfrad.20260504_000005.000_to_20260504_000452.999_Durban_SUR.nc"
)

loader = RadarLoader()
scan = loader.load(RADAR, elevation=2.3)
loader.summary(scan)



import xarray as xr

ds = xr.open_dataset(RADAR)

print()

print("Sweep variables")
for v in ds.variables:
    if "sweep" in v.lower() or "angle" in v.lower():
        print(v)

print()
print(ds["fixed_angle"])

print()

print(ds["sweep_start_ray_index"])

print()

print(ds["sweep_end_ray_index"])


print("Fixed angles:")
print(ds["fixed_angle"].values)

print()

print("Sweep start:")
print(ds["sweep_start_ray_index"].values)

print()

print("Sweep end:")
print(ds["sweep_end_ray_index"].values)