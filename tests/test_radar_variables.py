# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 00:29:02 2026

@author: Zamikhaya.Magogotya
"""

from pathlib import Path
import xarray as xr
import numpy as np

RADAR = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\DUR\cfrad.20260504_000005.000_to_20260504_000452.999_Durban_SUR.nc"
)

ds = xr.open_dataset(RADAR)

print()

for var in ["dBZ", "dBuZ", "V", "W"]:

    arr = ds[var].values

    print("="*60)
    print(var)

    print("shape:", arr.shape)

    print("NaNs :", np.isnan(arr).sum())

    print("Finite:", np.isfinite(arr).sum())

    if np.isfinite(arr).sum() > 0:
        print("min :", np.nanmin(arr))
        print("max :", np.nanmax(arr))
        
        
import xarray as xr

ds = xr.open_dataset(RADAR)

print(ds)


print(ds["ray_start_index"])
print(ds["ray_n_gates"])

for key, value in ds.attrs.items():
    if "lat" in key.lower() or "lon" in key.lower() or "alt" in key.lower():
        print(key, "=", value)
        
for v in ds.variables:
    if "lat" in v.lower() or "lon" in v.lower() or "alt" in v.lower():
        print(v)
        
        
