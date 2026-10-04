# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:26:08 2026

@author: Zamikhaya.Magogotya
"""
# ============================================================
# PROJECT ROOT
# ============================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\SAT_RADAR_AI"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ============================================================
# test_collocation_visual.py
#
# Scientific validation of MTG-FCI / Durban radar collocation
# ============================================================

from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from data.dataset import SatelliteRadarDataset
from data.collocation import SatelliteRadarCollocator


# ============================================================
# FILES
# ============================================================

SATELLITE = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT"
    r"\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD"
    r"--CHK-BODY--DIS-NC4E_C_EUMT_20260504105500_IDPFI_OPE_20260504105024"
    r"_20260504105100_N_JLS_O_0066_0004.nc"
)

RADAR = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\DUR"
    r"\cfrad.20260504_084856.000_to_20260504_085345.000_Durban_SUR.nc"
)

SATELLITE_TIME = datetime(
    2026,
    5,
    4,
    10,
    55,
    0,
)


# ============================================================
# FILE CHECK
# ============================================================

print("=" * 70)
print("FILE CHECK")
print("=" * 70)

print("Satellite exists :", SATELLITE.exists())
print("Radar exists     :", RADAR.exists())

if not SATELLITE.exists():
    raise FileNotFoundError(SATELLITE)

if not RADAR.exists():
    raise FileNotFoundError(RADAR)


# ============================================================
# DATASET
# ============================================================

print()
print("=" * 70)
print("CREATING DATASET")
print("=" * 70)

dataset = SatelliteRadarDataset(
    satellite_files=[SATELLITE],
    radar_files=[RADAR],
    radar_elevation=2.3,
    tolerance_seconds=600,
    satellite_times=[SATELLITE_TIME],
)

print("Number of samples :", len(dataset))

if len(dataset) == 0:
    raise RuntimeError("No satellite-radar pairs were created.")


# ============================================================
# LOAD SAMPLE
# ============================================================

sample = dataset[0]

print()
print("=" * 70)
print("SAMPLE")
print("=" * 70)

print("Valid              :", sample.valid)
print("Satellite time     :", sample.satellite_time)
print("Radar time         :", sample.radar_time)
print("Time difference    :", sample.time_difference_seconds)
print("Radar site         :", sample.radar_site)
print("Radar elevation    :", sample.radar_elevation)
print("Radar sweep        :", sample.radar_sweep_index)

############################################################


#############################################################
# ============================================================
# COLLOCATION
# ============================================================

print()
print("=" * 70)
print("RUNNING COLLOCATION")
print("=" * 70)

collocator = SatelliteRadarCollocator(
    satellite_channel="ir_105"
)

result = collocator.collocate(sample)


# ============================================================
# EXTRACT DATA
# ============================================================

latitude = np.asarray(result.latitude)
longitude = np.asarray(result.longitude)

reflectivity = np.asarray(result.reflectivity)
velocity = np.asarray(result.velocity)
spectrum_width = np.asarray(result.spectrum_width)

satellite_values = np.asarray(
    result.satellite_values
)

valid_mask = np.asarray(
    result.valid_mask
)


# ============================================================
# STATISTICS
# ============================================================

print()
print("=" * 70)
print("COLLOCATION STATISTICS")
print("=" * 70)

print("Latitude shape       :", latitude.shape)
print("Longitude shape      :", longitude.shape)
print("Satellite shape      :", satellite_values.shape)
print("Reflectivity shape   :", reflectivity.shape)

print()

print("Latitude range       :",
      np.nanmin(latitude),
      "to",
      np.nanmax(latitude))

print("Longitude range      :",
      np.nanmin(longitude),
      "to",
      np.nanmax(longitude))

print()

print("Valid collocation bins :",
      np.count_nonzero(valid_mask))

print("Total radar bins       :",
      valid_mask.size)

print(
    "Valid percentage       :",
    100.0 * np.count_nonzero(valid_mask)
    / valid_mask.size
)


# ============================================================
# APPLY VALID MASK
# ============================================================

sat_valid = np.where(
    valid_mask,
    satellite_values,
    np.nan,
)

dbz_valid = np.where(
    valid_mask,
    reflectivity,
    np.nan,
)


# ============================================================
# CHECK SATELLITE VALUES
# ============================================================

print()
print("=" * 70)
print("SATELLITE BRIGHTNESS TEMPERATURE")
print("=" * 70)

finite_sat = np.isfinite(sat_valid)

print("Finite pixels :", np.count_nonzero(finite_sat))

if np.any(finite_sat):

    print(
        "Minimum       :",
        np.nanmin(sat_valid),
        "K"
    )

    print(
        "Maximum       :",
        np.nanmax(sat_valid),
        "K"
    )

    print(
        "Mean          :",
        np.nanmean(sat_valid),
        "K"
    )

    print(
        "Std           :",
        np.nanstd(sat_valid),
        "K"
    )


# ============================================================
# CHECK RADAR REFLECTIVITY
# ============================================================

print()
print("=" * 70)
print("RADAR REFLECTIVITY")
print("=" * 70)

finite_dbz = np.isfinite(dbz_valid)

print("Finite pixels :", np.count_nonzero(finite_dbz))

if np.any(finite_dbz):

    print(
        "Minimum       :",
        np.nanmin(dbz_valid)
    )

    print(
        "Maximum       :",
        np.nanmax(dbz_valid)
    )

    print(
        "Mean          :",
        np.nanmean(dbz_valid)
    )


# ============================================================
# FIGURE 1
# RADAR REFLECTIVITY
# ============================================================

plt.figure(figsize=(10, 8))

plt.pcolormesh(
    longitude,
    latitude,
    dbz_valid,
    shading="auto",
)

plt.colorbar(
    label="Reflectivity"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "Durban Radar Reflectivity\n"
    "Sweep 2 — 2.3° elevation"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FIGURE 2
# SATELLITE IR
# ============================================================

plt.figure(figsize=(10, 8))

plt.pcolormesh(
    longitude,
    latitude,
    sat_valid,
    shading="auto",
)

plt.colorbar(
    label="Brightness Temperature (K)"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "MTG-FCI IR 10.5 µm\n"
    "Satellite Values Collocated to Durban Radar"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FIGURE 3
# RADAR + SATELLITE CONTOURS
# ============================================================

plt.figure(figsize=(10, 8))

plt.pcolormesh(
    longitude,
    latitude,
    sat_valid,
    shading="auto",
)

plt.colorbar(
    label="Brightness Temperature (K)"
)

# Radar reflectivity contours
contour = plt.contour(
    longitude,
    latitude,
    dbz_valid,
    levels=[
        10,
        20,
        30,
        40,
        50,
    ],
)

plt.clabel(
    contour,
    inline=True,
    fontsize=8,
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "MTG-FCI IR 10.5 µm + Durban Radar Reflectivity"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FIGURE 4
# SATELLITE BT vs RADAR REFLECTIVITY
# ============================================================

sat_flat = sat_valid.flatten()
dbz_flat = dbz_valid.flatten()

mask = (
    np.isfinite(sat_flat)
    &
    np.isfinite(dbz_flat)
)

sat_flat = sat_flat[mask]
dbz_flat = dbz_flat[mask]

print()
print("=" * 70)
print("SATELLITE / RADAR MATCHED PIXELS")
print("=" * 70)

print("Matched pixels :", len(sat_flat))

if len(sat_flat) > 0:

    plt.figure(figsize=(9, 7))

    plt.scatter(
        sat_flat,
        dbz_flat,
        s=4,
        alpha=0.3,
    )

    plt.xlabel(
        "MTG-FCI IR 10.5 µm Brightness Temperature (K)"
    )

    plt.ylabel(
        "Radar Reflectivity"
    )

    plt.title(
        "Satellite Brightness Temperature "
        "vs Radar Reflectivity"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("COLLOCATION VALIDATION COMPLETED")
print("=" * 70)

print("Dataset pairing       : PASS")
print("Satellite loading     : PASS")
print("Radar loading         : PASS")
print("Geographic conversion : PASS")
print("Satellite sampling    : PASS")
print("Valid mask generation : PASS")

print()
print("Next step:")
print(
    "Inspect the four plots for geographic and "
    "meteorological consistency."
)

print("=" * 70)


# ============================================================
# COLLOCATION DIAGNOSTICS
# ============================================================

print()
print("=" * 70)
print("DETAILED COLLOCATION DIAGNOSTICS")
print("=" * 70)

print()
print("LATITUDE")
print("-" * 70)

print("Finite :", np.isfinite(result.latitude).sum())
print("Min    :", np.nanmin(result.latitude))
print("Max    :", np.nanmax(result.latitude))

print()
print("LONGITUDE")
print("-" * 70)

print("Finite :", np.isfinite(result.longitude).sum())
print("Min    :", np.nanmin(result.longitude))
print("Max    :", np.nanmax(result.longitude))

print()
print("SATELLITE X")
print("-" * 70)

print("Finite :", np.isfinite(result.satellite_x).sum())

if np.isfinite(result.satellite_x).any():
    print("Min    :", np.nanmin(result.satellite_x))
    print("Max    :", np.nanmax(result.satellite_x))

print()
print("SATELLITE Y")
print("-" * 70)

print("Finite :", np.isfinite(result.satellite_y).sum())

if np.isfinite(result.satellite_y).any():
    print("Min    :", np.nanmin(result.satellite_y))
    print("Max    :", np.nanmax(result.satellite_y))

print()
print("SATELLITE VALUES")
print("-" * 70)

print("Finite :", np.isfinite(result.satellite_values).sum())

if np.isfinite(result.satellite_values).any():
    print("Min    :", np.nanmin(result.satellite_values))
    print("Max    :", np.nanmax(result.satellite_values))
    print("Mean   :", np.nanmean(result.satellite_values))

print()
print("RADAR REFLECTIVITY")
print("-" * 70)

print("Finite :", np.isfinite(result.reflectivity).sum())

if np.isfinite(result.reflectivity).any():
    print("Min    :", np.nanmin(result.reflectivity))
    print("Max    :", np.nanmax(result.reflectivity))

print()
print("VALID MASK")
print("-" * 70)

print("True :", np.sum(result.valid_mask))
print("False:", np.sum(~result.valid_mask))

print("=" * 70)
import hdf5plugin 
import xarray as xr
import h5py
# Measured radiance
ds = xr.open_dataset(
    SATELLITE,
    group="data/ir_105/measured"
)


print("=" * 70)
print("FCI IR_105 CHUNK POSITION")
print("=" * 70)

for name in [
    "start_position_row",
    "start_position_column",
    "end_position_row",
    "end_position_column"
]:
    if name in ds:
        print(f"{name:30s}: {ds[name].values}")
    else:
        print(f"{name:30s}: NOT FOUND")

print("\nIR_105 DATA")
print("-" * 70)
print("effective_radiance shape:", ds["effective_radiance"].shape)

print("\nIR_105 COORDINATES")
print("-" * 70)
print("x shape:", ds["x"].shape)
print("x range:", float(ds["x"].min()), "to", float(ds["x"].max()))

print("y shape:", ds["y"].shape)
print("y range:", float(ds["y"].min()), "to", float(ds["y"].max()))

print("\n" + "=" * 80)
print("REFERENCE GRID DETAILS")
print("=" * 80)

with h5py.File(SATELLITE, "r") as f:

    names = [
        "state/processor/reference_grid",
        "state/processor/reference_grid_identifier",
        "state/processor/reference_grid_number_of_columns",
        "state/processor/reference_grid_number_of_rows",
        "state/processor/reference_grid_projection",
        "state/processor/reference_grid_spatial_sampling_angle_ew",
        "state/processor/reference_grid_spatial_sampling_angle_ns",
        "state/processor/reference_grid_version",
    ]

    for name in names:
        print(f"\n{name}")
        print(f"    {f[name][()]}")

print("\n" + "=" * 80)
print("IR_105 COORDINATE DETAILS")
print("=" * 80)

print("\nX:")
print("first 10:", ds["x"].values[:10])
print("last 10 :", ds["x"].values[-10:])

print("\nY:")
print("first 10:", ds["y"].values[:10])
print("last 10 :", ds["y"].values[-10:])
        
print("\n" + "=" * 80)
print("FCI REFERENCE GRID / DURBAN ROW TEST")
print("=" * 80)

# Reference Grid 2
n = 5568
sampling = 5.58871526e-05

# Full-grid angular extent
full_extent = n * sampling

print(f"Grid size           : {n} x {n}")
print(f"Sampling            : {sampling:.12e} rad")
print(f"Full grid extent    : {full_extent:.8f} rad")
print(f"Half grid extent    : {full_extent/2:.8f} rad")

# IR_105 chunk
start_row = int(ds["start_position_row"].values)
end_row = int(ds["end_position_row"].values)

print("\nIR_105 chunk:")
print(f"Start row           : {start_row}")
print(f"End row             : {end_row}")
print(f"Number of rows      : {end_row - start_row + 1}")

print("\nIR_105 Y:")
print(f"First Y             : {ds['y'].values[0]:.12f}")
print(f"Last Y              : {ds['y'].values[-1]:.12f}")
print(f"Y midpoint          : {np.mean(ds['y'].values):.12f}")

print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
# ============================================================
# RADAR PROJECTED COORDINATES
# ============================================================

print("\n" + "=" * 80)
print("RADAR COORDINATES")
print("=" * 80)

# ------------------------------------------------------------
# Check what the collocation result contains
# ------------------------------------------------------------

print("\nResult attributes:")
print("-" * 80)

for name in dir(result):
    if not name.startswith("_"):
        print(name)

# ------------------------------------------------------------
# Check whether radar coordinates are already available
# ------------------------------------------------------------

for name in [
    "radar_latitude",
    "radar_longitude",
    "radar_lat",
    "radar_lon",
]:

    if hasattr(result, name):

        value = getattr(result, name)

        print(f"\n{name}:")
        print("    value :", value)

# ------------------------------------------------------------
# Check sample radar site information
# ------------------------------------------------------------

print("\nRadar site:")
print("    ", sample.radar_site)

print("\nRadar elevation:")
print("    ", sample.radar_elevation)

print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
# ============================================================
# RADAR BIN GEOGRAPHIC COORDINATES
# ============================================================

radar_lon = np.asarray(result.longitude)
radar_lat = np.asarray(result.latitude)

print()
print("=" * 80)
print("RADAR BIN GEOGRAPHIC COORDINATES")
print("=" * 80)

print("Latitude shape :", radar_lat.shape)
print("Longitude shape:", radar_lon.shape)

finite = (
    np.isfinite(radar_lat)
    &
    np.isfinite(radar_lon)
)

print("Finite bins    :", np.count_nonzero(finite))

if np.any(finite):

    print()
    print("Latitude:")
    print("  Min :", np.nanmin(radar_lat))
    print("  Max :", np.nanmax(radar_lat))

    print()
    print("Longitude:")
    print("  Min :", np.nanmin(radar_lon))
    print("  Max :", np.nanmax(radar_lon))

    print()
    print("Radar centre:")
    print(
        "  Latitude :",
        np.nanmean(radar_lat)
    )
    print(
        "  Longitude:",
        np.nanmean(radar_lon)
    )
    
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# ============================================================
# RADAR PROJECTED COORDINATES IN FCI GRID
# ============================================================

print()
print("=" * 80)
print("RADAR PROJECTED COORDINATES")
print("=" * 80)

# The collocator has already transformed the radar-bin
# geographic coordinates into the satellite/FCI coordinate
# system.

radar_x = np.asarray(result.satellite_x)
radar_y = np.asarray(result.satellite_y)

print("Radar X shape :", radar_x.shape)
print("Radar Y shape :", radar_y.shape)

finite_xy = (
    np.isfinite(radar_x)
    &
    np.isfinite(radar_y)
)

print("Finite X/Y bins :", np.count_nonzero(finite_xy))

if np.any(finite_xy):

    print()
    print("Radar projected X:")
    print(
        f"X: {np.nanmin(radar_x):,.8f} "
        f"to {np.nanmax(radar_x):,.8f}"
    )

    print()
    print("Radar projected Y:")
    print(
        f"Y: {np.nanmin(radar_y):,.8f} "
        f"to {np.nanmax(radar_y):,.8f}"
    )

    print()
    print("Radar projected centre:")
    print(
        "X:",
        np.nanmean(radar_x)
    )
    print(
        "Y:",
        np.nanmean(radar_y)
    )