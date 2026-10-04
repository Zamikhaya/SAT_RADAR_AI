# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:19:23 2026

@author: Zamikhaya.Magogotya
"""

import xarray as xr
import numpy as np
import hdf5plugin


from pathlib import Path
import xarray as xr
import hdf5plugin

DATA_DIR = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT"
)

FILE = list(DATA_DIR.glob("W_XX-EUMETSAT-Darmstadt*FCI*RRAD*FDHSI*NC4E*.nc"))

if not FILE:
    raise FileNotFoundError(
        f"No FCI RRAD NetCDF file found in:\n{DATA_DIR}"
    )

print("FCI FILE:")
for f in FILE:
    print(" ", f)

FILE = str(FILE[0])
GROUP = "data/ir_105/measured"



# ============================================================
# CONFIGURATION
# ============================================================

#FILE = r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY--DIS-NC4E_C_EUMT_20260504105622_IDPFI_OPE_20260504105244_....nc"

#GROUP = "data/ir_105"

# ============================================================
# OPEN
# ============================================================

print("=" * 70)
print("FCI IR_105 PHYSICAL VALIDATION")
print("=" * 70)

ds = xr.open_dataset(
    FILE,
    group=GROUP,
    engine="netcdf4",
    decode_cf=True
)


print("\nAVAILABLE VARIABLES:")
for name in ds.data_vars:
    print(f"  {name}")
print("\nDATASET")
print(ds)

# ============================================================
# RADIANCE
# ============================================================
# ============================================================
# RADIANCE
# ============================================================

if "effective_radiance" not in ds:
    raise KeyError(
        "effective_radiance not found. "
        f"Available variables: {list(ds.data_vars)}"
    )

rad = ds["effective_radiance"]


print("\n" + "=" * 70)
print("RADIANCE")
print("=" * 70)

r = rad.values

finite = np.isfinite(r)

print("Shape:", r.shape)
print("dtype:", r.dtype)

print("Total pixels:", r.size)
print("Finite pixels:", finite.sum())
print("NaN pixels:", (~finite).sum())

print("Finite %:", 100 * finite.sum() / r.size)
print("NaN %:", 100 * (~finite).sum() / r.size)

rv = r[finite]

print("\nRadiance statistics")
print("Minimum :", rv.min())
print("Maximum :", rv.max())
print("Mean    :", rv.mean())
print("Median  :", np.median(rv))
print("Std     :", rv.std())

# ============================================================
# PERCENTILES
# ============================================================

print("\nRadiance percentiles")

for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"{p:>3}% :", np.percentile(rv, p))

# ============================================================
# COORDINATES
# ============================================================

print("\n" + "=" * 70)
print("COORDINATES")
print("=" * 70)

print("x:")
print("  min:", float(ds.x.min()))
print("  max:", float(ds.x.max()))
print("  size:", ds.x.size)

print("\ny:")
print("  min:", float(ds.y.min()))
print("  max:", float(ds.y.max()))
print("  size:", ds.y.size)

# ============================================================
# CALIBRATION
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION")
print("=" * 70)

variables = [
    "radiance_unit_conversion_coefficient",
    "radiance_to_bt_conversion_coefficient_a",
    "radiance_to_bt_conversion_coefficient_b",
    "radiance_to_bt_conversion_coefficient_wavenumber",
    "radiance_to_bt_conversion_constant_c1",
    "radiance_to_bt_conversion_constant_c2",
]

for name in variables:
    value = ds[name].values
    print(f"{name}: {value}")

# ============================================================
# PIXEL QUALITY
# ============================================================

print("\n" + "=" * 70)
print("PIXEL QUALITY")
print("=" * 70)

pq = ds["pixel_quality"].values

print("Shape:", pq.shape)
print("dtype:", pq.dtype)

pq_finite = np.isfinite(pq)

print("Finite:", pq_finite.sum())
print("NaN:", (~pq_finite).sum())

if pq_finite.any():
    print("Minimum:", np.nanmin(pq))
    print("Maximum:", np.nanmax(pq))
    print("Unique:", np.unique(pq[pq_finite]))

else:
    print("WARNING: pixel_quality contains no finite values.")

# ============================================================
# NAN STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("NAN STRUCTURE")
print("=" * 70)

nan_mask = ~finite

nan_by_row = nan_mask.sum(axis=1)
nan_by_column = nan_mask.sum(axis=0)

print("\nNaNs by row:")
print("minimum:", nan_by_row.min())
print("maximum:", nan_by_row.max())
print("mean:", nan_by_row.mean())

print("\nNaNs by column:")
print("minimum:", nan_by_column.min())
print("maximum:", nan_by_column.max())
print("mean:", nan_by_column.mean())

# ============================================================
# FINITE PIXELS PER ROW
# ============================================================

finite_by_row = finite.sum(axis=1)

print("\nFinite pixels by row:")

for i in range(len(finite_by_row)):
    print(
        f"row {i:3d}: "
        f"{finite_by_row[i]:5d} / {r.shape[1]}"
    )

# ============================================================
# CONTIGUOUS VALID REGION
# ============================================================

print("\n" + "=" * 70)
print("VALID DATA EXTENT")
print("=" * 70)

rows, cols = np.where(finite)

print("Minimum row:", rows.min())
print("Maximum row:", rows.max())
print("Minimum column:", cols.min())
print("Maximum column:", cols.max())

# ============================================================
# CALCULATE BRIGHTNESS TEMPERATURE
# ============================================================

print("\n" + "=" * 70)
print("BRIGHTNESS TEMPERATURE")
print("=" * 70)

A = float(ds["radiance_to_bt_conversion_coefficient_a"])
B = float(ds["radiance_to_bt_conversion_coefficient_b"])
WN = float(ds["radiance_to_bt_conversion_coefficient_wavenumber"])
C1 = float(ds["radiance_to_bt_conversion_constant_c1"])
C2 = float(ds["radiance_to_bt_conversion_constant_c2"])
CONV = float(ds["radiance_unit_conversion_coefficient"])

# Radiance in W m-2 sr-1 um-1
radiance_um = r * CONV

valid = np.isfinite(radiance_um) & (radiance_um > 0)

# Planck BT using wavenumber formulation
bt_planck = (
    C2 * WN /
    np.log(
        1.0 +
        (C1 * WN**3) /
        r
    )
)

# FCI coefficient formulation
bt_fci = A * bt_planck + B

bt_valid = np.isfinite(bt_fci) & valid

bt = bt_fci[bt_valid]

print("Valid BT pixels:", bt.size)

print("BT minimum :", bt.min())
print("BT maximum :", bt.max())
print("BT mean    :", bt.mean())
print("BT median  :", np.median(bt))
print("BT std     :", bt.std())

print("\nBT percentiles")

for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"{p:>3}% :", np.percentile(bt, p))

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

print(ds["pixel_quality"].attrs)
print(ds["pixel_quality"].encoding)
print(ds["pixel_quality"].dtype)

print(ds["pixel_quality"].values)

import matplotlib.pyplot as plt
import numpy as np

radiance = ds["effective_radiance"].values

valid = np.isfinite(radiance)

plt.figure(figsize=(16, 5))

plt.imshow(
    valid,
    origin="upper",
    aspect="auto",
    interpolation="nearest"
)

plt.title("FCI IR_105 Effective Radiance – Valid Data Footprint")
plt.xlabel("Column")
plt.ylabel("Row")
plt.colorbar(label="Valid = 1 / NaN = 0")

plt.tight_layout()
plt.show()

valid = np.isfinite(radiance)

fig, ax = plt.subplots(figsize=(16, 5))

ax.imshow(
    valid,
    origin="upper",
    aspect="auto",
    interpolation="nearest"
)

ax.set_title(
    "MTG FCI IR_105 – Effective Radiance Validity Mask"
)

ax.set_xlabel("Column")
ax.set_ylabel("Row")

plt.tight_layout()
plt.show()

for row in range(radiance.shape[0]):

    valid_cols = np.where(np.isfinite(radiance[row, :]))[0]

    if len(valid_cols) > 0:

        first_valid = valid_cols.min()
        last_valid = valid_cols.max()
        count = len(valid_cols)

        print(
            f"row {row:3d}: "
            f"first={first_valid:4d}, "
            f"last={last_valid:4d}, "
            f"count={count:4d}"
        )
        
for row in range(radiance.shape[0]):

    valid_cols = np.where(np.isfinite(radiance[row]))[0]

    if len(valid_cols) == 0:
        continue

    first = valid_cols[0]
    last = valid_cols[-1]

    internal_nan = np.sum(
        ~np.isfinite(radiance[row, first:last+1])
    )

    print(
        f"row={row:3d} "
        f"first={first:4d} "
        f"last={last:4d} "
        f"valid={len(valid_cols):4d} "
        f"internal_NaN={internal_nan:4d}"
    )
    


# ============================================================
# RAW PIXEL QUALITY
# ============================================================

# ---------------------------------------------------------
# ROOT DATASET
# ---------------------------------------------------------
ds_root = xr.open_dataset(
    FILE,
    decode_cf=False,
    mask_and_scale=False
)

print("ROOT GROUPS")
print("===========")
print(ds_root.groups if hasattr(ds_root, "groups") else "Use netCDF4 for groups")


# ---------------------------------------------------------
# VIS04 MEASURED GROUP
# ---------------------------------------------------------
ds_vis04 = xr.open_dataset(
    FILE,
    group="data/vis_04/measured",
    decode_cf=False,
    mask_and_scale=False
)

print("\nVIS04 MEASURED GROUP")
print("====================")
print(ds_vis04)

print("\nVARIABLES")
print("=========")
print(list(ds_vis04.variables))


# ---------------------------------------------------------
# PIXEL QUALITY
# ---------------------------------------------------------
pq_raw = ds_vis04["pixel_quality"]

print("\nRAW PIXEL QUALITY")
print("=================")
print("dtype:", pq_raw.dtype)
print("shape:", pq_raw.shape)
print("dimensions:", pq_raw.dims)
print("attributes:")
print(pq_raw.attrs)


# ---------------------------------------------------------
# INDEX MAP
# ---------------------------------------------------------
idx_raw = ds_vis04["index_map"]

print("\nINDEX MAP")
print("=========")
print("dtype:", idx_raw.dtype)
print("shape:", idx_raw.shape)
print("dimensions:", idx_raw.dims)
print("attributes:")
print(idx_raw.attrs)

import numpy as np

pq = pq_raw.values

flags = {
    1:   "missing_warning",
    2:   "radiometric_warning",
    4:   "noise_warning",
    8:   "geolocation_warning",
    16:  "saturation_warning",
    32:  "straylight_correction_warning",
    64:  "extended_dynamic_range_warning",
    128: "encoding_saturation",
}

print("\nPIXEL QUALITY FLAG COUNTS")
print("=========================")

for mask, name in flags.items():

    count = np.count_nonzero((pq & mask) != 0)

    print(
        f"{name:35s} "
        f"count={count:10d} "
        f"percentage={100*count/pq.size:8.4f}%"
    )
    
import numpy as np

# Pixel quality
pq = ds_vis04["pixel_quality"].values

# Index map
idx = ds_vis04["index_map"].values

# Effective radiance
rad = ds_vis04["effective_radiance"].values

# Valid index-map pixels
valid_index = idx != 65535

# Quality flag = 0
quality_clean = pq == 0

# Radiometric warning
radiometric_warning = (pq & 2) != 0

print("VALID INDEX PIXELS")
print("==================")
print("count:", np.count_nonzero(valid_index))

print("\nQUALITY-CLEAN PIXELS")
print("====================")
print("count:", np.count_nonzero(quality_clean))

print("\nRADIOMETRIC WARNINGS")
print("====================")
print("count:", np.count_nonzero(radiometric_warning))

print("\nRADIOMETRIC WARNING AMONG INDEX-VALID PIXELS")
print("============================================")

warning_valid = radiometric_warning & valid_index

print("warning:", np.count_nonzero(warning_valid))
print("valid:", np.count_nonzero(valid_index))

percentage = (
    100.0 * np.count_nonzero(warning_valid)
    / np.count_nonzero(valid_index)
)

print(f"percentage: {percentage:.6f}%")


# =========================================================
# SCAN COVERAGE FROM INDEX MAP
# =========================================================

# A pixel is geometrically indexed when index_map != 65535
valid_index = idx != 65535

# Radiometric warning flag
radiometric_warning = (pq & 2) != 0

# ---------------------------------------------------------
# Determine first and last index-valid pixel in each row
# ---------------------------------------------------------

first_cols = np.full(pq.shape[0], -1, dtype=int)
last_cols = np.full(pq.shape[0], -1, dtype=int)

for r in range(pq.shape[0]):

    valid_positions = np.where(valid_index[r])[0]

    if valid_positions.size > 0:
        first_cols[r] = valid_positions[0]
        last_cols[r] = valid_positions[-1]


# ---------------------------------------------------------
# Build scan-valid mask
# ---------------------------------------------------------

valid_scan = np.zeros(pq.shape, dtype=bool)

for r in range(pq.shape[0]):

    first = first_cols[r]
    last = last_cols[r]

    if first >= 0 and last >= 0:
        valid_scan[r, first:last + 1] = True


# ---------------------------------------------------------
# Combined validity
# ---------------------------------------------------------

valid_combined = valid_index & valid_scan


# =========================================================
# STATISTICS
# =========================================================

print("\nSCAN COVERAGE VALIDATION")
print("========================")

for name, mask in [
    ("INDEX VALID", valid_index),
    ("SCAN VALID", valid_scan),
    ("COMBINED VALID", valid_combined),
]:

    n_valid = np.count_nonzero(mask)

    n_warning = np.count_nonzero(
        radiometric_warning & mask
    )

    percentage = (
        100.0 * n_warning / n_valid
        if n_valid > 0 else np.nan
    )

    print()
    print(name)
    print("-" * len(name))
    print(f"valid pixels:          {n_valid:,}")
    print(f"radiometric warnings: {n_warning:,}")
    print(f"warning percentage:    {percentage:.6f}%")
    
    
    # =========================================================
# RADIOMETRIC WARNING SPATIAL DISTRIBUTION
# =========================================================

warning_index_valid = radiometric_warning & valid_index

warning_rows = np.where(
    np.any(warning_index_valid, axis=1)
)[0]

warning_cols = np.where(
    np.any(warning_index_valid, axis=0)
)[0]

print("\nRADIOMETRIC WARNING SPATIAL DISTRIBUTION")
print("========================================")

print(
    "Affected rows:",
    len(warning_rows)
)

print(
    "First affected row:",
    warning_rows[0]
    if len(warning_rows) else "none"
)

print(
    "Last affected row:",
    warning_rows[-1]
    if len(warning_rows) else "none"
)

print(
    "Affected columns:",
    len(warning_cols)
)

print(
    "First affected column:",
    warning_cols[0]
    if len(warning_cols) else "none"
)

print(
    "Last affected column:",
    warning_cols[-1]
    if len(warning_cols) else "none"
)

print("\nWARNINGS PER ROW")
print("================")

for r in warning_rows:

    cols = np.where(warning_index_valid[r])[0]

    print(
        f"row={r:3d} "
        f"warnings={len(cols):5d} "
        f"first={cols[0]:5d} "
        f"last={cols[-1]:5d}"
    )
    
import numpy as np

pq = ds_vis04["pixel_quality"].values
idx = ds_vis04["index_map"].values
rad = ds_vis04["effective_radiance"].values

# Masks
valid_index = idx != 65535
radiometric_warning = (pq & 2) != 0

warning_valid = radiometric_warning & valid_index
clean_valid = (pq == 0) & valid_index

print("\nRADIOMETRIC WARNING RADIANCE ANALYSIS")
print("=====================================")

warning_rad = rad[warning_valid]
clean_rad = rad[clean_valid]

print("Warning pixels:", warning_rad.size)
print("Clean pixels:", clean_rad.size)

print("\nWARNING RADIANCE")
print("----------------")
print("min :", warning_rad.min())
print("max :", warning_rad.max())
print("mean:", warning_rad.mean())
print("std :", warning_rad.std())
print("median:", np.median(warning_rad))

print("\nCLEAN RADIANCE")
print("--------------")
print("min :", clean_rad.min())
print("max :", clean_rad.max())
print("mean:", clean_rad.mean())
print("std :", clean_rad.std())
print("median:", np.median(clean_rad))

print("\nWARNING FLAG VALUES")
print("===================")

warning_values, warning_counts = np.unique(
    pq[warning_valid],
    return_counts=True
)

for value, count in zip(warning_values, warning_counts):
    print(
        f"flag={value:3d} "
        f"binary={value:08b} "
        f"count={count:7d}"
    )
    
import numpy as np
import matplotlib.pyplot as plt

pq = ds_vis04["pixel_quality"].values
rad = ds_vis04["effective_radiance"].values
idx = ds_vis04["index_map"].values

# Radiometric warning = bit 2
warning = (pq & 2) != 0

# Geometrically valid pixels
valid_index = idx != 65535

# Warning pixels that are geometrically valid
warning_valid = warning & valid_index

print("Warning pixels:", np.count_nonzero(warning_valid))

# Bounding box
rows, cols = np.where(warning_valid)

print("Row range:", rows.min(), "to", rows.max())
print("Column range:", cols.min(), "to", cols.max())

plt.figure(figsize=(14, 5))
plt.imshow(warning_valid, aspect="auto")
plt.xlabel("X pixel")
plt.ylabel("Y scan row")
plt.title("MTG FCI VIS04 Radiometric Warning Pixels")
plt.colorbar(label="Radiometric warning")
plt.show()

warning_rad = rad[warning_valid]
clean_rad = rad[valid_index & ~warning]

print("WARNING RADIANCE")
print("================")
print("min   :", warning_rad.min())
print("max   :", warning_rad.max())
print("unique:", np.unique(warning_rad))
print("mean  :", warning_rad.mean())
print("median:", np.median(warning_rad))

print("\nCLEAN RADIANCE")
print("=============")
print("min   :", clean_rad.min())
print("max   :", clean_rad.max())

unique, counts = np.unique(warning_rad, return_counts=True)

print("\nWARNING RADIANCE FREQUENCY")
print("==========================")

for value, count in zip(unique, counts):
    print(f"radiance={value:5d}  count={count:6d}")
    
# =========================================================
# 5. SPATIAL ANALYSIS OF RADIOMETRIC WARNINGS
# =========================================================

warning = radiometric_warning & valid_combined

print()
print("SPATIAL DISTRIBUTION OF RADIOMETRIC WARNINGS")
print("============================================")

# Total warnings per row
warning_per_row = np.count_nonzero(warning, axis=1)

# Total warnings per column
warning_per_col = np.count_nonzero(warning, axis=0)

print("\nTOP ROWS WITH WARNINGS")
print("----------------------")

top_rows = np.argsort(warning_per_row)[::-1][:20]

for r in top_rows:
    if warning_per_row[r] > 0:
        print(
            f"row={r:3d} "
            f"warnings={warning_per_row[r]:5d}"
        )

print("\nTOP COLUMNS WITH WARNINGS")
print("-------------------------")

top_cols = np.argsort(warning_per_col)[::-1][:20]

for c in top_cols:
    if warning_per_col[c] > 0:
        print(
            f"column={c:5d} "
            f"warnings={warning_per_col[c]:5d}"
        )

# Bounding box of all warning pixels
warning_positions = np.where(warning)

if warning_positions[0].size > 0:

    rmin = warning_positions[0].min()
    rmax = warning_positions[0].max()

    cmin = warning_positions[1].min()
    cmax = warning_positions[1].max()

    print()
    print("WARNING BOUNDING BOX")
    print("--------------------")
    print(f"row:    {rmin} -> {rmax}")
    print(f"column: {cmin} -> {cmax}")
# =========================================================
# 6. WARNING PERCENTAGE PER ROW
# =========================================================

print()
print("ROW-WISE RADIOMETRIC WARNING ANALYSIS")
print("=====================================")

for r in range(pq.shape[0]):

    n_valid = np.count_nonzero(valid_combined[r])
    n_warning = np.count_nonzero(warning[r])

    if n_valid > 0:

        pct = 100.0 * n_warning / n_valid

        if n_warning > 0:

            print(
                f"row={r:3d} "
                f"valid={n_valid:5d} "
                f"warnings={n_warning:5d} "
                f"percentage={pct:8.4f}%"
            )
            
# =========================================================
# RADIOMETRIC WARNING SPATIAL DISTRIBUTION
# =========================================================

warning_mask = (pq & 2) != 0

print("\nRADIOMETRIC WARNING SPATIAL DISTRIBUTION")
print("========================================")

# Warning count per row
warning_rows = np.count_nonzero(warning_mask, axis=1)

print("\nWARNING COUNTS BY ROW")
print("---------------------")

for r, count in enumerate(warning_rows):
    if count > 0:
        cols = np.where(warning_mask[r])[0]

        print(
            f"row={r:3d} "
            f"warnings={count:5d} "
            f"first={cols[0]:5d} "
            f"last={cols[-1]:5d}"
        )

# Warning count per column
warning_cols = np.count_nonzero(warning_mask, axis=0)

print("\nWARNING COUNTS BY COLUMN")
print("------------------------")

active_cols = np.where(warning_cols > 0)[0]

print("columns containing warnings:", active_cols.size)

if active_cols.size > 0:
    print("first warning column:", active_cols[0])
    print("last warning column :", active_cols[-1])

# Bounding box
rows, cols = np.where(warning_mask)

print("\nWARNING BOUNDING BOX")
print("--------------------")

print("row min:", rows.min())
print("row max:", rows.max())
print("col min:", cols.min())
print("col max:", cols.max())

print("\nWARNING PIXELS VS INDEX MAP")
print("===========================")

warning_index = idx[warning_mask]

print("warning pixels:", warning_index.size)

print("index values:")
print("min     :", warning_index.min())
print("max     :", warning_index.max())
print("unique  :", np.unique(warning_index).size)

print("\nUNIQUE WARNING INDEX VALUES")
print("---------------------------")

unique_warning_index, counts = np.unique(
    warning_index,
    return_counts=True
)

for value, count in zip(unique_warning_index, counts):

    print(
        f"index={value:5d} "
        f"count={count:6d} "
        f"percentage={100*count/warning_index.size:8.4f}%"
    )
    
# =========================================================
# INDEX MAP SPATIAL ANALYSIS
# =========================================================

print("\nINDEX MAP SPATIAL ANALYSIS")
print("=========================")

for index_value in [29, 32, 43]:

    mask = idx == index_value

    n = np.count_nonzero(mask)

    rows_i, cols_i = np.where(mask)

    print()
    print(f"INDEX VALUE {index_value}")
    print("-" * 25)

    print(f"total pixels : {n:,}")

    if n > 0:
        print(f"row range    : {rows_i.min()} - {rows_i.max()}")
        print(f"column range : {cols_i.min()} - {cols_i.max()}")

        # Radiometric warnings within this index
        warning = warning_mask & mask

        n_warning = np.count_nonzero(warning)

        print(f"warnings     : {n_warning:,}")

        print(
            f"warning %    : "
            f"{100*n_warning/n:.6f}%"
        )
        
print("\nWARNING DISTRIBUTION WITHIN INDEX VALUES")
print("=========================================")

import numpy as np

pq = ds_vis04["pixel_quality"].values
rad = ds_vis04["effective_radiance"].values
idx = ds_vis04["index_map"].values

radiometric_warning = (pq & 2) != 0
valid_index = idx != 65535

# ---------------------------------------------------------
# WARNING PIXELS
# ---------------------------------------------------------
warning = radiometric_warning & valid_index

# ---------------------------------------------------------
# CLEAN PIXELS
# Exclude fill value and invalid index-map pixels
# ---------------------------------------------------------
clean = (
    (~radiometric_warning)
    & valid_index
    & (rad != 65535)
)

warning_rad = rad[warning].astype(np.float64)
clean_rad = rad[clean].astype(np.float64)

print("\nRADIANCE THRESHOLD ANALYSIS")
print("===========================")

print("\nWARNING PIXELS")
print("----------------")
print("count :", warning_rad.size)
print("min   :", warning_rad.min())
print("max   :", warning_rad.max())
print("mean  :", warning_rad.mean())
print("median:", np.median(warning_rad))
print("std   :", warning_rad.std())

print("\nUNIQUE WARNING RADIANCE VALUES")
print("------------------------------")

values, counts = np.unique(
    warning_rad.astype(np.uint16),
    return_counts=True
)

for value, count in zip(values, counts):
    print(
        f"radiance={int(value):5d} "
        f"count={count:7d} "
        f"percentage={100*count/warning_rad.size:8.4f}%"
    )

print("\nCLEAN RADIANCE NEAR WARNING RANGE")
print("----------------------------------")

# Look specifically around 190-220 DN
for value in range(190, 221):

    warning_count = np.count_nonzero(
        warning & (rad == value)
    )

    clean_count = np.count_nonzero(
        clean & (rad == value)
    )

    total = warning_count + clean_count

    if total > 0:

        warning_pct = (
            100.0 * warning_count / total
        )

        print(
            f"radiance={value:5d} "
            f"warning={warning_count:7d} "
            f"clean={clean_count:7d} "
            f"warning%={warning_pct:8.3f}"
        )

for index_value in [29, 32, 43]:

    index_mask = idx == index_value
    warning_mask_index = warning_mask & index_mask

    print()
    print(f"INDEX {index_value}")
    print("-" * 15)

    for r in range(pq.shape[0]):

        n_index = np.count_nonzero(index_mask[r])

        n_warning = np.count_nonzero(
            warning_mask_index[r]
        )

        if n_warning > 0:

            print(
                f"row={r:3d} "
                f"index_pixels={n_index:4d} "
                f"warnings={n_warning:3d} "
                f"warning_pct={100*n_warning/n_index:7.3f}%"
            )