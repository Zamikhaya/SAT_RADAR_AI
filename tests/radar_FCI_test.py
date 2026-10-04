# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:33:38 2026

@author: Zamikhaya.Magogotya
"""

from pathlib import Path
from netCDF4 import Dataset
import hdf5plugin
import os
import numpy as np

os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH

FCI_FILE = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY--DIS-NC4E_C_EUMT_20260504105547_IDPFI_OPE_20260504105141_20260504105216_N_JLS_O_0066_0010.nc"
)

with Dataset(FCI_FILE) as ds:

    # ==========================================================
    # FCI IR_105 measured group
    # ==========================================================

    measured = (
        ds.groups["data"]
          .groups["ir_105"]
          .groups["measured"]
    )

    x = measured.variables["x"][:]
    y = measured.variables["y"][:]

    radiance_var = measured.variables["effective_radiance"]

    # ==========================================================
    # FCI coordinate information
    # ==========================================================

    print("=" * 80)
    print("FCI IR_105 NATIVE COORDINATES")
    print("=" * 80)

    print("X shape:", x.shape)
    print("Y shape:", y.shape)

    print(f"X range: {x.min():.12f} to {x.max():.12f} rad")
    print(f"Y range: {y.min():.12f} to {y.max():.12f} rad")

    # ==========================================================
    # Durban radar coordinate
    # ==========================================================

    H = 35786400.0

    radar_x_m = 2722442.0
    radar_y_m = -2997771.0

    radar_x = radar_x_m / H
    radar_y = radar_y_m / H

    print()
    print("=" * 80)
    print("DURBAN RADAR → FCI COORDINATE")
    print("=" * 80)

    print(f"Radar X projected : {radar_x_m:,.3f} m")
    print(f"Radar Y projected : {radar_y_m:,.3f} m")

    print(f"Radar X native    : {radar_x:.12f} rad")
    print(f"Radar Y native    : {radar_y:.12f} rad")

    # ==========================================================
    # Check whether radar point falls inside this FCI chunk
    # ==========================================================

    inside_x = x.min() <= radar_x <= x.max()
    inside_y = y.min() <= radar_y <= y.max()

    print()
    print("=" * 80)
    print("RANGE CHECK")
    print("=" * 80)

    print("Inside X:", inside_x)
    print("Inside Y:", inside_y)

    if not (inside_x and inside_y):
        print()
        print("WARNING: Radar coordinate is outside this FCI chunk.")
        print("Stop here and check the correct spatial chunk.")
        raise SystemExit

    # ==========================================================
    # Find nearest FCI pixel
    # ==========================================================

    ix = np.argmin(np.abs(x - radar_x))
    iy = np.argmin(np.abs(y - radar_y))

    print()
    print("=" * 80)
    print("NEAREST FCI PIXEL")
    print("=" * 80)

    print("ix:", ix)
    print("iy:", iy)

    print(f"FCI X: {x[ix]:.12f} rad")
    print(f"FCI Y: {y[iy]:.12f} rad")

    print(
        f"X difference: "
        f"{abs(x[ix] - radar_x):.12e} rad"
    )

    print(
        f"Y difference: "
        f"{abs(y[iy] - radar_y):.12e} rad"
    )

    # ==========================================================
    # Read raw radiance
    # ==========================================================

    raw = radiance_var[iy, ix]

    print()
    print("=" * 80)
    print("IR_105 PIXEL")
    print("=" * 80)

    print("Raw radiance:", raw)

    # ==========================================================
    # Pixel quality
    # ==========================================================

    quality_var = measured.variables["pixel_quality"]
    quality = quality_var[iy, ix]

    print("Pixel quality:", quality)

    # ==========================================================
    # Radiance metadata
    # ==========================================================

    print()
    print("=" * 80)
    print("RADIANCE METADATA")
    print("=" * 80)

    for attr in [
        "_FillValue",
        "valid_range",
        "scale_factor",
        "add_offset",
        "units",
    ]:
        if hasattr(radiance_var, attr):
            print(
                f"{attr}: "
                f"{getattr(radiance_var, attr)}"
            )

    # ==========================================================
    # Convert if necessary
    # ==========================================================

    fill_value = getattr(
        radiance_var,
        "_FillValue",
        None
    )

    scale_factor = getattr(
        radiance_var,
        "scale_factor",
        1.0
    )

    add_offset = getattr(
        radiance_var,
        "add_offset",
        0.0
    )

    print()
    print("=" * 80)
    print("PHYSICAL VALUE")
    print("=" * 80)

    if fill_value is not None and raw == fill_value:

        print("Pixel is FILL VALUE.")

    else:

        physical = (
            float(raw) * float(scale_factor)
            + float(add_offset)
        )

        print(
            f"Physical radiance: "
            f"{physical:.6f}"
        )

    # ==========================================================
    # Brightness-temperature conversion coefficients
    # ==========================================================

    print()
    print("=" * 80)
    print("BRIGHTNESS TEMPERATURE COEFFICIENTS")
    print("=" * 80)

    for name in [
        "radiance_unit_conversion_coefficient",
        "radiance_to_bt_conversion_coefficient_a",
        "radiance_to_bt_conversion_coefficient_b",
        "radiance_to_bt_conversion_coefficient_wavenumber",
        "radiance_to_bt_conversion_constant_c1",
        "radiance_to_bt_conversion_constant_c2",
    ]:

        if name in measured.variables:

            value = measured.variables[name][:]

            print(f"{name}: {value}")
    
    # ==========================================================
    # NEIGHBOURHOOD QUALITY TEST
    # ==========================================================

    radiance_data = radiance_var[:]
    quality_data = quality_var[:]

    print()
    print("=" * 80)
    print("5 x 5 FCI NEIGHBOURHOOD")
    print("=" * 80)

    for j in range(max(0, iy - 2), min(len(y), iy + 3)):
        
        for i in range(max(0, ix - 2), min(len(x), ix + 3)):

            value = radiance_data[j, i]
            q = quality_data[j, i]

        print(
            f"iy={j:3d}, ix={i:4d} | "
            f"x={x[i]:.9f} | "
            f"y={y[j]:.9f} | "
            f"radiance={value} | "
            f"quality={q}"
        )
    
    print()
    print("=" * 80)
    print("PIXEL QUALITY VARIABLE")
    print("=" * 80)

    print("dtype:", quality_var.dtype)
    print("shape:", quality_var.shape)
    print("Fill value:", getattr(quality_var, "_FillValue", "None"))
    print("valid range:", getattr(quality_var, "valid_range", "None"))
    print("flag values:", getattr(quality_var, "flag_values", "None"))
    print("flag meanings:", getattr(quality_var, "flag_meanings", "None"))

    # Read a small neighbourhood
    qblock = quality_var[
        max(0, iy - 2):min(len(y), iy + 3),
        max(0, ix - 2):min(len(x), ix + 3)
        ]

    print()
    print("Quality block:")
    print(qblock)
    
    print()
    print("Masked:", np.ma.isMaskedArray(qblock))

    if np.ma.isMaskedArray(qblock):
        print("Mask:")
        print(np.ma.getmaskarray(qblock))
    
    print()
    print("=" * 80)
    print("RADIANCE VARIABLE")
    print("=" * 80)

    print("dtype:", radiance_var.dtype)
    print("shape:", radiance_var.shape)
    print("masked array:", np.ma.isMaskedArray(radiance_var[:]))

    rblock = radiance_var[
        max(0, iy - 2):min(len(y), iy + 3),
        max(0, ix - 2):min(len(x), ix + 3)
        ]

    print()
    print("Radiance block:")
    print(rblock)