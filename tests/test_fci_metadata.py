# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:40:16 2026

@author: Zamikhaya.Magogotya
"""

import os
import numpy as np
import xarray as xr


SAT_FILE = (
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT"
    r"\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD"
    r"--CHK-BODY--DIS-NC4E_C_EUMT_20260504105500_IDPFI_OPE_"
    r"20260504105024_20260504105100_N_JLS_O_0066_0004.nc"
)


print("=" * 70)
print("FCI METADATA DIAGNOSTIC")
print("=" * 70)

print("File exists:", os.path.exists(SAT_FILE))

ds = xr.open_dataset(SAT_FILE)

print("\nDATASET")
print("-" * 70)

print(ds)


print("\nVARIABLES")
print("-" * 70)

for name, var in ds.variables.items():

    print("\nVARIABLE:", name)
    print("  dimensions :", var.dims)
    print("  shape      :", var.shape)
    print("  dtype      :", var.dtype)

    if hasattr(var, "attrs"):
        print("  attributes:")

        for key, value in var.attrs.items():
            print(f"    {key}: {value}")


print("\nDATA VARIABLES")
print("-" * 70)

for name, var in ds.data_vars.items():

    print(
        name,
        "| dims =", var.dims,
        "| shape =", var.shape,
        "| dtype =", var.dtype
    )


print("\nCOORDINATES")
print("-" * 70)

for name, coord in ds.coords.items():

    print(
        name,
        "| dims =", coord.dims,
        "| shape =", coord.shape,
        "| dtype =", coord.dtype
    )

    for key, value in coord.attrs.items():
        print(f"    {key}: {value}")


print("\nIR_105 DIAGNOSTIC")
print("-" * 70)

if "ir_105" in ds:

    sat = ds["ir_105"]

    print("Dimensions:", sat.dims)
    print("Shape:", sat.shape)
    print("Dtype:", sat.dtype)

    values = sat.values

    finite = np.isfinite(values)

    print("\nFinite:", np.count_nonzero(finite))
    print("Total :", values.size)

    if np.any(finite):

        yy, xx = np.where(finite)

        print("\nFINITE INDEX EXTENT")
        print("Y:", yy.min(), "to", yy.max())
        print("X:", xx.min(), "to", xx.max())

        print("\nFINITE VALUE STATISTICS")

        vals = values[finite]

        print("Min   :", np.min(vals))
        print("Max   :", np.max(vals))
        print("Mean  :", np.mean(vals))
        print("Median:", np.median(vals))

        print("\nFINITE Y COORDINATES")

        ycoord = ds["y"].values

        print(
            "Y coordinate:",
            ycoord[yy.min()],
            "to",
            ycoord[yy.max()]
        )

        print("\nFINITE X COORDINATES")

        xcoord = ds["x"].values

        print(
            "X coordinate:",
            xcoord[xx.min()],
            "to",
            xcoord[xx.max()]
        )

else:

    print("ERROR: ir_105 does not exist")


print("\nGLOBAL ATTRIBUTES")
print("-" * 70)

for key, value in ds.attrs.items():
    print(f"{key}: {value}")


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)