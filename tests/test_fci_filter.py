# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:31:49 2026

@author: Zamikhaya.Magogotya
"""

import h5py

nc_file = r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY--DIS-NC4E_C_EUMT_20260504105500_IDPFI_OPE_20260504105024_20260504105100_N_JLS_O_0066_0004.nc"

print("=" * 60)
print("FCI HDF5 FILTER DIAGNOSTIC")
print("=" * 60)

with h5py.File(nc_file, "r") as f:

    path = "/data/ir_105/measured/effective_radiance"

    print("\nDATASET:")
    print(path)

    dset = f[path]

    print("\nShape:")
    print(dset.shape)

    print("\nDtype:")
    print(dset.dtype)

    print("\nChunks:")
    print(dset.chunks)

    print("\nCompression:")
    print(dset.compression)

    print("\nCompression options:")
    print(dset.compression_opts)

    print("\nCompression filters:")

    try:
        print(dset.id.get_create_plist().get_nfilters())

        for i in range(dset.id.get_create_plist().get_nfilters()):
            print(
                "Filter",
                i,
                ":",
                dset.id.get_create_plist().get_filter(i)
            )

    except Exception as e:
        print("Could not inspect filters:", repr(e))

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)



import h5py
import netCDF4
import xarray as xr
import numpy as np

print("Python:", __import__("sys").version)
print("NumPy:", np.__version__)
print("h5py:", h5py.__version__)
print("HDF5 used by h5py:", h5py.version.hdf5_version)
print("netCDF4:", netCDF4.__version__)
print("netCDF4 HDF5:", netCDF4.__hdf5libversion__)
print("xarray:", xr.__version__)

import h5py

print("\nHDF5 filter availability:")
print("32018:", h5py.h5z.filter_avail(32018))


import hdf5plugin
import h5py

print("hdf5plugin:", hdf5plugin.version)

print("HDF5:", h5py.version.hdf5_version)

print("Filter 32018 available:",
      h5py.h5z.filter_avail(32018))

import hdf5plugin
import xarray as xr
import numpy as np

print("=" * 60)
print("FCI EFFECTIVE RADIANCE READ TEST")
print("=" * 60)

ds = xr.open_dataset(
    nc_file,
    group="data/ir_105/measured",
    engine="netcdf4"
)

print("\nDATASET:")
print(ds)

rad = ds["effective_radiance"]

print("\nRADIANCE:")
print("shape:", rad.shape)
print("dtype:", rad.dtype)

# Force HDF5 to actually decompress/read the data
data = rad.values

print("\nDECOMPRESSED DATA:")
print("shape:", data.shape)
print("dtype:", data.dtype)

finite = np.isfinite(data)

print("finite pixels:", finite.sum())
print("total pixels:", data.size)
print("finite percentage:", 100.0 * finite.sum() / data.size)

if finite.any():
    print("minimum:", np.nanmin(data))
    print("maximum:", np.nanmax(data))
    print("mean:", np.nanmean(data))
    print("median:", np.nanmedian(data))

print("\nCALIBRATION VARIABLES:")

for name in [
    "radiance_unit_conversion_coefficient",
    "radiance_to_bt_conversion_coefficient_a",
    "radiance_to_bt_conversion_coefficient_b",
    "radiance_to_bt_conversion_coefficient_wavenumber",
    "radiance_to_bt_conversion_constant_c1",
    "radiance_to_bt_conversion_constant_c2",
    "channel_effective_solar_irradiance",
]:
    if name in ds:
        print(f"{name}: {ds[name].values}")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)

pq = ds["pixel_quality"]

print("=" * 60)
print("PIXEL QUALITY DIAGNOSTIC")
print("=" * 60)

print("shape:", pq.shape)
print("dtype:", pq.dtype)

pq_data = pq.values

print("finite pixels:", np.isfinite(pq_data).sum())
print("total pixels:", pq_data.size)

if np.isfinite(pq_data).any():
    print("minimum:", np.nanmin(pq_data))
    print("maximum:", np.nanmax(pq_data))
    print("mean:", np.nanmean(pq_data))

print("\nUNIQUE VALUES:")
values, counts = np.unique(pq_data[np.isfinite(pq_data)], return_counts=True)

for value, count in zip(values, counts):
    print(f"{value}: {count}")
    
    
print("=" * 60)
print("RADIANCE ATTRIBUTES")
print("=" * 60)

print(ds["effective_radiance"].attrs)

print("\nRADIANCE UNIT CONVERSION:")
print(ds["radiance_unit_conversion_coefficient"].attrs)

print("\nBT COEFFICIENT A:")
print(ds["radiance_to_bt_conversion_coefficient_a"].attrs)

print("\nBT COEFFICIENT B:")
print(ds["radiance_to_bt_conversion_coefficient_b"].attrs)

print("\nWAVENUMBER:")
print(ds["radiance_to_bt_conversion_coefficient_wavenumber"].attrs)

print("\nC1:")
print(ds["radiance_to_bt_conversion_constant_c1"].attrs)

print("\nC2:")
print(ds["radiance_to_bt_conversion_constant_c2"].attrs)

import numpy as np

# FCI calibration coefficients
A = float(ds["radiance_to_bt_conversion_coefficient_a"].values)
B = float(ds["radiance_to_bt_conversion_coefficient_b"].values)
nu = float(ds["radiance_to_bt_conversion_coefficient_wavenumber"].values)
C1 = float(ds["radiance_to_bt_conversion_constant_c1"].values)
C2 = float(ds["radiance_to_bt_conversion_constant_c2"].values)
radiance_conversion = float(
    ds["radiance_unit_conversion_coefficient"].values
)

# Effective radiance
L = ds["effective_radiance"].values

valid = np.isfinite(L) & (L > 0)

print("=" * 60)
print("FCI ir_105 BRIGHTNESS TEMPERATURE SANITY CHECK")
print("=" * 60)

print("A:", A)
print("B:", B)
print("Wavenumber:", nu)
print("C1:", C1)
print("C2:", C2)
print("Radiance conversion:", radiance_conversion)

print("\nEffective radiance:")
print("min:", np.nanmin(L))
print("max:", np.nanmax(L))
print("median:", np.nanmedian(L))

# Convert radiance units according to product coefficient
L_converted = L * radiance_conversion

# Planck inversion
T_planck = (C2 * nu) / np.log(
    1.0 + (C1 * nu**3) / L_converted
)

# Apply FCI A/B correction
BT = A * T_planck + B

BT_valid = BT[valid]

print("\nPLANCK TEMPERATURE:")
print("minimum:", np.nanmin(T_planck))
print("maximum:", np.nanmax(T_planck))
print("mean:", np.nanmean(T_planck))
print("median:", np.nanmedian(T_planck))

print("\nFCI BRIGHTNESS TEMPERATURE:")
print("minimum:", np.nanmin(BT_valid))
print("maximum:", np.nanmax(BT_valid))
print("mean:", np.nanmean(BT_valid))
print("median:", np.nanmedian(BT_valid))

print("\nVALID PIXELS:", valid.sum())
print("=" * 60)