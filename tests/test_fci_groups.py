import xarray as xr
import numpy as np

nc_file = r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY--DIS-NC4E_C_EUMT_20260504105500_IDPFI_OPE_20260504105024_20260504105100_N_JLS_O_0066_0004.nc"

channel_name = "ir_105"

print("=" * 60)
print("FCI CHANNEL DIAGNOSTIC")
print("=" * 60)

# ------------------------------------------------------------
# CHANNEL METADATA
# ------------------------------------------------------------

channel = xr.open_dataset(
    nc_file,
    group=f"data/{channel_name}"
)

print("\nCHANNEL:")
print(channel_name)

print("\nCHANNEL METADATA:")

for name in channel.data_vars:
    value = channel[name].values

    # Convert zero-dimensional arrays to normal Python values
    if np.ndim(value) == 0:
        value = value.item()

    print(f"{name}: {value}")


# ------------------------------------------------------------
# MEASURED DATA
# ------------------------------------------------------------

measured = xr.open_dataset(
    nc_file,
    group=f"data/{channel_name}/measured"
)

print("\n" + "=" * 60)
print("MEASURED DATA")
print("=" * 60)

print(measured)

print("\nVARIABLES:")

for name in measured.data_vars:
    print(
        f"{name}: "
        f"shape={measured[name].shape}, "
        f"dtype={measured[name].dtype}"
    )


# ------------------------------------------------------------
# RADIANCE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RADIANCE STATISTICS")
print("=" * 60)

rad = measured["effective_radiance"]

print("shape:", rad.shape)
print("dtype:", rad.dtype)

print("minimum:", float(rad.min(skipna=True)))
print("maximum:", float(rad.max(skipna=True)))
print("mean:", float(rad.mean(skipna=True)))
print("std:", float(rad.std(skipna=True)))


# ------------------------------------------------------------
# COORDINATES
# ------------------------------------------------------------

print("\nCOORDINATES:")

print("x:")
print(measured["x"])

print("\ny:")
print(measured["y"])


# ------------------------------------------------------------
# PIXEL QUALITY
# ------------------------------------------------------------

pq = measured["pixel_quality"]

print("\nPIXEL QUALITY:")
print("shape:", pq.shape)
print("dtype:", pq.dtype)

print("minimum:", float(pq.min(skipna=True)))
print("maximum:", float(pq.max(skipna=True)))


# ------------------------------------------------------------
# INDEX MAP
# ------------------------------------------------------------

index_map = measured["index_map"]

print("\nINDEX MAP:")
print("shape:", index_map.shape)
print("dtype:", index_map.dtype)

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)