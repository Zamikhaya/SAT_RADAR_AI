# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 09:07:14 2026

@author: Zamikhaya.Magogotya
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc


# ============================================================
# CONFIGURATION
# ============================================================

FILE = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT"
    r"\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY--DIS-NC4E_C_EUMT_20260504105500_IDPFI_OPE_20260504105024_20260504105100_N_JLS_O_0066_0004.nc"
)

CHANNELS = [
    "ir_105",
    "ir_133",
    "wv_73",
    "vis_06",
]


# ============================================================
# OPEN DATASET
# ============================================================

ds = nc.Dataset(FILE, "r")

print("\n==============================================")
print("MTG FCI L1C PROCESSING")
print("==============================================")
print("File:", FILE)


# ============================================================
# LIST CHANNELS
# ============================================================

available_channels = []

for ch in ds["data"].groups:
    available_channels.append(ch)

print("\nAvailable FCI channels:")
for ch in available_channels:
    print("  ", ch)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_effective_radiance(ds, channel):
    """
    Read FCI effective radiance.

    The netCDF library automatically applies scale_factor
    and add_offset when using the default auto scaling.
    """

    var = ds["data"][channel]["measured"]["effective_radiance"]

    radiance = var[:].astype(np.float64)

    return radiance


def get_scalar(group, name):

    value = group[name][...]

    return float(np.asarray(value).squeeze())


# ============================================================
# BRIGHTNESS TEMPERATURE
# ============================================================

def effective_radiance_to_bt(ds, channel):
    """
    Convert FCI effective radiance to effective
    brightness temperature.

    Formula according to EUMETSAT FCI L1C documentation:

        T = c2 * nu /
            (a * ln(1 + c1 * nu^3 / L))
            - b/a
    """

    measured = ds["data"][channel]["measured"]

    L = read_effective_radiance(ds, channel)

    a = get_scalar(
        measured,
        "radiance_to_bt_conversion_coefficient_a"
    )

    b = get_scalar(
        measured,
        "radiance_to_bt_conversion_coefficient_b"
    )

    nu = get_scalar(
        measured,
        "radiance_to_bt_conversion_coefficient_wavenumber"
    )

    c1 = get_scalar(
        measured,
        "radiance_to_bt_conversion_constant_c1"
    )

    c2 = get_scalar(
        measured,
        "radiance_to_bt_conversion_constant_c2"
    )

    print(f"\n{channel} calibration:")
    print("  a  =", a)
    print("  b  =", b)
    print("  nu =", nu)
    print("  c1 =", c1)
    print("  c2 =", c2)

    # Avoid invalid logarithms
    valid = np.isfinite(L) & (L > 0)

    bt = np.full(
        L.shape,
        np.nan,
        dtype=np.float32
    )

    bt[valid] = (
        (c2 * nu)
        /
        (
            a *
            np.log(
                1.0 +
                (c1 * nu**3 / L[valid])
            )
        )
        -
        b / a
    )

    return bt


# ============================================================
# PIXEL QUALITY
# ============================================================

def read_pixel_quality(ds, channel):

    q = ds["data"][channel]["measured"]["pixel_quality"]

    return q[:]


# ============================================================
# CHANNEL QC SUMMARY
# ============================================================

def channel_qc_summary(ds, channel):

    qc = ds["data"][channel]["quality_channel"]

    print(f"\nQC SUMMARY — {channel}")
    print("----------------------------------------------")

    for name in qc.variables:

        value = qc[name][...]

        try:
            value = int(np.asarray(value).squeeze())
        except Exception:
            pass

        print(f"{name}: {value}")


# ============================================================
# CHANNEL INFORMATION
# ============================================================

def channel_information(ds, channel):

    group = ds["data"][channel]

    print(f"\nCHANNEL: {channel}")
    print("----------------------------------------------")

    for name in [
        "central_wavelength_specified",
        "spectral_width_specified",
        "central_wavelength_actual",
        "spectral_width_actual",
    ]:

        if name in group.variables:

            value = group[name][...]

            print(
                f"{name}: "
                f"{np.asarray(value).squeeze()}"
            )


# ============================================================
# PROCESS CHANNEL
# ============================================================

results = {}


for channel in CHANNELS:

    if channel not in available_channels:

        print(
            f"\nWARNING: {channel} "
            "not found in file."
        )

        continue

    channel_information(ds, channel)

    channel_qc_summary(ds, channel)

    radiance = read_effective_radiance(
        ds,
        channel
    )

    print(f"\nRadiance statistics — {channel}")

    print("  shape :", radiance.shape)
    print("  min   :", np.nanmin(radiance))
    print("  max   :", np.nanmax(radiance))
    print("  mean  :", np.nanmean(radiance))
    print("  std   :", np.nanstd(radiance))

    quality = read_pixel_quality(
        ds,
        channel
    )

    print(
        "  pixel quality shape:",
        quality.shape
    )

    # --------------------------------------------------------
    # IR / WV
    # --------------------------------------------------------

    if channel.startswith("ir_") or channel.startswith("wv_"):

        bt = effective_radiance_to_bt(
            ds,
            channel
        )

        print(
            f"\nBrightness temperature — {channel}"
        )

        print(
            "  min  :",
            np.nanmin(bt),
            "K"
        )

        print(
            "  max  :",
            np.nanmax(bt),
            "K"
        )

        print(
            "  mean :",
            np.nanmean(bt),
            "K"
        )

        results[channel] = {
            "radiance": radiance,
            "brightness_temperature": bt,
            "quality": quality,
        }

    else:

        results[channel] = {
            "radiance": radiance,
            "quality": quality,
        }


# ============================================================
# SOLAR GEOMETRY
# ============================================================

celestial = ds["state"]["celestial"]

solar_elevation = celestial[
    "solar_elevation"
][:]

earth_sun_distance = celestial[
    "earth_sun_distance"
][:]

print("\n==============================================")
print("SOLAR GEOMETRY")
print("==============================================")

print(
    "Solar elevation:",
    solar_elevation
)

print(
    "Earth-Sun distance:",
    earth_sun_distance
)


# ============================================================
# PROCESSOR STRAYLIGHT STATUS
# ============================================================

processor = ds["state"]["processor"]

print("\n==============================================")
print("STRAYLIGHT PROCESSING")
print("==============================================")

if "earth_straylight_correction_enabled" in processor:

    print(
        "Earth straylight correction:",
        processor[
            "earth_straylight_correction_enabled"
        ][...]
    )

if "sun_straylight_correction_enabled" in processor:

    print(
        "Sun straylight correction:",
        processor[
            "sun_straylight_correction_enabled"
        ][...]
    )


# ============================================================
# PLOTS
# ============================================================

for channel, data in results.items():

    if "brightness_temperature" in data:

        image = data["brightness_temperature"]

        title = (
            f"{channel.upper()} "
            "Brightness Temperature"
        )

        cbar_label = "Brightness Temperature (K)"

    else:

        image = data["radiance"]

        title = (
            f"{channel.upper()} "
            "Effective Radiance"
        )

        cbar_label = "Effective Radiance"

    plt.figure(figsize=(15, 5))

    plt.imshow(
        image,
        origin="upper",
        aspect="auto"
    )

    plt.title(title)

    plt.xlabel("FCI X")
    plt.ylabel("FCI Y")

    plt.colorbar(
        label=cbar_label
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# CLOSE FILE
# ============================================================

ds.close()

print("\n==============================================")
print("PROCESSING COMPLETE")
print("==============================================")