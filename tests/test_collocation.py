# ============================================================
# SATELLITE-RADAR COLLOCATION VALIDATION
# ============================================================

import sys
from pathlib import Path
import pyproj
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\SAT_RADAR_AI"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from data.dataset import SatelliteRadarDataset
from data.collocation import SatelliteRadarCollocator


# ============================================================
# FILES
# ============================================================

SATELLITE = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY--DIS-NC4E_C_EUMT_20260504105547_IDPFI_OPE_20260504105141_20260504105216_N_JLS_O_0066_0010.nc"
)

RADAR = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\DUR\cfrad.20260504_084856.000_to_20260504_085345.000_Durban_SUR.nc"
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
# CREATE DATASET
# ============================================================

print()
print("=" * 70)
print("CREATING DATASET")
print("=" * 70)


dataset = SatelliteRadarDataset(
    satellite_files=[SATELLITE],
    radar_files=[RADAR],
    satellite_times=[
        __import__("datetime").datetime(
            2026, 5, 4, 10, 55, 0
        )
    ],
    radar_elevation=2.3,
    tolerance_seconds=600,
)

print("Number of samples :", len(dataset))

if len(dataset) == 0:
    raise RuntimeError("No valid radar-satellite pairs were found.")

sample = dataset[0]

print("Valid              :", sample.valid)
print("Satellite time     :", sample.satellite_time)
print("Radar time         :", sample.radar_time)

print(
    "Satellite obs start:",
    sample.satellite_observation_start,
)

print(
    "Satellite obs end  :",
    sample.satellite_observation_end,
)

print(
    "Time difference    :",
    sample.time_difference_seconds,
    "seconds",
)
if len(dataset) == 0:
    raise RuntimeError("No satellite/radar pairs were created.")


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
print(
    "Time difference    :",
    sample.time_difference_seconds,
)
print("Radar site         :", sample.radar_site)
print("Radar elevation    :", sample.radar_elevation)
print("Radar sweep        :", sample.radar_sweep_index)


# ============================================================
# SATELLITE INFORMATION
# ============================================================

satellite = sample.satellite

satellite_array = np.asarray(
    satellite.values,
    dtype=np.float32,
)


print()
print("=" * 70)
print("SATELLITE DATA")
print("=" * 70)

print("Shape :", satellite_array.shape)
print("Dtype :", satellite_array.dtype)

print(
    "Finite pixels :",
    np.count_nonzero(np.isfinite(satellite_array)),
)

if np.any(np.isfinite(satellite_array)):

    print(
        "Min :",
        np.nanmin(satellite_array),
    )

    print(
        "Max :",
        np.nanmax(satellite_array),
    )

    print(
        "Mean :",
        np.nanmean(satellite_array),
    )

    print(
        "Median :",
        np.nanmedian(satellite_array),
    )


# ============================================================
# RUN COLLOCATION
# ============================================================

print()
print("=" * 70)
print("RUNNING COLLOCATION")
print("=" * 70)


collocator = SatelliteRadarCollocator(
    satellite_channel="ir_105"
)



# ============================================================
# SATELLITE GRID DIAGNOSTIC
# ============================================================

satellite = sample.satellite

print()
print("=" * 70)
print("SATELLITE GRID COORDINATE DIAGNOSTIC")
print("=" * 70)

print()
print("Dimensions:")
print(satellite.dims)

print()
print("Coordinates:")
print(list(satellite.coords))

for coord_name in satellite.coords:

    coord = satellite.coords[coord_name]

    print()
    print("-" * 70)
    print("Coordinate:", coord_name)
    print("Dimensions :", coord.dims)
    print("Shape      :", coord.shape)
    print("Dtype      :", coord.dtype)

    try:
        values = np.asarray(coord.values)

        print("First 5    :", values.flatten()[:5])
        print("Last 5     :", values.flatten()[-5:])

        if np.issubdtype(values.dtype, np.number):
            finite = values[np.isfinite(values)]

            if finite.size > 0:
                print("Min        :", np.min(finite))
                print("Max        :", np.max(finite))

    except Exception as e:
        print("Could not inspect:", e)


print()
print("=" * 70)
print("SATELLITE X/Y")
print("=" * 70)

if "x" in satellite.coords:
    sat_x = np.asarray(satellite.coords["x"].values)

    print()
    print("X shape :", sat_x.shape)
    print("X first :", sat_x[:10])
    print("X last  :", sat_x[-10:])
    print("X min   :", np.nanmin(sat_x))
    print("X max   :", np.nanmax(sat_x))

if "y" in satellite.coords:
    sat_y = np.asarray(satellite.coords["y"].values)

    print()
    print("Y shape :", sat_y.shape)
    print("Y first :", sat_y[:10])
    print("Y last  :", sat_y[-10:])
    print("Y min   :", np.nanmin(sat_y))
    print("Y max   :", np.nanmax(sat_y))

print()
print("=" * 70)
print("MTG GRID SPACING")
print("=" * 70)

if "x" in satellite.coords:
    dx = np.diff(sat_x)
    print("X spacing:")
    print("  Median :", np.nanmedian(dx))
    print("  Min    :", np.nanmin(dx))
    print("  Max    :", np.nanmax(dx))

if "y" in satellite.coords:
    dy = np.diff(sat_y)
    print("Y spacing:")
    print("  Median :", np.nanmedian(dy))
    print("  Min    :", np.nanmin(dy))
    print("  Max    :", np.nanmax(dy))

result = collocator.collocate(sample)


# ============================================================
# EXTRACT RESULTS
# ============================================================

sat_values = np.asarray(
    result.satellite_values,
    dtype=np.float32,
)

reflectivity = np.asarray(
    result.reflectivity,
    dtype=np.float32,
)

latitude = np.asarray(
    result.latitude,
    dtype=np.float64,
)

longitude = np.asarray(
    result.longitude,
    dtype=np.float64,
)

satellite_x = np.asarray(
    result.satellite_x,
    dtype=np.float64,
)

satellite_y = np.asarray(
    result.satellite_y,
    dtype=np.float64,
)

valid_mask = np.asarray(
    result.valid_mask,
    dtype=bool,
)


# ============================================================
# COLLOCATION STATISTICS
# ============================================================

print()
print("=" * 70)
print("COLLOCATION STATISTICS")
print("=" * 70)

print("Radar shape       :", reflectivity.shape)
print("Satellite shape   :", sat_values.shape)

print()
print("Latitude:")
print("  Min :", np.nanmin(latitude))
print("  Max :", np.nanmax(latitude))

print()
print("Longitude:")
print("  Min :", np.nanmin(longitude))
print("  Max :", np.nanmax(longitude))


# ============================================================
# SATELLITE SAMPLE STATISTICS
# ============================================================

finite_sat = np.isfinite(sat_values)

print()
print("=" * 70)
print("COLLOCATED SATELLITE VALUES")
print("=" * 70)

print(
    "Finite satellite values :",
    np.count_nonzero(finite_sat),
)

print(
    "Total radar bins        :",
    sat_values.size,
)

print(
    "Valid satellite %       :",
    100.0 * np.count_nonzero(finite_sat)
    / sat_values.size,
)


if np.any(finite_sat):

    print()
    print("Satellite BT statistics")

    print(
        "Min    :",
        np.nanmin(sat_values),
    )

    print(
        "Max    :",
        np.nanmax(sat_values),
    )

    print(
        "Mean   :",
        np.nanmean(sat_values),
    )

    print(
        "Median :",
        np.nanmedian(sat_values),
    )


# ============================================================
# RADAR STATISTICS
# ============================================================

finite_radar = np.isfinite(reflectivity)

print()
print("=" * 70)
print("RADAR REFLECTIVITY")
print("=" * 70)

print(
    "Finite reflectivity :",
    np.count_nonzero(finite_radar),
)

if np.any(finite_radar):

    print(
        "Min :",
        np.nanmin(reflectivity),
    )

    print(
        "Max :",
        np.nanmax(reflectivity),
    )


# ============================================================
# FINAL VALID MASK
# ============================================================

print()
print("=" * 70)
print("VALID COLLOCATION MASK")
print("=" * 70)

print(
    "Valid mask bins :",
    np.count_nonzero(valid_mask),
)

print(
    "Total bins      :",
    valid_mask.size,
)

print(
    "Valid percentage:",
    100.0
    * np.count_nonzero(valid_mask)
    / valid_mask.size,
)


# ============================================================
# COMMON VALID DATA
# ============================================================

common_mask = (
    np.isfinite(latitude)
    & np.isfinite(longitude)
    & np.isfinite(reflectivity)
    & np.isfinite(sat_values)
)


print()
print("=" * 70)
print("COMMON SATELLITE-RADAR PIXELS")
print("=" * 70)

print(
    "Matched pixels :",
    np.count_nonzero(common_mask),
)

if np.any(common_mask):

    print(
        "Radar reflectivity range:",
        np.nanmin(reflectivity[common_mask]),
        "to",
        np.nanmax(reflectivity[common_mask]),
    )

    print(
        "Satellite BT range:",
        np.nanmin(sat_values[common_mask]),
        "to",
        np.nanmax(sat_values[common_mask]),
    )


# ============================================================
# PLOTS
# ============================================================

# ------------------------------------------------------------
# CREATE GEOGRAPHIC CELL EDGES
# ------------------------------------------------------------
# longitude and latitude are 2-D coordinates at the radar-bin
# centres. pcolormesh needs cell edges for a non-rectilinear
# geographic grid.

def make_cell_edges(values):
    """
    Calculate approximate cell-edge coordinates from 2-D
    cell-centre coordinates.

    Parameters
    ----------
    values : np.ndarray
        2-D array of cell-centre coordinates.

    Returns
    -------
    edges : np.ndarray
        2-D array with shape (ny + 1, nx + 1).
    """

    values = np.asarray(values, dtype=float)

    ny, nx = values.shape

    edges = np.empty(
        (ny + 1, nx + 1),
        dtype=float,
    )

    # Interior edges
    edges[1:-1, 1:-1] = (
        values[:-1, :-1]
        + values[:-1, 1:]
        + values[1:, :-1]
        + values[1:, 1:]
    ) / 4.0

    # Top and bottom rows
    edges[0, 1:-1] = (
        values[0, :-1]
        + values[0, 1:]
    ) / 2.0

    edges[-1, 1:-1] = (
        values[-1, :-1]
        + values[-1, 1:]
    ) / 2.0

    # Left and right columns
    edges[1:-1, 0] = (
        values[:-1, 0]
        + values[1:, 0]
    ) / 2.0

    edges[1:-1, -1] = (
        values[:-1, -1]
        + values[1:, -1]
    ) / 2.0

    # Four corners
    edges[0, 0] = (
        2 * values[0, 0]
        - edges[0, 1]
        - edges[1, 0]
        + values[1, 1]
    )

    edges[0, -1] = (
        2 * values[0, -1]
        - edges[0, -2]
        - edges[1, -1]
        + values[1, -2]
    )

    edges[-1, 0] = (
        2 * values[-1, 0]
        - edges[-1, 1]
        - edges[-2, 0]
        + values[-2, 1]
    )

    edges[-1, -1] = (
        2 * values[-1, -1]
        - edges[-1, -2]
        - edges[-2, -1]
        + values[-2, -2]
    )

    return edges


longitude_edges = make_cell_edges(longitude)
latitude_edges = make_cell_edges(latitude)


# ============================================================
# SATELLITE BRIGHTNESS TEMPERATURE
# ============================================================

fig = plt.figure(figsize=(10, 8))

plt.pcolormesh(
    longitude_edges,
    latitude_edges,
    sat_values,
    shading="flat",
)

plt.colorbar(
    label="Brightness Temperature (K)"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "MTG FCI IR 10.5 µm Sampled at Radar Bins"
)

plt.tight_layout()

plt.show()


# ============================================================
# RADAR REFLECTIVITY
# ============================================================

fig = plt.figure(figsize=(10, 8))

plt.pcolormesh(
    longitude_edges,
    latitude_edges,
    reflectivity,
    shading="flat",
)

plt.colorbar(
    label="Reflectivity"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "Durban Radar Reflectivity"
)

plt.tight_layout()

plt.show()


# ============================================================
# MATCHED PIXELS
# ============================================================

fig = plt.figure(figsize=(10, 8))

matched_sat = np.where(
    common_mask,
    sat_values,
    np.nan,
)

plt.pcolormesh(
    longitude_edges,
    latitude_edges,
    matched_sat,
    shading="flat",
)

plt.colorbar(
    label="Brightness Temperature (K)"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "Satellite-Radar Matched Pixels"
)

plt.tight_layout()

plt.show()


# ============================================================
# SCATTER: RADAR vs SATELLITE
# ============================================================

if np.count_nonzero(common_mask) > 0:

    plt.figure(figsize=(9, 7))

    plt.scatter(
        reflectivity[common_mask],
        sat_values[common_mask],
        s=5,
        alpha=0.4,
    )

    plt.xlabel(
        "Radar Reflectivity"
    )

    plt.ylabel(
        "Satellite Brightness Temperature (K)"
    )

    plt.title(
        "Radar Reflectivity vs MTG FCI Brightness Temperature"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print("Dataset pairing       : PASS")
print("Satellite loading     : PASS")
print("Radar loading         : PASS")
print("Geographic conversion : PASS")
print("MTG projection        : PASS")
print("Satellite sampling    : PASS")

if np.count_nonzero(common_mask) > 0:
    print("Satellite-radar match : PASS")
else:
    print("Satellite-radar match : CHECK REQUIRED")

print("=" * 70)

sat = sample.satellite

print()
print("=" * 70)
print("SAMPLE OBJECT CONTENTS")
print("=" * 70)

print("sample type :", type(sample))

print(
    "sample attributes :",
    [
        name
        for name in dir(sample)
        if not name.startswith("_")
    ]
)
sat_values = np.asarray(
    sat.load().values,
    dtype=np.float32,
)

sat_x = np.asarray(
    sat.coords["x"].values,
    dtype=np.float64,
)

sat_y = np.asarray(
    sat.coords["y"].values,
    dtype=np.float64,
)

finite = np.isfinite(sat_values)

print()
print("=" * 70)
print("FCI VALID PIXEL FOOTPRINT")
print("=" * 70)

print("Total pixels :", sat_values.size)

print(
    "Finite pixels:",
    np.count_nonzero(finite),
)

if np.any(finite):

    yy, xx = np.where(finite)

    print()
    print("VALID PIXEL INDEX RANGE")
    print("-----------------------")

    print(
        "Y index:",
        yy.min(),
        "to",
        yy.max(),
    )

    print(
        "X index:",
        xx.min(),
        "to",
        xx.max(),
    )

    print()
    print("VALID PIXEL PROJECTED RANGE")
    print("----------------------------")

    print(
        "X:",
        sat_x[xx].min(),
        "to",
        sat_x[xx].max(),
    )

    print(
        "Y:",
        sat_y[yy].min(),
        "to",
        sat_y[yy].max(),
    )

    print()
    print("VALID PIXEL PROJECTED CENTRE")
    print("-----------------------------")

    print(
        "X median:",
        np.median(sat_x[xx]),
    )

    print(
        "Y median:",
        np.median(sat_y[yy]),
    )

    print("=" * 70)
    
    finite = np.isfinite(satellite.values)

yy, xx = np.where(finite)

print()
print("=" * 70)
print("FCI FINITE PIXEL GEOGRAPHIC/GRID DIAGNOSTIC")
print("=" * 70)

print("Finite pixel count :", finite.sum())

print()
print("Finite Y indices:")
print("  Min :", yy.min())
print("  Max :", yy.max())

print()
print("Finite X indices:")
print("  Min :", xx.min())
print("  Max :", xx.max())

print()
print("Finite projected X:")
print("  Min :", satellite.x.values[xx].min())
print("  Max :", satellite.x.values[xx].max())

print()
print("Finite projected Y:")
print("  Min :", satellite.y.values[yy].min())
print("  Max :", satellite.y.values[yy].max())

print()
print("Finite X centre:",
      np.median(satellite.x.values[xx]))

print("Finite Y centre:",
      np.median(satellite.y.values[yy]))

print("=" * 70)

import pyproj

print()
print("=" * 70)
print("FCI VALID PIXEL GEOGRAPHIC DIAGNOSTIC")
print("=" * 70)

# ------------------------------------------------------------------
# Satellite CRS
# ------------------------------------------------------------------

sat_crs = satellite.coords["crs"].item()

print("Satellite CRS object:")
print(sat_crs)

print()
print("CRS type:")
print(type(sat_crs))

print()
print("CRS WKT:")
print(sat_crs.to_wkt())


# ------------------------------------------------------------------
# Geographic CRS
# ------------------------------------------------------------------

geographic_crs = pyproj.CRS.from_epsg(4326)

inverse_transformer = pyproj.Transformer.from_crs(
    sat_crs,
    geographic_crs,
    always_xy=True
)

# ------------------------------------------------------------------
# Use the ACTUAL finite-pixel centre calculated above
# ------------------------------------------------------------------

finite_x_centre = np.median(
    satellite.x.values[xx]
)

finite_y_centre = np.median(
    satellite.y.values[yy]
)

lon_valid, lat_valid = inverse_transformer.transform(
    finite_x_centre,
    finite_y_centre
)

print()
print("FCI finite-pixel centre:")
print("Projected X:", finite_x_centre)
print("Projected Y:", finite_y_centre)
print("Longitude  :", lon_valid)
print("Latitude   :", lat_valid)

print()
print("Expected consistency check:")
print("Finite X range:",
      satellite.x.values[xx].min(),
      "to",
      satellite.x.values[xx].max())

print("Finite Y range:",
      satellite.y.values[yy].min(),
      "to",
      satellite.y.values[yy].max())

print()
print("Centre inside finite X range:",
      satellite.x.values[xx].min()
      <= finite_x_centre
      <= satellite.x.values[xx].max())

print("Centre inside finite Y range:",
      satellite.y.values[yy].min()
      <= finite_y_centre
      <= satellite.y.values[yy].max())

print("=" * 70)