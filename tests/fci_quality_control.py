# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:49:59 2026

@author: Zamikhaya.Magogotya
"""

# -*- coding: utf-8 -*-
"""
FCI QUALITY CONTROL AND RADIANCE ANALYSIS
==========================================

Purpose
-------
Quality-control analysis of EUMETSAT MTG FCI Level 1C Rectified
Radiance NetCDF files.

The script performs TWO separate QC analyses:

1. OFFICIAL PIXEL QUALITY FLAGS
   --------------------------------
   Uses the FCI pixel_quality variable and decodes the documented
   bit masks.

2. RADIANCE / INDEX-MAP ANALYSIS
   --------------------------------
   Examines effective_radiance within each index_map region.
   This is intentionally kept separate from the official quality flags.

Outputs
-------
<INPUT>_FCI_QC/
    csv/
        FCI_all_channels_quality_summary.csv
        FCI_all_channels_radiance_summary.csv
        <channel>_DN_vs_quality.csv
        <channel>_index_map_analysis.csv
        <channel>_radiance_by_index.csv
        <channel>_radiance_warning_distribution.csv

    figures/
        <channel>_quality_map.png
        <channel>_index_map.png
        <channel>_warning_map.png
        <channel>_radiance_histogram.png
        <channel>_warning_radiance_histogram.png

    cleaned/
        <channel>_FCI_QC.nc

Important
---------
The script DOES NOT automatically classify radiance values such as
204, 205 or 206 as bad data.

Radiance anomalies are reported separately and can be investigated
against official FCI quality flags.

Author:
    SAWS / FCI QC analysis
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt


# ============================================================
# USER CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# INPUT FILE
# ------------------------------------------------------------

INPUT_FILE = (
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT"
    r"\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY--DIS-NC4E"
    r"_C_EUMT_20260504105500_IDPFI_OPE_20260504105024_20260504105100_N_JLS_O_0066_0004.nc"
)


# ------------------------------------------------------------
# CHANNELS
# ------------------------------------------------------------

# Set to None to process all discovered FCI channels.

CHANNELS_TO_PROCESS = None

# Example:
#
# CHANNELS_TO_PROCESS = [
#     "vis_09",
#     "vis_08",
#     "ir_105",
# ]


# ------------------------------------------------------------
# ANALYSIS SETTINGS
# ------------------------------------------------------------

# 65535 is normally a special/no-index value in the index map.
# We report it separately and do not treat it as a normal region.

EXCLUDE_INDEX_65535 = True


# Radiance threshold analysis
#
# IMPORTANT:
# These are NOT official EUMETSAT quality flags.
#
# The default values are deliberately conservative and are intended
# to identify values around the suspicious region observed in your
# previous analysis.
#
# Set to None if you do not want threshold analysis.

RADIANCE_WARNING_MIN = 200
RADIANCE_WARNING_MAX = 209


# If True, the script determines the threshold from official
# radiometric-warning pixels instead of using the fixed values above.
#
# Recommended for investigation:
#
# AUTO_THRESHOLD_FROM_OFFICIAL_WARNINGS = True
#
# This means:
#     min = minimum DN among official radiometric warnings
#     max = maximum DN among official radiometric warnings
#
AUTO_THRESHOLD_FROM_OFFICIAL_WARNINGS = True


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

SAVE_FIGURES = True

SHOW_FIGURES = False

DPI = 150


# ============================================================
# OFFICIAL FCI QUALITY FLAGS
# ============================================================

QUALITY_FLAGS = {
    1: "missing_warning",
    2: "radiometric_warning",
    4: "noise_warning",
    8: "geolocation_warning",
    16: "saturation_warning",
    32: "straylight_correction_warning",
    64: "extended_dynamic_range_warning",
    128: "encoding_saturation_warning",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    """Print a formatted section header."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_percentage(numerator, denominator):
    """Return percentage without division-by-zero errors."""

    if denominator == 0:
        return 0.0

    return 100.0 * numerator / denominator


def ensure_directory(path):
    """Create directory if it does not exist."""

    os.makedirs(path, exist_ok=True)

def discover_fci_channels(ds):

    channels = []

    # ---------------------------------------------------------
    # 1. FCI channel presence metadata
    # ---------------------------------------------------------
    if "l1c_channels_present" in ds.variables:

        values = ds["l1c_channels_present"].values

        for value in values:

            if value is None:
                continue

            channel = str(value).strip().lower()

            if channel and channel not in channels:
                channels.append(channel)

    # ---------------------------------------------------------
    # 2. Also inspect dataset variables for actual channels
    # ---------------------------------------------------------
    for variable in ds.variables:

        name = str(variable).lower()

        if (
            name.startswith("vis_")
            or name.startswith("nir_")
            or name.startswith("wv_")
            or name.startswith("ir_")
        ):

            if name not in channels:
                channels.append(name)

    return channels




def open_channel(channel):
    """
    Open an FCI channel.

    Attempts the most common FCI NetCDF group locations.
    """

    possible_paths = [
        f"data/measurement/{channel}",
        f"data/measurement_data/{channel}",
        f"measurement/{channel}",
        channel,
    ]

    last_error = None

    for group_path in possible_paths:

        try:

            ds = xr.open_dataset(
                INPUT_FILE,
                group=group_path,
                engine="netcdf4",
            )

            return ds, group_path

        except Exception as exc:

            last_error = exc

    raise RuntimeError(
        f"Unable to open channel {channel}. "
        f"Last error: {last_error}"
    )


def find_variable(ds, candidates):
    """
    Find the first available variable from a list of candidates.
    """

    for name in candidates:

        if name in ds.variables:
            return name

    return None


def get_data_array(ds, variable_name):
    """Return NumPy array from xarray DataArray."""

    return ds[variable_name].values


def decode_quality_flags(quality):
    """
    Decode FCI pixel_quality bit flags.

    Returns
    -------
    dict
        Dictionary containing boolean masks for each quality flag.
    """

    quality = np.asarray(quality)

    masks = {}

    for bit, name in QUALITY_FLAGS.items():

        masks[name] = (quality & bit) != 0

    return masks


def get_warning_mask(quality):
    """
    Return combined official FCI warning mask.
    """

    warning_mask = np.zeros(
        quality.shape,
        dtype=bool
    )

    for bit in QUALITY_FLAGS:

        warning_mask |= (quality & bit) != 0

    return warning_mask


def get_radiometric_warning_mask(quality):
    """
    Return official radiometric-warning mask.
    """

    return (quality & 2) != 0


def physical_radiance(
    raw_radiance,
    scale_factor,
    add_offset
):
    """
    Convert raw effective_radiance to physical units.

    physical = raw * scale_factor + add_offset
    """

    return (
        raw_radiance.astype(np.float64)
        * float(scale_factor)
        + float(add_offset)
    )


def get_scale_and_offset(ds):
    """
    Retrieve scale_factor and add_offset.

    Supports both xarray attributes and common FCI variable names.
    """

    da = ds["effective_radiance"]

    scale = da.attrs.get(
        "scale_factor",
        None
    )

    offset = da.attrs.get(
        "add_offset",
        None
    )

    if scale is None:

        scale = da.encoding.get(
            "scale_factor",
            None
        )

    if offset is None:

        offset = da.encoding.get(
            "add_offset",
            None
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if scale is None:
        scale = 1.0

    if offset is None:
        offset = 0.0

    return float(scale), float(offset)


def nan_statistics(values):
    """
    Calculate robust statistics.
    """

    values = np.asarray(values)

    values = values[np.isfinite(values)]

    if values.size == 0:

        return {
            "count": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
        }

    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def save_csv(df, path):

    ensure_directory(
        os.path.dirname(path)
    )

    df.to_csv(
        path,
        index=False
    )


# ============================================================
# QUALITY FLAG ANALYSIS
# ============================================================

def analyse_quality_flags(
    channel,
    quality,
    radiance,
    output_csv
):

    print_header(
        f"{channel}: QUALITY FLAG ANALYSIS"
    )

    masks = decode_quality_flags(
        quality
    )

    rows = []

    for bit, name in QUALITY_FLAGS.items():

        mask = masks[name]

        count = int(
            np.count_nonzero(mask)
        )

        percentage = safe_percentage(
            count,
            quality.size
        )

        if count > 0:

            yy, xx = np.where(mask)

            row_min = int(
                yy.min()
            )

            row_max = int(
                yy.max()
            )

            col_min = int(
                xx.min()
            )

            col_max = int(
                xx.max()
            )

        else:

            row_min = np.nan
            row_max = np.nan
            col_min = np.nan
            col_max = np.nan

        print()
        print(name)
        print(
            f"  mask       : {bit}"
        )
        print(
            f"  pixels     : {count:,}"
        )
        print(
            f"  percentage : {percentage:.6f}%"
        )
        print(
            f"  row range  : {row_min} - {row_max}"
        )
        print(
            f"  col range  : {col_min} - {col_max}"
        )

        rows.append(
            {
                "channel": channel,
                "mask": bit,
                "quality_flag": name,
                "pixels": count,
                "percentage": percentage,
                "row_min": row_min,
                "row_max": row_max,
                "column_min": col_min,
                "column_max": col_max,
            }
        )

    official_warning = get_warning_mask(
        quality
    )

    warning_count = int(
        np.count_nonzero(
            official_warning
        )
    )

    clean_count = (
        quality.size
        - warning_count
    )

    print_header(
        f"{channel}: CLEAN / WARNING SUMMARY"
    )

    print(
        f"Total pixels : {quality.size:,}"
    )

    print(
        f"Clean pixels : {clean_count:,} "
        f"({safe_percentage(clean_count, quality.size):.3f}%)"
    )

    print(
        f"Any warning  : {warning_count:,} "
        f"({safe_percentage(warning_count, quality.size):.3f}%)"
    )

    df = pd.DataFrame(rows)

    save_csv(
        df,
        output_csv
    )

    return (
        masks,
        official_warning,
        df
    )


# ============================================================
# RADIANCE INFORMATION
# ============================================================

def analyse_radiance_information(
    channel,
    ds
):

    print_header(
        f"{channel}: RADIANCE INFORMATION"
    )

    radiance_da = ds[
        "effective_radiance"
    ]

    scale_factor, add_offset = (
        get_scale_and_offset(ds)
    )

    print(
        "Radiance variable:"
    )

    print(
        "  effective_radiance"
    )

    print()

    print(
        "Datatype:"
    )

    print(
        f"  {radiance_da.dtype}"
    )

    print()

    print(
        "Scale factor:"
    )

    print(
        f"  {scale_factor}"
    )

    print()

    print(
        "Add offset:"
    )

    print(
        f"  {add_offset}"
    )

    print()

    units = radiance_da.attrs.get(
        "units",
        "unknown"
    )

    print(
        "Physical units:"
    )

    print(
        f"  {units}"
    )

    return (
        scale_factor,
        add_offset,
        units
    )


# ============================================================
# RAW DN VS QUALITY
# ============================================================

def analyse_dn_vs_quality(
    channel,
    radiance,
    official_warning,
    radiometric_warning,
    output_csv
):

    print_header(
        f"{channel}: RAW DN VS QUALITY FLAGS"
    )

    raw = np.asarray(
        radiance
    ).ravel()

    warning = np.asarray(
        official_warning
    ).ravel()

    radiometric = np.asarray(
        radiometric_warning
    ).ravel()

    valid = np.isfinite(
        raw
    )

    raw = raw[valid]
    warning = warning[valid]
    radiometric = radiometric[valid]

    unique_values = np.unique(
        raw
    )

    rows = []

    for value in unique_values:

        mask = (
            raw == value
        )

        total = int(
            np.count_nonzero(mask)
        )

        warning_count = int(
            np.count_nonzero(
                warning[mask]
            )
        )

        radiometric_count = int(
            np.count_nonzero(
                radiometric[mask]
            )
        )

        rows.append(
            {
                "channel": channel,
                "raw_dn": int(value),
                "total_pixels": total,
                "official_warning_pixels":
                    warning_count,
                "official_warning_percent":
                    safe_percentage(
                        warning_count,
                        total
                    ),
                "radiometric_warning_pixels":
                    radiometric_count,
                "radiometric_warning_percent":
                    safe_percentage(
                        radiometric_count,
                        total
                    ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    save_csv(
        df,
        output_csv
    )

    print()
    print(
        "Saved:"
    )

    print(
        output_csv
    )

    return df


# ============================================================
# INDEX MAP ANALYSIS
# ============================================================

def analyse_index_map(
    channel,
    index_map,
    official_warning,
    radiance,
    output_csv,
    radiance_by_index_csv
):

    print_header(
        f"{channel}: INDEX MAP ANALYSIS"
    )

    index_flat = (
        np.asarray(index_map)
        .ravel()
    )

    warning_flat = (
        np.asarray(official_warning)
        .ravel()
    )

    radiance_flat = (
        np.asarray(radiance)
        .ravel()
    )

    valid = (
        np.isfinite(radiance_flat)
    )

    index_flat = index_flat[valid]
    warning_flat = warning_flat[valid]
    radiance_flat = radiance_flat[valid]

    unique_indices = np.unique(
        index_flat
    )

    rows = []
    radiance_rows = []

    for index_value in unique_indices:

        index_mask = (
            index_flat == index_value
        )

        total = int(
            np.count_nonzero(
                index_mask
            )
        )

        warning_count = int(
            np.count_nonzero(
                warning_flat[index_mask]
            )
        )

        warning_pct = safe_percentage(
            warning_count,
            total
        )

        yy, xx = np.where(
            np.asarray(index_map)
            == index_value
        )

        if yy.size > 0:

            row_min = int(
                yy.min()
            )

            row_max = int(
                yy.max()
            )

            col_min = int(
                xx.min()
            )

            col_max = int(
                xx.max()
            )

        else:

            row_min = np.nan
            row_max = np.nan
            col_min = np.nan
            col_max = np.nan

        print()
        print(
            f"index = {int(index_value)}"
        )

        print(
            f"  total pixels : {total:,}"
        )

        print(
            f"  warnings     : {warning_count:,}"
        )

        print(
            f"  warning %    : {warning_pct:.6f}%"
        )

        print(
            f"  rows         : "
            f"{row_min} - {row_max}"
        )

        print(
            f"  columns      : "
            f"{col_min} - {col_max}"
        )

        rows.append(
            {
                "channel": channel,
                "index": int(index_value),
                "total_pixels": total,
                "warnings": warning_count,
                "warning_percentage": warning_pct,
                "row_min": row_min,
                "row_max": row_max,
                "column_min": col_min,
                "column_max": col_max,
            }
        )

        # ----------------------------------------------------
        # Radiance statistics
        # ----------------------------------------------------

        index_radiance = (
            radiance_flat[index_mask]
        )

        stats = nan_statistics(
            index_radiance
        )

        warning_radiance = (
            radiance_flat[
                index_mask
                & warning_flat
            ]
        )

        warning_stats = nan_statistics(
            warning_radiance
        )

        radiance_rows.append(
            {
                "channel": channel,
                "index": int(index_value),

                "total_pixels": total,

                "radiance_count":
                    stats["count"],

                "radiance_min":
                    stats["min"],

                "radiance_max":
                    stats["max"],

                "radiance_mean":
                    stats["mean"],

                "radiance_median":
                    stats["median"],

                "radiance_std":
                    stats["std"],

                "warning_radiance_count":
                    warning_stats["count"],

                "warning_radiance_min":
                    warning_stats["min"],

                "warning_radiance_max":
                    warning_stats["max"],

                "warning_radiance_mean":
                    warning_stats["mean"],

                "warning_radiance_median":
                    warning_stats["median"],

                "warning_radiance_std":
                    warning_stats["std"],
            }
        )

    df = pd.DataFrame(
        rows
    )

    df_radiance = pd.DataFrame(
        radiance_rows
    )

    save_csv(
        df,
        output_csv
    )

    save_csv(
        df_radiance,
        radiance_by_index_csv
    )

    return (
        df,
        df_radiance
    )


# ============================================================
# RADIANCE THRESHOLD ANALYSIS
# ============================================================

def determine_radiance_threshold(
    radiance,
    radiometric_warning
):

    raw = np.asarray(
        radiance
    )

    warning_values = raw[
        radiometric_warning
    ]

    warning_values = warning_values[
        np.isfinite(
            warning_values
        )
    ]

    if (
        AUTO_THRESHOLD_FROM_OFFICIAL_WARNINGS
        and warning_values.size > 0
    ):

        threshold_min = float(
            np.min(
                warning_values
            )
        )

        threshold_max = float(
            np.max(
                warning_values
            )
        )

        method = (
            "derived from official "
            "radiometric_warning pixels"
        )

    else:

        threshold_min = (
            RADIANCE_WARNING_MIN
        )

        threshold_max = (
            RADIANCE_WARNING_MAX
        )

        method = (
            "fixed user-defined range"
        )

    return (
        threshold_min,
        threshold_max,
        method
    )


def analyse_radiance_threshold(
    channel,
    radiance,
    official_warning,
    radiometric_warning,
    index_map,
    output_distribution_csv
):

    print_header(
        f"{channel}: RADIANCE THRESHOLD ANALYSIS"
    )

    (
        threshold_min,
        threshold_max,
        method
    ) = determine_radiance_threshold(
        radiance,
        radiometric_warning
    )

    print(
        f"Threshold method : {method}"
    )

    print(
        f"Threshold range  : "
        f"{threshold_min:g} - "
        f"{threshold_max:g}"
    )

    raw = np.asarray(
        radiance
    )

    warning_mask = (
        (raw >= threshold_min)
        &
        (raw <= threshold_max)
    )

    total_threshold_warning = int(
        np.count_nonzero(
            warning_mask
        )
    )

    print()
    print(
        "Radiance-threshold warning pixels:"
    )

    print(
        f"  {total_threshold_warning:,}"
    )

    # --------------------------------------------------------
    # Warning radiance statistics
    # --------------------------------------------------------

    warning_values = raw[
        warning_mask
    ]

    stats = nan_statistics(
        warning_values
    )

    print()
    print(
        "WARNING PIXELS"
    )

    print(
        "----------------"
    )

    print(
        f"count : {stats['count']:,}"
    )

    print(
        f"min   : {stats['min']}"
    )

    print(
        f"max   : {stats['max']}"
    )

    print(
        f"mean  : {stats['mean']}"
    )

    print(
        f"median: {stats['median']}"
    )

    print(
        f"std   : {stats['std']}"
    )

    # --------------------------------------------------------
    # Unique warning DN distribution
    # --------------------------------------------------------

    print()
    print(
        "UNIQUE WARNING RADIANCE VALUES"
    )

    print(
        "------------------------------"
    )

    distribution_rows = []

    unique_warning_values = np.unique(
        warning_values
    )

    for value in unique_warning_values:

        count = int(
            np.count_nonzero(
                warning_values == value
            )
        )

        percentage = safe_percentage(
            count,
            stats["count"]
        )

        print(
            f"radiance={int(value):4d} "
            f"count={count:8,d} "
            f"percentage={percentage:8.4f}%"
        )

        distribution_rows.append(
            {
                "channel": channel,
                "radiance": int(value),
                "warning_count": count,
                "warning_percentage": percentage,
            }
        )

    # --------------------------------------------------------
    # Clean versus warning radiance
    # --------------------------------------------------------

    print()
    print(
        "CLEAN RADIANCE NEAR WARNING RANGE"
    )

    print(
        "----------------------------------"
    )

    all_raw = raw.ravel()

    all_official_warning = (
        np.asarray(
            official_warning
        ).ravel()
    )

    # --------------------------------------------------------
    # Evaluate a slightly larger range
    # --------------------------------------------------------

    lower = int(
        np.floor(
            threshold_min
        )
    ) - 5

    upper = int(
        np.ceil(
            threshold_max
        )
    ) + 11

    for value in range(
        lower,
        upper + 1
    ):

        value_mask = (
            all_raw == value
        )

        total = int(
            np.count_nonzero(
                value_mask
            )
        )

        if total == 0:
            continue

        official_count = int(
            np.count_nonzero(
                all_official_warning[
                    value_mask
                ]
            )
        )

        clean_count = (
            total
            - official_count
        )

        warning_pct = safe_percentage(
            official_count,
            total
        )

        print(
            f"radiance={value:4d} "
            f"warning={official_count:8,d} "
            f"clean={clean_count:8,d} "
            f"warning%={warning_pct:8.3f}"
        )

    distribution_df = pd.DataFrame(
        distribution_rows
    )

    save_csv(
        distribution_df,
        output_distribution_csv
    )

    return (
        warning_mask,
        threshold_min,
        threshold_max
    )


# ============================================================
# SPATIAL MAPS
# ============================================================

def create_spatial_maps(
    channel,
    quality,
    index_map,
    warning_mask,
    figures_dir
):

    if not SAVE_FIGURES:
        return

    print_header(
        f"{channel}: CREATING SPATIAL MAPS"
    )

    ensure_directory(
        figures_dir
    )

    # --------------------------------------------------------
    # Official quality warning map
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 6)
    )

    plt.imshow(
        warning_mask,
        interpolation="nearest",
        aspect="auto"
    )

    plt.title(
        f"{channel} - Official FCI Quality Warnings"
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Row"
    )

    plt.colorbar(
        label="Warning"
    )

    plt.tight_layout()

    path = os.path.join(
        figures_dir,
        f"{channel}_quality_map.png"
    )

    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight"
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close()

    # --------------------------------------------------------
    # Index map
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 6)
    )

    plt.imshow(
        index_map,
        interpolation="nearest",
        aspect="auto"
    )

    plt.title(
        f"{channel} - FCI Index Map"
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Row"
    )

    plt.colorbar(
        label="Index"
    )

    plt.tight_layout()

    path = os.path.join(
        figures_dir,
        f"{channel}_index_map.png"
    )

    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight"
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close()

    # --------------------------------------------------------
    # Radiance-threshold warning map
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 6)
    )

    plt.imshow(
        warning_mask,
        interpolation="nearest",
        aspect="auto"
    )

    plt.title(
        f"{channel} - Radiance Threshold Warning Map"
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Row"
    )

    plt.colorbar(
        label="Radiance warning"
    )

    plt.tight_layout()

    path = os.path.join(
        figures_dir,
        f"{channel}_warning_map.png"
    )

    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight"
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close()


# ============================================================
# RADIANCE HISTOGRAMS
# ============================================================

def create_radiance_histograms(
    channel,
    radiance,
    official_warning,
    threshold_warning,
    figures_dir
):

    if not SAVE_FIGURES:
        return

    print_header(
        f"{channel}: RADIANCE HISTOGRAMS"
    )

    ensure_directory(
        figures_dir
    )

    raw = np.asarray(
        radiance
    ).ravel()

    valid = np.isfinite(
        raw
    )

    raw = raw[valid]

    official_warning = (
        np.asarray(
            official_warning
        ).ravel()[valid]
    )

    threshold_warning = (
        np.asarray(
            threshold_warning
        ).ravel()[valid]
    )

    # --------------------------------------------------------
    # All radiance values
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.hist(
        raw,
        bins=100
    )

    plt.title(
        f"{channel} - Effective Radiance Distribution"
    )

    plt.xlabel(
        "Raw effective radiance / DN"
    )

    plt.ylabel(
        "Pixel count"
    )

    plt.tight_layout()

    path = os.path.join(
        figures_dir,
        f"{channel}_radiance_histogram.png"
    )

    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight"
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close()

    # --------------------------------------------------------
    # Warning radiance
    # --------------------------------------------------------

    warning_values = raw[
        threshold_warning
    ]

    if warning_values.size > 0:

        plt.figure(
            figsize=(12, 6)
        )

        plt.hist(
            warning_values,
            bins=50
        )

        plt.title(
            f"{channel} - Radiance Values in "
            f"Threshold Warning Region"
        )

        plt.xlabel(
            "Raw effective radiance / DN"
        )

        plt.ylabel(
            "Pixel count"
        )

        plt.tight_layout()

        path = os.path.join(
            figures_dir,
            f"{channel}_warning_radiance_histogram.png"
        )

        plt.savefig(
            path,
            dpi=DPI,
            bbox_inches="tight"
        )

        if SHOW_FIGURES:
            plt.show()

        plt.close()


# ============================================================
# CLEAN DATASET
# ============================================================

def create_clean_dataset(
    channel,
    ds,
    official_warning,
    output_file
):

    print_header(
        f"{channel}: CREATING CLEAN DATASET"
    )

    # --------------------------------------------------------
    # Copy dataset
    # --------------------------------------------------------

    clean_ds = ds.copy(
        deep=False
    )

    # --------------------------------------------------------
    # Create quality mask
    # --------------------------------------------------------

    quality = ds[
        "pixel_quality"
    ]

    warning_mask = xr.DataArray(
        official_warning,
        dims=quality.dims,
        coords=quality.coords
    )

    # --------------------------------------------------------
    # Mask radiance
    # --------------------------------------------------------

    radiance = ds[
        "effective_radiance"
    ]

    clean_radiance = (
        radiance.where(
            ~warning_mask
        )
    )

    clean_ds[
        "effective_radiance_clean"
    ] = clean_radiance

    clean_ds[
        "official_quality_warning"
    ] = warning_mask.astype(
        np.uint8
    )

    clean_ds[
        "official_quality_warning"
    ].attrs[
        "description"
    ] = (
        "1 = at least one official FCI "
        "pixel_quality warning"
    )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    ensure_directory(
        os.path.dirname(
            output_file
        )
    )

    clean_ds.to_netcdf(
        output_file
    )

    print()
    print(
        "Clean/QC dataset written:"
    )

    print(
        output_file
    )

    return clean_ds


# ============================================================
# MAIN CHANNEL PROCESSING
# ============================================================

def process_channel(
    channel,
    output_root,
    quality_summary_rows,
    radiance_summary_rows
):

    print_header(
        f"PROCESSING CHANNEL: {channel}"
    )

    ds, group_path = open_channel(
        channel
    )

    print()
    print(
        "Measured dataset:"
    )

    print(
        ds
    )

    print()

    # --------------------------------------------------------
    # Find variables
    # --------------------------------------------------------

    quality_name = find_variable(
        ds,
        [
            "pixel_quality",
            "quality",
            "pixel_quality_flags",
        ]
    )

    radiance_name = find_variable(
        ds,
        [
            "effective_radiance",
            "radiance",
        ]
    )

    index_name = find_variable(
        ds,
        [
            "index_map",
            "index",
        ]
    )

    if quality_name is None:

        raise RuntimeError(
            f"{channel}: pixel_quality "
            "variable not found."
        )

    if radiance_name is None:

        raise RuntimeError(
            f"{channel}: effective_radiance "
            "variable not found."
        )

    if index_name is None:

        raise RuntimeError(
            f"{channel}: index_map "
            "variable not found."
        )

    print(
        f"Quality variable : {quality_name}"
    )

    print(
        f"Radiance variable: {radiance_name}"
    )

    print(
        f"Index map        : {index_name}"
    )

    quality = ds[
        quality_name
    ].values

    radiance = ds[
        radiance_name
    ].values

    index_map = ds[
        index_name
    ].values

    print()

    print(
        f"Shape          : {quality.shape}"
    )

    print(
        f"Total pixels   : {quality.size:,}"
    )

    # --------------------------------------------------------
    # Official quality flags
    # --------------------------------------------------------

    (
        masks,
        official_warning,
        quality_df
    ) = analyse_quality_flags(
        channel,
        quality,
        radiance,
        os.path.join(
            output_root,
            "csv",
            f"{channel}_quality_flags.csv"
        )
    )

    # --------------------------------------------------------
    # Radiance information
    # --------------------------------------------------------

    (
        scale_factor,
        add_offset,
        units
    ) = analyse_radiance_information(
        channel,
        ds
    )

    # --------------------------------------------------------
    # Official radiometric warning
    # --------------------------------------------------------

    radiometric_warning = (
        get_radiometric_warning_mask(
            quality
        )
    )

    # --------------------------------------------------------
    # DN vs quality
    # --------------------------------------------------------

    dn_df = analyse_dn_vs_quality(
        channel,
        radiance,
        official_warning,
        radiometric_warning,
        os.path.join(
            output_root,
            "csv",
            f"{channel}_DN_vs_quality.csv"
        )
    )

    # --------------------------------------------------------
    # Index map
    # --------------------------------------------------------

    (
        index_df,
        radiance_index_df
    ) = analyse_index_map(
        channel,
        index_map,
        official_warning,
        radiance,
        os.path.join(
            output_root,
            "csv",
            f"{channel}_index_map_analysis.csv"
        ),
        os.path.join(
            output_root,
            "csv",
            f"{channel}_radiance_by_index.csv"
        )
    )

    # --------------------------------------------------------
    # Radiance threshold
    # --------------------------------------------------------

    (
        threshold_warning,
        threshold_min,
        threshold_max
    ) = analyse_radiance_threshold(
        channel,
        radiance,
        official_warning,
        radiometric_warning,
        index_map,
        os.path.join(
            output_root,
            "csv",
            f"{channel}_radiance_warning_distribution.csv"
        )
    )

    # --------------------------------------------------------
    # Spatial figures
    # --------------------------------------------------------

    create_spatial_maps(
        channel,
        quality,
        index_map,
        official_warning,
        os.path.join(
            output_root,
            "figures"
        )
    )

    # --------------------------------------------------------
    # Radiance figures
    # --------------------------------------------------------

    create_radiance_histograms(
        channel,
        radiance,
        official_warning,
        threshold_warning,
        os.path.join(
            output_root,
            "figures"
        )
    )

    # --------------------------------------------------------
    # Clean dataset
    # --------------------------------------------------------

    clean_file = os.path.join(
        output_root,
        "cleaned",
        f"{channel}_FCI_QC.nc"
    )

    create_clean_dataset(
        channel,
        ds,
        official_warning,
        clean_file
    )

    # --------------------------------------------------------
    # Channel summary
    # --------------------------------------------------------

    total_pixels = quality.size

    official_warning_count = int(
        np.count_nonzero(
            official_warning
        )
    )

    radiometric_warning_count = int(
        np.count_nonzero(
            radiometric_warning
        )
    )

    threshold_warning_count = int(
        np.count_nonzero(
            threshold_warning
        )
    )

    radiance_stats = nan_statistics(
        radiance
    )

    warning_radiance_stats = nan_statistics(
        radiance[
            radiometric_warning
        ]
    )

    quality_summary_rows.append(
        {
            "channel": channel,
            "total_pixels": total_pixels,
            "official_warning_pixels":
                official_warning_count,
            "official_warning_percentage":
                safe_percentage(
                    official_warning_count,
                    total_pixels
                ),
            "radiometric_warning_pixels":
                radiometric_warning_count,
            "radiometric_warning_percentage":
                safe_percentage(
                    radiometric_warning_count,
                    total_pixels
                ),
            "radiance_threshold_warning_pixels":
                threshold_warning_count,
            "radiance_threshold_warning_percentage":
                safe_percentage(
                    threshold_warning_count,
                    total_pixels
                ),
        }
    )

    radiance_summary_rows.append(
        {
            "channel": channel,
            "total_pixels": total_pixels,
            "radiance_min":
                radiance_stats["min"],
            "radiance_max":
                radiance_stats["max"],
            "radiance_mean":
                radiance_stats["mean"],
            "radiance_median":
                radiance_stats["median"],
            "radiance_std":
                radiance_stats["std"],
            "official_radiometric_warning_min":
                warning_radiance_stats["min"],
            "official_radiometric_warning_max":
                warning_radiance_stats["max"],
            "official_radiometric_warning_mean":
                warning_radiance_stats["mean"],
            "official_radiometric_warning_median":
                warning_radiance_stats["median"],
            "radiance_threshold_min":
                threshold_min,
            "radiance_threshold_max":
                threshold_max,
        }
    )

    ds.close()

    print_header(
        f"{channel}: COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "FCI QUALITY CONTROL"
    )

    print(
        "Input file:"
    )

    print(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.isfile(
        INPUT_FILE
    ):

        print()
        print(
            "ERROR:"
        )

        print(
            "Input file does not exist:"
        )

        print(
            INPUT_FILE
        )

        sys.exit(
            1
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(
            INPUT_FILE
        )
    )[0]

    output_root = os.path.join(
        os.path.dirname(
            INPUT_FILE
        ),
        f"{base_name}_FCI_QC"
    )

    print()
    print(
        "Output directory:"
    )

    print(
        output_root
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    ensure_directory(
        output_root
    )

    ensure_directory(
        os.path.join(
            output_root,
            "csv"
        )
    )

    ensure_directory(
        os.path.join(
            output_root,
            "figures"
        )
    )

    ensure_directory(
        os.path.join(
            output_root,
            "cleaned"
        )
    )

    # --------------------------------------------------------
    # Open root dataset
    # --------------------------------------------------------

    print_header(
        "OPENING FCI FILE"
    )

    try:

        root_ds = xr.open_dataset(
            INPUT_FILE,
            engine="netcdf4"
        )

        print()
        print(
            "Root dataset opened successfully."
        )

        print(
            root_ds
        )

        root_ds.close()

    except Exception as exc:

        print()
        print(
            "ERROR opening root dataset:"
        )

        print(
            exc
        )

        traceback.print_exc()

        sys.exit(
            1
        )

    # --------------------------------------------------------
    # Discover channels
    # --------------------------------------------------------

    print_header(
        "DISCOVERING FCI CHANNELS"
    )

    channels = discover_fci_channels(
        root_ds if 'root_ds' in locals()
        else None
    )

    # --------------------------------------------------------
    # Fallback discovery
    # --------------------------------------------------------

    if not channels:

        # Direct test of known channels
        known_channels = [
            "ir_105",
            "ir_123",
            "ir_133",
            "ir_38",
            "ir_87",
            "ir_97",
            "vis_04",
            "vis_05",
            "vis_06",
            "vis_08",
            "vis_09",
        ]

        for channel in known_channels:

            try:

                test_ds, _ = open_channel(
                    channel
                )

                test_ds.close()

                channels.append(
                    channel
                )

            except Exception:
                pass

    channels = sorted(
        set(channels)
    )

    print()
    print(
        "Channels discovered:"
    )

    for channel in channels:

        print(
            f"  {channel}"
        )

    # --------------------------------------------------------
    # Apply user selection
    # --------------------------------------------------------

    if CHANNELS_TO_PROCESS is not None:

        channels = [
            c for c in channels
            if c in CHANNELS_TO_PROCESS
        ]

    print()
    print(
        "Channels selected for analysis:"
    )

    for channel in channels:

        print(
            f"  {channel}"
        )

    if not channels:

        print()
        print(
            "ERROR: No FCI channels found."
        )

        sys.exit(
            1
        )

    # --------------------------------------------------------
    # Global summaries
    # --------------------------------------------------------

    quality_summary_rows = []

    radiance_summary_rows = []

    # --------------------------------------------------------
    # Process channels
    # --------------------------------------------------------

    for channel in channels:

        try:

            process_channel(
                channel,
                output_root,
                quality_summary_rows,
                radiance_summary_rows
            )

        except Exception as exc:

            print()
            print(
                f"ERROR processing {channel}:"
            )

            print(
                exc
            )

            traceback.print_exc()

    # --------------------------------------------------------
    # Global summary files
    # --------------------------------------------------------

    print_header(
        "WRITING GLOBAL SUMMARY FILES"
    )

    quality_summary = pd.DataFrame(
        quality_summary_rows
    )

    radiance_summary = pd.DataFrame(
        radiance_summary_rows
    )

    quality_summary_file = os.path.join(
        output_root,
        "csv",
        "FCI_all_channels_quality_summary.csv"
    )

    radiance_summary_file = os.path.join(
        output_root,
        "csv",
        "FCI_all_channels_radiance_summary.csv"
    )

    save_csv(
        quality_summary,
        quality_summary_file
    )

    save_csv(
        radiance_summary,
        radiance_summary_file
    )

    print()
    print(
        "Quality summary:"
    )

    print(
        quality_summary_file
    )

    print()
    print(
        "Radiance summary:"
    )

    print(
        radiance_summary_file
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print_header(
        "FCI QUALITY CONTROL COMPLETE"
    )

    print()
    print(
        "Input:"
    )

    print(
        f"    {INPUT_FILE}"
    )

    print()
    print(
        "Output:"
    )

    print(
        f"    {output_root}"
    )

    print()
    print(
        "Figures:"
    )

    print(
        f"    {os.path.join(output_root, 'figures')}"
    )

    print()
    print(
        "CSV files:"
    )

    print(
        f"    {os.path.join(output_root, 'csv')}"
    )

    print()
    print(
        "Clean datasets:"
    )

    print(
        f"    {os.path.join(output_root, 'cleaned')}"
    )

    print()
    print(
        "Quality flags decoded:"
    )

    for bit, name in QUALITY_FLAGS.items():

        print(
            f"    {bit:3d} -> {name}"
        )

    print()
    print(
        "Analysis completed."
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()