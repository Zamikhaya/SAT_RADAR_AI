# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 01:17:34 2026

@author: Zamikhaya.Magogotya
"""

"""
==============================================================

SAT-RADAR AI

Export Predicted Radar to NetCDF

Author : Zamikhaya Magogotya

==============================================================
"""

import os
import numpy as np
from netCDF4 import Dataset
from datetime import datetime


##############################################################
# Export Prediction
##############################################################

def export_prediction(
        prediction,
        latitude,
        longitude,
        output_file,
        radar_name="Durban",
        source="SAT-RADAR AI"):

    """
    Parameters
    ----------
    prediction : ndarray

        Shape = (3,H,W)

        prediction[0] = Reflectivity

        prediction[1] = ZDR

        prediction[2] = RHOHV

    latitude : ndarray

        2D latitude grid

    longitude : ndarray

        2D longitude grid

    output_file : str

        Output NetCDF filename

    """

    H = prediction.shape[1]
    W = prediction.shape[2]

    ##########################################################

    nc = Dataset(

        output_file,

        "w",

        format="NETCDF4"

    )

    ##########################################################
    # Dimensions
    ##########################################################

    nc.createDimension("y", H)

    nc.createDimension("x", W)

    ##########################################################
    # Coordinates
    ##########################################################

    lat = nc.createVariable(

        "latitude",

        "f4",

        ("y", "x")

    )

    lon = nc.createVariable(

        "longitude",

        "f4",

        ("y", "x")

    )

    ##########################################################
    # Radar Variables
    ##########################################################

    dbz = nc.createVariable(

        "reflectivity",

        "f4",

        ("y", "x"),

        zlib=True

    )

    zdr = nc.createVariable(

        "zdr",

        "f4",

        ("y", "x"),

        zlib=True

    )

    rhohv = nc.createVariable(

        "rhohv",

        "f4",

        ("y", "x"),

        zlib=True

    )

    ##########################################################
    # Global Attributes
    ##########################################################

    nc.title = "Synthetic Weather Radar"

    nc.institution = "South African Weather Service"

    nc.source = source

    nc.history = "Created " + datetime.utcnow().isoformat()

    nc.radar = radar_name

    nc.Conventions = "CF-1.8"

    ##########################################################
    # Variable Attributes
    ##########################################################

    lat.units = "degrees_north"

    lon.units = "degrees_east"

    lat.standard_name = "latitude"

    lon.standard_name = "longitude"

    dbz.units = "dBZ"

    dbz.long_name = "Radar Reflectivity"

    zdr.units = "dB"

    zdr.long_name = "Differential Reflectivity"

    rhohv.units = "1"

    rhohv.long_name = "Correlation Coefficient"

    ##########################################################
    # Write Data
    ##########################################################

    lat[:] = latitude

    lon[:] = longitude

    dbz[:] = prediction[0]

    zdr[:] = prediction[1]

    rhohv[:] = prediction[2]

    ##########################################################

    nc.close()

    print("Saved:", output_file)