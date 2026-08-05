# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 01:26:46 2026

@author: Zamikhaya.Magogotya
"""
import os
import hdf5plugin

# Tell HDF5 where the plugins are BEFORE loading netCDF4/HDF5
os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH

print("HDF5_PLUGIN_PATH =", os.environ["HDF5_PLUGIN_PATH"])

import netCDF4

fname = r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL\SAT\W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY--DIS-NC4E_C_EUMT_20260504105839_IDPFI_OPE_20260504105540_20260504105629_N_JLS_O_0066_0025.nc"

ds = netCDF4.Dataset(fname)

var = ds["data"]["ir_105"]["measured"]["effective_radiance"]

print(var.shape)
print(var[0, 0])

import numpy as np

data = var[:]

print(data.shape)

rows, cols = np.where(~data.mask)

print(rows[0], cols[0])

print(data[rows[0], cols[0]])
