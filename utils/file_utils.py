# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 23:58:29 2026

@author: Zamikhaya.Magogotya
"""

from pathlib import Path

def get_nc_files(folder):

    return sorted(Path(folder).glob("*.nc"))