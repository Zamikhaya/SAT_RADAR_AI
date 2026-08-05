# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 23:52:46 2026

@author: Zamikhaya.Magogotya
"""

"""
Global configuration for SAT-RADAR AI
"""

from pathlib import Path
import torch

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

SATELLITE_DIR = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\SAT"
)

RADAR_DIR = Path(
    r"C:\Users\zamikhaya.magogotya\Desktop\000_PHD\VOL"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# =====================================================
# IMAGE SETTINGS
# =====================================================

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

RADAR_CHANNELS = 3

# =====================================================
# TIME SETTINGS
# =====================================================

TIME_TOLERANCE_SECONDS = 300

# =====================================================
# TRAINING
# =====================================================

BATCH_SIZE = 8

NUM_WORKERS = 4

EPOCHS = 100

LEARNING_RATE = 1e-4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =====================================================
# NORMALIZATION
# =====================================================

REFLECTIVITY_MIN = -10

REFLECTIVITY_MAX = 70