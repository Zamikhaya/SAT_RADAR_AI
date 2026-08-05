# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 00:40:53 2026

@author: Zamikhaya.Magogotya
"""

"""
==============================================================

SAT-RADAR AI

Inference Script

Author : Zamikhaya Magogotya

==============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.vision_transformer import SatRadarViT
from satellite_loader import load_satellite_file

##############################################################
# Configuration
##############################################################

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = "checkpoints/final_model.pth"

OUTPUT_FOLDER = "outputs"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

##############################################################
# Load Model
##############################################################

model = SatRadarViT()

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Supports both full checkpoint and state_dict
if isinstance(state_dict, dict) and "model" in state_dict:
    model.load_state_dict(state_dict["model"])
else:
    model.load_state_dict(state_dict)

model.to(DEVICE)

model.eval()

print("Model Loaded")

##############################################################
# Prediction Function
##############################################################

def predict_satellite(filename):

    ##########################################################

    satellite = load_satellite_file(filename)

    ##########################################################

    satellite = torch.tensor(
        satellite,
        dtype=torch.float32
    )

    satellite = satellite.unsqueeze(0)

    satellite = satellite.to(DEVICE)

    ##########################################################

    with torch.no_grad():

        prediction = model(satellite)

    ##########################################################

    prediction = prediction.squeeze(0)

    prediction = prediction.cpu().numpy()

    ##########################################################

    return prediction

##############################################################
# Plot Function
##############################################################

def plot_prediction(prediction):

    names = [

        "Reflectivity (dBZ)",

        "Differential Reflectivity (ZDR)",

        "Correlation Coefficient (RHOHV)"

    ]

    plt.figure(figsize=(15,5))

    for i in range(3):

        plt.subplot(1,3,i+1)

        plt.imshow(
            prediction[i],
            origin="lower",
            cmap="turbo"
        )

        plt.title(names[i])

        plt.colorbar()

    plt.tight_layout()

    plt.show()

##############################################################
# Save Function
##############################################################

def save_prediction(prediction, filename):

    output = os.path.join(

        OUTPUT_FOLDER,

        filename + ".npy"

    )

    np.save(

        output,

        prediction

    )

    print("Saved:", output)

##############################################################
# Example
##############################################################

if __name__ == "__main__":

    satellite_file = r"SAT/sample.nc"

    prediction = predict_satellite(

        satellite_file

    )

    print(

        prediction.shape

    )

    plot_prediction(

        prediction

    )

    save_prediction(

        prediction,

        "prediction"

    )