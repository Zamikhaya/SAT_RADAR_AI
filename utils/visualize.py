# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 01:14:49 2026

@author: Zamikhaya.Magogotya
"""

"""
==============================================================

SAT-RADAR AI

Visualization Module

Author : Zamikhaya Magogotya

==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


###############################################################
# Display a single image
###############################################################

def show_image(image,
               title="",
               cmap="turbo",
               vmin=None,
               vmax=None):

    plt.figure(figsize=(6,6))

    plt.imshow(
        image,
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax
    )

    plt.title(title)

    plt.colorbar()

    plt.tight_layout()

    plt.show()


###############################################################
# Display satellite RGB
###############################################################

def show_satellite(rgb):

    plt.figure(figsize=(8,8))

    plt.imshow(rgb)

    plt.title("Satellite")

    plt.axis("off")

    plt.tight_layout()

    plt.show()


###############################################################
# Compare Truth vs Prediction
###############################################################

def compare_prediction(target,
                       prediction,
                       title="Reflectivity"):

    fig,ax=plt.subplots(1,2,figsize=(12,6))

    im=ax[0].imshow(
        target,
        cmap="turbo",
        origin="lower"
    )

    ax[0].set_title("Observed")

    plt.colorbar(im,ax=ax[0])

    im=ax[1].imshow(
        prediction,
        cmap="turbo",
        origin="lower"
    )

    ax[1].set_title("Predicted")

    plt.colorbar(im,ax=ax[1])

    fig.suptitle(title)

    plt.tight_layout()

    plt.show()


###############################################################
# Difference Map
###############################################################

def difference_map(target,
                   prediction):

    diff=prediction-target

    plt.figure(figsize=(7,6))

    plt.imshow(

        diff,

        cmap="bwr",

        origin="lower",

        vmin=-20,

        vmax=20

    )

    plt.title("Prediction Error")

    plt.colorbar(label="dBZ")

    plt.tight_layout()

    plt.show()


###############################################################
# Three-panel Comparison
###############################################################

def three_panel(target,
                prediction):

    diff=prediction-target

    fig,ax=plt.subplots(1,3,figsize=(18,6))

    im=ax[0].imshow(target,
                    cmap="turbo",
                    origin="lower")

    ax[0].set_title("Observed")

    plt.colorbar(im,ax=ax[0])

    im=ax[1].imshow(prediction,
                    cmap="turbo",
                    origin="lower")

    ax[1].set_title("Predicted")

    plt.colorbar(im,ax=ax[1])

    im=ax[2].imshow(diff,
                    cmap="bwr",
                    origin="lower",
                    vmin=-20,
                    vmax=20)

    ax[2].set_title("Difference")

    plt.colorbar(im,ax=ax[2])

    plt.tight_layout()

    plt.show()


###############################################################
# Multi-channel Radar Display
###############################################################

def show_radar_channels(radar):

    """
    radar shape

    (3,H,W)

    channel0=dBZ

    channel1=ZDR

    channel2=RHOHV
    """

    names=[

        "Reflectivity (dBZ)",

        "Differential Reflectivity (ZDR)",

        "Correlation Coefficient (RHOHV)"

    ]

    cmaps=[

        "turbo",

        "viridis",

        "plasma"

    ]

    fig,ax=plt.subplots(

        1,

        3,

        figsize=(18,6)

    )

    for i in range(3):

        im=ax[i].imshow(

            radar[i],

            cmap=cmaps[i],

            origin="lower"

        )

        ax[i].set_title(names[i])

        plt.colorbar(im,ax=ax[i])

    plt.tight_layout()

    plt.show()


###############################################################
# Training Curves
###############################################################

def plot_loss(train_loss,
              val_loss):

    plt.figure(figsize=(8,5))

    plt.plot(train_loss,
             label="Training")

    plt.plot(val_loss,
             label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


###############################################################
# Histogram Comparison
###############################################################

def histogram(target,
              prediction):

    plt.figure(figsize=(8,5))

    plt.hist(

        target.flatten(),

        bins=60,

        alpha=0.5,

        label="Observed"

    )

    plt.hist(

        prediction.flatten(),

        bins=60,

        alpha=0.5,

        label="Predicted"

    )

    plt.xlabel("dBZ")

    plt.ylabel("Pixels")

    plt.legend()

    plt.tight_layout()

    plt.show()


###############################################################
# Scatter Plot
###############################################################

def scatter(target,
            prediction):

    plt.figure(figsize=(6,6))

    plt.scatter(

        target.flatten(),

        prediction.flatten(),

        s=2,

        alpha=0.3

    )

    lims=[

        np.min(target),

        np.max(target)

    ]

    plt.plot(lims,
             lims,
             "r--")

    plt.xlabel("Observed")

    plt.ylabel("Predicted")

    plt.grid(True)

    plt.tight_layout()

    plt.show()