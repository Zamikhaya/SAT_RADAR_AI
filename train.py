# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 00:37:18 2026

@author: Zamikhaya.Magogotya
"""

"""
==============================================================

SAT-RADAR AI

Training Script

Author : Zamikhaya Magogotya

==============================================================
"""

import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SatelliteRadarDataset
from models.vision_transformer import SatRadarViT
from losses import CombinedLoss


###############################################################
# Configuration
###############################################################

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

BATCH_SIZE = 2

EPOCHS = 50

LEARNING_RATE = 1e-4

CHECKPOINT_DIR = "checkpoints"

os.makedirs(

    CHECKPOINT_DIR,

    exist_ok=True,

)


###############################################################
# Dataset
###############################################################

train_dataset = SatelliteRadarDataset(

    satellite_folder="SAT",

    radar_folder="RADAR",

)

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

)


###############################################################
# Model
###############################################################

model = SatRadarViT()

model.to(DEVICE)


###############################################################
# Loss
###############################################################

criterion = CombinedLoss()


###############################################################
# Optimizer
###############################################################

optimizer = optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

)


###############################################################
# Scheduler
###############################################################

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=5,

)


###############################################################
# Mixed Precision
###############################################################

scaler = torch.cuda.amp.GradScaler(

    enabled=torch.cuda.is_available()

)


###############################################################
# Training Loop
###############################################################

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    start = time.time()

    for satellite, radar in train_loader:

        satellite = satellite.to(DEVICE)

        radar = radar.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(

            enabled=torch.cuda.is_available()

        ):

            prediction = model(satellite)

            loss = criterion(

                prediction,

                radar,

            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    scheduler.step(epoch_loss)

    elapsed = time.time() - start

    print(

        f"Epoch {epoch+1:03d}"

        f" | Loss = {epoch_loss:.5f}"

        f" | Time = {elapsed:.1f}s"

    )

    ###########################################################
    # Save checkpoint
    ###########################################################

    checkpoint = {

        "epoch": epoch,

        "model": model.state_dict(),

        "optimizer": optimizer.state_dict(),

    }

    torch.save(

        checkpoint,

        os.path.join(

            CHECKPOINT_DIR,

            "latest_model.pth",

        ),

    )


###############################################################
# Save Final Model
###############################################################

torch.save(

    model.state_dict(),

    os.path.join(

        CHECKPOINT_DIR,

        "final_model.pth",

    ),

)

print()

print("Training Complete")