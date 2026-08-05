# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 00:31:43 2026

@author: Zamikhaya.Magogotya
"""

"""
=========================================================
SAT-RADAR AI

Loss Functions

Author : Zamikhaya Magogotya

=========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


###########################################################
# Mean Squared Error
###########################################################

class MSELoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, prediction, target):

        return self.loss(prediction, target)


###########################################################
# Mean Absolute Error
###########################################################

class MAELoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, prediction, target):

        return self.loss(prediction, target)


###########################################################
# Weighted Reflectivity Loss
###########################################################

class ReflectivityLoss(nn.Module):

    """
    Gives more importance to strong echoes.

    Target is normalized between 0 and 1.
    """

    def __init__(self):

        super().__init__()

    def forward(self, prediction, target):

        weights = 1 + 4 * target

        loss = weights * (prediction - target) ** 2

        return loss.mean()


###########################################################
# Gradient Loss
###########################################################

class GradientLoss(nn.Module):

    """
    Preserves storm boundaries.
    """

    def __init__(self):

        super().__init__()

    def gradient_x(self, img):

        return img[:, :, :, 1:] - img[:, :, :, :-1]

    def gradient_y(self, img):

        return img[:, :, 1:, :] - img[:, :, :-1, :]

    def forward(self, prediction, target):

        pred_dx = self.gradient_x(prediction)

        pred_dy = self.gradient_y(prediction)

        true_dx = self.gradient_x(target)

        true_dy = self.gradient_y(target)

        loss_x = F.l1_loss(pred_dx, true_dx)

        loss_y = F.l1_loss(pred_dy, true_dy)

        return loss_x + loss_y


###########################################################
# Structural Similarity (Approximation)
###########################################################

class SSIMLoss(nn.Module):

    """
    Simplified differentiable SSIM.
    """

    def __init__(self):

        super().__init__()

    def forward(self, prediction, target):

        mu_x = prediction.mean()

        mu_y = target.mean()

        sigma_x = prediction.var()

        sigma_y = target.var()

        sigma_xy = ((prediction - mu_x) *
                    (target - mu_y)).mean()

        C1 = 0.01 ** 2

        C2 = 0.03 ** 2

        numerator = (
            (2 * mu_x * mu_y + C1) *
            (2 * sigma_xy + C2)
        )

        denominator = (
            (mu_x ** 2 + mu_y ** 2 + C1) *
            (sigma_x + sigma_y + C2)
        )

        ssim = numerator / denominator

        return 1 - ssim


###########################################################
# Combined Loss
###########################################################

class CombinedLoss(nn.Module):

    """
    Final loss used for training.

    Total Loss

    = MSE
    + Weighted Reflectivity
    + Gradient
    + SSIM
    """

    def __init__(self,
                 mse_weight=1.0,
                 refl_weight=2.0,
                 grad_weight=0.2,
                 ssim_weight=0.5):

        super().__init__()

        self.mse = MSELoss()

        self.reflectivity = ReflectivityLoss()

        self.gradient = GradientLoss()

        self.ssim = SSIMLoss()

        self.mse_weight = mse_weight

        self.refl_weight = refl_weight

        self.grad_weight = grad_weight

        self.ssim_weight = ssim_weight

    def forward(self,
                prediction,
                target):

        mse = self.mse(prediction, target)

        refl = self.reflectivity(
            prediction,
            target
        )

        grad = self.gradient(
            prediction,
            target
        )

        ssim = self.ssim(
            prediction,
            target
        )

        total = (

            self.mse_weight * mse +

            self.refl_weight * refl +

            self.grad_weight * grad +

            self.ssim_weight * ssim

        )

        return total