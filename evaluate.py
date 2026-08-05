# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 01:01:10 2026

@author: Zamikhaya.Magogotya
"""

"""
==============================================================

SAT-RADAR AI

Model Evaluation

Author : Zamikhaya Magogotya

==============================================================
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


##############################################################
# Basic Regression Metrics
##############################################################

def rmse(prediction, target):
    prediction = prediction.flatten()
    target = target.flatten()
    return np.sqrt(mean_squared_error(target, prediction))


def mae(prediction, target):
    prediction = prediction.flatten()
    target = target.flatten()
    return mean_absolute_error(target, prediction)


def bias(prediction, target):
    prediction = prediction.flatten()
    target = target.flatten()
    return np.mean(prediction - target)


def correlation(prediction, target):
    prediction = prediction.flatten()
    target = target.flatten()

    return np.corrcoef(
        prediction,
        target
    )[0,1]


def r2(prediction, target):

    prediction = prediction.flatten()

    target = target.flatten()

    return r2_score(

        target,

        prediction

    )


##############################################################
# Confusion Matrix
##############################################################

def confusion_matrix(prediction,
                     target,
                     threshold=35):

    prediction = prediction >= threshold

    target = target >= threshold

    TP = np.sum(prediction & target)

    TN = np.sum(~prediction & ~target)

    FP = np.sum(prediction & ~target)

    FN = np.sum(~prediction & target)

    return TP,FP,FN,TN


##############################################################
# Meteorological Scores
##############################################################

def pod(tp,fn):

    if tp+fn==0:

        return np.nan

    return tp/(tp+fn)


def far(tp,fp):

    if tp+fp==0:

        return np.nan

    return fp/(tp+fp)


def csi(tp,fp,fn):

    if tp+fp+fn==0:

        return np.nan

    return tp/(tp+fp+fn)


def precision(tp,fp):

    if tp+fp==0:

        return np.nan

    return tp/(tp+fp)


def recall(tp,fn):

    return pod(tp,fn)


def accuracy(tp,fp,fn,tn):

    total=tp+fp+fn+tn

    return (tp+tn)/total


##############################################################
# Heidke Skill Score
##############################################################

def hss(tp,fp,fn,tn):

    numerator=2*(tp*tn-fp*fn)

    denominator=(

        (tp+fn)*(fn+tn)+

        (tp+fp)*(fp+tn)

    )

    return numerator/denominator


##############################################################
# Equitable Threat Score
##############################################################

def ets(tp,fp,fn):

    random=((tp+fp)*(tp+fn))/(
        tp+fp+fn
    )

    return (tp-random)/(
        tp+fp+fn-random
    )


##############################################################
# Full Evaluation
##############################################################

def evaluate(prediction,
             target,
             threshold=35):

    TP,FP,FN,TN=confusion_matrix(

        prediction,

        target,

        threshold

    )

    results={

        "RMSE":rmse(prediction,target),

        "MAE":mae(prediction,target),

        "BIAS":bias(prediction,target),

        "Correlation":correlation(prediction,target),

        "R2":r2(prediction,target),

        "POD":pod(TP,FN),

        "FAR":far(TP,FP),

        "CSI":csi(TP,FP,FN),

        "Precision":precision(TP,FP),

        "Accuracy":accuracy(TP,FP,FN,TN),

        "HSS":hss(TP,FP,FN,TN),

        "ETS":ets(TP,FP,FN),

    }

    return results


##############################################################
# Pretty Printer
##############################################################

def print_results(results):

    print()

    print("="*55)

    print(" SAT-RADAR AI Evaluation")

    print("="*55)

    for key,value in results.items():

        print(f"{key:15s}: {value:.4f}")

    print("="*55)