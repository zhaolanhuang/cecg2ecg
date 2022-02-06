# Example code to reproduce the results of C. Hoog Antink, E. Breuer, D.U. Uguz, and S. Leonhardt:
# “Signal-Level Fusion with Convolutional Neural Networks for Capacitively Coupled ECG in the Car”
# Computing in Cardiology 2018;45: accepted for publication
#
# Code written by Erik Breuer, Birger Nordmann and Christoph Hoog Antink
#
# This code can be used for whatever purpose but comes with no warranty or support.
#
# If you use this code, citation of the original publication is much appreciated.
#
# The necessary file "UnoViS_auto2012.mat", like this code, can be downloaded from
# https://www.medit.hia.rwth-aachen.de/unovis

import argparse
import logging as log
import os
import unovis.preprocessing as pp
import unovis.training as train
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='e.g. python run_me.py ./UnoViS_auto2012.mat -e 2 4 -t 0.6 0.95')
    parser.add_argument('-e', '--epochs', default=[1,2,4,8], nargs='*', type=int, help="Defines the number of epochs per generation. Should be a list of integers. e.g. 1 2 4 8")
    parser.add_argument('-t', '--thresh', default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], nargs='*', type=float, help="Define the thresholds for the final peak detection. Should be a list of floats, e.g. 0.6 0.7 0.8 0.9")
    parser.add_argument('-c', '--cache', default='./cache/', help="Path to a directory to hold temporary files.")
    parser.add_argument('--reprint_only', action='store_true', help="Does not train and evaluate a network but print results of an earlier run stored in the cache directory.")
    parser.add_argument('-d', '--debug', action='store_true', help="Print additional debug information to the console.")
    parser.add_argument('mat_file', help="Path to the UnoViS_auto2012.mat file. (Can be downloaded from http://www.medit.hia.rwth-aachen.de/unovis)")
    params = parser.parse_args()

    if params.debug:
        log.basicConfig(level=log.DEBUG)
    else:
        log.basicConfig(level=log.INFO)

    if params.reprint_only:
        epochs, thresh, TP, FP, FN, nr_diff, diff_dum, abs_diff_sum, squared_diff_sum = train.load_results(params.cache)
        train.evaluation(epochs, thresh, TP, FP, FN, nr_diff, diff_dum, abs_diff_sum, squared_diff_sum)
        exit(0)

    if not os.path.isfile(params.mat_file):
        log.error("Mat file {} does not exist!".format(params.mat_file))
        exit(1)

    if not os.path.isdir(params.cache):
        log.info("Creating cache directory at {}".format(params.cache))
        os.makedirs(params.cache)
#     for filename in glob(params.cache + "model_s*.h5"):
#         os.remove(filename)

    pp.resample_unovis_auto2012(params.mat_file, params.cache)
    TP, FP, FN, nr_diff, diff_sum, abs_diff_sum, squared_diff_sum = train.train_UnoVis_3cECG(params.cache, params.epochs, params.thresh)
    train.save_results(params.cache, params.epochs, params.thresh, TP, FP, FN, nr_diff, diff_sum, abs_diff_sum, squared_diff_sum)
    train.evaluation(params.epochs, params.thresh, TP, FP, FN, nr_diff, diff_sum, abs_diff_sum, squared_diff_sum)
