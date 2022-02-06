import scipy.io
import numpy as np
from scipy.signal import butter, lfilter
import math


def evaluate_with_time(annotation_samples, peak_indices):
    ref = annotation_samples
    test = peak_indices

    TP = 0
    FP = 0
    FN = 0
    i = 0
    j = 0

    diff = list()

    while (i < ref.shape[0] - 1 and j < test.shape[0] - 1):
        t = test[j]
        T = ref[i]
        if (j != test.shape[0] - 1 and i != ref.shape[0] - 1):
            t_prime = test[j + 1]
            T_prime = ref[i + 1]

            if (t < T):
                if (T - t < (50) and (T - t < abs(T - t_prime) or abs(T_prime - t_prime) < abs(T - t_prime))):
                    # match a and A
                    TP = TP + 1
                    # get next t
                    j = j + 1
                    # get next T
                    i = i + 1

                    diff.append(T - t)
                else:
                    # no match for t
                    FP = FP + 1
                    # get next t
                    j = j + 1
            else:
                if (t - T <= (50) and (t - T < abs(t - T_prime) or abs(t_prime - T_prime) < abs(t - T_prime))):
                    # match a and A
                    TP = TP + 1
                    # get next t
                    j = j + 1
                    # get next T
                    i = i + 1

                    diff.append(T - t)

                else:
                    # no match for T
                    FN = FN + 1
                    # get next T
                    i = i + 1

    FN = annotation_samples.shape[0] - TP

    diff_sum = np.sum(diff)
    squared_diff_sum = 0
    abs_diff_sum = 0
    for i in diff:
        abs_diff_sum = abs_diff_sum + math.sqrt(i * i)
        squared_diff_sum = squared_diff_sum + (i * i)
    #squared_diff_sum = np.sum(math.sqrt(diff * diff))
    nr_diff = len(diff)

    return TP, FP, FN, diff_sum, abs_diff_sum, squared_diff_sum, nr_diff

def differentiator(data):
    b = [1, 1, -1, -1]
    a = [1]
    y = lfilter(b, a, data)
    return y