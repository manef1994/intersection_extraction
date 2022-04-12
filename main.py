import bisect
import math
import statistics
# import neurokit2 as nk
from shapely.geometry import LineString
import numpy as np
from matplotlib import *
from pathlib import Path
import matplotlib.pyplot as plt
import entropy as ent
import wfdb
import antropy as ant
from mne.io import read_raw_edf

def read_wfdb(path):
    #, sampfrom = start, sampto = end
    record = wfdb.rdsamp(path)
    annotation = wfdb.rdann(path, 'ecg')
    # annotation = wfdb.rdann(path, 'dat')

    # Read an annotation as an Annotation object
    sig = record[0]
    rr = record[1]

    fs = rr['fs']
    print('fs equals to: ', fs)

    i = 0
    # l_n = end - start
    start = 0
    end = rr['sig_len']
    l_n = rr['sig_len']
    signal = []

    while i < l_n:
        an = sig[i]
        #signal.append(an[1])
        signal.append(an)
        i = i + 1

    annotation.sample = annotation.sample - start
    anno = ["|", "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"]
    peak = []

    for i in range(len(annotation.symbol)):

        if (annotation.symbol[i] in anno):
            peak.append(annotation.sample[i])

    return signal, peak, fs

def features(peaks, fs):

    NN50, approximate, = ([] for i in range(2))
    start = 0
    end = 120 * fs

    while True:

        go = bisect.bisect_left(peaks, start)
        out = bisect.bisect_left(peaks, end)

        RRi = []
        peaks_in = []
        ff = 0
        peaks_in = peaks[go:out]

        for i in range(len(peaks_in) - 1):
            new = peaks_in[i + 1] - peaks_in[i]
            new = (new / fs)
            RRi.append(new)
            if (new > 0.05):
                ff = ff+1

        NN50.append(ff)

        #approximate.append(nk.entropy_approximate(RRi))
        approximate.append(ant.app_entropy(RRi))

        start = start + (10 * fs)
        end = end + (10 * fs)
        i += 1

        if (end > peaks[-1]):
            break

    return approximate, NN50

def std_compute(xx):
    i = 0
    j = 6
    STD = []

    while (True):
        go_in = []

        go_in = xx[i:j]
        std = np.std(go_in)
        STD.append(std)

        j = j + 6
        i = i + 6

        if j > len(app):
            break
    return STD

def threshold(array):

    sorted = np.sort(array)
    # == computing Q1 and Q3
    print("length of sorted", len(sorted))
    Q1 = (len(sorted) + 1) / 4
    Q3 = ((3 * len(sorted)) + 1) / 4

    if (Q1 % 2 != 0):
        new = (sorted[int(Q1)] + sorted[int(Q1) + 1]) / 2
        lower = new
    else:
        lower = sorted[Q1]

    if (Q3 % 2 != 0):
        new = (sorted[int(Q3)] + sorted[int(Q3) + 1]) / 2
        upper = new
    else:
        upper = sorted[Q3]

    print("Q1 equals to \t", Q1)
    print("lower value equals to \t", lower)

    print("Q3 equals to \t", Q3)
    print("upper value equals to \t", upper)

    IQR = upper - lower
    print("IQR equals to \t", IQR)
    thresh = upper + (1.5 * IQR)
    print("the new threshold is", thresh)

    return thresh

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
# =============================================================================================================
# == loading epileptic patients acquisitions

# path = "E:\\data\\post-ictal-heart-rate-oscillations-in-partial-epilepsy-1.0.0\\sz01"
path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\siena_vs_fantasia\\Peaks_RR\\PN09\\"
ID = "09-2"
fs = 512
# == load the epileptic patient
with open(path + "peaks-PN" + ID + ".txt", 'r') as file1:
    peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]



print("plotting done")
# =============================================================================================================

# load healthy subjects
# path = "/home/manef/tests/siena_vs_fantasia/fantasia_features/"
#
# path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\data\\fantasia-database-1.0.0(1)\\young_to_try_in_test\\f1y04"
#
# ID = "f1y01"
#
# peaks = []
# signal_input, peaks, fs = read_wfdb(path)

# =============================================================================================================

# == healthy subjects
NN50_ep, pNN50_ep, SDNN_ep, SDNN_test, nn50_test_sei, fft_ep, welch_ep, RMSSD_ep = ([] for i in range(8))

# == healthy subjects

NN50, app = ([] for i in range(2))

pre_ictal = []
STD = []
STD_healthy = []

app, NN50 = features(peaks,fs)

# =============================================================================================================
# == threshold computing
#  working on ApEn

STD_app = std_compute(app)
STD_NN50 = std_compute(NN50)

SD_app = STD_app[0:len(STD_app) - 10]

thresh_AP = threshold(SD_app)

# =============================================================================================================
# == working on NN50
SD_NN50 = STD_NN50[0:len(STD_NN50) - 10]

thresh_nn = threshold(SD_NN50)

# ==============================
# == normality test

print('ApEn threshold equals to:\t', thresh_AP)
print('NN50 threshold equals to:\t', thresh_nn)
# =============================================================================================================
# == finding the intersections of the threshold value computed

first = 0
tt = [first]
i = 1
while i <= len(app) - 1:
    first = first + 10
    tt.append(first)
    i = i + 1

first = 0
tt_healthy = [first]
i = 1
while i <= len(STD_app) - 1:
    first = first + 60
    tt_healthy.append(first)
    i = i + 1

first = 0
tt_healthy = [first]
i = 1
while i <= len(STD_app) - 1:
    first = first + 1
    tt_healthy.append(first)
    i = i + 1

# ____ woerking on the approximate thresholding
x = numpy.zeros(len(STD_app), dtype=float, order='C')
xy = numpy.zeros(len(app), dtype=float, order='C')

arr=[]
arr1=[]

for i in range (len(STD_app)):
    arr.append(thresh_AP)
    arr1.append(0)

first_line = LineString(np.column_stack((tt_healthy,STD_app)))
second_line = LineString(np.column_stack((tt_healthy,arr)))
intersection = first_line.intersection(second_line)

x, y = LineString(intersection).xy
print(" x \t", x)

arr_n=[]
arr1_n=[]

for i in range (len(STD_NN50)):
    arr_n.append(thresh_nn)
    arr1_n.append(0)

first_line_n = LineString(np.column_stack((tt_healthy, STD_NN50)))
second_line_n = LineString(np.column_stack((tt_healthy, arr_n)))
intersection_n = first_line_n.intersection(second_line_n)
x_n, y_n = LineString(intersection_n).xy
print((x_n))


round_app = []
round_NN50 = []

# =============================================================================================================
# == plotting the results

fig, axs = plt.subplots(2, 1)
axs[0].plot(tt, app, label="approximate entropy of healthy signal", marker='o')
axs[1].plot(tt_healthy,STD_app, label="STD of approximate entropy", marker='o')
axs[0].axvline(x=tt[-1] - 600, color='red', linestyle='--')
axs[1].axvline(x=len(STD_app) - 10, color='red', linestyle='--')

axs[0].set_title('Subject ' + ID,fontsize=24, y=1)

axs[1].axhline(y=thresh_AP, color='blue', linestyle='--')

axs[0].set_xlabel('sample per min')
axs[0].set_ylabel('entropy value')
axs[1].set_xlabel('sample per min')
axs[1].set_ylabel('entropy value')
axs[0].grid(True)
axs[1].grid(True)
axs[0].legend()
axs[1].legend()

ymin = 0.1
ymax = 1.3
axs[0].set_ylim([ymin,ymax])

ymin_1 = 0
ymax_1 = 0.25
axs[1].set_ylim([ymin_1,ymax_1])

if intersection.geom_type == 'MultiPoint':
    axs[1].plot(*LineString(intersection).xy, 'o')
elif intersection.geom_type == 'Point':
    axs[1].plot(*intersection.xy, 'o')


# == plotting NN50
fig, axs = plt.subplots(2, 1)
axs[0].plot(tt_healthy,STD_app, label="STD of approximate entropy", marker='o')
axs[1].plot(tt_healthy, STD_NN50, label="STD of NN50", marker='o')

axs[0].set_title('Subject ' + ID,fontsize=24, y=1)

axs[0].axvline(x= len(STD_app) - 10, color='red', linestyle='--')
axs[1].axvline(x= len(STD_NN50) - 10, color='red', linestyle='--')

axs[0].set_xlabel('sample per min')
axs[0].set_ylabel('entropy value')
axs[1].set_xlabel('sample per min')
axs[1].set_ylabel('entropy value')
axs[0].grid(True)
axs[1].grid(True)
axs[0].legend()
axs[1].legend()

axs[0].axhline(y=thresh_AP, color='blue', linestyle='--')
axs[1].axhline(y=thresh_nn, color='blue', linestyle='--')

ymin_1 = 0; ymax_1 = 0.25
axs[0].set_ylim([ymin_1,ymax_1])

ymin_1 = 0; ymax_1 = 14
axs[1].set_ylim([ymin_1,ymax_1])

if intersection.geom_type == 'MultiPoint':
    axs[0].plot(*LineString(intersection).xy, 'o')

elif intersection.geom_type == 'Point':
    axs[0].plot(*intersection.xy, 'o')


if intersection_n.geom_type == 'MultiPoint':
    axs[1].plot(*LineString(intersection_n).xy, 'o')

elif intersection_n.geom_type == 'Point':
    axs[1].plot(*intersection_n.xy, 'o')


# == plotting NN50
fig, axs = plt.subplots(2, 1)
axs[0].plot(tt, NN50, label=" NN50 of healthy signal", marker='o')
axs[1].plot(tt_healthy, STD_NN50, label="STD of NN50", marker='o')
axs[0].set_title('Subject ' + ID,fontsize=24, y=1)

axs[0].axvline(x=tt[-1] - 600, color='red', linestyle='--')
axs[1].axvline(x= len(STD_NN50) - 10, color='red', linestyle='--')

axs[0].set_xlabel('sample en seconde')
axs[0].set_ylabel('valeur de entropie')
axs[1].set_xlabel('sample en seconde')
axs[1].set_ylabel('valeur de entropie')
axs[0].grid(True)
axs[1].grid(True)
axs[0].legend()
axs[1].legend()

axs[1].axhline(y=thresh_nn, color='blue', linestyle='--')

ymin = 70 ; ymax = 275
axs[0].set_ylim([ymin,ymax])

ymin_1 = 0; ymax_1 = 14
axs[1].set_ylim([ymin_1,ymax_1])

if intersection_n.geom_type == 'MultiPoint':
    axs[1].plot(*LineString(intersection_n).xy, 'o')
elif intersection_n.geom_type == 'Point':
    axs[1].plot(*intersection_n.xy, 'o')


plt.show()