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
# import pyhrv

def read_wfdb(path, start, end):

    record = wfdb.rdsamp(path, sampfrom= start, sampto= end)
    # record = wfdb.rdsamp(path)
    # annotation = wfdb.rdann(path, 'ecg')

    # annotation = wfdb.rdann(path, 'ari')
    annotation = wfdb.rdann(path, 'ari', sampfrom= start, sampto= end)
    print('annotion')
    # print(annotation.sample)
    # print(annotation.symbol)

    # Read an annotation as an Annotation object
    sig = record[0]
    rr = record[1]
    print('rr\t', rr)

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
        signal.append(an)
        i = i + 1

    annotation.sample = annotation.sample - start
    anno = ["|", "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"]
    peak = []

    for i in range(len(annotation.symbol)):

        if (annotation.symbol[i] in anno):
            peak.append(annotation.sample[i])

    print('peaks length\t', len(peak))
    return signal, peak, fs
def features(peaks, fs):

    NN50, approximate, = ([] for i in range(2))
    start = 0
    end = 120 * fs

    print("length of peaks:\t", len(peaks))
    XX = 1

    while True:

        go = bisect.bisect_left(peaks, start)
        out = bisect.bisect_left(peaks, end)

        RRi = []
        ff = 0
        peaks_in = peaks[go:out]
        XY = 0

        for i in range(len(peaks_in) - 1):
            new = peaks_in[i + 1] - peaks_in[i]
            new = (new / fs)
            RRi.append(new)
            XY = XY + 1

            if (new > 0.05):
                ff = ff + 1

        # print("XY equals to\t")

        NN50.append(ff)
        # print("after filtering\t", len(RRi) - ff)
        approximate.append(ant.app_entropy(RRi))
        # print("lenght of RRi", len(RRi))
        # print("lenght of NN50", len(RRi) - ff)
        # print("segment lenght\t", XX)
        XX = XX+1

        # compute the R-R intervals
        # rr = np.diff(peaks_in)
        # gg = [x /fs for x in rr]
        # compute the
        # rri = np.abs(np.diff(gg))

        # print("rri\t", RRi)

        # for i in range (len(RRi)):
        #     if ( rri[i]> 0.05):
        #         ff = ff+1
        # NN50.append(ff)

        start = start + (10 * fs)
        end = end + (10 * fs)
        # i += 1

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
def thresh(array, thresh):

    mean = np.mean(array)
    percent = (mean / 100) * thresh

    return (mean + percent)
def reading_MIt_data(path, ID):

    fs = 128
    with open(path + "peaks-" + ID + ".txt", 'r') as file1:
        peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    return peaks
def reading_seina_data(path, ID):
    fs = 512
    with open(path + "peaks-" + ID + ".txt", 'r') as file1:
        peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    return peaks
def reading_timone_data(path, ID):
    fs = 256
    with open(path + "peaks-" + ID + ".txt", 'r') as file1:
        peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    return peaks

# =============================================================================================================
# # == loading epileptic patients acquisitions taken from INS " DAvid Olivier"
# path = "C:\\Users\\Ftay\Desktop\\PhD\\tests\\INS - David\\"
# fs = 512
# ID = "0252GRE-EEG_13"
# with open(path + "peaks-" + ID + ".txt", 'r') as file1:
#     peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

# path = "C:\\Users\\Ftay\Desktop\\PhD\\tests\\INS - David\\inter-ictal\\"
# ID = "0252GRE"
# fs = 512
# # == load the epileptic patient
# with open(path + "peaks-" + ID + "-end-EEG-14.txt", 'r') as file1:BQ
#     peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

thresh_AP = 0.07
thresh_nn = 5

# =============================================================================================================
# # == loading epileptic patients acquisitions taken from INS " Fabrice"
# path = "E:\\data\\tests\\Peaks_RR\\fabrice\\pre-ictal\\PN02\\"
# ID = "EEEEEEE140528A-DEX_0001"
#
# path = "E:\\data\\tests\\Peaks_RR\\fabrice\\inter-ictal\\PN01\\"
# ID = "PN02-08-correction"
#
# fs = 256
# with open(path + "peaks-" + ID + ".txt", 'r') as file1:
#     peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

# =============================================================================================================
# = Working on the Healthy Mit data
# fs = 250
# path = "E:\\data\\tests\\Peaks_RR\\Fantasia\\"
#
# ID = "PNf1o06-1h"
# peaks = reading_MIt_data(path,ID)
#
# thresh_AP = 0.07
# thresh_nn = 1.8
#
onset = 10 + (0/60)

path = "E:\\data\\tests\\Peaks_RR\\fabrice\\pre-ictal\\PN01\\"
# path = "E:\\data\\tests\\siena_vs_fantasia\\Peaks_RR\\PN06\\"
ID = "PN01-03+04"

##
# path = "E:\\data\\tests\\Peaks_RR\\Interictal\\PN05\\"
# ID = "PN05-2-10_40"
#
fs = 256
# # # == load the epileptic patient
with open(path + "peaks-" + ID + ".txt", 'r') as file1:
    peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]
#
# thresh_AP = 0.14
# thresh_nn = 7
# =============================================================================================================
# == loading epileptic patients acquisitions C:\Users\Ftay\Desktop\PhD\tests\INS - David

# # path = "E:\\data\\post-ictal-heart-rate-oscillations-in-partial-epilepsy-1.0.0\\sz01"
# path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\siena_vs_fantasia\\Peaks_RR\\PN00\\"
# path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\siena_vs_fantasia\\unhealthy\\"
# path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\siena_vs_fantasia\\Peaks_RR\\PN10\\"
# ID = "PN10-5"
# # path = "C:\\Users\\Ftay\\Desktop\\PhD\\tests\\siena_vs_fantasia\\Peaks_RR\\Interictal\\PN06\\"
# # ID = "PN06-3"
# fs = 512
# # # == load the epileptic patient
# with open(path + "peaks-" + ID + ".txt", 'r') as file1:
#     peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]
#
# thresh_AP = 0.05
#
# thresh_nn = 0

# print("first peak\t", peaks[0])
# print("last peak\t", peaks[-1])
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
# == testing the result of our approach on the post-ictal database

# path = 'E://data//post-ictal-heart-rate-oscillations-in-partial-epilepsy-1.0.0//sz01'
# ID = "sz01"
# fs = 200
# # =====================
#  # 00:51:25
# end = (fs * 60 * 60 * 0) + (fs * 60 * 24) + (36 * fs)
# start = (fs * 60 * 60 * 0) + (fs * 60 * 0) + (36 * fs)

# end = (fs * 60 * 60 * 3) + (fs * 60 * 5) + (51 * fs)
# start = (fs * 60 * 60 * 1) + (fs * 60 * 55) + (51 * fs)

#00:51:25
# seizure1 = (fs * 60 * 60 * 0) + (fs * 60 * 51) + (25 * fs)
# # 02:04:45
# seizure2 = (fs * 60 * 60 * 2) + (fs * 60 * 4) + (45 * fs)

# thresh_AP = 0.14
# thresh_nn = 7

# == reading the signal
# signal_input, peaks, fs = read_wfdb(path, start, end)
# signal_input, peaks, fs = read_wfdb(path, 0, 0)

# print('lenght of signal\t', (len(signal_input)/200)/60)
# print(len(signal_input))
# print(len(peaks))
# pl = [1.5] * len(peaks)
# peaks = [x - start for x in peaks]
#

print("FS equals to\t", fs)
print("plotting done")

# =============================================================================================================
# == healthy subjects
NN50_ep, pNN50_ep, SDNN_ep, SDNN_test, nn50_test_sei, fft_ep, welch_ep, RMSSD_ep = ([] for i in range(8))

# == healthy subjects

NN50, app = ([] for i in range(2))


pre_ictal = []
STD = []
STD_healthy = []

print("peaks lenght:\t", len(peaks))
app, NN50 = features(peaks,fs)

print("lenght of NN50\t" , len(NN50))

# fig, axs = plt.subplots()
# axs.plot(NN50, label="the input signal")
# axs.set_ylabel('Amplitude')
# axs.legend()
# plt.show()
# print('seizure index')
# print ('stop')
# plt.show()
# =============================================================================================================
# == threshold computing
#  working on ApEn

STD_app = std_compute(app)
STD_NN50 = std_compute(NN50)

# SD_app = STD_app[0:len(STD_app) - 10]
#
# thresh_AP = threshold(SD_app)

# =============================================================================================================
# == working on NN50
# SD_NN50 = STD_NN50[0:len(STD_NN50) - 10]
#
# thresh_nn = threshold(SD_NN50)

# ==============================
# == normality test

print('ApEn threshold equals to:\t', thresh_AP)
print('NRRi threshold equals to:\t', thresh_nn)
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
# x_n, y_n = LineString(intersection_n).xy
# print((x_n))



# =============================================================================================================
# == plotting the results

# onset =  (fs * 60 * 5) + (41 * fs)

# onset = 10

import csv
EEG = []
with open("D:\\fabrise\\PN.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    EEG.append(row)

print("row")
fig, axs = plt.subplots(2, 1)
axs[0].plot(tt, app, label="ApEn feature", marker='o')
axs[1].plot(tt_healthy,STD_app, label="STD of ApEn feature", marker='o')
axs[0].axvline(x=tt[-1] - 600, color='red', linestyle='--')
axs[1].axvline(x=len(STD_app) - onset, color='red', linestyle='--')

axs[0].set_title('Acquisition ' + ID,fontsize=24, y=1)

# axs[1].axhline(y=thresh_AP, color='blue', linestyle='--')

axs[0].set_xlabel('Progress per 10s')
axs[0].set_ylabel('entropy value')
axs[1].set_xlabel('Time in min')
axs[1].set_ylabel('STD value')
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

# if intersection.geom_type == 'MultiPoint':
#     axs[1].plot(*LineString(intersection).xy, 'o')
# elif intersection.geom_type == 'Point':
#     axs[1].plot(*intersection.xy, 'o')

# == plotting NN50
fig, axs = plt.subplots(2, 1)
axs[0].plot(tt_healthy,STD_app, label="STD of ApEn feature", marker='o')
axs[1].plot(tt_healthy, STD_NN50, label="STD of NRRi feature", marker='o')

axs[0].set_title('Acquisition ' + ID,fontsize=24, y=1)

axs[0].axvline(x= len(STD_app) - onset, color='red', linestyle='--')
axs[1].axvline(x= len(STD_NN50) - onset, color='red', linestyle='--')

axs[0].set_xlabel('Progress per 10s')
axs[0].set_ylabel('STD value')
axs[1].set_xlabel('Time in min')
axs[1].set_ylabel('STD value')
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

# if intersection.geom_type == 'MultiPoint':
#     axs[0].plot(*LineString(intersection).xy, 'o')
#
# elif intersection.geom_type == 'Point':
#     axs[0].plot(*intersection.xy, 'o')
#
#
# if intersection_n.geom_type == 'MultiPoint':
#     axs[1].plot(*LineString(intersection_n).xy, 'o')
#
# elif intersection_n.geom_type == 'Point':
#     axs[1].plot(*intersection_n.xy, 'o')


# == plotting NN50
fig, axs = plt.subplots(2, 1)
axs[0].plot(tt, NN50, label=" NRRi feature", marker='o')
axs[1].plot(tt_healthy, STD_NN50, label="STD of NRRi feature", marker='o')
axs[0].set_title('Acquisition ' + ID,fontsize=24, y=1)

axs[0].axvline(x=tt[-1] - 600, color='red', linestyle='--')
axs[1].axvline(x= len(STD_NN50) - onset, color='red', linestyle='--')

axs[0].set_xlabel('Progress per 10s')
axs[0].set_ylabel('NRRi value')
axs[1].set_xlabel('Time in min')
axs[1].set_ylabel('STD value')
axs[0].grid(True)
axs[1].grid(True)
axs[0].legend()
axs[1].legend()

# axs[1].axhline(y=thresh_nn, color='blue', linestyle='--')

ymin = 70 ; ymax = 275
axs[0].set_ylim([ymin,ymax])

ymin_1 = 0; ymax_1 = 14
axs[1].set_ylim([ymin_1,ymax_1])

# if intersection_n.geom_type == 'MultiPoint':
#     axs[1].plot(*LineString(intersection_n).xy, 'o')
# elif intersection_n.geom_type == 'Point':
#     axs[1].plot(*intersection_n.xy, 'o')

plt.show()

# path = "C:\\Users\\Administrator\\Desktop\\figures\\fabrice\\curves - correlation\\ECG\\"
# path = "E:\\data\\tests\\features-healthy-epileptic\\"
# Path(path).mkdir(parents=True, exist_ok=True)

# normalizedData = (STD_app -np.min(STD_app ))/(np.max(STD_app)-np.min(STD_app))
# normalizedDataNN50 = (STD_NN50 -np.min(STD_NN50 ))/(np.max(STD_NN50)-np.min(STD_NN50))
#
# np.savetxt(path + '\\ApEn-' + ID + '.txt', np.array(app))
# np.savetxt(path + '\\NRRi-' + ID + '.txt', np.array(NN50))
