import bisect
import math
import statistics
# import neurokit2 as nk
from shapely.geometry import LineString
import numpy as np
from matplotlib import *
from pathlib import Path
import matplotlib.pyplot as plt
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

    NNRi, approximate, = ([] for i in range(2))
    start = 0
    end = 120 * fs

    while True:

        go = bisect.bisect_left(peaks, start)
        out = bisect.bisect_left(peaks, end)

        peaks_in = peaks[go:out]
        RRi = np.diff(peaks_in)

        NNRi.append(len(RRi))
        approximate.append(ant.app_entropy(RRi))

        start = start + (10 * fs)
        end = end + (10 * fs)

        if (end > peaks[-1]):
            break

    return approximate, NNRi

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
# == loading epileptic patients acquisitions taken from INS " DAvid Olivier"

path = "E:\\data\\tests\\Peaks_RR\\fabrice\\pre-ictal\\PN01\\"

ID = "PN01-03+04"

fs = 256
# # # == load the epileptic patient
with open(path + "peaks-" + ID + ".txt", 'r') as file1:
    peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

"in this part, you need to precise how many minute the difference between the end of the acquisition and the seizures time"
"In my worek, i always use 10 min as ictal (len(signal) - seizure time)"
onset = 10

"those value are the threshold values to be used for both ApEn and NNRi features"

thresh_AP = 0.14
thresh_nn = 7

# =============================================================================================================
"computing the featurees here"
NNRi, app = ([] for i in range(2))

pre_ictal = []
STD = []
STD_healthy = []

app, NNRi = features(peaks,fs)

# =============================================================================================================
"computing the STD based on the feautres computed in the previous step"

STD_app = std_compute(app)
STD_NNRi = std_compute(NNRi)


print('ApEn threshold equals to:\t', thresh_AP)
print('NRRi threshold equals to:\t', thresh_nn)
# =============================================================================================================
# == finding the intersections of the threshold value computed
"this is step is important to mmake the plotting better visually"

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

for i in range (len(STD_NNRi)):
    arr_n.append(thresh_nn)
    arr1_n.append(0)

first_line_n = LineString(np.column_stack((tt_healthy, STD_NNRi)))
second_line_n = LineString(np.column_stack((tt_healthy, arr_n)))
intersection_n = first_line_n.intersection(second_line_n)
x_n, y_n = LineString(intersection_n).xy
print((x_n))

# =============================================================================================================
# == plotting the results

"plotting the ApEn curve and the STD ApEn"
fig, axs = plt.subplots(2, 1)
axs[0].plot(tt, app, label="ApEn feature", marker='o')
axs[1].plot(tt_healthy,STD_app, label="STD of ApEn feature", marker='o')
axs[0].axvline(x=tt[-1] - 600, color='red', linestyle='--')
axs[1].axvline(x=len(STD_app) - onset, color='red', linestyle='--')
axs[0].set_title('Acquisition ' + ID,fontsize=24, y=1)
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

if intersection.geom_type == 'MultiPoint':
    axs[1].plot(*LineString(intersection).xy, 'o')
elif intersection.geom_type == 'Point':
    axs[1].plot(*intersection.xy, 'o')

# == == == == == == == == == == == == == == == == == == == == == == == ==
"Plotting the NNRi curve an,d the corresponding STD curve "
fig, axs = plt.subplots(2, 1)
axs[0].plot(tt, NNRi, label=" NRRi feature", marker='o')
axs[1].plot(tt_healthy, STD_NNRi, label="STD of NRRi feature", marker='o')
axs[0].set_title('Acquisition ' + ID,fontsize=24, y=1)

axs[0].axvline(x=tt[-1] - 600, color='red', linestyle='--')
axs[1].axvline(x= len(STD_NNRi) - onset, color='red', linestyle='--')

axs[0].set_xlabel('Progress per 10s')
axs[0].set_ylabel('NRRi value')
axs[1].set_xlabel('Time in min')
axs[1].set_ylabel('STD value')
axs[0].grid(True)
axs[1].grid(True)
axs[0].legend()
axs[1].legend()

ymin = 70 ; ymax = 275
axs[0].set_ylim([ymin,ymax])

ymin_1 = 0; ymax_1 = 14
axs[1].set_ylim([ymin_1,ymax_1])

if intersection_n.geom_type == 'MultiPoint':
    axs[1].plot(*LineString(intersection_n).xy, 'o')
elif intersection_n.geom_type == 'Point':
    axs[1].plot(*intersection_n.xy, 'o')


# == == == == == == == == == == == == == == == == == == == == == == == ==
"Plotting the STD curves of the ApEN and the NRRI features and the thier corresponding thresholds"
fig, axs = plt.subplots(2, 1)
axs[0].plot(tt_healthy,STD_app, label="STD of ApEn feature", marker='o')
axs[1].plot(tt_healthy, STD_NNRi, label="STD of NRRi feature", marker='o')

axs[0].set_title('Acquisition ' + ID,fontsize=24, y=1)

axs[0].axvline(x= len(STD_app) - onset, color='red', linestyle='--')
axs[1].axvline(x= len(STD_NNRi) - onset, color='red', linestyle='--')

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

if intersection.geom_type == 'MultiPoint':
    axs[0].plot(*LineString(intersection).xy, 'o')

elif intersection.geom_type == 'Point':
    axs[0].plot(*intersection.xy, 'o')


if intersection_n.geom_type == 'MultiPoint':
    axs[1].plot(*LineString(intersection_n).xy, 'o')

elif intersection_n.geom_type == 'Point':
    axs[1].plot(*intersection_n.xy, 'o')

plt.show()
