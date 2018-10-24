import sys
import pywt                             # for wavelet operations
import numpy as np

from statsmodels.robust import mad      # median absolute deviation from array
from scipy.optimize import curve_fit    # for interpolating muons

from functions_fitting import *

# function to split muons from each other
def SplitMuons(indices):
    # create multidimensional list
    grouped_array = [[]]

    # muon counter
    muons = 0

    # loop through list and find gaps in the list to group the muons
    for i in range(0, len(indices) - 1):
        # as long as the index is increasing by one the indices belong to one muon
        if indices[i] + 1 == indices[i + 1]:
            grouped_array[muons].append(indices[i])
        # as soon as there is a jump, a new muon was found and is added to the list
        else:
            grouped_array[muons].append(indices[i])
            grouped_array.append([])
            muons += 1
    if len(indices) > 0:
        # add the last element to the list and
        grouped_array[muons].append(indices[-1])
        # print the number of muons found
        print(str(muons + 1) + ' muons have been found.')

    return grouped_array

# detect muons for removal and returns non vanishing indices
def WaveletMuon(noisydata, thresh_mod, wavelet='sym8', level=1):
    # calculate wavelet coefficients
    coeff = pywt.wavedec(noisydata, wavelet)    # symmetric signal extension mode

    # calculate a threshold (1.5 the size of usual threshold)
    sigma = mad(coeff[-level])
    threshold = sigma * np.sqrt(2 * np.log(len(noisydata))) * thresh_mod

    # detect spikes on D1 details (written in the last entry of coeff)
    # calculate thresholded coefficients
    for i in range(1, len(coeff)):
        coeff[i] = pywt.threshold(coeff[i], value=threshold, mode='soft')
    # set everything but D1 level to zero
    for i in range(0, len(coeff)-1):
        coeff[i] = np.zeros_like(coeff[i])

    # reconstruct the signal using the thresholded coefficients
    denoised = pywt.waverec(coeff, wavelet)

    if (len(noisydata) % 2) == 0:
        # get non vanishing indices
        indices = np.nonzero(denoised)[0]
        grouped = SplitMuons(indices)
        # return the value of denoised and the non vanishing indices
        return denoised, grouped
    else:
        # get non vanishing indices
        indices = np.nonzero(denoised[:-1])[0]
        grouped = SplitMuons(indices)
        # return the value of denoised and the non vanishing indices
        return denoised[:-1], grouped

# linear function for muon approximation
def linear(x, m, b):
    return x * m + b

# approximate muon by linear function and return modified y
def RemoveMuon(xdata, y, indices):
    # prevent python from working on original data
    ydata = np.copy(y)

    for muon in indices:
        # calculate limits for indices to use for fitting
        limit = int(len(muon)/4)
        lower = muon[:limit]
        upper = muon[-limit:]
        fit_indices = np.append(lower, upper)

        # fit to the data
        popt, pcov = curve_fit(linear, xdata[fit_indices], ydata[fit_indices])

        # calculate approximated y values and remove muon
        for index in muon[limit:-limit]:
            ydata[index] = linear(xdata[index], *popt)

    return ydata

# smooth spectrum by using wavelet transform and soft threshold
# returns the denoised spectrum and sigma
def WaveletSmooth(noisydata, wavelet='sym8', level=1):
    # calculate wavelet coefficients
    coeff = pywt.wavedec(noisydata, wavelet)

    # calculate a threshold
    sigma = mad(coeff[-level])
    threshold = sigma * np.sqrt(2 * np.log(len(noisydata)))

    # calculate thresholded coefficients
    for i in range(1,len(coeff)):
        coeff[i] = pywt.threshold(coeff[i], value=threshold, mode='soft')

    # reconstruct the signal using the thresholded coefficients
    denoised = pywt.waverec(coeff, wavelet)

    # return the value of denoised except for the last value
    if (len(noisydata) % 2) == 0:
        return denoised, sigma
    else:
        return denoised[:-1], sigma

# plot noisy and denoised data
def WaveletPlot(x, noisydata, denoised, title=None):
    f, ax = plt.subplots()
    ax.plot(x, noisydata, color='b', alpha=0.5)
    ax.plot(x, denoised, color='b')
    if title:
        ax.set_title(title)
    ax.set_xlim(min(x), max(x))
    plt.show()

def DenoiseSpectrum(x, ynormed, thresh_mod=1.5, level=2):
    # find and remove muons
    muonrec, indices = WaveletMuon(ynormed, thresh_mod)
    if len(indices[0]) > 0:
        ymuon = RemoveMuon(x, ynormed, indices)
    else:
        ymuon = np.copy(ynormed)
    # smooth the spectra
    yrec, sigma = WaveletSmooth(ymuon, level=level)

    return ymuon, yrec

def renormalize(y, ymuon):
    return (y - np.min(ymuon)) / (np.max(ymuon) - np.min(ymuon))

if __name__ == '__main__':
    # name of the spectra to be analyzed
    label = sys.argv[1]

    # initialize data
    x, y, maxyvalue = initialize(label + '/' + label + '_0017.txt')

    # denoise spectrum
    ymuon, yrec = DenoiseSpectrum(x, y)

    region = range(700, 900)

    # plot muonfree and reconstructed spectra
    WaveletPlot(x[region], renormalize(ymuon[region], ymuon[region]), renormalize(yrec[region], ymuon[region]))
