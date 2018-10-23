import sys
import pywt                             # for wavelet operations
import numpy as np

from statsmodels.robust import mad      # median absolute deviation from array
from scipy.optimize import curve_fit    # for interpolating muons

from functions_fitting import *

# detect muons for removal and returns non vanishing indices
def WaveletMuon(noisydata, wavelet='sym8', level=1):
    # calculate wavelet coefficients
    coeff = pywt.wavedec(noisydata, wavelet)    # symmetric signal extension mode

    # calculate a threshold
    sigma = mad(coeff[-level])
    threshold = sigma * np.sqrt(2 * np.log(len(noisydata)))

    # detect spikes on D1 details (written in the last entry of coeff)
    # calculate thresholded coefficients
    for i in range(1, len(coeff)):
        coeff[i] = pywt.threshold(coeff[i], value=threshold, mode='soft')
    # set everything but D1 level to zero
    for i in range(0, len(coeff)-1):
        coeff[i] = np.zeros_like(coeff[i])

    # reconstruct the signal using the thresholded coefficients
    denoised = pywt.waverec(coeff, wavelet)

    # get non vanishing indices
    indices = np.nonzero(denoised[:-1])[0]

    # return the value of denoised and the non vanishing indices
    return denoised[:-1], indices

# linear function for muon approximation
def linear(x, m, b):
    return x * m + b

# approximate muon by linear function and return modified y
def RemoveMuon(xdata, y, indices):
    # prevent python from working on original data
    ydata = np.copy(y)

    # calculate limits for indices to use for fitting
    limit = int(len(indices)/4)
    lower = indices[:limit]
    upper = indices[-limit:]
    fit_indices = np.append(lower, upper)

    # fit to the data
    popt, pcov = curve_fit(linear, xdata[fit_indices], ydata[fit_indices])

    # calculate approximated y values and remove muon
    for index in indices[limit:-limit]:
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

if __name__ == '__main__':
    # name of the spectra to be analyzed
    label = sys.argv[1]
    # level of smoothing
    level = int(sys.argv[2])

    # initialize data
    x, y, maxyvalue = initialize(label + '/data_' + label + '.txt')
    # find and remove muons
    muonrec, indices = WaveletMuon(y)
    if len(indices) > 0 :
        ymuon = RemoveMuon(x, y, indices)
    else:
        ymuon = np.copy(y)
    # smooth the spectra
    yrec, sigma = WaveletSmooth(ymuon, level=level)

    print('level: ' + str(level))
    print('sigma: ' + str(sigma))
    # plot muonfree and reconstructed spectra
    WaveletPlot(x, ymuon, yrec)
