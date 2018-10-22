import sys
import pywt                         # for wavelet operations
import numpy as np
from statsmodels.robust import mad # median absolute deviation from array

from functions_fitting import *

def WaveletSmooth(noisydata, wavelet="sym8", level=1):
    # calculate wavelet coefficients
    coeff = pywt.wavedec(noisydata, wavelet, mode="per")

    # calculate a threshold
    sigma = mad(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(noisydata)))

    # calculate thresholded coefficients
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])

    # reconstruct the signal using the thresholded coefficients
    denoised = pywt.waverec(coeff, wavelet, mode="per")

    return denoised, sigma

def WaveletPlot(noisydata, denoised, title=None):
    f, ax = plt.subplots()
    ax.plot(noisydata, color="b", alpha=0.5)
    ax.plot(denoised, color="b")
    if title:
        ax.set_title(title)
    ax.set_xlim((0, len(denoised)))
    plt.show()

if __name__ == '__main__':
    label = sys.argv[1]
    level = int(sys.argv[2])

    x, y, maxyvalue = initialize(label + '/' + label + '_0001.txt')
    yrec, sigma = WaveletSmooth(y, level=level)

    print("level: " + str(level))
    print("sigma: " + str(sigma))
    WaveletPlot(y, yrec)
