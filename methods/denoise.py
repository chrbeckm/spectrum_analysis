import pywt                         # for wavelet operations
import numpy as np
import statsmodels.robust import mad # median absolute deviation from array


def waveletSmooth(x, wavelet="db4", level=1, title=None):
    # calculate wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")

    # calculate a threshold
    sigma = mad(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))

    # calculate thresholded coefficients
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])

    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec(coeff, wavelet, mode="per")

    f, ax = plt.subplots()
    plot(x, color="b", alpha=0.5)
    plot(y, color="b")
    if title:
        ax.set_title(title)
    ax.set_xlim((0, len(y)))
