from functions_fitting import *
import sys
import time

# possible peaks
peaks = ['voigt', 'fano', 'lorentzian', 'gaussian']

# Combination of the methods provided by functions_fitting.py
def analyze(label):

    # function that initializes data for evaluation
    x, y, maxyvalue = initialize(label + '/data_' + label + '.txt')

    # Select the interesting region in the spectrum, by clicking on the plot
    xred, yred = SelectSpectrum(x, y, label)

    # Function opens a window with the data,
    # you can select the regions that do not belong to
    # the third degree polynominal background signal
    # by clicking in the plot
    baselinefile = SelectBaseline(xred, yred, label)

    # fit the baseline
    fitresult_background = FitBaseline(xred, yred, baselinefile, show = False)

    # Function that opens a Window with the data,
    # you can choose initial values for the peaks by clicking on the plot.
    SelectPeaks(xred, yred, fitresult_background, label, peaks)

    # Fit Spectrum with initial values provided by SelectBaseline()
    # and SelectPeaks()
    fitresult_peaks = FitSpectrum(xred, yred, maxyvalue,
                                  fitresult_background, label)

    # Save the Results of the fit in a .zip file using numpy.savez()
    # and in additional txt-files (in folder results_fitparameter)
    SaveFitParams(xred, yred, maxyvalue, fitresult_peaks,
                  fitresult_background, label)

    # delete temporary files
    DeleteTempFiles(label)


if __name__ == '__main__':
    label = sys.argv[1]
    # label is the typed in name of the data file to analyze
    analyze(label)
