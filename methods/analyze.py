from functions import *
import sys
import time



def analyze(label):
    # Combination of the methods provided by functions.py


    x, y, maxyvalue = initialize(label + '/data_' + label + '.txt')
    # function that initializes data for evaluation

    xred, yred = SelectSpectrum(x, y, label)
    # Select the interesting region in the spectrum, by clicking on the plot

    baselinefile = SelectBaseline(xred, yred, label)
    # Function opens a window with the data,
    # you can select the regions that do not belong to the (linear!)
    # background signal by clicking in the plot

    fitresult_background = Fitbaseline(xred, yred, baselinefile, show = False)

    SelectPeaks(xred, yred, fitresult_background, label)
    # Function that opens a Window with the data,
    # you can choose initial values for the peaks by clicking on the plot.

    fitresult_peaks = FitSpectrum(xred, yred, maxyvalue, fitresult_background, label)
    # Fit Spectrum with initial values provided by SelectBaseline()
    # and SelectPeaks()

    SaveFitParams(xred, yred, maxyvalue, fitresult_peaks, fitresult_background, label)
    #Save the Results of the fit in a .zip file using numpy.savez().

    DeleteTempFiles(label)


if __name__ == '__main__':
    label = sys.argv[1] 
    # label is the typed in name of the data file to analyze
    analyze(label)
