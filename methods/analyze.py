from functions_fitting import *
import sys
import time



def analyze(label):
    # Combination of the methods provided by functions.py

    x, y, maxyvalue = initialize(label + '/data_' + label + '.txt')
    xred, yred = SelectSpectrum(x, y, label)
    baselinefile = SelectBaseline(xred, yred, label)
    SelectPeaks(xred, yred, label)
    fitresult = FitSpectrum(xred, yred, maxyvalue, label)
    SaveFitParams(xred, yred, maxyvalue, fitresult, label)


if __name__ == '__main__':
    label = sys.argv[1]
    analyze(label)
