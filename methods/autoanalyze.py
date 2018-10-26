from functions_fitting import *
import sys

# Method to analyze several spectra with equal shape.
# Run python 'autonalyze.py labels.txt'. Fit the first spectrum
# interactivilly, the next spectra will be analyzed automatically using
# the fit params of the previous spectrum as initial values for the next one.

labelfile = sys.argv[1]
with open(labelfile, 'r') as file:
	labels = file.readlines()
for i in range(len(labels)):
	labels[i] = labels[i].rstrip()

for i in range(len(labels)):
	label = labels[i]
	label = label.split(r'\n')[0]
	x, y, maxyvalue = initialize(label + '/data_' + label + '.txt')
	# show in terminal which spectrum is currently analyzed
	print('\n ----- ANALYZING ' + label + ' ----- \n')

	# analyze the first spectrum
	if i == 0:
		xred, yred = SelectSpectrum(x, y, label)
		baselinefile = SelectBaseline(xred, yred, label)
		fitresult_background = Fitbaseline(xred, yred, baselinefile, show = False)
		SelectPeaks(xred, yred, fitresult_background, label)
		fitresult_peaks = FitSpectrum(xred, yred, maxyvalue, fitresult_background, label)
		SaveFitParams(xred, yred, maxyvalue, fitresult_peaks, fitresult_background, label)

	# analyze all following spectra with initial values from the previous spectrum
	else:
		fitresult_peaks, fitresult_background = FitSpectrumInit(x, y, maxyvalue, labels[i - 1], label, baselinefile)
		SaveFitParams(x, y, maxyvalue, fitresult_peaks, fitresult_background, label)
