from functions import *
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
	x, y = initialize(label + '/data_' + label + '.txt')
	print('analyzing ' + label)

	if i == 0:
		xred, yred = SelectSpectrum(x, y, label)
		baselinefile = SelectBaseline(xred, yred, label)
		SelectPeaks(xred, yred, label)
		fitresult = FitSpectrum(xred, yred, label)
		SaveFitParams(xred, yred, fitresult, label)

	else:
		fitresult = FitSpectrumInit(x, y, labels[i - 1], label)
		SaveFitParams(x, y, fitresult, label)
