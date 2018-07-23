from functions import *
import sys

# Run python 'autoplot.py labels.txt'. Creates simple plots of the spectra.

label = sys.argv[1]
with open('labels.txt', 'r') as file:
	labels = file.readlines()
for i in range(len(labels)):
	labels[i] = labels[i].rstrip()

for label in labels:
	label = label.split(r'\n')[0]
	x, y = initialize(label + '/data_' + label + '.txt')
	plt.clf()
	plt.plot(x, y, 'b-')
	plt.xlim(x[0], 550)
	plt.savefig(label + '/simpleplot_' + label + '.pdf')
