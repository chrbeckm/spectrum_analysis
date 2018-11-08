from spectrum import *

# possible peaks (breit_wigner == fano)
# implemented are: breit_wigner, lorentzian, gaussian, voigt
peaks = ['breit_wigner', 'lorentzian']

# select folder you want to analyze and initialize everything
# it doesn't matter if there is one or more files in the folder
spec = spectrum('smallmap')

# Select the interesting region in the spectrum,
# by clicking on the plot
spec.SelectSpectrum()

# Function opens a window with the data,
# you can select the regions that do not belong to
# the third degree polynominal background signal
# by clicking in the plot
spec.SelectBaseline()

# fit the baselines
spec.FitAllBaselines()

# Function that opens a Window with the data,
# you can choose initial values for the peaks by clicking on the plot.
# You have to choose peaks for all spectra to get the proper starting
# values. -> Improvement needed
spec.SelectAllPeaks(peaks)

# Fit all spectra with initial values provided by SelectBaseline()
# and SelectAllPeaks()
spec.FitAllSpectra(peaks)

# Save the results of the fit in txt-files
spec.SaveAllFitParams(peaks)

# plot mapping
# input values are
# xdim:     the number of Spectra in x direction
# ydim:     the number of Spectra in y direction
# stepsize: the interval at which the mapping was collected in µm
# xmin:     the lowest wavenumber to be used in the mapping
# xmax:     the highest wavenumber to be used in the mapping
#spec.PlotMapping(2, 2, 10, 1550, 1620)
