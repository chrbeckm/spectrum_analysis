from spectrum import *

# possible peaks (breit_wigner == fano)
# implemented are: breit_wigner, lorentzian, gaussian, voigt
peaks = ['voigt']

# select folder you want to analyze and initialize everything
# it doesn't matter if there is one or more files in the folder
spec = spectrum('smallmap')
# choose the spectrum you want to analyze
spectrum = 2

# calculate the correct values
spectrum = spectrum - 1
label = str(spectrum + 1).zfill(4)

# Select the interesting region in the spectrum,
# by clicking on the plot
spec.SelectSpectrum(spectrum=spectrum)

# find all Muons and remove them
spec.DetectMuonsWavelet(spectrum=spectrum)
spec.RemoveMuons(spectrum=spectrum)

# Function opens a window with the data,
# you can select the regions that do not belong to
# the third degree polynominal background signal
# by clicking in the plot
spec.SelectBaseline(spectrum=spectrum)

# fit the baselines
spec.FitBaseline(spectrum=spectrum, degree=1)

# select a frequency of the fourier spectrum (only needed if there is one)
spec.SelectFrequency(spectrum=spectrum)
# remove the selected and unwanted frequency (only needed if there is one)
spec.RemoveFrequency(spectrum=spectrum)

# Function that opens a Window with the data,
# you can choose initial values for the peaks by clicking on the plot.
# You have to choose peaks for all spectra to get the proper starting
# values. -> Improvement needed
spec.SelectPeaks(peaks, spectrum=spectrum, label=label)

# Fit all spectra with initial values provided by SelectBaseline()
# and SelectAllPeaks()
# if you need to see the fit results set report to True,
# otherwise set it to false
spec.FitSpectrum(peaks, report=False, spectrum=spectrum, label=label)

# Save the results of the fit in txt-files
spec.SaveFitParams(peaks, spectrum=spectrum, label=label)
