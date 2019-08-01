from spectrum import *

# possible peaks (breit_wigner == fano)
# implemented are: breit_wigner, lorentzian, gaussian, voigt
#peaks = ['voigt']

# select folder you want to analyze and initialize everything
# it doesn't matter if there is one or more files in the folder
spec = spectrum('smallmap')
# choose the spectrum you want to analyze
spectrum = 2

# calculate the correct values
spectrum = spectrum - 1

# Select the interesting region in the spectrum,
# by clicking on the plot
spec.SelectSpectrum(spectrum=spectrum)

# find all Muons and remove them
spec.DetectMuonsWavelet(spectrum=spectrum)
spec.RemoveMuons(spectrum=spectrum)

# normalize all spectra
spec.NormalizeAll()

# denoise spectrum and save the plot
spec.WaveletSmoothSpectrum(spectrum=spectrum, sav=True)
spec.PlotSmoothed(spectrum=spectrum)
