from spectrum import *

# possible peaks (breit_wigner == fano)
# implemented are: breit_wigner, lorentzian, gaussian, voigt
#peaks = ['voigt']

# select folder you want to analyze and initialize everything
# it doesn't matter if there is one or more files in the folder
spec = spectrum('smallmap')

# Select the interesting region in the spectrum,
# by clicking on the plot
spec.SelectSpectrum()

# find all Muons and remove them
spec.DetectAllMuons()
spec.RemoveAllMuons()

# normalize all spectra
spec.NormalizeAll()

spec.SelectBaseline()
spec.FitAllBaselines()

# denoise all spectra and save them
spec.WaveletSmoothAllSpectra(sav=True)
spec.PlotAllSmoothedSpectra()
