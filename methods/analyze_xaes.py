from spectrum import *

# select folder you want to analyze and initialize everything
# it doesn't matter if there is one or more files in the folder
# define xps as measurement
spec = spectrum('0XPS/highres_450/01_1', measurement='xps')

# Select the interesting region in the spectrum,
# by clicking on the plot
spec.SelectSpectrum()

spec.WaveletSmoothAllSpectra(level=0, sav=True)

spec.PlotAllXaes()
