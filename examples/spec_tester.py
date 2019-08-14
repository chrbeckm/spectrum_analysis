from spectrum_analysis import spectrum as sp

import matplotlib.pyplot as plt

peaks=['breit_wigner', 'lorentzian']

spec = sp.spectrum(filename='testdata/0005')

# get x- and y-data
x, y = spec.GetData()

# reduce the data to the region of interest
xmin, xmax = spec.SelectRegion(x, y)
x_red, y_red = spec.ReduceRegion(x, y, xmin, xmax)

# remove muons
y_cleaned = spec.RemoveMuons(x_red, y_red)

# select and fit baseline
xregion = spec.SelectBaseline(x_red, y_cleaned)
baseline = spec.FitBaseline(x_red, y_cleaned, xregion)
y_basefree = y_cleaned - baseline

# smooth the baseline corrected spectrum
y_smooth = spec.WaveletSmooth(y_basefree)

# normalize the baseline free and smoothed spectra
y_bfn, ymax = spec.Normalize(y_basefree)
#y_sn, ignore = spec.Normalize(y_smooth, ymax=ymax)

# select peaks for fitting
spec.SelectPeaks(x_red, y_bfn, peaks=peaks)

# fit the spectrum
fitresults = spec.FitSpectrum(x_red, y_bfn, peaks=peaks)

spec.PlotFit(x_red, y_bfn, ymax, fitresults, show=True)

spec.SaveFitParams(ymax, fitresults, peaks=peaks)
