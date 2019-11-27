from spectrum_analysis import spectrum as sp
from spectrum_analysis import data

import matplotlib.pyplot as plt

peaks = ['gaussian', 'lorentzian']

spec = sp.spectrum('testdata/1/0005')

# get x- and y-data
x, y = data.GetData(spec.file)

# reduce the data to the region of interest
region = spec.SelectRegion(x, y)
xmin, xmax = spec.ExtractRegion(x, region)
x_red, y_red = spec.ReduceRegion(x, y, xmin, xmax)

# remove muons
y_cleaned = spec.RemoveMuons(x_red, y_red)

# select and fit baseline
xregion = spec.SelectBaseline(x_red, y_cleaned)
basefit = spec.FitBaseline(x_red, y_cleaned, xregion)
baseline = spec.EvaluateBaseline(x_red, basefit)
y_basefree = y_cleaned - baseline

# smooth the baseline corrected spectrum
y_smooth = spec.WaveletSmooth(y_basefree)

# normalize the baseline free and smoothed spectra
y_bfn, ymax = spec.Normalize(y_basefree)
#y_sn, ignore = spec.Normalize(y_smooth, ymax=ymax)

# remove frequency
x_fft, y_fft = spec.SelectFrequency(x_red, y_bfn)
y_freqfree = spec.RemoveFrequency(x_fft, y_fft, prnt=True)

# select peaks for fitting
spec.SelectPeaks(x_red, y_freqfree, peaks=peaks)

# fit the spectrum
fitresults = spec.FitSpectrum(x_red, y_freqfree, peaks=peaks)

spec.PlotFit(x_red, y_freqfree, ymax, fitresults, show=True)

spec.SaveFitParams(ymax, fitresults, peaks=peaks)
spec.SaveBackground(basefit, fitresults, ymax)
