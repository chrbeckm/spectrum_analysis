from spectrum_analysis import mapping as mp

import matplotlib.pyplot as plt

peaks=['breit_wigner', 'lorentzian']

map = mp.mapping(foldername='testdata')

# get x- and y-data
x, y = map.GetAllData()

# reduce the data to the region of interest
x_red, y_red = map.ReduceAllRegions(x, y)

# remove muons
y_cleaned = map.RemoveAllMuons(x_red, y_red)

# select and fit baseline
xregion = map.SelectAllBaselines(x_red, y_cleaned)
baselines = map.FitAllBaselines(x_red, y_cleaned, xregion)
y_basefree = y_cleaned - baselines

# smooth the baseline corrected spectrum
y_smooth = map.WaveletSmoothAll(y_basefree)

# normalize the baseline free and smoothed spectra
y_bfn, ymax = map.NormalizeAll(y_basefree)
#y_sn, ignore = map.NormalizeAll(y_smooth, ymax=ymax)

# select peaks for fitting
map.SelectAllPeaks(x_red, y_bfn, peaks=peaks)

# fit the spectrum
fitresults = map.FitAllSpectra(x_red, y_bfn, peaks=peaks)

map.PlotAllFits(x_red, y_bfn, ymax, fitresults)

map.SaveAllFitParams(ymax, fitresults, peaks=peaks)
