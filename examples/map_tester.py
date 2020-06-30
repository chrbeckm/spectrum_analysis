import os
from spectrum_analysis import mapping as mp
from spectrum_analysis import data

import matplotlib.pyplot as plt

peaks=['breit_wigner', 'lorentzian']

map = mp.mapping(foldername=os.path.join('testdata', '1'))

# get x- and y-data
x, y = data.GetAllData(map.listOfFiles)

# plot raw data
map.PlotAllRawSpectra(x, y)

# reduce the data to the region of interest
x_red, y_red = map.ReduceAllRegions(x, y)

# remove muons
y_cleaned = map.RemoveAllMuons(x_red, y_red)

# select and fit baseline
xregion = map.SelectAllBaselines(x_red, y_cleaned)
basefits = map.FitAllBaselines(x_red, y_cleaned, xregion)
baselines = map.EvaluateAllBaselines(x_red, basefits)
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
map.SaveAllBackgrounds(basefits, fitresults, ymax, peaks=peaks)
