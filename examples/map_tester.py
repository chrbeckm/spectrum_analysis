import os
from spectrum_analysis import mapping as mp
from spectrum_analysis import data

import matplotlib.pyplot as plt

peaks=['breit_wigner', 'lorentzian']

mapp = mp.mapping(foldername=os.path.join('testdata', '1'))

# get x- and y-data
x, y = data.GetAllData(mapp.listOfFiles)

# plot raw data
mapp.PlotAllRawSpectra(x, y)

# reduce the data to the region of interest
x_red, y_red = mapp.ReduceAllRegions(x, y)

# remove muons
y_cleaned = mapp.RemoveAllMuons(x_red, y_red)

# select and fit baseline
xregion = mapp.SelectAllBaselines(x_red, y_cleaned)
basefits = mapp.FitAllBaselines(x_red, y_cleaned, xregion)
baselines = mapp.EvaluateAllBaselines(x_red, basefits)
y_basefree = y_cleaned - baselines

# smooth the baseline corrected spectrum
y_smooth = mapp.WaveletSmoothAll(y_basefree)

# normalize the baseline free and smoothed spectra
y_bfn, ymax = mapp.NormalizeAll(y_basefree)
#y_sn, ignore = mapp.NormalizeAll(y_smooth, ymax=ymax)

# select peaks for fitting
mapp.SelectAllPeaks(x_red, y_bfn, peaks=peaks)

# fit the spectrum
fitresults = mapp.FitAllSpectra(x_red, y_bfn, peaks=peaks)

mapp.PlotAllFits(x_red, y_bfn, ymax, fitresults)

mapp.SaveAllFitParams(ymax, fitresults, peaks=peaks)
mapp.SaveAllBackgrounds(basefits, fitresults, ymax, peaks=peaks)
