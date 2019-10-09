import os
import re
import shutil

from spectrum_analysis import mapping as mp
from spectrum_analysis import data

map = mp.mapping(foldername='testdata', plot=True)
mapdims = (4, 4)
step = 10

# get and plot raw data
x, y = data.GetAllData(map.listOfFiles)
map.PlotMapping('raw', y, mapdims, step, x=x, xmin=1300, xmax=1400)

# plot all colormaps
#map.PlotAllColormaps('raw', y, mapdims, step, x=x, xmin=1300, xmax=1400)

# get fit data and plot one map per peak parameter
peakList, numberOfPeakFiles = data.GetFolderContent(map.pardir_peak,
                                                    filetype='dat',
                                                    object='peakparameter')

parameters, errors = data.GetAllData(peakList)
missingvalue = map.missingvalue

for i, mapping in enumerate(peakList):
    mapping = re.sub(map.pardir_peak, '', mapping)
    mapping = re.sub('.dat', '', mapping)
    map.PlotMapping('params', parameters[i], mapdims, step, name=mapping)
    # calculate relative error, set missing values and plot errors
    relative_error = errors[i]/parameters[i]
    missingindices = [i for i, x in enumerate(errors[i]) if (x == missingvalue)]
    for index in missingindices:
        relative_error[index] = missingvalue
    map.PlotMapping('errs', relative_error, mapdims, step, name=mapping,
                    numbered=True)
