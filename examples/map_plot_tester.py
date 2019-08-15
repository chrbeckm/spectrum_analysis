import os
import re
import shutil

from spectrum_analysis import mapping as mp
from spectrum_analysis import data

map = mp.mapping(foldername='testdata', plot=True)
mapdims = (3, 3)
step = 10

# get and plot raw data
x, y = data.GetAllData(map.listOfFiles)
map.PlotMapping('raw', y, mapdims, step, x=x, xmin=1300, xmax=1400)

# get fit data and plot one map per peak parameter
peakList, numberOfPeakFiles = data.GetFolderContent(map.pardir_peak,
                                                    filetype='dat',
                                                    object='peakparameter')

parameters, errors = data.GetAllData(peakList)

for i, mapping in enumerate(peakList):
    mapping = re.sub(map.pardir_peak, '', mapping)
    mapping = re.sub('.dat', '', mapping)
    map.PlotMapping('params', parameters[i], mapdims, step, name=mapping)
