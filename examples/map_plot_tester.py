import os
import re
import shutil

from spectrum_analysis import mapping as mp
from spectrum_analysis import data

map = mp.mapping(foldername='testdata', plot=True)

# get x- and y-data
x, y = data.GetAllData(map.listOfFiles)

map.PlotMapping(x, y, 3, 3, 10, 1300, 1400)

# get fit data, one map per peak parameter
peakList, numberOfPeakFiles = data.GetFolderContent(map.pardir_peak,
                                                         filetype='dat')

parameters, errors = data.GetAllData(peakList)

for mapping in peakList:
    mapping = re.sub(map.pardir_peak, '', mapping)
    mapping = re.sub('.dat', '', mapping)
    print(mapping)
    #map.PlotMapping(parameters, 3, 3, 10)
