import os
import re
import shutil

from spectrum_analysis import mapping as mp
from spectrum_analysis import data

map = mp.mapping(foldername='testdata', plot=True)
mapdims = (4, 4)
step = 10

# plot ratios
top = 'breit_wigner_p1_height'
bot = 'lorentzian_p1_height'
div = 'div'

# get and plot raw data
x, y = data.GetAllData(map.listOfFiles)
map.PlotMapping('raw', y, mapdims, step, x=x, xmin=1300, xmax=1400)

# plot all colormaps
#map.PlotAllColormaps('raw', y, mapdims, step, x=x, xmin=1300, xmax=1400)

# get fit data and plot one map per peak parameter
peakFileList, numberOfPeakFiles = data.GetFolderContent(map.pardir_peak,
                                                        filetype='dat',
                                                        object='peakparameter')

parameters, errors = data.GetAllData(peakFileList)
peakList = map.CreatePeakList(peakFileList)

def PlotParameterMappings():
    for i, mapping in enumerate(peakList):
        map.PlotMapping('params', parameters[i], mapdims, step, name=mapping)
        # calculate relative error, set missing values and plot errors
        relative_error = map.ModifyValues(errors[i], parameters[i], 'div')
        map.PlotMapping('errs', relative_error, mapdims, step, name=mapping,
                        numbered=True)
PlotParameterMappings()

def PlotParameterOperations(first, second, operation):
    a = peakList.index(first)
    b = peakList.index(second)
    ratio = map.ModifyValues(parameters[a], parameters[b], operation)
    filename = first + '_' + operation + '_' + second
    map.PlotMapping(operation, ratio, mapdims, step, name=filename,
                    numbered=False)
PlotParameterOperations(top, bot, div)
