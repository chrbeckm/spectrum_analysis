import os
import re
import shutil

from spectrum_analysis import mapping as mp
from spectrum_analysis import data

mapFolderList = ['testdata']
mapdims = (4, 4)
step = 10

# plot ratios
top = 'lorentzian_p1_height'
bot = 'breit_wigner_p1_height'
opt = 'div'

def PlotParameterMappings(peakList):
    """
    Plot all parameters and the corresponding relative errors of a mapping.
    """
    for i, mapping in enumerate(peakList):
        map.PlotMapping('params', parameters[i], mapdims, step, name=mapping)
        # calculate relative error, set missing values and plot errors
        relative_error = map.ModifyValues(errors[i], parameters[i], 'div')
        map.PlotMapping('errs', relative_error, mapdims, step, name=mapping,
                        numbered=True)

def PlotParameterOperations(first, second, operation):
    """
    Plot a mapping calculated from two parameters (like height_a/height_b).
    """
    a = peakList.index(first)
    b = peakList.index(second)
    ratio = map.ModifyValues(parameters[a], parameters[b], operation)
    filename = first + '_' + operation + '_' + second
    map.PlotMapping(operation, ratio, mapdims, step, name=filename,
                    numbered=False)

for folder in mapFolderList:
    print(folder + ' mappings are plotted now.')
    map = mp.mapping(foldername=folder, plot=True)

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
    PlotParameterMappings(peakList)

    # plot one mapping calculated by selected option
    # (opt=['div', 'mult', 'add', 'sub']).
    PlotParameterOperations(top, bot, opt)

    linebreaker ='============================================================'
    print(linebreaker + '\n' + linebreaker)
