import os
import re
import shutil

import numpy as np

from spectrum_analysis import mapping as mp
from spectrum_analysis import data

mapFolderList = ['testdata']
mapdims = (4, 4)
step = 10

# plot ratios
top = 'lorentzian_p1_height'
bot = 'breit_wigner_p1_height'
opt = 'div'

dict_minmax = {}

def PlotParameterMappings(params, peakList, name='', dict=None):
    """
    Plot all parameters of a mapping.
    """
    for i, mapping in enumerate(peakList):
        vmin = None
        vmax = None
        if dict is not None:
            vmin = dict[mapping][0]
            vmax = dict[mapping][1]
        map.PlotMapping('params', params[i], mapdims, step, name=name + mapping,
                        vmin=vmin, vmax=vmax)

def PlotErrorMappings(params, errors, peakList):
    """
    Plot all relative errors of a mapping.
    """
    for i, mapping in enumerate(peakList):
        # calculate relative error, set missing values and plot errors
        relative_error = map.ModifyValues(errors[i], params[i], 'div')
        map.PlotMapping('errs', relative_error, mapdims, step, name=mapping,
                        numbered=True)

def PlotParameterOperations(params, peakList, first, second, operation):
    """
    Plot a mapping calculated from two parameters (like height_a/height_b).
    """
    a = peakList.index(first)
    b = peakList.index(second)
    ratio = map.ModifyValues(params[a], params[b], operation)
    filename = first + '_' + operation + '_' + second
    map.PlotMapping(operation, ratio, mapdims, step, name=filename,
                    numbered=False)

def CreateMinMaxDict(params, paramList):
    """
    Create a dictionary containing all parameters with the global min and max.
    """
    for param in paramList:
        i = paramList.index(param)
        min = np.min(params[i])
        max = np.max(params[i])
        if param in dict_minmax:
            if dict_minmax[param][0] < min:
                min = dict_minmax[param][0]
            if dict_minmax[param][1] > max:
                max = dict_minmax[param][1]
        content = {param : (min, max)}
        dict_minmax.update(content)

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
    parameterList = map.CreatePeakList(peakFileList)
    PlotParameterMappings(parameters, parameterList)
    PlotErrorMappings(parameters, errors, parameterList)

    CreateMinMaxDict(parameters, parameterList)

    # plot one mapping calculated from two parameters linked by selected
    # operation (opt=['div', 'mult', 'add', 'sub']).
    PlotParameterOperations(parameters, parameterList, top, bot, opt)

    linebreaker ='============================================================'
    print(linebreaker + '\n' + linebreaker)

if len(mapFolderList) > 1:
    for folder in mapFolderList:
        print(folder + ' mappings with same scale are plotted now.')
        map = mp.mapping(foldername=folder, plot=True)

        # get fit data and plot one map per peak parameter
        peakFileList, numberOfPeakFiles = data.GetFolderContent(map.pardir_peak,
                                                        filetype='dat',
                                                        object='peakparameter')
        parameters, errors = data.GetAllData(peakFileList)
        parameterList = map.CreatePeakList(peakFileList)
        PlotParameterMappings(parameters, parameterList,
                              name='scaled_',
                              dict=dict_minmax)

        linebreaker ='============================================================'
        print(linebreaker + '\n' + linebreaker)
