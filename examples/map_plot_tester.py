import os
import re
import shutil

import numpy as np

from spectrum_analysis import mapping as mp
from spectrum_analysis import data

mapFolderList = ['testdata/1',
#                 'testdata/2'
                 ]
dims = [(4, 4),
#        (8, 2)
        ]
stepsize = [10,
#           10
            ]

# plot ratios
top = 'lorentzian_p1_height'
bot = 'breit_wigner_p1_height'
opt = 'div'

dict_minmax_global = {}

linebreaker ='============================================================'

def CalculateSpectraNumber(dimensions):
    sum = 0
    for spectrum in dimensions:
        sum += spectrum[0] * spectrum[1]
    return sum

def PlotParameterMappings(params, peakList, mapdims, step, name='', dict=None):
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

def PlotErrorMappings(params, errors, peakList, mapdims, step):
    """
    Plot all relative errors of a mapping.
    """
    for i, mapping in enumerate(peakList):
        # calculate relative error, set missing values and plot errors
        relative_error = map.ModifyValues(errors[i], params[i], 'div')
        map.PlotMapping('errs', relative_error, mapdims, step, name=mapping,
                        numbered=True)

def PlotParameterOperations(params, peakList, mapdims, step,
                            first, second, operation):
    """
    Plot a mapping calculated from two parameters (like height_a/height_b).
    """
    a = peakList.index(first)
    b = peakList.index(second)
    ratio = map.ModifyValues(params[a], params[b], operation)
    filename = first + '_' + operation + '_' + second
    map.PlotMapping(operation, ratio, mapdims, step, name=filename,
                    numbered=False)

def CreateMinMaxDict(params, paramList, mapping):
    """
    Create a dictionary containing all parameters with the global min and max.
    """
    dict = {}
    # go through all parameters
    for param in paramList:
        # get index of parameter and corresponding min and max
        i = paramList.index(param)
        nonMissing = [x for x in params[i] if not (x == map.missingvalue)]
        min = np.min(nonMissing)
        max = np.max(nonMissing)
        minfile = mapping
        maxfile = mapping
        # create content and update dictionary
        content = {param : (min, max, minfile, maxfile)}
        dict.update(content)
    return dict

def UpdateGlobalDict(globaldict, dict):
    for param in dict.keys():
        min = dict[param][0]
        max = dict[param][1]
        minfile = dict[param][2]
        maxfile = dict[param][3]
        # check if parameter already in dictionary
        if param in globaldict:
            # check if parameter smaller/bigger than current value
            # and update values and mappings
            if ((globaldict[param][0] > min)
            and not (globaldict[param][0] == map.missingvalue)):
                pass
            else:
                min = globaldict[param][0]
                minfile = globaldict[param][2]
            if globaldict[param][1] < max:
                pass
            else:
                max = globaldict[param][1]
                maxfile = globaldict[param][3]
        content = {param : (min, max, minfile, maxfile)}
        globaldict.update(content)
    return globaldict

def PrintMinMax(dict, list):
    for param in list:
        print(param + '\n'
                    + '\tMin: ' + str(dict[param][0])
                    + ' ({})'.format(dict[param][2]) + '\n'
                    + '\tMax: ' + str(dict[param][1])
                    + ' ({})'.format(dict[param][3]))

print('There are ' + str(CalculateSpectraNumber(dims)) + ' spectra at all.')
print(linebreaker + '\n' + linebreaker)

for folder in mapFolderList:
    print('Mapping ' + str(mapFolderList.index(folder) + 1) + ' of '
        + str(len(mapFolderList)) + '\n')
    print(folder + ' mappings are plotted now.')
    mapdims = dims[mapFolderList.index(folder)]
    step = stepsize[mapFolderList.index(folder)]
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
    PlotParameterMappings(parameters, parameterList, mapdims, step)
    PlotErrorMappings(parameters, errors, parameterList, mapdims, step)

    dict_minmax = CreateMinMaxDict(parameters, parameterList, folder)
    dict_minmax_global = UpdateGlobalDict(dict_minmax_global, dict_minmax)

    # plot one mapping calculated from two parameters linked by selected
    # operation (opt=['div', 'mult', 'add', 'sub']).
    PlotParameterOperations(parameters, parameterList, mapdims, step,
                            top, bot, opt)
    PlotParameterOperations(parameters, parameterList, mapdims, step,
                            bot, top, opt)

    print('\nList of minima and maxima.')
    PrintMinMax(dict_minmax, parameterList)

    print(linebreaker + '\n' + linebreaker)

if len(mapFolderList) > 1:
    print('List of global minima and maxima '
        + 'and the mappings they are taken from.')
    PrintMinMax(dict_minmax_global, dict_minmax_global.keys())
    print(linebreaker + '\n' + linebreaker)

    for folder in mapFolderList:
        print('Scaled mapping ' + str(mapFolderList.index(folder) + 1) + ' of '
            + str(len(mapFolderList)) + '\n')
        print(folder + ' mappings with same scale are plotted now.')
        mapdims = dims[mapFolderList.index(folder)]
        step = stepsize[mapFolderList.index(folder)]
        map = mp.mapping(foldername=folder, plot=True)

        # get fit data and plot one map per peak parameter
        peakFileList, numberOfPeakFiles = data.GetFolderContent(map.pardir_peak,
                                                        filetype='dat',
                                                        object='peakparameter')
        parameters, errors = data.GetAllData(peakFileList)
        parameterList = map.CreatePeakList(peakFileList)
        PlotParameterMappings(parameters, parameterList, mapdims, step,
                              name='scaled_',
                              dict=dict_minmax_global)

        print(linebreaker + '\n' + linebreaker)
