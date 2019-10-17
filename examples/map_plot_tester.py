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

dict_minmax = {}

linebreaker ='============================================================'

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
    # go through all parameters
    for param in paramList:
        # get index of parameter and corresponding min and max
        i = paramList.index(param)
        nonMissing = [x for x in params[i] if not (x == map.missingvalue)]
        min = np.min(nonMissing)
        max = np.max(nonMissing)
        # check if parameter already in dictionary
        if param in dict_minmax:
            # check if parameter smaller/bigger than current value
            # and update values and mappings
            if ((dict_minmax[param][0] < min)
            and not (dict_minmax[param][0] == map.missingvalue)):
                min = dict_minmax[param][0]
                minfile = mapping
            else:
                minfile = dict_minmax[param][2]
            if dict_minmax[param][1] > max:
                max = dict_minmax[param][1]
                maxfile = mapping
            else:
                maxfile = dict_minmax[param][3]
        else:
            minfile = mapping
            maxfile = mapping
        # create content and update dictionary
        content = {param : (min, max, minfile, maxfile)}
        dict_minmax.update(content)

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

    CreateMinMaxDict(parameters, parameterList, folder)

    # plot one mapping calculated from two parameters linked by selected
    # operation (opt=['div', 'mult', 'add', 'sub']).
    PlotParameterOperations(parameters, parameterList, mapdims, step,
                            top, bot, opt)
    PlotParameterOperations(parameters, parameterList, mapdims, step,
                            bot, top, opt)

    print(linebreaker + '\n' + linebreaker)

print('List of minima and maxima and the mappings they are taken from.')
for key in dict_minmax.keys():
    print(key + '\n'
              + '\tMin: ' + str(dict_minmax[key][0])
              + ' ({})'.format(dict_minmax[key][2]) + '\n'
              + '\tMax: ' + str(dict_minmax[key][1])
              + '({})'.format(dict_minmax[key][3]))

print(linebreaker + '\n' + linebreaker)

if len(mapFolderList) > 1:
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
                              dict=dict_minmax)

        print(linebreaker + '\n' + linebreaker)
