import os
import re
import shutil

import numpy as np

from spectrum_analysis import mapping as mp
from spectrum_analysis import data
from peaknames import *

mapFolderList = ['testdata/1',
#                 'testdata/2'
                 ]
dims = [(4, 4),
#        (8, 2)
        ]
stepsize = [10,
#            10
            ]

# images need to be in folders specified in mapFolderList
# best is to use png backgrounds, but jpgs work as well
backgrounds = ['bg_test.png',
#               'bg_test.jpg'
              ]

msizes = [2.0,
#          2.0
]

# True if background should be plotted
bg_plot = False

# True if additional plots should be created
# with the same scale for each parameter
scaled = False

# True if all raw spectra should be plotted
# careful if plotting many spectra. Your PC might freeze
plotrawspectra = False

# plot ratios
top = 'lorentzian_p1_height'
bot = 'breit_wigner_p1_height'
opt = 'div'

# plot peak distance
dist1 = 'breit_wigner_p1_center'
dist2 = 'lorentzian_p1_center'
subst = 'sub'

dict_minmax_global = {}

linebreaker ='============================================================'

def CalculateSpectraNumber(dimensions):
    sum = 0
    for spectrum in dimensions:
        sum += spectrum[0] * spectrum[1]
    return sum

def PlotParameterMappings(params, peakList, mapdims, step, background='',
                          msize=2.1, name='', dict=None, area=None):
    """
    Plot all parameters of a mapping.
    """
    for i, mapping in enumerate(peakList):
        vmin = None
        vmax = None
        if dict is not None:
            vmin = dict[mapping][0]
            vmax = dict[mapping][1]
        plot_matrix, plotname = map.PlotMapping('params',
                        params[i], mapdims, step,
                        name=name + mapping,
                        vmin=vmin, vmax=vmax, grid=False)
        map.PlotMapping('params', params[i], mapdims, step,
                        name=name + 'grid_' + mapping, alpha=0.75,
                        vmin=vmin, vmax=vmax, grid=True,
                        background=background, msize=msize,
                        plot_missing=False, area=area)
        map.PlotHistogram(plot_matrix, plotname)

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
                            first, second, operation, background='',
                            msize=2.1, name='', dict=None, area=None):
    """
    Plot a mapping calculated from two parameters (like height_a/height_b).
    """
    a = peakList.index(first)
    b = peakList.index(second)
    ratio = map.ModifyValues(params[a], params[b], operation)
    filename = first + '_' + operation + '_' + second
    vmin = None
    vmax = None
    if dict is not None:
        vmin = dict[filename][0]
        vmax = dict[filename][1]
    plot_matrix, plotname = map.PlotMapping(operation,
                    ratio, mapdims, step,
                    name=name + filename,
                    numbered=False, vmin=vmin, vmax=vmax, grid=False)
    map.PlotMapping(operation, ratio, mapdims, step,
                    name=name + 'grid_' + filename, alpha=0.75,
                    numbered=False, vmin=vmin, vmax=vmax, grid=True,
                    background=background, msize=msize,
                    plot_missing=False, area=area)
    map.PlotHistogram(plot_matrix, plotname)
    return filename, ratio

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
        # get index and correct to get the correct spectrum number
        min_idx = np.where(params[i] == min)
        max_idx = np.where(params[i] == max)
        min_idx = [i + 1 for i in min_idx[0]]
        max_idx = [i + 1 for i in max_idx[0]]
        # create content and update dictionary
        content = {param : (min, max, minfile, maxfile, min_idx, max_idx)}
        dict.update(content)
    return dict

def UpdateGlobalDict(globaldict, dict):
    for param in dict.keys():
        min = dict[param][0]
        max = dict[param][1]
        minfile = dict[param][2]
        maxfile = dict[param][3]
        min_idx = dict[param][4]
        max_idx = dict[param][5]
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
                min_idx = globaldict[param][4]
            if globaldict[param][1] < max:
                pass
            else:
                max = globaldict[param][1]
                maxfile = globaldict[param][3]
                max_idx = globaldict[param][5]
        content = {param : (min, max, minfile, maxfile, min_idx, max_idx)}
        globaldict.update(content)
    return globaldict

def PrintMinMax(dict, list):
    for param in list:
        print(param + '\n'
                    + '\tMin: ' + str(dict[param][0])
                    + ' ({})'.format(dict[param][2])
                    + '\tSpectra: ' + str(dict[param][4]) + '\n'
                    + '\tMax: ' + str(dict[param][1])
                    + ' ({})'.format(dict[param][3])
                    + '\tSpectra: ' + str(dict[param][5]))

print('There are ' + str(CalculateSpectraNumber(dims)) + ' spectra at all.')
print(linebreaker + '\n' + linebreaker)

for folder in mapFolderList:
    index = mapFolderList.index(folder)
    print('Mapping ' + str(index + 1) + ' of '
        + str(len(mapFolderList)) + '\n')
    print(folder + ' mappings are plotted now.')
    mapdims = dims[index]
    step = stepsize[index]
    background = folder + '/' + backgrounds[index]
    msize = msizes[index]

    map = mp.mapping(foldername=folder, plot=True, peaknames=peaknames)

    # get and plot raw data
    x, y = data.GetAllData(map.listOfFiles)
    map.PlotMapping('raw', y, mapdims, step, x=x, xmin=1300, xmax=1400)
    if plotrawspectra:
        map.PlotAllRawSpectra(x, y)

    # plot all colormaps
    #map.PlotAllColormaps('raw', y, mapdims, step, x=x, xmin=1300, xmax=1400)

    # get fit data and plot one map per peak parameter
    peakFileList, numberOfPeakFiles = data.GetFolderContent(map.pardir_peak,
                                                        filetype='dat',
                                                        object='peakparameter')
    parameters, errors = data.GetAllData(peakFileList)
    parameterList = map.CreatePeakList(peakFileList)

    # calculate area under the curve
    # area is used to scale the linewidth of the grid marker
    area = 0
    for i, parameter in enumerate(parameterList):
        test = re.findall('amplitude', parameter)
        if test:
            area += parameters[i]

    PlotParameterMappings(parameters, parameterList, mapdims, step,
                          background=background, msize=msize, area=area)
    PlotErrorMappings(parameters, errors, parameterList, mapdims, step)

    dict_minmax = CreateMinMaxDict(parameters, parameterList, folder)
    dict_minmax_global = UpdateGlobalDict(dict_minmax_global, dict_minmax)

    # plot one mapping calculated from two parameters linked by selected
    # operation (opt=['div', 'mult', 'add', 'sub']).
    parameter_name, values = PlotParameterOperations(parameters, parameterList,
                                                     mapdims, step,
                                                     top, bot, opt,
                                                     background=background,
                                                     msize=msize, area=area)
    dict_topbot = CreateMinMaxDict([values], [parameter_name], folder)
    dict_minmax_global = UpdateGlobalDict(dict_minmax_global, dict_topbot)

    parameter_name, values = PlotParameterOperations(parameters, parameterList,
                                                     mapdims, step,
                                                     bot, top, opt,
                                                     background=background,
                                                     msize=msize, area=area)
    dict_bottop = CreateMinMaxDict([values], [parameter_name], folder)
    dict_minmax_global = UpdateGlobalDict(dict_minmax_global, dict_bottop)

    parameter_name, values = PlotParameterOperations(parameters, parameterList,
                                                     mapdims, step,
                                                     dist1, dist2, subst,
                                                     background=background,
                                                     msize=msize, area=area)
    dict_topbot = CreateMinMaxDict([values], [parameter_name], folder)
    dict_minmax_global = UpdateGlobalDict(dict_minmax_global, dict_topbot)

    # plot background values from fits
    if bg_plot:
        peakFileList, numberOfPeakFiles = data.GetFolderContent(map.pardir_peak_bg,
                                                            filetype='dat',
                                                            quiet=True)
        parameters, errors = data.GetAllData(peakFileList)
        parameterList_bg = map.CreatePeakList(peakFileList)
        PlotParameterMappings(parameters, parameterList_bg, mapdims, step,
                              background=background, msize=msize, area=area)

        dict_bg = CreateMinMaxDict(parameters, parameterList_bg, folder)
        dict_minmax_global = UpdateGlobalDict(dict_minmax_global, dict_bg)

    print('\nList of minima and maxima.')
    PrintMinMax(dict_minmax, parameterList)

    print(linebreaker + '\n' + linebreaker)

if scaled:
    print('List of global minima and maxima '
        + 'and the mappings they are taken from.')
    PrintMinMax(dict_minmax_global, dict_minmax_global.keys())
    print(linebreaker + '\n' + linebreaker)

    for folder in mapFolderList:
        index = mapFolderList.index(folder)
        print('Scaled mapping ' + str(index + 1) + ' of '
            + str(len(mapFolderList)) + '\n')
        print(folder + ' mappings with same scale are plotted now.')
        mapdims = dims[index]
        step = stepsize[index]
        background = folder + '/' + backgrounds[index]
        msize = msizes[index]

        map = mp.mapping(foldername=folder, plot=True, peaknames=peaknames)

        # get fit data and plot one map per peak parameter
        peakFileList, numberOfPeakFiles = data.GetFolderContent(map.pardir_peak,
                                                        filetype='dat',
                                                        object='peakparameter')
        parameters, errors = data.GetAllData(peakFileList)
        parameterList = map.CreatePeakList(peakFileList)
        PlotParameterMappings(parameters, parameterList, mapdims, step,
                              background=background, msize=msize,
                              name='scaled_',
                              dict=dict_minmax_global)
        PlotParameterOperations(parameters, parameterList, mapdims, step,
                                top, bot, opt,
                                background=background, msize=msize,
                                name='scaled_',
                                dict=dict_minmax_global)

        print(linebreaker + '\n' + linebreaker)
