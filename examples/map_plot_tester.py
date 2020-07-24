import os
import re
import shutil

import numpy as np
import matplotlib

from spectrum_analysis import mapping as mp
from spectrum_analysis import data
from peaknames import *

import pandas as pd
import tracemalloc
from pympler import muppy, summary

matplotlib.use('Agg')  # might need adjustment in case of memory leakage
                       # as some backends are leaking

debug = False

if debug:
    tracemalloc.start()

mapFolderList = [os.path.join('testdata', '1'),
                 os.path.join('testdata', '2')
                 ]
dims = [(4, 4),
        (8, 2)
        ]
stepsize = [10,
            10
            ]

# images need to be in folders specified in mapFolderList
# best is to use png backgrounds, but jpgs work as well
backgrounds = ['bg_test.png',
               'bg_test.jpg'
              ]

msizes = [1.04,
          1.04,
]

# number of bins
bins = 20

# True if background should be plotted
bg_plot = True

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

top2 = 'breit_wigner_p1_fwhm'
bot2 = 'lorentzian_p1_fwhm'
opt2 = 'div'

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
                          msize=2.1, name='', dicti=None, area=None):
    """
    Plot all parameters of a mapping.
    """
    plot_mats = []
    plot_nams = []
    for i, mapping in enumerate(peakList):
        vmin = None
        vmax = None
        if dicti is not None:
            vmin = dicti[mapping][0]
            vmax = dicti[mapping][1]
        plot_matrix, plotname = mapp.PlotMapping('params',
                        params[i], mapdims, step,
                        name=name + mapping,
                        vmin=vmin, vmax=vmax, grid=False)
        plot_mats.append(plot_matrix)
        plot_nams.append(plotname)
        mapp.PlotMapping('params', params[i], mapdims, step,
                        name=name + 'grid_' + mapping, alpha=0.75,
                        vmin=vmin, vmax=vmax, grid=True,
                        background=background, msize=msize,
                        plot_missing=False, area=area)
    # create plot ranges dictionary
    param_types = []
    for param in peakList:
        param_types.append(param.split('_')[-1])
    param_types = set(param_types)
    plot_ranges = dict.fromkeys(param_types, {'min': None, 'max': None})

    # fill plot ranges
    for key in plot_ranges.keys():
        for keys in peakList:
            if key in keys:
                # get index of parameter and corresponding min and max
                i = peakList.index(keys)
                nonMissing = [x for x in params[i] if not (x == mapp.missingvalue)]
                minval = np.min(nonMissing)
                maxval = np.max(nonMissing)
                if plot_ranges[key]['min'] is None:
                    content = {key: {'min': minval,
                                     'max': maxval}}
                else:
                    content = {key: {'min': np.min((plot_ranges[key]['min'],
                                                    minval)),
                                     'max': np.max((plot_ranges[key]['max'],
                                                    maxval))}}
                plot_ranges.update(content)

    # plot histograms with the same plot ranges
    for mat, nam in zip(plot_mats, plot_nams):
        mapp.PlotHistogram(mat, nam, bins=bins,
                          rng=(plot_ranges[nam.split('_')[-1]]['min'],
                               plot_ranges[nam.split('_')[-1]]['max']))

def PlotErrorMappings(params, errors, peakList, mapdims, step):
    """
    Plot all relative errors of a mapping.
    """
    for i, mapping in enumerate(peakList):
        # calculate relative error, set missing values and plot errors
        relative_error = mapp.ModifyValues(errors[i], params[i], 'div')
        mapp.PlotMapping('errs', relative_error, mapdims, step, name=mapping,
                        numbered=True)

def PlotParameterOperations(params, peakList, mapdims, step,
                            first, second, operation, background='',
                            msize=2.1, name='', dicti=None, area=None):
    """
    Plot a mapping calculated from two parameters (like height_a/height_b).
    """
    a = peakList.index(first)
    b = peakList.index(second)
    ratio = mapp.ModifyValues(params[a], params[b], operation)
    filename = first + '_' + operation + '_' + second
    vmin = None
    vmax = None
    if dicti is not None:
        vmin = dicti[filename][0]
        vmax = dicti[filename][1]
    plot_matrix, plotname = mapp.PlotMapping(operation,
                    ratio, mapdims, step,
                    name=name + filename,
                    numbered=False, vmin=vmin, vmax=vmax, grid=False)
    mapp.PlotMapping(operation, ratio, mapdims, step,
                    name=name + 'grid_' + filename, alpha=0.75,
                    numbered=False, vmin=vmin, vmax=vmax, grid=True,
                    background=background, msize=msize,
                    plot_missing=False, area=area)
    mapp.PlotHistogram(plot_matrix, plotname, bins=bins)
    return filename, ratio

def CreateMinMaxDict(params, paramList, mapping):
    """
    Create a dictionary containing all parameters with the global min and max.
    """
    dicti = {}
    # go through all parameters
    for param in paramList:
        # get index of parameter and corresponding min and max
        i = paramList.index(param)
        nonMissing = [x for x in params[i] if not (x == mapp.missingvalue)]
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
        dicti.update(content)
    return dicti

def UpdateGlobalDict(globaldict, dicti):
    for param in dicti.keys():
        min = dicti[param][0]
        max = dicti[param][1]
        minfile = dicti[param][2]
        maxfile = dicti[param][3]
        min_idx = dicti[param][4]
        max_idx = dicti[param][5]
        # check if parameter already in dictionary
        if param in globaldict:
            # check if parameter smaller/bigger than current value
            # and update values and mappings
            if ((globaldict[param][0] > min)
            and not (globaldict[param][0] == mapp.missingvalue)):
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

def PrintMinMax(dicti, list):
    for param in list:
        print(param + '\n'
                    + '\tMin: ' + str(dicti[param][0])
                    + ' ({})'.format(dicti[param][2])
                    + '\tSpectra: ' + str(dicti[param][4]) + '\n'
                    + '\tMax: ' + str(dicti[param][1])
                    + ' ({})'.format(dicti[param][3])
                    + '\tSpectra: ' + str(dicti[param][5]))

print('There are ' + str(CalculateSpectraNumber(dims)) + ' spectra at all.')
print(linebreaker + '\n' + linebreaker)

for folder in mapFolderList:
    index = mapFolderList.index(folder)
    print('Mapping ' + str(index + 1) + ' of '
        + str(len(mapFolderList)) + '\n')
    print(folder + ' mappings are plotted now.')
    mapdims = dims[index]
    step = stepsize[index]
    background = folder + os.sep + backgrounds[index]
    msize = msizes[index]

    mapp = mp.mapping(foldername=folder, plot=True, peaknames=peaknames)

    # get and plot raw data
    x, y = data.GetAllData(mapp.listOfFiles)
    mapp.PlotMapping('raw', y, mapdims, step, x=x, xmin=1300, xmax=1400)
    if plotrawspectra:
        mapp.PlotAllRawSpectra(x, y)

    # plot all colormaps
    #mapp.PlotAllColormaps('raw', y, mapdims, step, x=x, xmin=1300, xmax=1400)

    # get fit data and plot one map per peak parameter
    peakFileList, numberOfPeakFiles = data.GetFolderContent(mapp.pardir_peak,
                                                        filetype='dat',
                                                        objects='peakparameter')
    parameters, errors = data.GetAllData(peakFileList)
    parameterList = mapp.CreatePeakList(peakFileList)

    # calculate area under the curve
    # area is used to scale the linewidth of the grid marker
    area = 0
    for i, parameter in enumerate(parameterList):
        test = re.findall('amplitude', parameter)
        if test:
            area += parameters[i]
    if debug:
        time1 = tracemalloc.take_snapshot()
    PlotParameterMappings(parameters, parameterList, mapdims, step,
                          background=background, msize=msize, area=area)
    PlotErrorMappings(parameters, errors, parameterList, mapdims, step)
    if debug:
        time2 = tracemalloc.take_snapshot()

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
                                                     top2, bot2, opt2,
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
        peakFileList, numberOfPeakFiles = data.GetFolderContent(mapp.pardir_peak_bg,
                                                            filetype='dat',
                                                            quiet=True)
        parameters, errors = data.GetAllData(peakFileList)
        parameterList_bg = mapp.CreatePeakList(peakFileList)
        PlotParameterMappings(parameters, parameterList_bg, mapdims, step,
                              background=background, msize=msize, area=area)

        dict_bg = CreateMinMaxDict(parameters, parameterList_bg, folder)
        dict_minmax_global = UpdateGlobalDict(dict_minmax_global, dict_bg)

    print('\nList of minima and maxima.')
    PrintMinMax(dict_minmax, parameterList)

    print(linebreaker + '\n' + linebreaker)

    if debug:
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)# Prints out a summary of the large objects
        summary.print_(sum1)# Get references to certain types of objects such as dataframe
        dataframes = [ao for ao in all_objects if isinstance(ao, pd.DataFrame)]

        for dat in dataframes:
            print(dat.columns.values)
            print(len(dat))
        stats = time2.compare_to(time1, 'lineno')
        for stat in stats[:10]:
            print(stat)

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
        background = folder + os.sep + backgrounds[index]
        msize = msizes[index]

        mapp = mp.mapping(foldername=folder, plot=True, peaknames=peaknames)

        # get fit data and plot one map per peak parameter
        peakFileList, numberOfPeakFiles = data.GetFolderContent(mapp.pardir_peak,
                                                        filetype='dat',
                                                        objects='peakparameter')
        parameters, errors = data.GetAllData(peakFileList)
        parameterList = mapp.CreatePeakList(peakFileList)
        PlotParameterMappings(parameters, parameterList, mapdims, step,
                              background=background, msize=msize,
                              name='scaled_',
                              dicti=dict_minmax_global)
        PlotParameterOperations(parameters, parameterList, mapdims, step,
                                top, bot, opt,
                                background=background, msize=msize,
                                name='scaled_',
                                dicti=dict_minmax_global)

        print(linebreaker + '\n' + linebreaker)
