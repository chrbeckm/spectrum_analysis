import os
import glob
import numpy as np
import pandas as pd

def GetData(file, measurement='', prnt=False, xdata='', ydata=''):
    """
    Get data of one specified spectrum.

    Parameters
    ----------
    file : string
        Filename of the data to be read in.

    measurement : string, default : ''
        Type of measurement. Supported are '' for simple x, y data,
        'xps' for XPS-Data from BL11 at DELTA and 'tribo' for tribometer
        data from LWT.

    prnt : boolean, default : False
        Print what data was read.

    xdata : string, default : ''
        Only needed for 'tribo' measurements. Possible keywords are 'Time',
        'Distance', 'laps', 'Sequence ID', 'Cycle ID', 'Max linear speed',
        'Nominal Load', 'µ' or 'Normal force'. The selected column will be
        taken as data for the x-axis.

    ydata : string, default : ''
        Only needed for 'tribo' measurements. Possible keywords are 'Time',
        'Distance', 'laps', 'Sequence ID', 'Cycle ID', 'Max linear speed',
        'Nominal Load', 'µ' or 'Normal force'. The selected column will be
        taken as data for the y-axis.

    Returns
    -------
    x : numpy.array
        Read in data of the x-axis.

    y : numpy.array
        Read in data of the y-axis.
    """

    if os.path.isfile(file):
        if measurement == '':
            x, y = np.genfromtxt(file, unpack=True)
            if prnt:
                print('Simple xy Data was read.')
        elif measurement == 'xps':
            x, y = np.genfromtxt(file, unpack=True,
                                       skip_header=1,
                                       usecols=(0,10))
            if prnt:
                print('XPS Data from DELTA was read.')
        if measurement == 'tribo':
            dataframe = pd.read_csv(file, skiprows=54,
                                    delimiter='\t', decimal=',')
            labels = dataframe.columns
            xlabel = [s for s in labels if xdata in s]
            ylabel = [s for s in labels if ydata in s]
            x = dataframe[xlabel[0]].to_numpy(copy=True)
            y = dataframe[ylabel[0]].to_numpy(copy=True)

        return x, y

    else:
        print('The spectrum you have chosen doesn\'t exist.\n'
              'You need to choose a different spectrum to read in.')
        return np.empty([1]), np.empty([1])

def Teller(number, kind, location='folder'):
    """
    Function that prints out how many instances there are in a
    location.
    Parameters
    ----------
    number : int
        Number of instances in a requested location.
    kind : string
        Instance that is analyzed.
    location : string, default : 'folder'
        Location where the instance can be found.
    """
    if number > 1:
        print('There are {} {}s in this {}.'.format(number, kind,
                                                    location))
        print()
    else:
        print('There is {} {} in this {}.'.format(number, kind,
                                                  location))
        print()

def GetFolderContent(folder, filetype,
                     object='spectra', quiet=False):
    """
    Get a list of all files of a defined type from a folder
    Parameters
    ----------
    folder : string
        Name of the folder that should be analyzed.
    filetype : string
        Ending of the file types that should be analyzed.
        For example 'txt', 'csv' or 'dat'.
    object : string, default : 'spectra'
        Type of the objects that are analyzed.
    quiet : boolean, default : False
        Whether the command line should tell how many files are in
        the folder or not.
    Returns
    -------
    listOfFiles : sorted list of strings
        List of strings containing the names of all the files in
        the folder.
    numberOfFiles : int
        The number of files in the requested folder.
    """
    # generate list of files in requested folder
    files = folder + '/*.' + filetype
    listOfFiles = sorted(glob.glob(files))
    numberOfFiles = len(listOfFiles)

    # tell the number of files in the requested folder
    if not quiet:
        Teller(numberOfFiles, object)

    return listOfFiles, numberOfFiles

def VStack(i, x, xtemp):
    """
    Stacks arrays
    """
    if i != 0:
        x = np.vstack((x, xtemp))
    else:
        x = np.array(xtemp)
        if len(x.shape) == 1:
            x = x.reshape(1,len(x))
        if len(x.shape) == 0:
            x = x.reshape(1,1)
    return x

def GetAllData(listOfFiles, measurement='', prnt=False):
    """
    Get data of the mapping.
    Parameters
    ----------
    measurement : string, default : ''
        Type of measurement. Supported are '' for simple x, y data
        and 'xps' for XPS-Data from BL11 at DELTA.
    prnt : boolean, default : False
        Print what data was read.
    Returns
    -------
    x : numpy.ndarray
        Read in data of the x-axis.
    y : numpy.ndarray
        Read in data of the y-axis.
    """
    # define arrays to hold data from the files
    x = np.array([])
    y = np.array([])

    for i, spec in enumerate(listOfFiles):
        xtemp, ytemp = GetData(spec, measurement=measurement,
                               prnt=prnt)
        x = VStack(i, x, xtemp)
        y = VStack(i, y, ytemp)

    return x, y
