import os
import numpy as np

def GetData(file, measurement='', prnt=False):
    """
    Get data of one specified spectrum.

    Parameters
    ----------
    measurement : string, default : ''
        Type of measurement. Supported are '' for simple x, y data
        and 'xps' for XPS-Data from BL11 at DELTA.

    prnt : boolean, default : False
        Print what data was read.

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
        return x, y

    else:
        print('The spectrum you have chosen doesn\'t exist.\n'
              'You need to choose a different spectrum to read in.')
        return np.empty([1]), np.empty([1])
