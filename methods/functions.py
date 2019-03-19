import glob
import numpy as np

# print out the number of files in folder
def Teller(number, kind, location):
    if number != 1:
        print('There are {} {}s in this {}.'.format(number, kind, location))
        print()
    else:
        print('There is {} {} in this {}.'.format(number, kind, location))
        print()

# get a list of files with defined type in the folder
def GetFolderContent(folder, filetype, object='file', where='folder', quiet=False):
    #generate list of files in requested folder
    files = folder + '/*.' + filetype
    listOfFiles = sorted(glob.glob(files))
    numberOfFiles = len(listOfFiles)
    # tell the number of files in the requested folder
    if not quiet:
        Teller(numberOfFiles, object, where)

    return listOfFiles, numberOfFiles

# returns arrays containing the measured data
def GetMonoData(listOfFiles, **kwargs):
    # define arrays to hold data from the files
    inversecm = np.array([])
    intensity = np.array([])

    # read all files
    for fileName in listOfFiles:
        # read one file
        index = listOfFiles.index(fileName)
        if kwargs == {}:
            cm, inty = np.genfromtxt(listOfFiles[index], unpack=True)
        elif kwargs['measurement'] == 'xps':
            cm, inty = np.genfromtxt(listOfFiles[index], unpack=True,
                                                         skip_header=1,
                                                         usecols=(0,10))
        if index != 0:
            inversecm = np.vstack((inversecm, cm))
            intensity = np.vstack((intensity, inty))
        else:
            inversecm = cm
            intensity = inty

    return inversecm, intensity
