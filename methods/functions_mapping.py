import glob
import numpy as np

# print some stuff
def teller(number, kind, location):
    if number != 1:
        print('There are {} {}s in this {}.'.format(number, kind, location))
        print()
    else:
        print('There is {} {} in this {}.'.format(number, kind, location))
        print()

# return a list of all files in a folder
def get_folder_content(foldername, filetype):
    #generate list of txt-files in requested folder
    foldername = foldername + '*.' + filetype
    listOfFiles = sorted(glob.glob(foldername))
    numberOfFiles = len(listOfFiles)
    # tell the number of files in the requested folder
    teller(numberOfFiles, 'file', 'folder')

    return listOfFiles

# returns arrays containing the measured data
def get_mono_data(listOfFiles):
    # define arrays to hold data from the files
    inversecm = np.array([])
    intensity = np.array([])

    # read all files
    for fileName in listOfFiles:
        # read one file
        index = listOfFiles.index(fileName)
        cm, inty = np.genfromtxt(listOfFiles[index], unpack=True)
        if index != 0:
            inversecm = np.vstack((inversecm, cm))
            intensity = np.vstack((intensity, inty))
        else:
            inversecm = cm
            intensity = inty

    return inversecm, intensity
