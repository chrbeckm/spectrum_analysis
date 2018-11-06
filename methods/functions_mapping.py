import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from functions_denoising import *

# print some stuff
def teller(number, kind, location):
    if number != 1:
        print('There are {} {}s in this {}.'.format(number, kind, location))
        print()
    else:
        print('There is {} {} in this {}.'.format(number, kind, location))
        print()

# return a list of all files in a folder
def GetFolderContent(foldername, filetype):
    #generate list of txt-files in requested folder
    foldername = foldername + '/*.' + filetype
    listOfFiles = sorted(glob.glob(foldername))
    numberOfFiles = len(listOfFiles)
    # tell the number of files in the requested folder
    teller(numberOfFiles, 'file', 'folder')

    return listOfFiles

# returns arrays containing the measured data
def GetMonoData(listOfFiles):
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

# generate all denoised spectra
def DenoiseMapping(x, ynormed, ymax, folder,
                   sav=True, display=False, prnt=True):
    ymuon_scaled = np.empty_like(ynormed)    # save muon free spectra
    yden_scaled = np.empty_like(ynormed)     # save denoised spectra

    # create dirs for muon removed and denoised spectra
    # if they not already exist
    if not os.path.exists(folder + '/muons-removed/'):
        os.makedirs(folder + '/muons-removed/')
        os.makedirs(folder + '/denoised/')
        os.makedirs(folder + '/temp/')

    # loop through all spectra
    iterator = 0
    for spectrum in ynormed:
        # print the current spectrum
        if prnt:
            print('spectrum ' + str(iterator + 1).zfill(4))
        # denoise single spectrum
        muon, denoised = DenoiseSpectrum(x[0], spectrum, thresh_mod=2)

        # rescale spectra back former values
        ymuon_scaled[iterator] = muon * ymax[iterator]
        yden_scaled[iterator] = denoised * ymax[iterator]

        # print the results
        if prnt:
            print('Maximum: %.2f at index: %4d and position: %4d' %
                 (max(yden_scaled[iterator]),
                  np.argmax(yden_scaled[iterator]),
                  x[iterator][np.argmax(yden_scaled[iterator])]))

        if sav:
            # save the muons removed spectra
            np.savetxt(folder + '/muons-removed/' +
                       str(iterator + 1).zfill(4) + '.dat',
                       np.transpose([x[iterator], ymuon_scaled[iterator]]),
                       fmt='%3.3f')

            # save the denoised spectra
            np.savetxt(folder + '/denoised/' +
                       str(iterator + 1).zfill(4) + '.dat',
                       np.transpose([x[iterator], yden_scaled[iterator]]),
                       fmt='%3.3f')
        if display:
            # plot muonfree and denoised spectra
            WaveletPlot(x[0], ymuon_scaled[iterator],
                              yden_scaled[iterator],
                              title=('Spectrum ' + str(iterator + 1)))
        iterator += 1
    return ymuon_scaled, yden_scaled

# remove the baseline
def RemoveBaselineMapping(x, y, baselinefile, degree=3, display=False):
    y_bgfree = np.empty_like(y) # save background free spectra

    # loop through all spectra
    iterator = 0
    for spectrum in y:
        # fit a baseline through the data
        baseline_fit = FitBaseline(x[iterator], spectrum, baselinefile)
        background = PolynomialModel(degree = degree)
        background_line = background.eval(baseline_fit.params, x = x[iterator])

        # remove background from data
        y_bgfree[iterator] = y[iterator] - background_line

        # plot background and original data
        if display:
            f, ax = plt.subplots()
            # plot denoised data
            ax.plot(x[iterator], y[iterator], 'b.')
            # plot background
            ax.plot(x[iterator], background_line, 'r-')
            # plot denoised data - background
            ax.plot(x[iterator], y_bgfree[iterator] + min(background_line), 'y--')
            f.show()

        iterator += 1

    return y_bgfree

# plot pca reduced data with cluster labels
# returns the sum for each cluster
def PlotClusteredPCA(x, yden, folder, pca, algorithm, cluster_algorithm,
                     n_clusters, colors):
    cluster_sum = np.empty([n_clusters, yden.shape[1]])

    f_algorithm, ax_algorithm = plt.subplots()
    # plot algorithm labeled pca analysis
    for point in range(0, len(pca)):
        # get cluster from algorithm
        clust = algorithm.labels_[point]

        # calculate sum spectra for each cluster
        cluster_sum[clust] = cluster_sum[clust] + yden[point, :]

        # plot each pca point
        ax_algorithm.scatter(pca[point, 0], pca[point, 1],
                    color=colors[clust], alpha=.8)

    # set title and show image
    ax_algorithm.set_title('PCA of ' + folder + ' Dataset with '
                           + cluster_algorithm + ' coloring')
    f_algorithm.show()

    return cluster_sum

# plot a mapping of data
#def PlotMapping():
