import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from functions import *
from functions_denoising import *

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
