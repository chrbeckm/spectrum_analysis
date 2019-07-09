import os
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from functions import *

# Class for mappings (under development)
class mapping(object):
    """
    This class provides methods to plot spectral xy-mappings.

    Before you can execute most of the code you need to analyze the spectra with
    the methods given by the spectrum class.

    Parameters
    ----------

    foldername : string
        A folder containing the data for the mapping.

    xdim : int
        Number of steps in x direction.

    ydim : int
        Number of steps in y direction.

    stepsize : int
        Size of one step in µm.

    raw : boolean
        Set to true if you want to plot the raw data. Otherwise the processed
        data will be analyzed.
    """
    def __init__(self, foldername, xdim, ydim, stepsize, raw=False):
        self.folder = foldername
        self.xdim = xdim
        self.ydim = ydim
        self.stepsize = stepsize
        self.raw = raw
        self.listOfFiles, self.numberOfFiles = GetFolderContent(self.folder,
                                                                'txt')
        self.missingvalue = np.genfromtxt(self.folder +
                                          '/temp/missingvalue.dat',
                                          unpack = True)

        # create results folders
        if not os.path.exists(self.folder + '/results/plot'):
            os.makedirs(self.folder + '/results/plot')

    # function to label the z-axis with label
    def LabelZ(self, plt, ax, label='Integrated Intensity\n(arb. u.)', nbins=5,
               linear=False):
        """
        Function to label the z-axis of the Plot.

        Parameters
        ----------
        plt : matplotlib.figure.Figure
        Plot that should be labeled.

        ax : matplotlib.axes.Axes
        Axis of interest.

        label : string
        Label that should be used for the z-axis.
        """
        tick_locator = matplotlib.ticker.MaxNLocator(nbins=nbins)
        if linear:
            tick_locator = matplotlib.ticker.LinearLocator(numticks=nbins)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = plt.colorbar(cax=cax)
        clb.set_label(label)
        clb.locator = tick_locator
        clb.update_ticks()

    # plot mapping
    # input values are
    # xmin:     the lowest wavenumber to be used in the mapping
    # xmax:     the highest wavenumber to be used in the mapping
    def PlotMapping(self, xmin=None, xmax=None,     # set x min and xmax if you want to integrate a region
                    maptype='',                     # maptypes accordingly to fitparameter/peakwise/*
                    top='', bot='',                 # define these if you want to calculate a ratio
                    clustered = False, colorlist=['w'],  # True if clustered should be plotted
                    distance=False,
                    label='',
                    xticker=2, colormap='Reds'):
        """
        Method to plot different mappings.

        Parameters
        ----------

        xmin : int
            Lowest wavenumber that should be used for integrating a spectral
            region.

        xmax : int
            Highest wavenumber that should be used for integrating a spectral
            region.

        maptype : string
            Plot any of the parameters in fitparameter/peakwise/

        top : string
            If you want to plot a ratio this will be the file name of the
            numerator located in fitparameter/peakwise/.

        bot : string
            If you want to plot a ratio this will be the file name of the
            divisor located in fitparameter/peakwise/.

        distance : boolean
            If you want to plot a mapping of the distance of two peaks set True.

        label : string

        xticker : int

        colormap : string
            Defines the coloring of the mapping according to the `matplotlib
            colormaps <https://matplotlib.org/users/colormaps.html>`_
        """
        # create x and y ticks accordingly to the parameters of the mapping
        x_ticks = np.arange(self.stepsize, self.stepsize * (self.xdim + 1), step=xticker*self.stepsize)
        y_ticks = np.arange(self.stepsize, self.stepsize * (self.ydim + 1), step=self.stepsize)
        y_ticks = y_ticks[::-1]

        plot_value = np.empty(self.numberOfFiles)

        savefile = ''

        # create figure for mapping
        fig, ax = plt.subplots(figsize=(self.xdim,self.ydim))
        ax.set_aspect('equal')

        # if fitlines should be integrated
        if (xmin != None) and (xmax != None):
            # get data from rawfiles or fitlines
            if self.raw:
                x, y = GetMonoData(self.listOfFiles)

                # define save file
                savefile = self.folder + '/results/plot/map_raw'
            else:
                folder = self.folder + '/results/fitlines'
                self.listOfFiles, self.numberOfFiles = GetFolderContent(folder, 'dat')
                x, y = GetMonoData(self.listOfFiles)

                # define save file
                savefile = self.folder + '/results/plot/map'

            # sum up each spectrum
            iterator = 0
            for spectrum in y:
                selectedvalues = spectrum[(x[0] > xmin) and (x[0] < xmax)]
                plot_value[iterator] = sum(selectedvalues)
                iterator += 1
        elif maptype != '':
            # get the selected file and the corresponding values
            file = (self.folder + '/results/fitparameter/peakwise/'
                     + maptype
                     + '.dat')
            plot_value, error = GetMonoData([file])

            # define save file
            savefile = self.folder + '/results/plot/map_' + maptype + label
        elif (top != '') and (bot != ''):
            # get files that should be divided
            file1 = self.folder + '/results/fitparameter/peakwise/' + top
            file2 = self.folder + '/results/fitparameter/peakwise/' + bot
            plot1, error1 = GetMonoData([file1])
            plot2, error2 = GetMonoData([file2])

            # check for missing indices
            missingindices_p1 = [i for i, x in enumerate(plot1) if (x == self.missingvalue)]
            missingindices_p2 = [i for i, x in enumerate(plot2) if (x == self.missingvalue)]

            # set the missing values vice versa
            for index in missingindices_p1:
                plot2[index] = self.missingvalue
            for index in missingindices_p2:
                plot1[index] = self.missingvalue

            # used to define the save file
            file1 = re.sub('.dat', '', top)
            file2 = re.sub('.dat', '', bot)

            # decide if the distance or the ratio should be plotted
            if distance:
                # calculate the difference
                plot_value = plot1 - plot2
                savefile = (self.folder + '/results/plot/map_'
                            + file1 + '_-_' + file2)
            else:
                # calculate the ratio
                plot_value = plot1 / plot2
                savefile = (self.folder + '/results/plot/map_'
                           + file1 + '_div_' + file2)


        elif clustered:
            plot_value = self.clustered.labels_
            savefile = self.folder + '/results/plot/map_clustered'

        # check if any value in plot_value is a missing value or 1
        missingindices = [i for i, x in enumerate(plot_value)
                          if ((x == self.missingvalue) or (x == 1.0)
                                                      or (x == 0.0))
                                                      and not clustered]
        existingindices = [i for i, x in enumerate(plot_value)
                           if (x != self.missingvalue) and (x != 1.0)
                                                       and (x != 0.0)]
        # calculate the mean of the existing values
        fitmean = 0
        for index in existingindices:
            fitmean += plot_value[index]
        fitmean = fitmean / len(existingindices)

        # set the missing values as mean
        for index in missingindices:
            plot_value[index] = fitmean

        # create matrix for plotting
        plot_matrix = np.reshape(plot_value, (self.ydim, self.xdim))
        plot_matrix = np.flipud(plot_matrix)

        # create matrix with missing values
        missing_matrix = np.full_like(plot_matrix, False, dtype=bool)
        if not clustered:
            missing_matrix = (plot_matrix == fitmean)

        # set font and parameters
        matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
        matplotlib.rcParams.update({'font.size': 22})

        # plot the selected mapping
        if clustered:
            # make a color map of fixed colors
            colorlist = colorlist[0:self.clustered.n_clusters]
            cmap = colors.ListedColormap(colorlist)
            ticks = [i for i, x in enumerate(colorlist)]
            plt.imshow(plot_matrix, cmap=cmap)
        else:
            plt.imshow(plot_matrix, cmap=colormap)
        plt.xticks(np.arange(self.xdim, step=xticker), x_ticks)
        plt.yticks(np.arange(self.ydim), y_ticks)

        # Create list for all the missing values as missing patches
        missingboxes = []

        # find all fields not containing signals and append to
        for iy in range(0,self.ydim):
            for ix in range(0,self.xdim):
                if missing_matrix[iy][ix]:
                    # calculate position correction for the patches
                    corr = 0.5
                    linecorr = matplotlib.rcParams['axes.linewidth']/fig.dpi/4
                    # create the missing patch and add to list
                    rect = matplotlib.patches.Rectangle((ix - corr + linecorr,
                                              iy - corr - linecorr), 1, 1)
                    missingboxes.append(rect)

        # Create patch collection with specified colour/alpha
        pc = matplotlib.collections.PatchCollection(missingboxes,
                                                    facecolor='black')

        # Add collection to axes
        ax.add_collection(pc)

        # label the x and y axis
        plt.ylabel('y-Position ($\mathrm{\mu}$m)')
        plt.xlabel('x-Position ($\mathrm{\mu}$m)')

        # set title and z-axis properly
        if (xmin != None) and (xmax != None):
            plt.title('Mapping of ' + self.folder, fontsize='small')
            self.LabelZ(plt, ax)
        elif maptype != '':
            plt.title('Mapping of ' + self.folder + ' ' + maptype, fontsize='small')
            unwanted = re.findall('(?s:.*)_', maptype)[0]
            type = re.sub(unwanted, '', maptype)

            if (type == 'center') or (type == 'fwhm') or (type == 'sigma'):
                self.LabelZ(plt, ax, label=type + ' (cm$^{-1}$)')
            else:
                self.LabelZ(plt, ax, label=type + ' (arb. u.)')
        elif (top != '') and (bot != ''):
            if distance:
                plt.title('Mapping of ' + self.folder + ' ' + top + '-' + bot,
                        fontsize='small')
                self.LabelZ(plt, ax, label='Distance  (cm$^{-1}$)')
            else:
                plt.title('Mapping of ' + self.folder + ' ' + top + '/' + bot,
                        fontsize='small')
                self.LabelZ(plt, ax, label='Ratio (arb. u.)')
        elif clustered:
            plt.title('Mapping of the clustered ' + self.decompose + ' data.')
            self.LabelZ(plt, ax, label='Cluster (arb. u.)', linear=True, nbins=3)

        # have a tight layout
        plt.tight_layout()

        # save everything and show the plot
        if self.raw:
            plt.savefig(savefile + '.pdf', format='pdf')
            plt.savefig(savefile + '.png')
        elif maptype != '':
            plt.savefig(savefile + '.pdf', format='pdf')
            plt.savefig(savefile + '.png')
        elif (top != '') and (bot != ''):
            plt.savefig(savefile + '.pdf', format='pdf')
            plt.savefig(savefile + '.png')
        elif clustered:
            plt.savefig(savefile + '.pdf', format='pdf')
            plt.savefig(savefile + '.png')
        else:
            plt.savefig(self.folder + '/results/plot/map.pdf', format='pdf')
            plt.savefig(self.folder + '/results/plot/map.png')
        plt.clf()

        savefile = re.sub(self.folder + '/results/plot/map_', '', savefile)
        print(savefile)

    def PlotAllMappings(self, colormap='Reds'):
        """

        """
        folder = self.folder + '/results/fitparameter/peakwise/'
        listOfFiles, numberOfFiles = GetFolderContent(folder, 'dat', object='parameter', where='fit')
        print('The following maps have been plotted:')
        for map in listOfFiles:
            map = re.sub(folder, '', map)
            map = re.sub('.dat', '', map)
            self.PlotMapping(maptype=map, colormap=colormap)

    def PlotAllColormaps(self, map):
        """

        """
        map = re.sub('.dat', '', map)
        cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]
        for category in cmaps:
            print(category[0])
            for colormap in category[1]:
                self.PlotMapping(maptype=map, label=category[0] + colormap, colormap=colormap)

    def DecomposePCA(self, decompose='raw', n_components=2):
        """
        Decompose the spectra of a given mapping.

        Parameters
        ----------
        decompose : string
            string that names what should be decomposed.
            For example raw, baselines, fitlines, fitspectra or fitpeaks

        n_components : int
            Number of components for the PCA analysis
        """
        # create the pca analysis
        pipeline = make_pipeline(StandardScaler(),
                                 PCA(n_components=n_components))
        self.decompose = decompose

        # get requested data
        self.x = np.empty(self.numberOfFiles)
        self.y = np.empty(self.numberOfFiles)

        folder = self.folder
        type = 'txt'

        printstring = 'Decompose ' + self.decompose

        # create strings to get the requested data
        if self.decompose == 'raw':
            printstring += ' data.'
        elif self.decompose == 'baselines':
            printstring += '.'
            folder += '/results/baselines'
            type = 'dat'
        elif self.decompose == 'fitlines':
            printstring += '.'
            folder += '/results/fitlines'
            type = 'dat'

        print(printstring)

        # get the files and data
        self.listOfFiles, self.numberOfFiles = GetFolderContent(folder, type,
                                                                quiet=True)
        self.x, self.y = GetMonoData(self.listOfFiles)

        # do the pca analysis
        self.pca_analysis = pipeline.fit(self.y).transform(self.y)
        pca = pipeline.named_steps['pca']

        # print the result
        print('Explained variance ratio (first two components): %s'
              % str(pca.explained_variance_ratio_))

    def ClusterPCA(self, cluster='kmeans', n_clusters=3):
        """
        Parameters
        ----------
        cluster : string
            Name of the clustering Algorithm. At the moment only kmeans from
            'sklearn.cluster <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster>'_
            is defined

        n_clusters : int
            Number of clusters that should be used for clustering.
        """
        if cluster == 'kmeans':
            self.clustered = KMeans(init='k-means++', n_clusters=n_clusters)
        else:
            print('Use different cluster algorithm.')

        self.clustered.fit(self.pca_analysis)

    def PlotClusteredPCA(self, colorlist):
        """
        Plot the clustered data. And calculate the sum of each cluster.
        """
        self.cluster_sum = np.empty([self.clustered.n_clusters, len(self.y)])

        fig, ax = plt.subplots(figsize=(12,9))
        for point in range(0, len(self.pca_analysis)):
            # get cluster calculated from ClusterPCA
            clust = self.clustered.labels_[point]

            # calculate the sum spectra for each cluster
            self.cluster_sum[clust] += sum(self.y[point][:])

            # plot each pca point into a scatter plot
            ax.scatter(self.pca_analysis[point, 0], self.pca_analysis[point, 1],
                       color=colorlist[clust], alpha=.8)

        # set the labels
        plt.title('PCA of ' + self.folder + ' with ' + self.clustered.init
                     + ' coloring')
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')

        # save the figures
        fig.savefig(self.folder + '/results/plot/pca_analysis.pdf')
        fig.savefig(self.folder + '/results/plot/pca_analysis.png', dpi=150)

    def PlotClusteredPCAMapping(self, colorlist, cluster='kmeans', n_clusters=3):
        """
        Plot a mapping PCA decomposed mapping clustered with a cluster
        algorithm.
        """

        self.DecomposePCA()
        self.ClusterPCA(cluster=cluster, n_clusters=n_clusters)
        self.PlotClusteredPCA(colorlist=colorlist)
        self.PlotMapping(clustered=True, colorlist=colorlist)
