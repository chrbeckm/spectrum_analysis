import os
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from functions import *

# Class for spectra (under development)
class mapping(object):

    # xdim:     the number of Spectra in x direction
    # ydim:     the number of Spectra in y direction
    # stepsize: the interval at which the mapping was collected in µm
    def __init__(self, foldername, xdim, ydim, stepsize, raw=False):

        self.folder = foldername
        self.xdim = xdim
        self.ydim = ydim
        self.stepsize = stepsize
        self.raw = raw
        self.listOfFiles, self.numberOfFiles = GetFolderContent(self.folder, 'txt')

        # create results folders
        if not os.path.exists(self.folder + '/results/plot'):
            os.makedirs(self.folder + '/results/plot')

    # plot mapping
    # input values are
    # xmin:     the lowest wavenumber to be used in the mapping
    # xmax:     the highest wavenumber to be used in the mapping
    def PlotMapping(self, xmin=None, xmax=None,     # set x min and xmax if you want to integrate a region
                    maptype='',                     # maptypes accordingly to fitparameter/peakwise/*
                    xticker=2, colormap='RdYlGn'):
        # create x and y ticks accordingly to the parameters of the mapping
        x_ticks = np.arange(self.stepsize, self.stepsize * (self.xdim + 1), step=xticker*self.stepsize)
        y_ticks = np.arange(self.stepsize, self.stepsize * (self.ydim + 1), step=self.stepsize)
        y_ticks = y_ticks[::-1]

        plot_value = np.empty(self.numberOfFiles)

        # if fitlines should be integrated
        if (xmin != None) & (xmax != None):
            # get data from rawfiles or fitlines
            if self.raw:
                x, y = GetMonoData(self.listOfFiles)
            else:
                folder = self.folder + '/results/fitlines'
                self.listOfFiles, self.numberOfFiles = GetFolderContent(folder, 'dat')
                x, y = GetMonoData(listOfFiles)

            # sum up each spectrum
            iterator = 0
            for spectrum in y:
                selectedvalues = spectrum[(x[0] > xmin) & (x[0] < xmax)]
                plot_value[iterator] = sum(selectedvalues)
                iterator += 1
        if maptype != '':
            folder = self.folder + '/results/fitparameter/peakwise/' + maptype + '.dat'
            plot_value, error = GetMonoData([folder])

        # create matrix for plotting
        plot_matrix = np.reshape(plot_value, (self.ydim, self.xdim))
        plot_matrix = np.flipud(plot_matrix)

        # set font and parameters
        matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
        matplotlib.rcParams.update({'font.size': 22})
        tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)

        # create mapping
        fit, ax = plt.subplots(figsize=(18,6))
        ax.set_aspect('equal')
        plt.imshow(plot_matrix, cmap=colormap)
        plt.xticks(np.arange(self.xdim, step=xticker), x_ticks)
        plt.yticks(np.arange(self.ydim), y_ticks)

        # label everything
        plt.title('Mapping of ' + self.folder + ' ' + maptype, fontsize='small')
        plt.ylabel('y-Position ($\mathrm{\mu}$m)')
        plt.xlabel('x-Position ($\mathrm{\mu}$m)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = plt.colorbar(cax=cax)
        clb.set_label('Integrated Intensity\n(arb. u.)')
        clb.locator = tick_locator
        clb.update_ticks()

        # have a tight layout
        plt.tight_layout()

        # save everything and show the plot
        if self.raw:
            plt.savefig(self.folder + '/results/plot/map_raw.pdf', format='pdf')
            plt.savefig(self.folder + '/results/plot/map_raw.png', dpi=300)
        if maptype != '':
            plt.savefig(self.folder + '/results/plot/map_' + maptype + '.pdf', format='pdf')
            plt.savefig(self.folder + '/results/plot/map_' + maptype + '.png', dpi=300)
        else:
            plt.savefig(self.folder + '/results/plot/map.pdf', format='pdf')
            plt.savefig(self.folder + '/results/plot/map.png', dpi=300)
        plt.clf()

    def PlotAllMappings(self):
        folder = self.folder + '/results/fitparameter/peakwise/'
        listOfFiles, numberOfFiles = GetFolderContent(folder, 'dat', object='parameter', where='fit')
        print('The following maps have been plotted:')
        for map in listOfFiles:
            map = re.sub(folder, '', map)
            map = re.sub('.dat', '', map)
            self.PlotMapping(maptype=map)
            print(map)
