import os

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
        self.savefolder = foldername

        # create results folders
        if not os.path.exists(self.savefolder + '/results/plot'):
            os.makedirs(self.savefolder + '/results/plot')

    # plot mapping
    # input values are
    # xmin:     the lowest wavenumber to be used in the mapping
    # xmax:     the highest wavenumber to be used in the mapping
    def PlotMapping(self, xmin=None, xmax=None,     # set x min and xmax if you want to integrate a region
                    test=0,
                    xticker=2, colormap='RdYlGn'):
        # create x and y ticks accordingly to the parameters of the mapping
        x_ticks = np.arange(self.stepsize, self.stepsize * (self.xdim + 1), step=xticker*self.stepsize)
        y_ticks = np.arange(self.stepsize, self.stepsize * (self.ydim + 1), step=self.stepsize)
        y_ticks = y_ticks[::-1]

        # if fitlines should be integrated
        if (xmin != None) & (xmax != None):

            # get data from rawfiles or fitlines
            if self.raw:
                self.listOfFiles, self.numberOfFiles = GetFolderContent(self.folder, 'txt')
                self.x, self.y = GetMonoData(self.listOfFiles)
                self.ymax = np.max(self.y, axis=1)
            else:
                self.folder = self.folder + '/results/fitlines'
                self.listOfFiles, self.numberOfFiles = GetFolderContent(self.folder, 'dat')
                self.x, self.y = GetMonoData(self.listOfFiles)
                self.ymax = np.max(self.y, axis=1)

            # sum up each spectrum
            ysum = np.empty(self.numberOfFiles)
            iterator = 0
            for spectrum in self.y:
                selectedvalues = spectrum[(self.x[0] > xmin) & (self.x[0] < xmax)]
                ysum[iterator] = sum(selectedvalues)
                iterator += 1

        # create matrix for plotting
        ysum_matrix = np.reshape(ysum, (self.ydim, self.xdim))
        ysum_matrix = np.flipud(ysum_matrix)

        # create figure
        fig = plt.figure(figsize=(18,6))
        plt.suptitle('Mapping of ' + self.folder)

        # set font and parameters
        matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
        matplotlib.rcParams.update({'font.size': 22})
        tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)

        # create mapping
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(ysum_matrix, cmap=colormap)
        plt.xticks(np.arange(self.xdim, step=xticker), x_ticks)
        plt.yticks(np.arange(self.ydim), y_ticks)

        # label everything
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
            plt.savefig(self.savefolder + '/results/plot/mapping_map_raw.pdf', format='pdf')
        else:
            plt.savefig(self.savefolder + '/results/plot/mapping_map.pdf', format='pdf')
        plt.show()
