import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from spectrum import *

# Class for spectra (under development)
class mapping(object):

    def __init__(self, foldername, raw=False):
        if raw:
            self.raw = True
            self.folder = foldername
            self.listOfFiles, self.numberOfFiles = spectrum.GetFolderContent(self, 'txt')
        else:
            self.raw=False
            self.folder = foldername + '/results/fitlines'
            self.listOfFiles, self.numberOfFiles = spectrum.GetFolderContent(self, 'dat')

        self.savefolder = foldername
        self.x, self.y = spectrum.GetMonoData(self)
        self.ymax = np.max(self.y, axis=1)

        # create results folders
        if not os.path.exists(self.savefolder + '/results/plot'):
            os.makedirs(self.savefolder + '/results/plot')

    # plot mapping
    # input values are
    # xdim:     the number of Spectra in x direction
    # ydim:     the number of Spectra in y direction
    # stepsize: the interval at which the mapping was collected in µm
    # xmin:     the lowest wavenumber to be used in the mapping
    # xmax:     the highest wavenumber to be used in the mapping
    def PlotMapping(self, xdim, ydim, stepsize, xmin, xmax,
                    xticker=2, colormap='RdYlGn'):
        # create x and y ticks accordingly to the parameters of the mapping
        x_ticks = np.arange(stepsize, stepsize * (xdim + 1), step=xticker*stepsize)
        y_ticks = np.arange(stepsize, stepsize * (ydim + 1), step=stepsize)
        y_ticks = y_ticks[::-1]

        # sum up each spectrum and create matrix
        ysum = np.empty(self.numberOfFiles)
        iterator = 0
        for spectrum in self.y:
            selectedvalues = spectrum[(self.x[0] > xmin) & (self.x[0] < xmax)]
            ysum[iterator] = sum(selectedvalues)
            iterator += 1
        ysum_matrix = np.reshape(ysum, (ydim, xdim))
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
        plt.xticks(np.arange(xdim, step=xticker), x_ticks)
        plt.yticks(np.arange(ydim), y_ticks)

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
