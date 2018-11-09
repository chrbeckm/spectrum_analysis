import glob
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from lmfit.models import *

from starting_params import *

# Class for spectra (under development)
class spectrum(object):

    def __init__(self, foldername):
        self.folder = foldername
        self.listOfFiles, self.numberOfFiles = self.GetFolderContent('txt')
        self.x, self.y = self.GetMonoData()
        if self.numberOfFiles == 1:
            self.x = np.array([self.x])
            self.y = np.array([self.y])

        # get maximum and norm from each spectrum
        self.ymax = np.max(self.y, axis=1)
        self.ynormed = self.y/self.ymax[:,None]
        # selected spectrum
        self.xreduced = None
        self.yreduced = None

        # create temporary folders
        if not os.path.exists(self.folder + '/temp'):
            os.makedirs(self.folder + '/temp')
        # create results folders
        if not os.path.exists(self.folder + '/results'):
            os.makedirs(self.folder + '/results/baselines')
            os.makedirs(self.folder + '/results/fitlines')
            os.makedirs(self.folder + '/results/fitparameter/spectra')
            os.makedirs(self.folder + '/results/fitparameter/peakwise')
            os.makedirs(self.folder + '/results/plot')

        # names of files created during the procedure
        self.fSpectrumBorders = None
        self.fBaseline = None

        # fit parameters
        self.fitresult_bg = [None] * self.numberOfFiles
        self.baseline = [None] * self.numberOfFiles
        self.fitresult_peaks = [None] * self.numberOfFiles
        self.fitline = [None] * self.numberOfFiles
        self.confidence = [None] * self.numberOfFiles

    # print out the number of files in folder
    def Teller(self, number, kind, location):
        if number != 1:
            print('There are {} {}s in this {}.'.format(number, kind, location))
            print()
        else:
            print('There is {} {} in this {}.'.format(number, kind, location))
            print()

    # get a list of files with defined type in the folder
    def GetFolderContent(self, filetype):
        #generate list of files in requested folder
        files = self.folder + '/*.' + filetype
        listOfFiles = sorted(glob.glob(files))
        numberOfFiles = len(listOfFiles)
        # tell the number of files in the requested folder
        spectrum.Teller(self, numberOfFiles, 'file', 'folder')

        return listOfFiles, numberOfFiles

    # returns arrays containing the measured data
    def GetMonoData(self):
        # define arrays to hold data from the files
        inversecm = np.array([])
        intensity = np.array([])

        # read all files
        for fileName in self.listOfFiles:
            # read one file
            index = self.listOfFiles.index(fileName)
            cm, inty = np.genfromtxt(self.listOfFiles[index], unpack=True)
            if index != 0:
                inversecm = np.vstack((inversecm, cm))
                intensity = np.vstack((intensity, inty))
            else:
                inversecm = cm
                intensity = inty

        return inversecm, intensity

    # function that plots regions chosen by clicking into the plot
    def PlotVerticalLines(self, color, fig):
        xregion = []                            # variable to save chosen region
        ax = plt.gca()                          # get current axis
        plt_ymin, plt_ymax = ax.get_ylim()      # get plot min and max

        def onclickbase(event):                 # choose region by clicking
            if event.button:                    # if clicked
                xregion.append(event.xdata)     # append data to region
                # plot vertical lines to mark chosen region
                plt.vlines(x = event.xdata,
                           color = color,
                           linestyle = '--',
                           ymin = plt_ymin, ymax = plt_ymax)
                # fill selected region with transparent colorbar
                if(len(xregion) % 2 == 0 & len(xregion) != 1):
                    # define bar height
                    barheight = np.array([plt_ymax - plt_ymin])
                    # define bar width
                    barwidth = np.array([xregion[-1] - xregion[-2]])
                    # fill region between vertical lines with prior defined bar
                    plt.bar(xregion[-2],
                            height = barheight, width = barwidth,
                            bottom = plt_ymin,
                            facecolor = color,
                            alpha = 0.2,
                            align = 'edge',
                            edgecolor = 'black',
                            linewidth = 5)
                fig.canvas.draw()

        # actual execution of the defined function onclickbase
        cid = fig.canvas.mpl_connect('button_press_event', onclickbase)
        figManager = plt.get_current_fig_manager()  # get current figure
        figManager.window.showMaximized()           # show it maximized

        return xregion

    # Select the interesting region in the spectrum, by clicking on the plot
    def SelectSpectrum(self, spectrum=0, label=''):
        if spectrum >= self.numberOfFiles:
            print('You need to choose a smaller number for spectra to select.')
        else:
            # plot spectrum
            fig, ax = plt.subplots()
            ax.plot(self.x[spectrum], self.ynormed[spectrum],
                    'b.', label = 'Data')
            ax.set_title('Select the part of the spectrum you wish to consider\
            by clicking into the plot.')

            # select region of interest
            xregion = self.PlotVerticalLines('green', fig)

            plt.legend(loc='upper right')
            plt.show()
            self.yreduced = self.ynormed[:, (self.x[spectrum] > xregion[0]) &
                                            (self.x[spectrum] < xregion[-1])]
            self.xreduced = self.x[:, (self.x[spectrum] > xregion[0]) &
                                      (self.x[spectrum] < xregion[-1])]
            # save spectrum borders
            self.fSpectrumBorders = self.folder + '/temp/spectrumborders' +\
                                    label + '.dat'
            np.savetxt(self.fSpectrumBorders, np.array(xregion))

    #function to select the data that is relevent for the background
    def SelectBaseline(self, spectrum=0, label='', color='b'):
        if spectrum >= self.numberOfFiles:
            print('You need to choose a smaller number for spectra to select.')
        else:
            # plot the reduced spectrum
            fig, ax = plt.subplots()
            ax.plot(self.xreduced[spectrum], self.yreduced[spectrum],
                    '.', label='Data', color=color)
            ax.set_title('Normalized spectrum\n Select the area of the spectrum\
                         you wish to exclude from the background by clicking\
                        into the plot\n (3rd-degree polynomial assumed)')

            # choose the region
            xregion = self.PlotVerticalLines('red', fig)

            plt.legend(loc = 'upper right')
            plt.show()
            self.fBaseline = self.folder + '/temp/baseline' + label + '.dat'
            np.savetxt(self.fBaseline, np.array(xregion))

    # actual fit of the baseline
    def FitBaseline(self, spectrum=0, show=False):
        if spectrum >= self.numberOfFiles:
            print('You need to choose a smaller number for spectra to select.')
        else:
            # Load the bounderies for the relevent data from SelectBaseline()
            bed = np.genfromtxt(self.fBaseline, unpack = True)

            # generate mask for the baseline fit,
            # for relevent data relevant = True,
            # else relevant = False
            # bed[0] is the lowest border
            relevant = (self.xreduced[spectrum] <= bed[0])
            for i in range(1, len(bed) - 2, 2): # upper borders i
                # take only data between the borders
                relevant = relevant | ((self.xreduced[spectrum] >= bed[i]) &
                                       (self.xreduced[spectrum] <= bed[i + 1]))
            # bed[-1] is the highest border
            relevant = relevant | (self.xreduced[spectrum] >= bed[-1])

            # Third-degree polynomial to model the background
            background = PolynomialModel(degree = 3)
            pars = background.guess(self.yreduced[spectrum, relevant],
                                x = self.xreduced[spectrum, relevant])
            self.fitresult_bg[spectrum] = background.fit(self.yreduced[spectrum, relevant],
                                               pars, x = self.xreduced[spectrum, relevant])

            # create baseline
            self.baseline[spectrum] = background.eval(self.fitresult_bg[spectrum].params,
                                                  x = self.xreduced[spectrum])

            # plot the fitted function in the selected range
            if show:
                plt.plot(self.xreduced[spectrum], self.yreduced[spectrum],
                         'b.', label = 'Data')
                plt.plot(self.xreduced[spectrum], self.baseline[spectrum], 'r-', label = 'Baseline')
                plt.show()

    # fit all baselines
    def FitAllBaselines(self, show=False):
        for i in range(self.numberOfFiles):
            self.FitBaseline(spectrum=i, show=show)

    # function that plots the dots at the peaks you wish to fit
    def PlotPeaks(self, fig):
        xpeak = []  # x and
        ypeak = []  # y arrays for peak coordinates

        def onclickpeaks(event):
            if event.button:
                xpeak.append(event.xdata)               # append x data and
                ypeak.append(event.ydata)               # append y data
                plt.plot(event.xdata, event.ydata, 'ro')# plot the selected peak
                fig.canvas.draw()                       # and show it

        # actual execution of the defined function oneclickpeaks
        cid = fig.canvas.mpl_connect('button_press_event', onclickpeaks)
        figManager = plt.get_current_fig_manager()  # get current figure
        figManager.window.showMaximized()           # show it maximized

        return xpeak, ypeak

    # function that allows you to select Voigt-, Fano-, Lorentzian-,
    # and Gaussian-peaks for fitting
    def SelectPeaks(self, peaks, spectrum=0, label=''):
        if spectrum >= self.numberOfFiles:
            print('You need to choose a smaller number for spectra to select.')
        else:
            # loop over all peaks and save the selected positions
            for peaktype in peaks:
                # create plot and baseline
                fig, ax = plt.subplots()
                # plot corrected data
                ax.plot(self.xreduced[spectrum],
                        self.yreduced[spectrum] - self.baseline[spectrum], 'b.')
                ax.set_title('Background substracted, normalized spectrum\n\
                              Select the maxima of the ' + peaktype +\
                              '-PEAKS to fit.')
                # arrays of initial values for the fits
                xpeak, ypeak = self.PlotPeaks(fig)
                plt.show()
                # store the chosen initial values
                peakfile = self.folder + '/temp/locpeak_' + peaktype + '_' +\
                           label + '.dat'
                np.savetxt(peakfile, np.transpose([np.array(xpeak),
                                                   np.array(ypeak)]))

    # select all peaks
    def SelectAllPeaks(self, peaks):
        for i in range(self.numberOfFiles):
            self.SelectPeaks(peaks, spectrum=i, label=str(i+1).zfill(4))

    # Fit the peaks selected before
    # for detailed describtions see:
    # https://lmfit.github.io/lmfit-py/builtin_models.html
    def FitSpectrum(self, peaks, spectrum=0, label='', show=True, report=False):
        if spectrum >= self.numberOfFiles:
            print('You need to choose a smaller number for spectra to select.')
        else:
            # values from the background fit and the SelectPeak-funtion are used
            # in the following
            y_fit = self.yreduced[spectrum] - self.baseline[spectrum]

            # Create a composed model of a ConstantModel plus
            # models supplied by lmfit.models
            ramanmodel = ConstantModel() # Add a constant for a better fit

            # go through all defined peaks
            for peaktype in peaks:
                peakfile = self.folder + '/temp/locpeak_' + peaktype + '_' +\
                           label + '.dat'
                # check, if the current peaktype has been selected
                if(os.stat(peakfile).st_size > 0):
                    # get the selected peak positions
                    xpeak, ypeak = np.genfromtxt(peakfile, unpack = True)

                    # necessary if only one peak is selected
                    if type(xpeak) == np.float64:
                        xpeak = [xpeak]
                        ypeak = [ypeak]

                    #define starting values for the fit
                    for i in range(0, len(xpeak)):
                        # prefix for the different peaks from one model
                        prefix = peaktype + '_p'+ str(i + 1) + '_'
                        temp = ChoosePeakType(peaktype, prefix)
                        temp = StartingParameters(xpeak, ypeak, i, temp, peaks)

                        ramanmodel += temp # add the models to 'ramanmodel'

            # create the fit parameters of the background substracted fit
            pars = ramanmodel.make_params()
            # fit the data to the created model
            self.fitresult_peaks[spectrum] = ramanmodel.fit(y_fit, pars,
                                                  x = self.xreduced[spectrum],
                                                  method = 'leastsq',
                                                  scale_covar = True)
            # print which fit is conducted
            print('Spectrum ' + label + ' fitted')

            # calculate the fit line
            self.fitline[spectrum] = ramanmodel.eval(self.fitresult_peaks[spectrum].params,
                                                    x = self.xreduced[spectrum])
            # calculate all components
            self.comps = self.fitresult_peaks[spectrum].eval_components(x = self.xreduced[spectrum])

            # show fit report in terminal
            if report:
                print(self.fitresult_peaks[spectrum].fit_report(min_correl=0.5))

            # Plot the raw sprectrum, the fitted data, the background, and the confidence interval
            fig, ax = plt.subplots()
            ax.plot(self.xreduced[spectrum],
                    self.yreduced[spectrum] * self.ymax[spectrum],
                    'b.', alpha = 0.8, markersize = 1, zorder = 0, label = 'Data') # Measured data
            ax.plot(self.xreduced[spectrum],
                    (self.baseline[spectrum] + self.comps['constant']) * self.ymax[spectrum],
                    'k-', linewidth = 1, zorder = 0, label = 'Background') # Fitted background
            ax.plot(self.xreduced[spectrum],
                    (self.fitline[spectrum] + self.baseline[spectrum]) * self.ymax[spectrum],
                    'r-', linewidth = 0.5, zorder = 1, label = 'Fit') # Fitted spectrum

            # plot the single peaks
            for name in self.comps.keys():
                if (name != 'constant'):
                    ax.plot(self.xreduced[spectrum],
                            (self.comps[name] + self.baseline[spectrum] + self.comps['constant']) * self.ymax[spectrum],
                            'k-', linewidth = 0.5, zorder = 0)


            # check if errors exist.
            # calculate and plot confidence band
            if self.fitresult_peaks[spectrum].params['c'].stderr is not None:
                # calculate confidence band
                self.confidence[spectrum] = self.fitresult_peaks[spectrum].eval_uncertainty(x = self.xreduced[spectrum],
                                                        sigma=3)
                # plot confidence band
                ax.fill_between(self.xreduced[spectrum],
                     (self.fitline[spectrum] + self.baseline[spectrum] + self.confidence[spectrum]) * self.ymax[spectrum],
                     (self.fitline[spectrum] + self.baseline[spectrum] - self.confidence[spectrum]) * self.ymax[spectrum],
                     color = 'r', linewidth = 1, alpha = 0.5, zorder = 1, label = '3$\sigma$') # plot confidence band

            fig.legend(loc = 'upper right')
            fig.savefig(self.folder + '/results/plot/fitplot_' + label + '.pdf')
            fig.savefig(self.folder + '/results/plot/fitplot_' + label + '.png', dpi=300)

            if show:
                figManager = plt.get_current_fig_manager()  # get current figure
                figManager.window.showMaximized()           # show it maximized
                plt.show()

    # fit all spectra
    def FitAllSpectra(self, peaks, show=False, report=False):
        for i in range(self.numberOfFiles):
            self.FitSpectrum(peaks, spectrum=i, label=str(i+1).zfill(4), show=show, report=report)

    # Save the Results of the fit in a file using
    def SaveFitParams(self, peaks, label='', spectrum=0):
        if spectrum >= self.numberOfFiles:
            print('You need to choose a smaller number for spectra to select.')
        else:
            # get the data to be stored
            fitparams_back = self.fitresult_bg[spectrum].params     # Fitparameter Background
            fitparams_peaks = self.fitresult_peaks[spectrum].params # Fitparamter Peaks

            # save background parameters
            f = open(self.folder + '/results/fitparameter/spectra/' + label + '_background.dat','a')
            # iterate through all the background parameters
            for name in fitparams_back:
                # get parameters for saving
                parametervalue = fitparams_back[name].value * self.ymax[spectrum]
                parametererror = fitparams_back[name].stderr * self.ymax[spectrum]

                # add background from peaks fit
                if name == 'c0':
                    parametervalue += fitparams_peaks['c'].value * self.ymax[spectrum]
                    if fitparams_peaks['c'].stderr is not None:
                        parametererror = np.sqrt(parametererror**2 +\
                                         (fitparams_peaks['c'].stderr*self.ymax[spectrum])**2)

                f.write(name.ljust(5) + '{:>13.5f}'.format(parametervalue)
                                      + ' +/- ' + '{:>11.5f}'.format(parametererror)
                                      + '\n')
            f.close()

            # find all prefixes used in the current model
            modelpeaks = re.findall('prefix=\'(.*?)\'',
                                    self.fitresult_peaks[spectrum].model.name)

            # iterate through all peaks used in the current model
            for peak in modelpeaks:
                peakfile = self.folder + '/results/fitparameter/spectra/' + label +\
                           '_' + peak + '.dat'
                f = open(peakfile, 'a')
                # iterate through all fit parameters
                for name in fitparams_peaks.keys():
                    # and find the current peak
                    peakparameter = re.findall(peak, name)
                    if peakparameter:
                        # create file for each parameter
                        allpeaks = self.folder + '/results/fitparameter/peakwise/' + name + '.dat'
                        g = open(allpeaks, 'a')

                        # get parameters for saving
                        peakparameter = name.replace(peak, '')
                        parametervalue = fitparams_peaks[name].value
                        parametererror = fitparams_peaks[name].stderr

                        # if parameter is height or amplitude or intensity
                        # it has to be scaled properly as the fit was normalized
                        if (peakparameter == 'amplitude') or (peakparameter == 'height') or (peakparameter == 'intensity'):
                            parametervalue = parametervalue * self.ymax[spectrum]
                            if parametererror is not None:
                                parametererror = parametererror * self.ymax[spectrum]

                        # if there is no error set the value to -1
                        if parametererror is None:
                            parametererror = -1.0

                        # write to file
                        f.write(peakparameter.ljust(12) + '{:>13.5f}'.format(parametervalue)
                                              + ' +/- ' + '{:>11.5f}'.format(parametererror)
                                              + '\n')
                        g.write('{:>13.5f}'.format(parametervalue) + '\t' + '{:>11.5f}'.format(parametererror) + '\n')
                        g.close()
                f.close()

            # save the fitlines
            for line in self.fitline:
                file = self.folder + '/results/fitlines/' + label + '_fitline.dat'
                np.savetxt(file, np.column_stack([self.xreduced[spectrum], line * self.ymax[spectrum]]))

            # save the fitlines
            for line in self.baseline:
                file = self.folder + '/results/baselines/' + label + '_baseline.dat'
                np.savetxt(file, np.column_stack([self.xreduced[spectrum], line * self.ymax[spectrum]]))

            # print which spectrum is saved
            print('Spectrum ' + label + ' saved')

    # Save all the results
    def SaveAllFitParams(self, peaks):
        for i in range(self.numberOfFiles):
            self.SaveFitParams(peaks, spectrum=i, label=str(i+1).zfill(4))
