"""
This module contains the spectrum class to work with spectral data.
"""

import os

import pywt                             # for wavelet operations
from statsmodels.robust import mad      # median absolute deviation from array
from scipy.optimize import curve_fit    # for interpolating muons

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from lmfit.models import *

from starting_params import *
from functions import *

import decimal                          # to get exponent of missingvalue

# Class for spectra (under development)
class spectrum(object):
    """
    Class for working with spectral data.

    Parameters
    ----------
    foldername : string
        The folder of interest has to be in the current directory.
        The data will be prepared to analyze spectral data.
    """

    def __init__(self, foldername):
        self.folder = foldername
        self.listOfFiles, self.numberOfFiles = GetFolderContent(self.folder, 'txt')
        self.x, self.y = GetMonoData(self.listOfFiles)
        if self.numberOfFiles == 1:
            self.x = np.array([self.x])
            self.y = np.array([self.y])

        # set value of missing values
        self.missingvalue = 1.11

        # get maximum and norm from each spectrum
        self.ymax = np.max(self.y, axis=1)
        self.ynormed = self.y/self.ymax[:,None]

        # selected spectrum
        self.xreduced = None
        self.yreduced = None

        # array to contain denoised data
        self.ydenoised = [None] * self.numberOfFiles

        # denoised values
        self.muonsgrouped = [None] * self.numberOfFiles

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
            os.makedirs(self.folder + '/results/denoised/')

        # save missing value
        self.missingvalueexponent = decimal.Decimal(str(self.missingvalue)).as_tuple().exponent * (-1)
        np.savetxt(self.folder + '/temp/missingvalue.dat', [self.missingvalue],
                   fmt='%.{}f'.format(self.missingvalueexponent))

        # names of files created during the procedure
        self.fSpectrumBorders = None
        self.fBaseline = None

        # fit parameters
        self.fitresult_bg = [None] * self.numberOfFiles
        self.baseline = [None] * self.numberOfFiles
        self.fitresult_peaks = [None] * self.numberOfFiles
        self.fitline = [None] * self.numberOfFiles
        self.confidence = [None] * self.numberOfFiles

    # function that plots regions chosen by clicking into the plot
    def PlotVerticalLines(self, color, fig):
        """
        Function that plots regions chosen by clicking into the plot

        Parameters
        ----------
        color : string
            Defines color of the vertical lines and the region.
        fig : matplotlib.figure.Figure
            Figure to choose the region from.

        Returns
        -------
        xregion : array
            Points selected from the user.
        """
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
        """
        Function that lets the user select a region of interest. It saves the
        selected region to '/temp/spectrumborders' + label + '.dat'

        Parameters
        ----------
        spectrum : int, default: 0
            Defines which spectrum in the analysis folder is chosen.
        label : string, default: ''
            Label for the spectrumborders file in case you want to have
            different borders for different files.

        """
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

    # function to split muons from each other
    def SplitMuons(self, indices, prnt=False):
        """

        """
        # create multidimensional list
        grouped_array = [[]]

        # muon counter
        muons = 0

        # loop through list and find gaps in the list to group the muons
        for i in range(0, len(indices) - 1):
            # as long as the index is increasing by one the indices belong to one muon
            if indices[i] + 1 == indices[i + 1]:
                grouped_array[muons].append(indices[i])
            # as soon as there is a jump, a new muon was found and is added to the list
            else:
                grouped_array[muons].append(indices[i])
                grouped_array.append([])
                muons += 1
        if len(indices) > 0:
            # add the last element to the list and
            grouped_array[muons].append(indices[-1])
            # print the number of muons found
            if prnt:
                print(str(muons + 1) + ' muons have been found.')

        return grouped_array

    # detect muons for removal and returns non vanishing indices
    def DetectMuonsWavelet(self, spectrum=0, thresh_mod=1, wavelet='sym8', level=1, prnt=False):
        """

        """
        # calculate wavelet coefficients
        coeff = pywt.wavedec(self.yreduced[spectrum], wavelet)    # symmetric signal extension mode

        # calculate a threshold
        sigma = mad(coeff[-level])
        threshold = sigma * np.sqrt(2 * np.log(len(self.yreduced[spectrum]))) * thresh_mod

        # detect spikes on D1 details (written in the last entry of coeff)
        # calculate thresholded coefficients
        for i in range(1, len(coeff)):
            coeff[i] = pywt.threshold(coeff[i], value=threshold, mode='soft')
        # set everything but D1 level to zero
        for i in range(0, len(coeff)-1):
            coeff[i] = np.zeros_like(coeff[i])

        # reconstruct the signal using the thresholded coefficients
        muonfree = pywt.waverec(coeff, wavelet)

        if (len(self.yreduced[spectrum]) % 2) == 0:
            # get non vanishing indices
            indices = np.nonzero(muonfree)[0]
            self.muonsgrouped[spectrum] = self.SplitMuons(indices, prnt=prnt)
        else:
            # get non vanishing indices
            indices = np.nonzero(muonfree[:-1])[0]
            self.muonsgrouped[spectrum] = self.SplitMuons(indices, prnt=prnt)

    # detect all muons in all spectra
    def DetectAllMuons(self, prnt=False):
        """
        Wrapper around :func:`~spectrum.DetectMuonsWavelet` that iterates over all spectra
        given.
        """
        for i in range(self.numberOfFiles):
            self.DetectMuonsWavelet(spectrum=i, prnt=prnt)

    # linear function for muon approximation
    def linear(self, x, m, b):
        """

        """
        return x * m + b

    # approximate muon by linear function
    def RemoveMuons(self, spectrum=0, prnt=False):
        """

        """
        # check if there are any muons in the spectrum given
        if len(self.muonsgrouped[spectrum][0]) > 0:
            # remove each muon
            for muon in self.muonsgrouped[spectrum]:
                # calculate limits for indices to use for fitting
                limit = int(len(muon)/4)
                lower = muon[:limit]
                upper = muon[-limit:]
                fit_indices = np.append(lower, upper)

                # fit to the data
                popt, pcov = curve_fit(linear, self.xreduced[spectrum, fit_indices], self.yreduced[spectrum, fit_indices])

                # calculate approximated y values and remove muon
                for index in muon[limit:-limit]:
                    self.yreduced[spectrum, index] = linear(self.xreduced[spectrum, index], *popt)
        elif prnt:
            print('No muons found.')

    # remove all muons
    def RemoveAllMuons(self, prnt=False):
        """
        Wrapper around :func:`~spectrum.RemoveMuons` that iterates over all spectra
        given.
        """
        for i in range(self.numberOfFiles):
            self.RemoveMuons(spectrum=i, prnt=prnt)

    # smooth spectrum by using wavelet transform and soft threshold
    def WaveletSmoothSpectrum(self, spectrum=0, wavelet='sym8', level=2, sav=False):
        """

        """
        # calculate wavelet coefficients
        coeff = pywt.wavedec(self.yreduced[spectrum], wavelet)

        # calculate a threshold
        sigma = mad(coeff[-level])
        threshold = sigma * np.sqrt(2 * np.log(len(self.yreduced[spectrum])))

        # calculate thresholded coefficients
        for i in range(1,len(coeff)):
            coeff[i] = pywt.threshold(coeff[i], value=threshold, mode='soft')

        # reconstruct the signal using the thresholded coefficients
        denoised = pywt.waverec(coeff, wavelet)

        # return the value of denoised except for the last value
        if (len(self.yreduced) % 2) == 0:
            self.ydenoised[spectrum] = denoised
        else:
            self.ydenoised[spectrum] = denoised[:-1]

        # save denoised data
        if sav:
            savefile = self.folder + '/results/denoised/' + str(spectrum + 1).zfill(4) + '.dat'
            np.savetxt(savefile, np.column_stack([self.xreduced[spectrum], self.ydenoised[spectrum]]))

    # smooth all spectra
    def WaveletSmoothAllSpectra(self, level=2, sav=False, wavelet='sym8'):
        """
        Wrapper around :func:`~spectrum.WaveletSmoothSpectrum` that iterates
        over all spectra given.
        """
        for i in range(self.numberOfFiles):
            self.WaveletSmoothSpectrum(spectrum=i, level=level, sav=sav, wavelet=wavelet)

    #function to select the data that is relevent for the background
    def SelectBaseline(self, spectrum=0, label='', color='b'):
        """
        Function that lets the user distinguish between the background and the signal. It saves the
        selected regions to '/temp/baseline' + label + '.dat'.

        Parameters
        ----------
        spectrum : int, default: 0
            Defines which spectrum in the analysis folder is chosen.
        label : string, default: ''
            Label for the spectrumborders file in case you want to have
            different borders for different files.
        color : string, default 'b'
            Color of the plotted spectrum.

        """
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
    def FitBaseline(self, spectrum=0, show=False, degree=3):
        """
        Fit of the baseline by using the 
        `PolynomalModel() <https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.PolynomialModel>`_ 
        from lmfit.

        Parameters
        ----------
        spectrum : int, default: 0
            Defines which spectrum in the analysis folder is chosen.
        show : boolean, default: False
            Decides whether the a window with the fitted baseline is opened or not.
        degree : int, default: 3
            Degree of the polynomial that describes the background.

        """

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
            background = PolynomialModel(degree=degree)
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
    def FitAllBaselines(self, show=False, degree=3):
        """
        Wrapper around :func:`~spectrum.FitBaseline` that iterates over all spectra
        given.
        """
        for i in range(self.numberOfFiles):
            self.FitBaseline(spectrum=i, show=show, degree=degree)

    # function that plots the dots at the peaks you wish to fit
    def PlotPeaks(self, fig):
        """
        Plot the selected peaks while :func:`~spectrum.SelectPeaks` is running.

        Parameters
        ----------
        fig : string
            Currently displayed window that shows the spectrum as well as the selected peaks.

        """
        xpeak = []  # x and
        ypeak = []  # y arrays for peak coordinates

        def onclickpeaks(event):
            if event.button:
                xpeak.append(event.xdata)               # append x data and
                ypeak.append(event.ydata)               # append y data
                plt.plot(event.xdata, event.ydata, 'ro',# plot the selected peak
                        picker=5)
                fig.canvas.draw()                       # and show it

        # actual execution of the defined function oneclickpeaks
        cid = fig.canvas.mpl_connect('button_press_event', onclickpeaks)
        figManager = plt.get_current_fig_manager()  # get current figure
        figManager.window.showMaximized()           # show it maximized

        return xpeak, ypeak

    # function that allows you to select Voigt-, Fano-, Lorentzian-,
    # and Gaussian-peaks for fitting
    def SelectPeaks(self, peaks, spectrum=0, label=''):
        """
        Function that lets the user select the maxima of the peaks to fit 
        according to their line shape (Voigt, Fano, Lorentzian, Gaussian). 
        The positions (x- and y-value) are taken as initial values in the function
        :func:`~spectrum.FitSpectrum`.
        It saves the selected positions to '/temp/locpeak_' + peaktype + '_' +\label + '.dat'.

        Parameters
        ----------
        peaks : list, default: ['breit_wigner', 'lorentzian']
            Possible line shapes of the peaks to fit are 
            'breit_wigner', 'lorentzian', 'gaussian', and 'voigt'.
        spectrum : int, default: 0
            Defines which spectrum in the analysis folder is chosen.
        label : string, default: ''
            Label for the spectrumborders file in case you want to have
            different borders for different files.

        """
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
                ax.set_title('Spectrum ' + label +
                             '\nBackground substracted, normalized spectrum\n\
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
        """
        Wrapper around :func:`~spectrum.SelectPeaks` that iterates over all spectra
        given.
        """
        for i in range(self.numberOfFiles):
            self.SelectPeaks(peaks, spectrum=i, label=str(i+1).zfill(4))


    def FitSpectrum(self, peaks, spectrum=0, label='', show=True, report=False):
        """
        Conducts the actual fit of the spectrum. A `CompositeModel() <https://lmfit.github.io/lmfit-py/model.html#lmfit.model.CompositeModel>`_ 
        consisting of an offset (`ConstantModel() <https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.ConstantModel>`_) 
        and the line shape of the selected peaks is used.
        The fit functions of the selectable peaks are described in detail in :func:`~starting_params.ChoosePeakType`
        and the choice of the initial values in :func:`~starting_params.StartingParameters`.
        In addition a plot of the fitted spectrum is created including the :math:`3\sigma`-confidence-band.  

        

        It saves the figures to '/results/plot/fitplot_' + label + '.pdf' and '/results/plot/fitplot_' + label + '.png'.
        The fit parameters are saved in the function :func:`~spectrum.SaveFitParams`.

        Parameters
        ----------
        peaks : list, default: ['breit_wigner', 'lorentzian']
            Possible line shapes of the peaks to fit are 
            'breit_wigner', 'lorentzian', 'gaussian', and 'voigt'.
        spectrum : int, default: 0
            Defines which spectrum in the analysis folder is chosen.
        label : string, default: ''
            Label for the spectrumborders file in case you want to have
            different borders for different files.
        show : boolean, default=True
            If True the plot of the fitted spectrum is shown.
        report : boolean, default = False
            If True the `fit_report <https://lmfit.github.io/lmfit-py/fitting.html#getting-and-printing-fit-reports>`_ is shown in the terminal including the correlations of the fit parameters.

        """
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
                        temp = ChoosePeakType(peaktype, i)
                        temp = StartingParameters(temp, peaks, xpeak, ypeak, i)

                        ramanmodel += temp # add the models to 'ramanmodel'

            # create the fit parameters of the background substracted fit
            pars = ramanmodel.make_params()
            # fit the data to the created model
            self.fitresult_peaks[spectrum] = ramanmodel.fit(y_fit, pars,
                                                            x = self.xreduced[spectrum],
                                                            method = 'leastsq',
                                                            scale_covar = True)
            # calculate the fit line
            self.fitline[spectrum] = ramanmodel.eval(self.fitresult_peaks[spectrum].params,
                                                     x = self.xreduced[spectrum])

            # calculate all components
            self.comps = self.fitresult_peaks[spectrum].eval_components(x = self.xreduced[spectrum])

            # check if ramanmodel was only a constant
            if ramanmodel.name == ConstantModel().name:
                # set fitline and constant to zero
                self.fitline[spectrum] = np.zeros_like(self.baseline[spectrum])
                self.comps['constant'] = 0

            # print which fit is conducted
            print('Spectrum ' + label + ' fitted')

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
            plt.title('Fit to ' + self.folder + ' spectrum ' + str(spectrum + 1))

            # label the x and y axis
            plt.ylabel('Scattered light intensity (arb. u.)')
            plt.xlabel('Raman shift (cm$^{-1}$)')

            # save figures
            fig.savefig(self.folder + '/results/plot/fitplot_' + label + '.pdf')
            fig.savefig(self.folder + '/results/plot/fitplot_' + label + '.png', dpi=300)

            if show:
                figManager = plt.get_current_fig_manager()  # get current figure
                figManager.window.showMaximized()           # show it maximized
                plt.show()

    # fit all spectra
    def FitAllSpectra(self, peaks, show=False, report=False):
        """
        Wrapper around :func:`~spectrum.FitSpectrum` that iterates over all spectra
        given.
        """
        for i in range(self.numberOfFiles):
            self.FitSpectrum(peaks, spectrum=i, label=str(i+1).zfill(4), show=show, report=report)

    # Save the Results of the fit in a file using
    def SaveFitParams(self, peaks, usedpeaks=[], label='', spectrum=0):
        """
        The optimized line shapes as well as the corresponding fit parameters with uncertainties are saved in several folders, all contained in the folder 'results/'.
        The optimized baseline from each spectrum can be found in '/results/baselines/' + label + '_baseline.dat' and the 
        line shape of the background subtracted spectrum in '/results/fitlines/' + label + '_fitline.dat'. 
        The folder '/results/fitparameter/spectra/' + label + '_' + peak + '.dat' contains files each of which with one parameter including its uncertainty.
        The parameters are also sorted by peak for different spectra. This is stored in '/results/fitparameter/peakwise/' + name + '.dat'.

        Parameters
        ----------
        peaks : list, default: ['breit_wigner', 'lorentzian']
            Possible line shapes of the peaks to fit are 
            'breit_wigner', 'lorentzian', 'gaussian', and 'voigt'.
        usedpeaks : 
            ?
        label : string, default: ''
            Label for the spectrumborders file in case you want to have
            different borders for different files.
        spectrum : int, default: 0
            Defines which spectrum in the analysis folder is chosen. including the correlations of the fit parameters.

        """
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

            # enter value for non used peaks
            if usedpeaks != []:
                # calculate the peaks that have not been used
                unusedpeaks = list(set(usedpeaks)-set(modelpeaks))

                # save default value for each parameter of unused peaks
                for peak in unusedpeaks:
                    # get the peaktype and number of the peak
                    number = int(re.findall('\d', peak)[0]) - 1
                    peaktype = re.sub('_p.*_', '', peak)

                    # create model with parameters as before
                    model = ChoosePeakType(peaktype, number)
                    model = StartingParameters(model, peaks)
                    model.make_params()

                    # go through all parameters and write missing values
                    for parameter in model.param_names:
                        peakfile = self.folder + '/results/fitparameter/peakwise/' + parameter + '.dat'
                        # open file and write missing values
                        f = open(peakfile, 'a')
                        f.write('{:>13.5f}'.format(self.missingvalue) + '\t' + '{:>11.5f}'.format(self.missingvalue) + '\n')
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
        """
        Wrapper around :func:`~spectrum.SaveFitParams` that iterates over all spectra
        given.
        """
        # find all peaks that were fitted and generate a list
        allpeaks = []
        for i in range(self.numberOfFiles):
            allpeaks.extend(re.findall('prefix=\'(.*?)\'',
                                    self.fitresult_peaks[i].model.name))
        allusedpeaks = list(set(allpeaks))

        for i in range(self.numberOfFiles):
            self.SaveFitParams(peaks, usedpeaks=allusedpeaks, spectrum=i, label=str(i+1).zfill(4))
