"""
This module contains the spectrum class to work with spectral data.
"""

import os

import pywt                             # for wavelet operations
from statsmodels.robust import mad      # median absolute deviation from array
from scipy.optimize import curve_fit    # for interpolating muons
from scipy.interpolate import UnivariateSpline  # for estimating FWHM of
                                                # unwanted frequencies

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from lmfit.models import *

from starting_params import *
from functions import *

import decimal                          # to get exponent of missingvalue

from sklearn import preprocessing


# Class for spectra (under development)
class spectrum(object):
    """
    Class for working with spectral data.

    Parameters
    ----------
    foldername : string
        The folder of interest has to be in the current directory.
        The data will be prepared to analyze spectral data.
    measurement : sting
        Defines the type of measurement used to generate the data that
        should be analyzed now.
        Possible measurements are
        - xps
    """

    def __init__(self, foldername, **kwargs):
        self.second_analysis = False #boolean value for exception handling if several spectra are analyzed for the second time
        self.folder = foldername
        self.listOfFiles, self.numberOfFiles = GetFolderContent(self.folder,
                                                                'txt')
        self.labels = [files.split('/')[-1].split('.')[-2] for files in self.listOfFiles]

        if os.path.exists(self.folder + '/results'):
            self.indices = np.arange(self.numberOfFiles) #indices of the files that are analyzed again, default are all spectra
            self.second_analysis = True
            answer = input('These spectra have been analyzed already. Do you want to analyze all of them again? (y/n) \n')
            if answer == 'y':
                pass
            elif answer == 'n':
                for label in self.labels:
                    print(f'{label} \n')
                print('Enter the spectra that you want to analyze again. (Finish the selection with x).')
                list_of_incides = []
                list_of_labels = []
                list_of_filenames = []
                while True:
                    label = input()
                    if label == 'x':
                        break
                    if label in self.labels:
                        list_of_labels.append(label)
                        index = self.labels.index(label)
                        list_of_incides.append(index)
                        list_of_filenames.append(self.listOfFiles[index])
                    else:
                        print('This spectrum does not exist.')
                self.listOfFiles = list_of_filenames #update listOfFiles
                self.labels = list_of_labels #update labels
                self.indices = list_of_incides ##update indices of the files that are analyzed again
                self.numberOfFiles = len(self.labels) #update number of files

        # default are raman measurements
        if kwargs == {}:
            print('Raman measurements')
            self.x, self.y = GetMonoData(self.listOfFiles)
        elif kwargs['measurement'] == 'xps':
            print('XPS measurements')
            self.x, self.y = GetMonoData(self.listOfFiles, measurement='xps')



        self.x, self.y = GetMonoData(self.listOfFiles)
        if self.numberOfFiles == 1:
            self.x = np.array([self.x])
            self.y = np.array([self.y])

        # set value of missing values
        self.missingvalue = 1.11

        # selected spectrum
        self.xreduced = None
        self.yreduced = None

        # array to contain denoised data
        self.ydenoised = [None] * self.numberOfFiles

        # array containing yderived
        self.dy = [None] * self.numberOfFiles

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
            os.makedirs(self.folder + '/results/grouped_spectra/')
            os.makedirs(self.folder + '/results/derived/')


        # save missing value
        missingvaluedecimal = decimal.Decimal(str(self.missingvalue))
        self.missingvalueexponent = missingvaluedecimal.as_tuple().exponent*(-1)
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

        # boolean values for exception handling
        self.critical = [False for files in self.listOfFiles]

    # function that plots regions chosen by clicking into the plot
    def PlotVerticalLines(self, color, fig):
        """
        Function to select horizontal regions by clicking into the plot.

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
        Function that lets the user select a region by running the
        method :func:`PlotVerticalLines() <spectrum.spectrum.PlotVerticalLines()>`.
        The region of interest is only chosen for one specific spectrum and assumed to be the same for all others.
        The borders of the selected region is saved to '/temp/spectrumborders' + label + '.dat'

        Parameters
        ----------
        spectrum : int, default: 0
            Defines which of the spectra is chosen to select the region of interest.

        label : string, default: ''
            Label for the spectrumborders file in case you want to have
            different borders for different files.

        """
        if spectrum >= self.numberOfFiles:
            print('You need to choose a smaller number for spectra to select.')
        else:
            # plot spectrum
            fig, ax = plt.subplots()
            ax.plot(self.x[spectrum], self.y[spectrum],
                    'b.', label = 'Data')
            ax.set_title('Select the part of the spectrum you wish to consider\
                          by clicking into the plot.')

            # select region of interest
            xregion = self.PlotVerticalLines('green', fig)

            plt.legend(loc='upper right')
            plt.show()

            self.yreduced = self.y[:, (self.x[spectrum] > xregion[0]) &
                                      (self.x[spectrum] < xregion[-1])]
            self.xreduced = self.x[:, (self.x[spectrum] > xregion[0]) &
                                      (self.x[spectrum] < xregion[-1])]
            # save spectrum borders
            self.fSpectrumBorders = (self.folder + '/temp/spectrumborders'
                                     + label + '.dat')
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
            # as the indices are incrementing they belong to the same muon
            if indices[i] + 1 == indices[i + 1]:
                grouped_array[muons].append(indices[i])
            # as soon as there is a jump, a new muon was found
            # and is added to the list
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
    def DetectMuonsWavelet(self, spectrum=0, thresh_mod=1, wavelet='sym8',
                                 level=1, prnt=False):
        """

        """
        # calculate wavelet coefficients
        # with symmetric signal extension mode
        coeff = pywt.wavedec(self.yreduced[spectrum], wavelet)

        # calculate a threshold
        sigma = mad(coeff[-level])
        threshold = (sigma * np.sqrt(2 * np.log(len(self.yreduced[spectrum])))
                     * thresh_mod)

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
        Wrapper around :func:`~spectrum.DetectMuonsWavelet` that iterates
        over all spectra given.
        """
        for i in range(self.numberOfFiles):
            self.DetectMuonsWavelet(spectrum=i, prnt=prnt)

    # linear function for muon approximation
    def linear(self, x, slope, intercept):
        """
        Parameters
        ----------
        x : float

        slope : float
            Slope of the linear model.

        intercept : float
            Y-intercept of the linear model.

        Returns
        -------
        x * slope + intercept : float
            Calculated y value for inserted x, slope and intercept.
        """
        return x * slope + intercept

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
                popt, pcov = curve_fit(linear,
                                       self.xreduced[spectrum, fit_indices],
                                       self.yreduced[spectrum, fit_indices])

                # calculate approximated y values and remove muon
                for index in muon[limit:-limit]:
                    self.yreduced[spectrum, index] = linear(
                                                self.xreduced[spectrum, index],
                                                *popt)
        elif prnt:
            print('No muons found.')

    # remove all muons
    def RemoveAllMuons(self, prnt=False):
        """
        Wrapper around :func:`~spectrum.RemoveMuons` that iterates over
        all spectra given.
        """
        for i in range(self.numberOfFiles):
            self.RemoveMuons(spectrum=i, prnt=prnt)

    # Normalize all spectra
    def NormalizeAll(self):
        self.ymax = np.max(self.yreduced, axis=1)
        self.yreduced = self.yreduced/self.ymax[:, None]

    # smooth spectrum by using wavelet transform and soft threshold
    def WaveletSmoothSpectrum(self, spectrum=0, wavelet='sym8', level=2,
                              sav=False):
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
        if (len(self.yreduced[spectrum]) % 2) == 0:
            self.ydenoised[spectrum] = denoised
        else:
            self.ydenoised[spectrum] = denoised[:-1]

        # save denoised data
        if sav:
            savefile = (self.folder + '/results/denoised/den_'
                        + self.labels[spectrum] + '.dat')
            np.savetxt(savefile, np.column_stack([self.xreduced[spectrum],
                                                  self.ydenoised[spectrum] * self.ymax[spectrum]]))

    # smooth all spectra
    def WaveletSmoothAllSpectra(self, level=2, sav=False, wavelet='sym8'):
        """
        Wrapper around :func:`~spectrum.WaveletSmoothSpectrum` that iterates
        over all spectra given.
        """
        for i in range(self.numberOfFiles):
            self.WaveletSmoothSpectrum(spectrum=i, level=level, sav=sav,
                                       wavelet=wavelet)

    # plot the smoothed spectrum
    def PlotSmoothed(self, spectrum=0, level=2):
        # create figure
        fig, ax = plt.subplots()

        # plot the selected and the denoised data
        ax.plot(self.xreduced[spectrum],
                self.yreduced[spectrum] * self.ymax[spectrum],
                'b.', alpha = 0.8, markersize = 1, zorder = 0,
                label = 'Data')
        ax.plot(self.xreduced[spectrum],
                self.ydenoised[spectrum] * self.ymax[spectrum],
                'r-', linewidth = 1, zorder = 0, label = 'Denoised')

        # add legend, title and labels
        fig.legend()
        plt.title('Denoised data of ' + self.folder + ' spectrum '
                  + str(spectrum + 1))
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel('Scattered light intensity (arb. u.)')

        # save the figure
        fig.savefig(self.folder + '/results/plot/denoised_' + self.labels[spectrum] + '.pdf')
        fig.savefig(self.folder + '/results/plot/denoised_' + self.labels[spectrum] + '.png',
                    dpi=300)

        plt.close()

        print('Spectrum ' + self.labels[spectrum] + ' smoothed')

    def PlotAllSmoothedSpectra(self, level=2):
        for i in range(self.numberOfFiles):
            self.PlotSmoothed(spectrum=i, level=level)

    #function to select the data that is relevent for the background
    def SelectBaseline(self, spectrum=0, label='', color='b'):
        """
        Function that lets the user distinguish between the background and
        the signal. It runs the method :func:`PlotVerticalLines() <spectrum.spectrum.PlotVerticalLines()>`
        to select the regions that do
        not belong to the background and are therefore not used for background fit.
        The selected regions will be saved to '/temp/baseline' + label
        + '.dat'.

        Parameters
        ----------
        spectrum : int, default: 0
            Defines which of the spectra is chosen to distinguish
            between the background and the signal.

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
                        into the plot\n (1st-degree polynomial assumed)')

            # choose the region
            xregion = self.PlotVerticalLines('red', fig)

            plt.legend(loc = 'upper right')
            plt.show()
            self.fBaseline = self.folder + '/temp/baseline' + label + '.dat'
            np.savetxt(self.fBaseline, np.array(xregion))

    # actual fit of the baseline
    def FitBaseline(self, spectrum=0, show=False, degree=1):
        """
        Fit of the baseline by using the
        `PolynomalModel()
        <https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.PolynomialModel>`_
        from lmfit.

        Parameters
        ----------
        spectrum : int, default: 0
            Defines which of the spectra is modeled.

        show : boolean, default: False
            Decides whether the a window with the fitted baseline is opened
            or not.

        degree : int, default: 1
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
            self.fitresult_bg[spectrum] = background.fit(
                                        self.yreduced[spectrum, relevant],
                                        pars,
                                        x = self.xreduced[spectrum, relevant])

            # create baseline
            self.baseline[spectrum] = background.eval(
                                            self.fitresult_bg[spectrum].params,
                                            x = self.xreduced[spectrum])

            # plot the fitted function in the selected range
            if show:
                plt.plot(self.xreduced[spectrum], self.yreduced[spectrum],
                         'b.', label = 'Data')
                plt.plot(self.xreduced[spectrum], self.baseline[spectrum],
                         'r-', label = 'Baseline')
                plt.show()

    # fit all baselines
    def FitAllBaselines(self, show=False, degree=1):
        """
        Wrapper around :func:`~spectrum.FitBaseline` that iterates over
        all spectra given.
        """
        for i in range(self.numberOfFiles):
            self.FitBaseline(spectrum=i, show=show, degree=degree)

    # function that plots the dots at the peaks you wish to fit
    def PlotPeaks(self, fig, ax):
        """
        Plot the selected peaks while :func:`~spectrum.SelectPeaks` is running.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Currently displayed window that shows the spectrum as well as
            the selected peaks.

        ax : matplotlib.axes.Axes
            Corresponding Axes object to Figure object fig.

        Returns
        -------
        xpeak, ypeak : array
            Peak position (xpeak) and height (ypeak) selected from the user.

        """
        xpeak = []  # x and
        ypeak = []  # y arrays for peak coordinates
        global line
        line, = ax.plot(xpeak, ypeak, 'ro', markersize = 10)


        def onclickpeaks(event):

            if event.button == 1: # left mouse click to add data point
                xpeak.append(event.xdata)               # append x data and
                ypeak.append(event.ydata)               # append y data
                line.set_xdata(xpeak)
                line.set_ydata(ypeak)
                plt.draw()                       # and show it


            if event.button == 3: #right mouse click to remove data point
                if xpeak != []:
                    xdata_nearest_index = ( np.abs(xpeak - event.xdata) ).argmin() #nearest neighbour
                    del xpeak[xdata_nearest_index] #delte x
                    del ypeak[xdata_nearest_index] #and y data point
                    line.set_xdata(xpeak) #update x
                    line.set_ydata(ypeak) #and y data in plot
                    plt.draw()   # and show it


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
        The positions (x- and y-value) are taken as initial values in the
        function :func:`~spectrum.FitSpectrum`.
        It saves the selected positions to
        '/temp/locpeak_' + peaktype + '_' + label + '.dat'.

        Usage: Select peaks with left mouse click, remove them with right mouse click.

        Parameters
        ----------
        peaks : list, default: ['breit_wigner', 'lorentzian']
            Possible line shapes of the peaks to fit are
            'breit_wigner', 'lorentzian', 'gaussian', and 'voigt'.
            See lmfit documentation (https://lmfit.github.io/lmfit-py/builtin_models.html) for details.

        spectrum : int, default: 0
            Defines the spectrum which peaks are selected.

        label : string, default: ''
            Name of the spectrum is N if spectrum is (N-1).

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
                xpeak, ypeak = self.PlotPeaks(fig, ax)
                plt.show()
                # store the chosen initial values
                peakfile = self.folder + '/temp/locpeak_' + peaktype + '_' +\
                           label + '.dat'
                np.savetxt(peakfile, np.transpose([np.array(xpeak),
                                                   np.array(ypeak)]))

    # select all peaks
    def SelectAllPeaks(self, peaks):
        """
        Wrapper around :func:`~spectrum.SelectPeaks` that iterates over
        all spectra given.
        """
        for i in range(self.numberOfFiles):
            self.SelectPeaks(peaks, spectrum=i, label=self.labels[i])


    def FitSpectrum(self, peaks, spectrum=0, show=True, report=False, init_spectrum = None):
        """
        Conducts the actual fit of the spectrum. A `CompositeModel()
        <https://lmfit.github.io/lmfit-py/model.html#lmfit.model.CompositeModel>`_
        consisting of an offset (`ConstantModel()
        <https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.ConstantModel>`_)
        and the line shape of the selected peaks is used.
        The fit functions of the selectable peaks are described in detail in
        :func:`~starting_params.ChoosePeakType` and the choice of the initial
        values in :func:`~starting_params.StartingParameters`.
        In addition, a plot of the fitted spectrum is created including the
        :math:`3\sigma`-confidence-band.



        It saves the figures to '/results/plot/fitplot\_' + label + '.pdf' and
        '/results/plot/fitplot\_' + label + '.png'.
        The fit parameters are saved in the function
        :func:`~spectrum.SaveFitParams`.
        The fit parameters values that are derived from the fit parameters
        are individual for each line shape.
        Especially parameters of the BreitWignerModel() is adapted to our research.

        **VoigtModel():**
            |'center': x value of the maximum
            |'heigt': fit-function evaluation at 'center'
            |'amplitude': area under fit-function
            |'sigma': parameter related to gaussian-width
            |'gamma': parameter related to lorentzian-width
            |'fwhm_g': gaussian-FWHM
            |'fwhm_l': lorentzian-FWHM
            |'fwhm': FWHM

        **GaussianModel():**
            |'center': x value of the maximum
            |'heigt': fit-function evaluation at 'center'
            |'amplitude': area under fit-function
            |'sigma': parameter related to gaussian-width (variance)
            |'fwhm': FWHM

        **LorentzianModel():**
            |'center': x value of the maximum
            |'heigt': fit-function evaluation at 'center'
            |'amplitude': area under fit-function
            |'sigma': parameter related to lorentzian-width
            |'fwhm': FWHM

        **BreitWigner():**
            |'center': position of BWF resonance (not the maximum)
            |'sigma': FWHM of BWF resonance
            |'q': coupling coefficient of BWF is q^{-1}
            |'amplitude': A
            |'intensity': fit-function evaluation at 'center' (is A^2)
            |'heigt': y-value of the maximum (is A^2+1)

        Parameters
        ----------
        peaks : list, default: ['breit_wigner', 'lorentzian']
            Possible line shapes of the peaks to fit are
            'breit_wigner', 'lorentzian', 'gaussian', and 'voigt'.
        spectrum : int, default: 0
            Defines which spectrum to be modeled.
        label : string, default: ''
            Name of the spectrum is N if spectrum is (N-1).
        show : boolean, default=True
            If True the plot of the fitted spectrum is shown.
        report : boolean, default = False
            If True the `fit_report
            <https://lmfit.github.io/lmfit-py/fitting.html#getting-and-printing-fit-reports>`_
            is shown in the terminal including the correlations of
            the fit parameters.

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

                if init_spectrum != None:
                    init_peakfile = self.folder + '/temp/locpeak_' + peaktype + '_' +\
                               self.labels[init_spectrum] + '.dat'

                    # check, if the current peaktype has been selected
                    if(os.stat(init_peakfile).st_size > 0):
                        # get the selected peak positions
                        xpeak, ypeak = np.genfromtxt(init_peakfile, unpack = True)
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

                else:
                    peakfile = self.folder + '/temp/locpeak_' + peaktype + '_' +\
                           self.labels[spectrum] + '.dat'

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
            if init_spectrum != None:
                pars = self.fitresult_peaks[init_spectrum].params
            else:
                pars = ramanmodel.make_params()

            lower_bounds = np.array([pars[key].min for key in pars.keys()]) #arrays of lower and upper bounds of the start parameters
            upper_bounds = np.array([pars[key].max for key in pars.keys()])
            inf_mask = (upper_bounds != float('inf')) & (lower_bounds != float('-inf'))
            range_bounds = upper_bounds[inf_mask] - lower_bounds[inf_mask]


            # fit the data to the created model
            self.fitresult_peaks[spectrum] = ramanmodel.fit(y_fit, pars,
                                                    x = self.xreduced[spectrum],
                                                    method = 'leastsq',
                                                    scale_covar = True)


            best_values = np.array([self.fitresult_peaks[spectrum].params[key].value for key in self.fitresult_peaks[spectrum].params.keys()]) #best values of all parameters in the spectrum
            names = np.array([self.fitresult_peaks[spectrum].params[key].name for key in self.fitresult_peaks[spectrum].params.keys()]) #names of all parameters in the spectrum
            limit = 0.01 #percentage distance to the bounds leading to a warning
            lower_mask = best_values[inf_mask] <= lower_bounds[inf_mask] + limit * range_bounds #mask = True if best value is near lower bound
            upper_mask = best_values[inf_mask] >= lower_bounds[inf_mask] + (1 - limit) * range_bounds #mask = True if best value is near upper bound



            if True in lower_mask: #warn if one of the parameters has reached the lower bound
                warn(f'The parameter(s) {(names[inf_mask])[lower_mask]} of spectrum {self.listOfFiles[spectrum]} are close to chosen lower bounds.', ParameterWarning)
                self.critical[spectrum] = True

            if True in upper_mask: #warn if one of the parameters has reached the upper bound
                warn(f'The parameter(s) {(names[inf_mask])[upper_mask]} of spectrum {self.listOfFiles[spectrum]} are close to chosen upper bounds.', ParameterWarning)
                self.critical[spectrum] = True


            # calculate the fit line
            self.fitline[spectrum] = ramanmodel.eval(
                                        self.fitresult_peaks[spectrum].params,
                                        x = self.xreduced[spectrum])

            # calculate all components
            self.comps = self.fitresult_peaks[spectrum].eval_components(
                            x = self.xreduced[spectrum])

            # check if ramanmodel was only a constant
            if ramanmodel.name == ConstantModel().name:
                # set fitline and constant to zero
                self.fitline[spectrum] = np.zeros_like(self.baseline[spectrum])
                self.comps['constant'] = 0

            # print which fit is conducted
            print('Spectrum ' + self.labels[spectrum] + ' fitted')

            # show fit report in terminal
            if report:
                print(self.fitresult_peaks[spectrum].fit_report(min_correl=0.5))

            # Plot the raw sprectrum, the fitted data, the background,
            #and the confidence interval
            fig, ax = plt.subplots()
            # Measured data
            ax.plot(self.xreduced[spectrum],
                    self.yreduced[spectrum] * self.ymax[spectrum],
                    'b.', alpha = 0.8, markersize = 1, zorder = 0,
                    label = 'Data')
            # Fitted background
            ax.plot(self.xreduced[spectrum],
                    (self.baseline[spectrum]
                     + self.comps['constant']) * self.ymax[spectrum],
                    'k-', linewidth = 1, zorder = 0, label = 'Background')
            # Fitted spectrum
            ax.plot(self.xreduced[spectrum],
                    (self.fitline[spectrum]
                     + self.baseline[spectrum]) * self.ymax[spectrum],
                    'r-', linewidth = 0.5, zorder = 1, label = 'Fit')

            # plot the single peaks
            for name in self.comps.keys():
                if (name != 'constant'):
                    ax.plot(self.xreduced[spectrum],
                            (self.comps[name] + self.baseline[spectrum]
                             + self.comps['constant']) * self.ymax[spectrum],
                            'k-', linewidth = 0.5, zorder = 0)

            # check if errors exist.
            # calculate and plot confidence band
            if self.fitresult_peaks[spectrum].params['c'].stderr is not None:
                # calculate confidence band
                self.confidence[spectrum] = self.fitresult_peaks[spectrum].eval_uncertainty(
                                                x = self.xreduced[spectrum],
                                                sigma=3)
                # plot confidence band
                ax.fill_between(self.xreduced[spectrum],
                     (self.fitline[spectrum] + self.baseline[spectrum]
                      + self.confidence[spectrum]) * self.ymax[spectrum],
                     (self.fitline[spectrum] + self.baseline[spectrum]
                      - self.confidence[spectrum]) * self.ymax[spectrum],
                     color = 'r', linewidth = 1, alpha = 0.5, zorder = 1,
                     label = '3$\sigma$') # plot confidence band

            fig.legend(loc = 'upper right')
            plt.title('Fit to ' + self.folder + ' spectrum '
                      + str(spectrum + 1))

            # label the x and y axis
            plt.ylabel('Scattered light intensity (arb. u.)')
            plt.xlabel('Raman shift (cm$^{-1}$)')

            # save figures
            fig.savefig(self.folder + '/results/plot/fitplot_' + self.labels[spectrum] + '.pdf')
            fig.savefig(self.folder + '/results/plot/fitplot_' + self.labels[spectrum] + '.png',
                        dpi=300)

            if show:
                figManager = plt.get_current_fig_manager()  # get current figure
                figManager.window.showMaximized()           # show it maximized
                plt.show()

            # close the figure after it is saved and shown
            plt.close()

    # fit all spectra
    def FitAllSpectra(self, peaks, show=False, report=False):
        """
        Wrapper around :func:`~spectrum.FitSpectrum` that iterates over all spectra
        given.
        """
        for i in range(self.numberOfFiles):
            self.FitSpectrum(peaks, spectrum=i, show=show, report=report)

    # Save the Results of the fit in a file using
    def SaveFitParams(self, peaks, usedpeaks=[], label='', spectrum=0):
        """
        The optimized line shapes as well as the corresponding fit parameters
        with uncertainties are saved in several folders, all contained in the
        folder 'results/'.
        The optimized baseline from each spectrum can be found in
        '/results/baselines/' + label + '_baseline.dat' and the line shape
        of the background subtracted spectrum in
        '/results/fitlines/' + label + '_fitline.dat'.
        The folder '/results/fitparameter/spectra/' + label + '_' + peak
        + '.dat' contains files each of which with one parameter including
        its uncertainty.
        The parameters are also sorted by peak for different spectra.
        This is stored in '/results/fitparameter/peakwise/' + name + '.dat'
        including the correlations of the fit parameters.

        Parameters
        ----------
        peaks : list, default: ['breit_wigner', 'lorentzian']
            Possible line shapes of the peaks to fit are
            'breit_wigner', 'lorentzian', 'gaussian', and 'voigt'.

        usedpeaks : list, default []
            List of all actually used peaks.

        label : string, default: ''
            Name of the spectrum is N if spectrum is (N-1).

        spectrum : int, default: 0
            Defines which spectrum is chosen.

        """
        if spectrum >= self.numberOfFiles:
            print('You need to choose a smaller number for spectra to select.')
        elif self.fitresult_peaks[spectrum] == None:
            return
        else:
            # get the data to be stored
            fitparams_back = self.fitresult_bg[spectrum].params     # Background
            fitparams_peaks = self.fitresult_peaks[spectrum].params # Peaks

            # save background parameters
            f = open(self.folder + '/results/fitparameter/spectra/' + label
                     + '_background.dat','w')
            # iterate through all the background parameters
            for name in fitparams_back:
                # get parameters for saving
                parametervalue = (fitparams_back[name].value
                                 * self.ymax[spectrum])
                if fitparams_back[name].stderr is None:
                    parametererror = self.missingvalue
                else:
                    parametererror = (fitparams_back[name].stderr
                                     * self.ymax[spectrum])

                # add background from peaks fit
                if name == 'c0':
                    parametervalue += (fitparams_peaks['c'].value
                                      * self.ymax[spectrum])
                    if fitparams_peaks['c'].stderr is not None:
                        parametererror = np.sqrt(parametererror**2 +\
                                         (fitparams_peaks['c'].stderr
                                         * self.ymax[spectrum])**2)

                f.write(name.ljust(5) + '{:>13.5f}'.format(parametervalue)
                                      + ' +/- '
                                      + '{:>11.5f}'.format(parametererror)
                                      + '\n')
            f.close()

            # find all prefixes used in the current model
            modelpeaks = re.findall('prefix=\'(.*?)\'',
                                    self.fitresult_peaks[spectrum].model.name)

            # iterate through all peaks used in the current model
            for peak in modelpeaks:
                peakfile = (self.folder + '/results/fitparameter/spectra/'
                            + label + '_' + peak + '.dat')
                f = open(peakfile, 'w')
                # iterate through all fit parameters
                for name in fitparams_peaks.keys():
                    # and find the current peak
                    peakparameter = re.findall(peak, name)
                    if peakparameter:
                        # create file for each parameter
                        allpeaks = (self.folder
                                    + '/results/fitparameter/peakwise/'
                                    + name + '.dat')
                        if self.second_analysis == False:
                            g = open(allpeaks, 'a')

                        # get parameters for saving
                        peakparameter = name.replace(peak, '')
                        parametervalue = fitparams_peaks[name].value
                        parametererror = fitparams_peaks[name].stderr

                        # if parameter is height or amplitude or intensity
                        # it has to be scaled properly as the fit was normalized
                        if ((peakparameter == 'amplitude')
                            or (peakparameter == 'height')
                            or (peakparameter == 'intensity')):
                            parametervalue = (parametervalue
                                             * self.ymax[spectrum])
                            if parametererror is not None:
                                parametererror = (parametererror
                                                 * self.ymax[spectrum])

                        # if there is no error set the value to -1
                        if parametererror is None:
                            parametererror = -1.0

                        # write to file
                        f.write(peakparameter.ljust(12)
                                + '{:>13.5f}'.format(parametervalue)
                                + ' +/- ' + '{:>11.5f}'.format(parametererror)
                                + '\n')
                        if self.second_analysis == True: #if several spectra are analyzed again, the new values have to be put on the right position in the peakwise files for the parameters
                            values, stderrs = np.genfromtxt(allpeaks, unpack = True) #read existing values
                            values[self.indices[spectrum]] = parametervalue #update values
                            stderrs[self.indices[spectrum]] = parametererror
                            with open(allpeaks, 'w') as g: #write updated values to file
                                for i in range(len(values)):
                                    g.write('{:>13.5f}'.format(values[i])
                                    + '\t' + '{:>11.5f}'.format(stderrs[i])
                                    + '\n')
                        else:
                            g.write('{:>13.5f}'.format(parametervalue)
                                    + '\t' + '{:>11.5f}'.format(parametererror)
                                    + '\n')
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
                        peakfile = (self.folder
                                    + '/results/fitparameter/peakwise/'
                                    + parameter + '.dat')

                        if self.second_analysis == True:   #if several spectra are analyzed again, the new values have to be put on the right position in the peakwise files for the parameters
                            values, stderrs = np.genfromtxt(peakfile, unpack = True)
                            values[self.indices[spectrum]] = self.missingvalue
                            stderrs[self.indices[spectrum]] = self.missingvalue
                            with open(peakfile, 'w') as g:
                                for i in range(len(values)):
                                    g.write('{:>13.5f}'.format(values[i])
                                    + '\t' + '{:>11.5f}'.format(stderrs[i])
                                    + '\n')

                        else:
                            # open file and write missing values
                            f = open(peakfile, 'a')
                            f.write('{:>13.5f}'.format(self.missingvalue)
                                    + '\t' + '{:>11.5f}'.format(self.missingvalue)
                                    + '\n')
                            f.close()

            # save the fitline
            file = (self.folder + '/results/fitlines/'
                        + label + '_fitline.dat')
            np.savetxt(file, np.column_stack([self.xreduced[spectrum],
                           self.fitline[spectrum] * self.ymax[spectrum]]))

            # save the baseline
            file = (self.folder + '/results/baselines/'
                        + label + '_baseline.dat')
            np.savetxt(file, np.column_stack([self.xreduced[spectrum],
                           self.baseline[spectrum] * self.ymax[spectrum]]))

            # print which spectrum is saved
            print('Spectrum ' + label + ' saved')

    def GroupSpectra(self, sigma = 1.5, denoised=False):
        """
        Method uses principle component analysis (PCA) to group equal spectra. Therefore the first and second principle components are used.
        Since the different principle components contain more information if the corresponding eigenvalues are large,
        five splits on PC1 and three splits on PC2 are performed.

        Parameters
        ----------
        sigma : float, default: 1.5
            Number of standard deviations to mark outlined spectra.
            In order to detect outlined spectra, all the spectra with principle components
            outside of sigma standard deviations of the normal distributed sample are
            marked as outliners.
        """
        print('group spectra')

        #calculation of the priniple components
        if denoised:
            c = np.cov(self.ydenoised, rowvar = False)
        else:
            c = np.cov(self.yreduced, rowvar = False) #covariance matrix of the data set
        l, W = np.linalg.eigh(c) # eigenvalues and eigenvectors (columns of matrix W)

        l = l[::-1]
        W = W[:, ::-1] #because np.linalg.eigh() returns eigenvalues and corresponding eigenvectors in decending order, the order is inverted

        if denoised:
            y_Prime = self.ydenoised @ W
        else:
            y_Prime = self.yreduced @ W #transform the data set
        y_Prime = preprocessing.RobustScaler().fit_transform(y_Prime) #scale the data to a standard normal distribution


        #create groups by performing splits in the first two principle components
        rows = np.arange(np.shape(self.y)[0])
        self.groups = []
        interval_x = np.linspace(-sigma, sigma, 5) #array of split points in first principle component
        interval_y = np.linspace(-sigma, sigma, 3) #array of split points in second principle component
        for iter_x in range(len(interval_x)-1):
            for iter_y in range(len(interval_y)-1):
               self.groups.append(rows[(y_Prime[:, 0] >= interval_x[iter_x]) & (y_Prime[:, 0] < interval_x[iter_x + 1]) & (y_Prime[:, 1] >= interval_y[iter_y]) & (y_Prime[:, 1] < interval_y[iter_y + 1])])

               if (self.groups[-1].size != 0):

                    #plot the results
                    fig = plt.figure(figsize = (15, 10))
                    ax1 = fig.add_subplot(122)
                    ax2 = fig.add_subplot(223)
                    ax1.set_title(f'{len(self.groups[-1])} spectra')
                    ax1.set_xlabel('Raman Shift / $cm^{-1}$')
                    ax1.set_ylabel('Normed Intensity / a.u.')
                    for spectrum in self.groups[-1]:
                        if denoised:
                            pl = ax1.plot(self.xreduced[spectrum], self.ydenoised[spectrum], linewidth = 0.8)
                        else:
                            pl = ax1.plot(self.xreduced[spectrum], self.yreduced[spectrum], linewidth = 0.8)
                        clr = pl[0].get_color()
                        ax2.plot(y_Prime[:, 0][spectrum], y_Prime[:, 1][spectrum], marker = 'o', color = clr, markersize = 10)
                    ax3 = fig.add_subplot(221)
                    ax3.scatter(y_Prime[:, 0], y_Prime[:, 1], color =  'k')
                    ax3.fill_between(np.linspace(interval_x[iter_x], interval_x[iter_x+1], 100), interval_y[iter_y], interval_y[iter_y+1], color = 'g', alpha = 0.5)
                    ax3.set_xlabel('PC 1')
                    ax3.set_ylabel('PC 2')
                    ax2.set_xlabel('PC 1')
                    ax2.set_ylabel('PC 2')
                    fig.savefig(f'{self.folder}/results/grouped_spectra/group{len(self.groups)}.png', bbox_inches = 'tight', pad_inches = 0)
                    plt.close(fig)




        fig = plt.figure(figsize = (15, 10))
        ax1 = fig.add_subplot(122)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(221)
        ax3.scatter(y_Prime[:, 0], y_Prime[:, 1], color =  'k')
        ax3.fill_between(np.linspace(min([-sigma, min(y_Prime[:, 0])]), max([sigma, max(y_Prime[:, 0])]), 100), min(y_Prime[:, 1]), -sigma, color = 'r', alpha = 0.5, linewidth = 0)
        ax3.fill_between(np.linspace(min(y_Prime[:, 0]), -sigma, 100), max([-sigma, min(y_Prime[:, 1])]), sigma, color = 'r', alpha = 0.5, linewidth = 0)
        ax3.fill_between(np.linspace(sigma, max(y_Prime[:, 0]), 100), max([-sigma, min(y_Prime[:, 1])]), sigma, color = 'r', alpha = 0.5, linewidth = 0)
        ax3.fill_between(np.linspace(min([-sigma, min(y_Prime[:, 0])]), max([sigma, max(y_Prime[:, 0])]), 100), sigma , max([sigma, max(y_Prime[:, 1])]), color = 'r', alpha = 0.5, linewidth = 0)
        ax3.set_xlabel('PC 1')
        ax3.set_ylabel('PC 2')
        ax2.set_xlabel('PC 1')
        ax2.set_ylabel('PC 2')
        for spectrum in rows[(abs(y_Prime[:, 0]) > sigma) | (abs(y_Prime[:, 1]) > sigma)]:
             ax1.set_xlabel('Raman Shift / $cm^{-1}$')
             ax1.set_ylabel('Normed Intensity / a.u.')
             if denoised:
                 pl = ax1.plot(self.xreduced[spectrum], self.ydenoised[spectrum], linewidth = 0.8)
             else:
                 pl = ax1.plot(self.xreduced[spectrum], self.yreduced[spectrum], linewidth = 0.8)
             clr = pl[0].get_color()
             ax2.plot(y_Prime[:, 0][spectrum], y_Prime[:, 1][spectrum], marker = 'o', color = clr, markersize = 10)


        ax1.set_title(f'{len(rows[(abs(y_Prime[:, 0]) > sigma) | (abs(y_Prime[:, 1]) > sigma)])} spectra (outliners)')
        fig.savefig(f'{self.folder}/results/grouped_spectra/outliners.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close(fig)


    def SelectGroupedPeaks(self, peaks, groups = None):
        """
        Wrapper around :func:`~spectrum.SelectPeaks` that iterates over
        all grouped spectra by PCA analysis (see :func:`~spectrum.GroupSpectra`).
        """
        if groups == None:
            for i in range(len(self.groups)):
                if self.groups[i].size != 0:
                    self.SelectPeaks(peaks, spectrum = (self.groups[i])[0], label=self.labels[(self.groups[i])[0]])

        else:
            for group in groups:
                if self.groups[group - 1].size != 0:
                    self.SelectPeaks(peaks, spectrum = (self.groups[group - 1])[0], label=self.labels[(self.groups[group - 1])[0]])

    def FitAllGroupedSpectra(self, peaks, show=False, report=False):
        """
        Wrapper around :func:`~spectrum.FitSpectrum` that iterates over all grouped spectra by PCA analysis (see :func:`~spectrum.GroupSpectra`).
        """
        for i in range(len(self.groups)):
            if self.groups[i].size != 0:
                self.FitSpectrum(peaks, spectrum=(self.groups[i])[0], show=show, report=report)
                for j in range(1, len(self.groups[i])):
                    self.FitSpectrum(peaks, spectrum=(self.groups[i])[j], show=show, report=report, init_spectrum = (self.groups[i])[0])

    def FitGroups(self, peaks, groups, show=False, report=False):
        """
        Wrapper around :func:`~spectrum.FitSpectrum` that iterates over given groups (see :func:`~spectrum.GroupSpectra`) of spectra.
        """
        for group in groups:
            if self.groups[group - 1].size != 0:
                self.FitSpectrum(peaks, spectrum=(self.groups[group - 1])[0], show=show, report=report)
                for j in range(1, len(self.groups[group - 1])):
                    self.FitSpectrum(peaks, spectrum=(self.groups[group - 1])[j], show=show, report=report, init_spectrum = (self.groups[group - 1])[0])


    # Save all the results
    def SaveAllFitParams(self, peaks):
        """
        Wrapper around :func:`~spectrum.SaveFitParams` that iterates over
        all spectra that were analyzed.
        """
        # find all peaks that were fitted and generate a list
        allpeaks = []
        for i in range(self.numberOfFiles):
            if self.fitresult_peaks[i] != None:
                allpeaks.extend(re.findall('prefix=\'(.*?)\'', self.fitresult_peaks[i].model.name))

        allusedpeaks = list(set(allpeaks))

        for i in range(self.numberOfFiles):
            self.SaveFitParams(peaks, usedpeaks=allusedpeaks, spectrum=i,
                               label=self.labels[i])


    # function that allows you select an unwanted frequency in the spectrum
    def SelectFrequency(self, spectrum=0):
        """
        Function that lets the user select an unwanted frequency in the
        spectrum given.
        It saves the selected positions to
        '/temp/fftpeak_' + spectrum + '.dat'.

        Parameters
        ----------

        """
        if spectrum >= self.numberOfFiles:
            print('You need to choose a smaller number for spectra to select.')
        else:
            # create plot and baseline
            fig, ax = plt.subplots()
            # fourier transform data
            self.fourierSpec = np.fft.fft(self.yreduced[spectrum] -
                                          self.baseline[spectrum])
            self.fourierFreq = np.fft.fftfreq(self.xreduced[spectrum].shape[-1])

            # plot fourier transformed data
            ax.plot(self.fourierFreq,
                    abs(self.fourierSpec), 'b-')
            ax.set_title('Fourier transform of Spectrum ' + str(spectrum) +\
                         ' Select the maximum of the unwanted frequency.')
            # arrays of initial values for the fits
            xpeak, ypeak = self.PlotPeaks(fig, ax)
            plt.show()
            # store the chosen values
            peakfile = self.folder + '/temp/fftpeak_' +\
                       self.labels[spectrum] + '.dat'
            if xpeak != []:
                np.savetxt(peakfile, np.transpose([np.array(xpeak),
                                                   np.array(ypeak)]))

    # select an unwanted frequency in all spectra
    def SelectAllFrequencies(self):
        for i in range(self.numberOfFiles):
            self.SelectFrequency(spectrum=i)

    # remove an unwanted frequency from the spectrum given
    def RemoveFrequency(self, spectrum=0, tolerance=6, prnt=False):
            """
            Function that removes an unwanted frequency in the
            spectrum given within a given tolerance

            Parameters
            ----------
            tolerance : int, default : 6
                tolerance times 2 defines the width of the unwanted frequency
                (in number of values not in the frequency regieme)
            prnt : bool, default : False
                plots some more information to process the images. You might
                want to check if the tolerance is set properly.
            """
            # generate name of the fft peak file
            peakfile = self.folder + '/temp/fftpeak_' +\
                                   self.labels[spectrum] + '.dat'

            # check if there is an unwanted frequency
            if os.path.exists(peakfile):
                # get the selected peak position
                xpeak, ypeak = np.genfromtxt(peakfile, unpack = True)
                # search the closest index and generate min and max values
                index = np.abs(self.fourierFreq - xpeak).argmin()
                indexmin = index - tolerance
                indexmax = index + tolerance

                # calculate mean and replace unwanted frequencies
                ymean = np.mean(self.fourierSpec)
                self.modifiedSpec = self.fourierSpec.copy()

                # as fourier is symmetric the indices have
                # to be removed symmetrically
                self.modifiedSpec[indexmin:indexmax] = ymean
                self.modifiedSpec[-indexmax:-indexmin] = ymean

                if prnt:
                    plt.plot(self.fourierFreq, abs(self.fourierSpec),
                             label='Fourier frequencies')
                    plt.plot(self.fourierFreq, abs(self.modifiedSpec),
                             label='Modified frequencies')
                    plt.title('Frequency spectrum of the selected spectrum.')
                    plt.legend()
                    figManager = plt.get_current_fig_manager()  # get current figure
                    figManager.window.showMaximized()           # show it maximized
                    plt.show()

                    plt.plot(self.xreduced[spectrum],
                             self.yreduced[spectrum]-self.baseline[spectrum],
                             label='raw signal')

                # calculate the inverse fft
                self.yreduced[spectrum] = np.fft.ifft(self.modifiedSpec) +\
                                          self.baseline[spectrum]

                if prnt:
                    plt.plot(self.xreduced[spectrum],
                             self.yreduced[spectrum] - self.baseline[spectrum],
                             label='frequency removed signal')
                    plt.title('Comparison of original and frequency removed Spectrum.')
                    plt.legend()
                    figManager = plt.get_current_fig_manager()  # get current figure
                    figManager.window.showMaximized()           # show it maximized
                    plt.show()

    # remove an unwanted frequency in all spectra
    def RemoveAllFrequencies(self):
        for i in range(self.numberOfFiles):
            self.RemoveFrequency(spectrum=i)

# plot XAES data and its derivative
    def PlotXaes(self, spectrum):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(self.x[spectrum], self.ynormed[spectrum],
                'b.', label = 'Data')
        ax1.plot(self.x[spectrum], self.ydenoised[spectrum],
                'r-', label = 'Wavelet-Denoised')
        fig.legend(loc='upper right')

        # calculate and plot derivative
        self.dy[spectrum] = np.diff(self.ydenoised[spectrum])
        ax2.plot(self.x[spectrum][:-1], self.dy[spectrum],
                 'g-', label = 'Derivative')

        plt.legend(loc='upper right')
        #plt.show()
        fig.savefig(self.folder + '/results/plot/derivative_'
                                + self.labels[spectrum]
                                + '.png', dpi=300)
        savefile = (self.folder + '/results/derived/der_'
                    + self.labels[spectrum] + '.dat')
        np.savetxt(savefile, np.column_stack([self.x[spectrum][:-1],
                                              self.dy[spectrum]]))

# plot all XAES data and the corresponding derivative
    def PlotAllXaes(self):
        for i in range(self.numberOfFiles):
            self.PlotXaes(spectrum=i)
