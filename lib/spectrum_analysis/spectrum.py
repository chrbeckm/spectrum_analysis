import os
import numpy as np

import matplotlib.pyplot as plt

import pywt                             # for wavelet operations
from statsmodels.robust import mad      # median absolute deviation from array
from scipy.optimize import curve_fit    # for interpolating muons

from lmfit.models import *

from spectrum_analysis.starting_params import *

"""
This module contains the spectrum class to work with any x, y structured
data.
"""

class spectrum(object):
    """
    Class for working with x, y structured data.

    Attributes
    ----------
    file : string
        name of the file of the data to be analyzed

    Parameters
    ----------
    filename : string
        The file of interest has to be in the current directory.
        The data will be prepared to analyze spectral data.

    datatype : string, default : 'txt'
        Type of the datafiles that should be used, like 'txt', 'csv',
        or 'dat'
    """

    def __init__(self, filename, datatype='txt'):
        self.file = filename + '.' + datatype
        self.folder = self.file[:-(len(self.label) + len(datatype) + 2)]

        self.tmpdir = self.folder + '/temp'
        self.resdir = self.folder + '/results'
        self.basdir = self.resdir + '/baselines'
        self.fitdir = self.resdir + '/fitlines'
        self.pardir = self.resdir + '/fitparameter'
        self.pardir_spec = self.pardir + '/spectra'
        self.pltdir = self.resdir + '/plot'

        self.tmploc ='locpeak'
        self.tmpfft = 'fftpeak'
        self.pltname = 'fitplot'

        # set value of missing values
        self.missingvalue = 1.11

        # create temporary folders
        if not os.path.exists(self.tmpdir):
            os.makedirs(self.tmpdir)
        # create results folders
        if not os.path.exists(self.resdir):
            os.makedirs(self.basdir)
            os.makedirs(self.fitdir)
            os.makedirs(self.pardir_spec)
            os.makedirs(self.pltdir)

    @property
    def label(self):
        return self.file.split('/')[-1].split('.')[-2]

    def get_file(self, dir, prefix, datatype, suffix='', label=''):
        """
        Returns a filename

        Parameters
        ----------
        dir : string
            Directory where to find the file.

        prefix : string
            prefix of the filename

        datatype : string
            Type of the data. For example 'txt', 'csv', 'png' or 'pdf'

        suffix : string, default : ''
            suffix added after prefix

        Returns
        -------
        : string
            Retruns a string constructed as defined by the function.
        """
        if (suffix == '') and (prefix != ''):
            return (dir + '/' + prefix
                    + '_' + self.label + '.' + datatype)
        elif (prefix == '') and (suffix != ''):
            return (dir + '/' + self.label + '_' + suffix + '.' + datatype)
        elif label != '':
            return (dir + '/' + label + '.' + datatype)
        else:
            return (dir + '/' + prefix + '_' + suffix
                    + '_' + self.label + '.' + datatype)

    def PlotVerticalLines(self, color, fig, jupyter=False):
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
            Points selected from the user containing the selected
            region in x-dimension.
        """
        xregion = []
        ax = plt.gca()                          # get current axis
        plt_ymin, plt_ymax = ax.get_ylim()      # get plot min and max

        def onclickbase(event):
            """
            Choose region if clicked.
            Append data to xregion and plot vertical lines.
            """
            if event.button:
                xregion.append(event.xdata)
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
        if jupyter == True:
            pass
        else:
            figManager = plt.get_current_fig_manager()  # get current figure
            figManager.window.showMaximized()           # show it maximized

        return xregion

    def SelectRegion(self, x, y, **kwargs):
        """
        Function that lets the user select a region by running the
        method :func:`PlotVerticalLines() <spectrum.spectrum.PlotVerticalLines()>`.

        Parameters
        ----------
        x : np.array
            x-values of the requested spectrum.

        y : np.array
            y-values of the requested spectrum.

        Returns
        -------
        xregion[0] : numpy.float
            Minimum of the selected x-region.

        xregion[-1] : numpy.float
            Maximum of the selected x-region.
        """
        # plot spectrum
        fig, ax = plt.subplots()
        ax.plot(x, y, 'b.', label='Data')
        ax.set_title('Spectrum ' + self.label
                     + '\nSelect the part of the spectrum you wish to '
                     + 'consider by clicking into the plot.')

        # select region of interest
        xregion = self.PlotVerticalLines('green', fig, **kwargs)

        plt.legend(loc='upper right')
        plt.show()

        return xregion

    def ExtractRegion(self, x, xregion):
        """
        Function to extract region.
        """
        xmin = x[0]
        xmax = x[-1]

        if xregion != []:
            xmin = xregion[0]
            xmax = xregion[-1]

        return xmin, xmax

    def ReduceRegion(self, x, y, xmin, xmax):
        """
        Function that calculates the reduced spectra, as selected before
        by the method :func:`SelectRegion() <spectrum.spectrum.SelectRegion()>`.

        Parameters
        ----------
        x : numpy.array
            x-values of the selected spectrum.

        y : numpy.array
            y-values of the selected spectrum.

        Returns
        -------
        xreduced : numpy.array
            Reduced x-values of the spectrum.

        yreduced : numpy.array
            Reduced y-values of the spectrum.
        """
        yreduced = y[(x > xmin) &
                     (x < xmax)]
        xreduced = x[(x > xmin) &
                     (x < xmax)]
        return xreduced, yreduced

    def SplitMuons(self, indices, prnt=False):
        """
        Function to separate muons from each other.

        Parameters
        ----------
        indices : numpy.array
            Array of indices to be splitted.

        prnt : boolean
            Set to true if you want to know the number of muons
            detected in a spectrum.

        Returns
        -------
        grouped array : list
            Multidimensional list containing the indices of one muon
            in each dimension.
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

    def DetectMuonsWavelet(self, y, thresh_mod=1.0, wavelet='sym8',
                                 level=1, prnt=False):
        """
        Detect muons for removal an returns non vanishing indices.

        Parameters
        ----------
        y : numpy.array
            Array of interest from which the muons should be removed.

        thresh_mod : float, default : 1.0
            Multiplies the threshhold by thresh_mod, to optimize
            muon detection

        wavelet : string, default : 'sym8'
            Wavelet to be used in the Transformation. See pywt
            documentation for possible wavelets.

        level : int, default : 1
            Used to vary the coefficient-level. 1 is the highest level,
            2 the second highest, etc. Depends on the wavelet used.

        prnt : string, default : False
            Is handed over to SplitMuons.

        Returns
        -------
        muonsgrouped : list
            Multidimensional list containing the indices of one muon
            in each dimension.
        """
        # calculate wavelet coefficients
        # with symmetric signal extension mode
        coeff = pywt.wavedec(y, wavelet)

        # calculate a threshold
        sigma = mad(coeff[-level])
        threshold = (sigma * np.sqrt(2 * np.log(len(y)))
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

        # get non vanishing indices
        if (len(y) % 2) == 0:
            indices = np.nonzero(muonfree)[0]
            muonsgrouped = self.SplitMuons(indices, prnt=prnt)
        else:
            indices = np.nonzero(muonfree[:-1])[0]
            muonsgrouped = self.SplitMuons(indices, prnt=prnt)

        return muonsgrouped

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

    def RemoveMuons(self, x, y, prnt=False, **kwargs):
        """
        Removes muons from a spectrum and approximates linearly
        in the muon region.

        Parameters
        ----------
        x : numpy.array
            x-data of the selected spectrum.

        y : numpy.array
            y-data that contains muons which should be removed.

        prnt : boolean
            Prints if muons were found in the spectrum of interest.

        **kwargs
            see method :func:`DetectMuonsWavelet() <spectrum.spectrum.DetectMuonsWavelet()>`

        Returns
        -------
        y : numpy.array
            Muon-free y-data.
        """
        muonsgrouped = self.DetectMuonsWavelet(y, **kwargs)
        # check if there are any muons in the spectrum given
        if len(muonsgrouped[0]) > 0:
            # remove each muon
            for muon in muonsgrouped:
                # calculate limits for indices to use for fitting
                limit = int(len(muon)/4)
                lower = muon[:limit]
                upper = muon[-limit:]
                fit_indices = np.append(lower, upper)

                # fit to the data
                popt, pcov = curve_fit(self.linear,
                                       x[fit_indices],
                                       y[fit_indices])

                # calculate approximated y values and remove muon
                for index in muon[limit:-limit]:
                    y[index] = self.linear(x[index],*popt)
        elif prnt:
            print('No muons found.')

        return y

    def SelectBaseline(self, x, y, color='b', degree=1, **kwargs):
        """
        Function that lets the user distinguish between the background
        and the signal. It runs the
        method :func:`PlotVerticalLines() <spectrum.spectrum.PlotVerticalLines()>`
        to select the regions that do not belong to the background and
        are therefore not used for background fit.

        Parameters
        ----------
        x : numpy.array
            x-data of the selected spectrum.

        y : numpy.array
            y-data that should be cleaned from background.

        label : string, default: ''
            Label for the spectrumborders file in case you want to have
            different borders for different files.

        color : string, default 'b'
            Color of the plotted spectrum.

        Returns
        -------
        xregion : numpy.array
            Array containing the min and max x-values which should be excluded
            from background calculations.
        """
        # plot the reduced spectrum
        fig, ax = plt.subplots()
        ax.plot(x, y, label='Data', color=color)
        ax.set_title('Spectrum ' + self.label +
                     '\nSelect the area of the spectrum you wish to exclude'
                     'from the background by clicking into the plot\n'
                     '({} degree polynomial assumed)'.format(degree))

        # choose the region
        xregion = self.PlotVerticalLines('red', fig, **kwargs)

        plt.legend(loc = 'upper right')
        plt.show()

        return xregion

    def CreateMask(self, x, xregion):
        """
        Function to generate booloean mask for the baseline fit.

        Parameters
        ----------
        x : numpy.array
            x-values of the spectrum that should be analyzed.

        xregion : numpy.array
            Array containing the min and max x-values which should be
            excluded from background calculations.

        Returns
        -------
        relevant : boolean array
            Boolean array containing Trues and Falses that mask the
            inserted data.
        """
        # xregion[0] is the lowest border
        relevant = (x <= xregion[0])
        for i in range(1, len(xregion) - 2, 2): # upper borders i
            # take only data between the borders
            relevant = relevant | ((x >= xregion[i]) &
                                   (x <= xregion[i + 1]))
        # xregion[-1] is the highest border
        relevant = relevant | (x >= xregion[-1])

        return relevant

    def FitBaseline(self, x, y, xregion, show=False, degree=1):
        """
        Fit of the baseline by using the
        `PolynomalModel()
        <https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.PolynomialModel>`_
        from lmfit.

        Parameters
        ----------
        x : numpy.array
            x-values of spectrum which should be background-corrected.

        y : numpy.array
            y-values of spectrum which should be background-corrected.

        show : boolean, default: False
            Decides whether the a window with the fitted baseline is opened
            or not.

        degree : int, default: 1
            Degree of the polynomial that describes the background.

        Returns
        -------
        baseline : numpy.array
            Baseline of the input spectrum.
        """
        relevant = self.CreateMask(x, xregion)
        # polynomial to model the background
        background = PolynomialModel(degree=degree)
        pars = background.guess(y[relevant], x=x[relevant])
        fitresult = background.fit(y[relevant], pars, x=x[relevant])

        # create baseline
        baseline = background.eval(fitresult.params, x=x)

        # plot the fitted function in the selected range
        if show:
            plt.plot(x, y, 'b.', label = 'Data')
            plt.plot(x, baseline, 'r-', label = 'Baseline')
            plt.show()

        # save the baseline
        file = self.get_file(self.basdir, prefix='',
                        suffix='baseline', datatype='dat')
        np.savetxt(file, np.column_stack([x, baseline]))

        return baseline

    def WaveletSmooth(self, y, wavelet='sym8', level=2):
        """
        Smooth array by using wavelet transformation and soft threshold.

        Parameters
        ----------
        y : numpy.array
            Array that should be denoised.

        wavelet : string, default : 'sym8'
            Wavelet for the transformation, see pywt documentation for
            different wavelets.

        level : int, default : 2
            Used to vary the coefficient-level. 1 is the highest level,
            2 the second highest, etc. Depends on the wavelet used.

        Returns
        -------
        ydenoised : numpy.array
            Denoised array of the input array.
        """
        # calculate wavelet coefficients
        coeff = pywt.wavedec(y, wavelet)

        # calculate a threshold
        sigma = mad(coeff[-level])
        threshold = sigma * np.sqrt(2 * np.log(len(y)))

        # calculate thresholded coefficients
        for i in range(1,len(coeff)):
            coeff[i] = pywt.threshold(coeff[i], value=threshold, mode='soft')

        # reconstruct the signal using the thresholded coefficients
        denoised = pywt.waverec(coeff, wavelet)

        # return the value of denoised except for the last value
        if (len(y) % 2) == 0:
            ydenoised = denoised
        else:
            ydenoised = denoised[:-1]

        return ydenoised

    def Normalize(self, y, ymax=None):
        """
        Divides any input array by its maximum and returns the resulting
        array and its maximum.

        Parameters
        ----------
        y : numpy.array
            Array that should be normalized.

        Returns
        -------
        ynormed : numpy.array
            Normed array.

        ymax :
            Maximum of the input array.
        """
        if ymax == None:
            ymax = np.max(y)
        ynormed = y/ymax

        return ynormed, ymax

    def SelectFrequency(self, x, y):
        """
        Function that lets the user select an unwanted frequency in the
        spectrum given.
        It saves the selected positions to
        self.tmpdir + '/' + self.tmpfft + '.dat'.
        Parameters
        ----------
        """
        # create plot
        fig, ax = plt.subplots()
        # fourier transform data
        x_fft = np.fft.fftfreq(x.shape[-1])
        y_fft = np.fft.fft(y)

        # plot fourier transformed data
        ax.plot(x_fft, abs(self.RemoveIndex(y_fft)), 'b-')
        ax.set_title('Fourier transform of Spectrum ' + self.label
                     + ' Select the maximum of the unwanted frequency.\n'
                     + ' Frequencies close to 0 are removed for better'
                     + ' visibility. Only one frequency selectable!')
        # arrays of initial values for the fits
        xpeak, ypeak = self.PlotPeaks(fig, ax)
        plt.show()
        # store the chosen values
        if xpeak != []:
            np.savetxt(self.get_file(dir=self.tmpdir, prefix=self.tmpfft,
                                     datatype='dat'),
                       np.transpose([np.array(xpeak),
                                     np.array(ypeak)]))

        return x_fft, y_fft

    def RemoveIndex(self, y_fft, index=0, tolerance=40):
        # calculate mean and replace unwanted frequencies
        ymean = np.mean(y_fft)
        modifiedSpec = y_fft.copy()

        if index == 0:
            indexmin = 1
            indexmax = tolerance
            modifiedSpec[0] = ymean
            modifiedSpec[-1] = ymean
        else:
            indexmin = index - tolerance
            indexmax = index + tolerance

        # as fourier is symmetric the indices have
        # to be removed symmetrically
        modifiedSpec[indexmin:indexmax] = ymean
        modifiedSpec[-indexmax:-indexmin] = ymean

        return modifiedSpec

    def RemoveFrequency(self, x_fft, y_fft, prnt=False, **kwargs):
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
        # get the selected peak position
        peakfile = self.get_file(dir=self.tmpdir, prefix=self.tmpfft,
                                 datatype='dat')
        if os.path.exists(peakfile):
            xpeak, ypeak = np.genfromtxt(peakfile, unpack = True)
            # search the closest index and generate min and max values
            index = np.abs(x_fft - xpeak).argmin()
            modifiedSpec = self.RemoveIndex(y_fft, index=index, **kwargs)

            # calculate the inverse fft
            yreduced = np.fft.ifft(modifiedSpec)

            if prnt:
                plt.plot(x_fft, abs(y_fft),
                         label='Fourier frequencies')
                plt.plot(x_fft, abs(modifiedSpec),
                         label='Modified frequencies')
                plt.title('Frequency spectrum of the selected spectrum.')
                plt.legend()
                figManager = plt.get_current_fig_manager()  # get current figure
                figManager.window.showMaximized()           # show it maximized
                plt.show()
        else:
            yreduced = np.fft.ifft(y_fft)

        return yreduced

    def PlotPeaks(self, fig, ax, jupyter=False):
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
            """
            Function that defines behavoir if the left and right mouse
            button are clicked.

            -> left click: Add data point
            -> right click: Remove closest data point
            """
            if event.button == 1:
                xpeak.append(event.xdata)
                ypeak.append(event.ydata)
                line.set_xdata(xpeak)
                line.set_ydata(ypeak)
                plt.draw()

            if event.button == 3:
                if xpeak != []:
                    xdata_nearest_index = (np.abs(xpeak - event.xdata)).argmin()
                    del xpeak[xdata_nearest_index]
                    del ypeak[xdata_nearest_index]
                    line.set_xdata(xpeak)
                    line.set_ydata(ypeak)
                    plt.draw()

        # actual execution of the defined function oneclickpeaks
        cid = fig.canvas.mpl_connect('button_press_event', onclickpeaks)
        if jupyter == True:
            pass
        else:
            figManager = plt.get_current_fig_manager()  # get current figure
            figManager.window.showMaximized()           # show it maximized

        return xpeak, ypeak

    def SelectPeaks(self, x, y, peaks, **kwargs):
        """
        Function that lets the user select the maxima of the peaks to fit
        according to their line shape (Voigt, Fano, Lorentzian, Gaussian).
        The positions (x- and y-value) are taken as initial values in the
        function :func:`~spectrum.FitSpectrum`.
        It saves the selected positions to
        '/temp/locpeak_' + peaktype + '_' + label + '.dat'.

        Usage: Select peaks with left mouse click, remove them with right
        mouse click.

        Parameters
        ----------
        peaks : list, default: ['breit_wigner', 'lorentzian']
            Possible line shapes of the peaks to fit are
            'breit_wigner', 'lorentzian', 'gaussian', and 'voigt'.
            See lmfit documentation
            (https://lmfit.github.io/lmfit-py/builtin_models.html)
            for details.

        x : numpy.array
            x-values of the mapping

        y : numpy.array
            y-values of the mapping
        """
        # loop over all peaks and save the selected positions
        for peaktype in peaks:
            # create plot and baseline
            fig, ax = plt.subplots()
            # plot corrected data
            ax.plot(x, y, 'b')
            ax.set_title('Spectrum ' + self.label +
                         '\nBackground substracted, smoothed,'
                         ' normalized spectrum\n Select the maxima of the '
                         + peaktype + '-PEAKS to fit.')
            # arrays of initial values for the fits
            xpeak, ypeak = self.PlotPeaks(fig, ax, **kwargs)
            plt.show()

            # store the chosen initial values
            np.savetxt(self.get_file(dir=self.tmpdir, prefix=self.tmploc,
                                     datatype='dat', suffix=peaktype),
                       np.transpose([np.array(xpeak),
                                     np.array(ypeak)]))

    def GenerateModel(self, peaks):
        """
        Generates a fit Model using lmfit.

        Parameters
        ----------
        peaks : list of strings
            A list of strings containing the peaks that should be used
            for fitting.

        Returns
        -------
        model : lmfit.models.CompositeModel
            Fitmodel for the spectrum.
        """
        model = ConstantModel() # Add a constant for a better fit

        # go through all defined peaks
        for peaktype in peaks:
            peakfile = self.get_file(dir=self.tmpdir, prefix=self.tmploc,
                                     datatype='dat', suffix=peaktype)

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
                    model += temp # add the models to 'model'

        return model

    def TestLimits(self, pars, fit):
        """
        Tests if starting limits were reached and prints out warnings.

        Parameters
        ----------
        pars : lmfit.parameter.Parameters
            Parameters of a fitmodel

        fit : lmfit.model.ModelResult
            Fitresults of a fit.
        """
        # arrays of low and high limits of the start parameters
        low_lmt = np.array([pars[key].min for key in pars.keys()])
        hig_lmt = np.array([pars[key].max for key in pars.keys()])

        inf_mask = (hig_lmt != float('inf')) & (low_lmt != float('-inf'))
        range_lmt = hig_lmt[inf_mask] - low_lmt[inf_mask]

        fit_val = np.array([fit.params[key].value for key in fit.params.keys()])
        names = np.array([fit.params[key].name for key in fit.params.keys()])

        lmt = 0.01 # percentage distance to the bounds leading to a warning

        # mask = True if best value is near upper or lower bound
        low_mask = fit_val[inf_mask] <= low_lmt[inf_mask] + lmt * range_lmt
        hig_mask = fit_val[inf_mask] >= low_lmt[inf_mask] + (1-lmt) * range_lmt

        # warn if one of the parameters has reached the lower bound
        if True in low_mask:
            warn(f'The parameter(s) {(names[inf_mask])[low_mask]} of '
                  'spectrum {self.label} are close to chosen low limits.',
                  ParameterWarning)

        # warn if one of the parameters has reached the upper bound
        if True in hig_mask:
            warn(f'The parameter(s) {(names[inf_mask])[hig_mask]} of '
                  'spectrum {self.label} are close to chosen high limits.',
                  ParameterWarning)

    def FitSpectrum(self, x, y, peaks):
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

        Returns
        -------
        fitresults : lmfit.model.ModelResult
            The results of the fit.
        """
        # Create a composed model of lmfit.models
        model = self.GenerateModel(peaks)

        # create the fit parameters and fit
        pars = model.make_params()
        fitresults = model.fit(y, pars, x=x,
                                    method = 'leastsq',
                                    scale_covar = True)

        # test if any starting limits were reached and print to console
        self.TestLimits(pars, fitresults)

        # print which fit is conducted
        print('Spectrum ' + self.label + ' fitted')

        return fitresults

    def PlotComponents(self, x, ax, ymax, components):
        """
        Plots all the components handed to it.

        Parameters
        ----------
        x : numpy.array
            x-values

        ax : matplotlib.axes_subplots.AxesSubplot
            Axis where to plot on.

        components : collections.OrderedDict
            Components of a lmfit.CompositeModel
        """
        for name in components.keys():
            if (name != 'constant'):
                ax.plot(x, (components[name] + components['constant']) * ymax,
                        'k-', linewidth = 0.5, zorder = 0)

    def PlotConfidence(self, x, ax, ymax, fitresults, fitline):
        """
        Plots the confidence band, if it can be calculated.

        Parameters
        ----------
        x : numpy.array
            x-values

        ax : matplotlib.axes_subplots.AxesSubplot
            Axis where to plot on.

        fitresults : lmfit.model.ModelResult
            The results of the requested fit.

        fitline : numpy.array
            calculated fitline from fitresults.eval(x=x)
        """
        # check if errors exist.
        if fitresults.params['c'].stderr is not None:
            # calculate confidence band
            confidence = fitresults.eval_uncertainty(x=x, sigma=3)

            # plot confidence band
            ax.fill_between(x, (fitline + confidence) * ymax,
                               (fitline - confidence) * ymax,
                 color = 'r', linewidth = 1, alpha = 0.5, zorder = 1,
                 label = '3$\sigma$')

    def PlotFit(self, x, y, ymax, fitresults, show=False, jupyter=False):
        """
        Plot the fitresults of a fit with lmfit.

        Parameters
        ----------
        x : numpy.ndarray
            x-values of the spectrum.

        y : numpy.ndarray
            y-values of the spectrum.

        fitresults : lmfit.model.ModelResult
            Results from a fit with lmfit.

        show : boolean, default : False
            Show the resulting plot.
        """
        # calculate the fit line and all components lines
        fitline = fitresults.eval(x=x)
        comps = fitresults.eval_components(x=x)

        # check if model was only a constant and set to the value
        if type(fitline) == np.float64:
            fitline = np.ones_like(y) * comps['constant']

        # Plot the raw sprectrum, the fitted data, the background,
        # the different components and the confidence interval
        fig, ax = plt.subplots()
        ax.plot(x, y * ymax,
                'b.', alpha = 0.8, markersize = 1, zorder = 0, label = 'Data')
        ax.plot(x, np.ones_like(y) * comps['constant'] * ymax,
                'k-', linewidth = 1, zorder = 0, label = 'Background')
        ax.plot(x, fitline * ymax,
                'r-', linewidth = 0.5, zorder = 1, label = 'Fit')
        self.PlotComponents(x, ax, ymax, comps)
        self.PlotConfidence(x, ax, ymax, fitresults, fitline)

        fig.legend(loc = 'upper right')
        plt.title('Fit to spectrum ' + self.label)
        plt.ylabel('Scattered light intensity (arb. u.)')
        plt.xlabel('Raman shift (cm$^{-1}$)')

        # save figures
        fig.savefig(self.get_file(dir=self.pltdir, prefix=self.pltname,
                                  datatype='pdf'))
        fig.savefig(self.get_file(dir=self.pltdir, prefix=self.pltname,
                                  datatype='png'), dpi=300)

        if show:
            if jupyter == True:
                pass
            else:
                figManager = plt.get_current_fig_manager()  # get current figure
                figManager.window.showMaximized()           # show it maximized
            plt.show()

        plt.close()

        # save the fitline
        file = self.get_file(self.fitdir, prefix='',
                        suffix='fitline', datatype='dat')
        np.savetxt(file, np.column_stack([x, fitline * ymax]))

    def ScaleParameters(self, ymax, peakparameter, value, error):
        # if parameter is height or amplitude or intensity
        # it has to be scaled properly as the fit was normalized
        if ((peakparameter == 'amplitude')
            or (peakparameter == 'height')
            or (peakparameter == 'intensity')):
            value = value * ymax
            if error is not None:
                error = error * ymax

        # if there is no error set the value to self.missingvalue
        if error is None:
            error = self.missingvalue

        return value, error

    def SaveSpec(self, ymax, peak, params):
        file = self.get_file(dir=self.pardir_spec, prefix='',
                             suffix=peak[:-1], datatype='dat')
        with open(file, 'w') as f:
            # iterate through all fit parameters
            for name in params.keys():
                # and find the current peak
                peakparameter = re.findall(peak, name)

                if peakparameter:
                    # get parameters for saving
                    peakparameter = name.replace(peak, '')
                    value = params[name].value
                    error = params[name].stderr

                    value, error = self.ScaleParameters(ymax, peakparameter,
                                                        value, error)

                    # write to file
                    f.write(peakparameter.ljust(12)
                            + '{:>13.5f}'.format(value)
                            + ' +/- ' + '{:>11.5f}'.format(error)
                            + '\n')

    def SaveFuncParams(self, savefunc, ymax, fitresults, peaks):
        """
        The optimized line shapes as well as the corresponding fit parameters
        with uncertainties are saved in several folders, all contained in the
        folder self.redir.
        The optimized baseline from each spectrum can be found in
        self.basdir and the line shape of the background subtracted spectrum
        in self.fitdir.
        The folder self.pardir_spec contains files each of which with one
        parameter including its uncertainty.
        The parameters are also sorted by peak for different spectra.
        This is stored in self.pardir_peak including the correlations of the
        fit parameters.

        Parameters
        ----------
        fitresults : lmfit.model.ModelResult
            Result of a fit.

        peaks : list
            Possible line shapes of the peaks to fit are
            'breit_wigner', 'lorentzian', 'gaussian', and 'voigt'.
        """
        # find all prefixes used in the model
        modelpeaks = re.findall('prefix=\'(.*?)\'', fitresults.model.name)

        # iterate through all peaks used in the current model
        for peak in modelpeaks:
            savefunc(ymax, peak, fitresults.params)

        # print which spectrum is saved
        print('Spectrum '
              + self.label + ' '
              + str(savefunc).split('Save')[-1].split(' ')[0]
              + 's saved')

    def SaveFitParams(self, ymax, fitresults, peaks):
        self.SaveFuncParams(self.SaveSpec, ymax, fitresults, peaks)
