import numpy as np

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spectrum_analysis.spectrum import *
from spectrum_analysis.dictionaries import *
from spectrum_analysis import data

from PIL import Image


"""
This module contains the mapping class to work multiple x, y structured
data sets.
"""

class mapping(spectrum):
    """
    Class for working with x, y structured data sets.

    Attributes
    ----------
    folder : string
        name of the folder of the data to be analyzed

    listOfFiles : string
        List of files that are in the requested folder

    numberOfFiles : int
        Number of Files in the requested folder

    spectrum : int, default : 0
        Spectrum which is used as the reference spectrum for region selection.

    spectra : spectrum
        List containing all spectra of the mapping

    Parameters
    ----------
    foldername : string
        The folder of interest has to be in the current directory.
        The data will be prepared to analyze spectral data.

    datatype : string, default : 'txt'
        Type of the datafiles that should be used, like 'txt', 'csv',
        or 'dat'
    """

    def __init__(self, foldername, plot=False, datatype='txt'):
        self.folder = foldername
        self.second_analysis = False
        self.answer = 'n'
        self.listOfFiles, self.numberOfFiles = data.GetFolderContent(
                                                        self.folder,
                                                        datatype)
        if os.path.exists(self.folder + '/results') and not plot:
            self.second_analysis = True
            self.listOfFiles, self.numberOfFiles, self.indices = self.Get2ndLabels()

        self.spectrum = 0
        self.spectra = []
        for spec in self.listOfFiles:
            self.spectra.append(spectrum(spec.split('.')[-2]))

        self.pardir_peak = self.pardir + '/peakwise'

        if not os.path.exists(self.pardir_peak):
            os.makedirs(self.pardir_peak)

    @property
    def label(self):
        return self.spectra[self.spectrum].label

    @label.setter
    def label(self, spectrum):
        self.spectrum = spectrum

    @property
    def tmpdir(self):
        return self.spectra[self.spectrum].tmpdir

    @property
    def resdir(self):
        return self.spectra[self.spectrum].resdir

    @property
    def basdir(self):
        return self.spectra[self.spectrum].basdir

    @property
    def fitdir(self):
        return self.spectra[self.spectrum].fitdir

    @property
    def pardir(self):
        return self.spectra[self.spectrum].pardir

    @property
    def pardir_spec(self):
        return self.spectra[self.spectrum].pardir_spec

    @property
    def pltdir(self):
        return self.spectra[self.spectrum].pltdir

    @property
    def dendir(self):
        return self.spectra[self.spectrum].dendir

    @property
    def tmploc(self):
        return self.spectra[self.spectrum].tmploc

    @property
    def pltname(self):
        return self.spectra[self.spectrum].pltname

    @property
    def missingvalue(self):
        return self.spectra[self.spectrum].missingvalue

    def SplitLabel(self, file):
        return file.split('/')[-1].split('.')[-2]

    def Get2ndLabels(self):
        """
        Function to get a list of indices for the second analysis.
        """
        list_of_files = []
        list_of_indices = []

        self.answer = input('These spectra have been analyzed already.\n'
                            'Do you want to fit all of them again? (y/n)\n')

        if self.answer == 'y':
            list_of_files = self.listOfFiles
            number_of_files = self.numberOfFiles
            list_of_indices = np.arange(self.numberOfFiles)
        elif self.answer == 'n':
            for i, label in enumerate(self.listOfFiles):
                print(f'{self.SplitLabel(label)} \n')

            print('Enter the spectra that you want to analyze again.\n'
                  'It is enough to enter the appendant four letter number.\n'
                  '(Finish the selection with x).')

            while True:
                label = input()
                if label == 'x':
                    break
                if any(label in file for file in self.listOfFiles):
                    index = [i for i, file in enumerate(self.listOfFiles) if label in file]
                    list_of_files.append(self.listOfFiles[index[0]])
                    list_of_indices.append(index[0])
                    print('Added ' + self.SplitLabel(self.listOfFiles[index[0]]))
                else:
                    print('This spectrum does not exist.')
            number_of_files = len(list_of_files)

        return list_of_files, number_of_files, list_of_indices

    def ReduceAllRegions(self, x, y):
        """
        Function that calculates the reduced spectra, as selected before
        by the method :func:`SelectRegion() <spectrum.spectrum.SelectRegion()>`.

        Parameters
        ----------
        x : numpy.ndarray
            x-values of the selected spectrum.

        y : numpy.ndarray
            y-values of the selected spectrum.

        Returns
        -------
        xreduced : numpy.ndarray
            Reduced x-values of the spectrum.

        yreduced : numpy.ndarray
            Reduced y-values of the spectrum.
        """
        xregion = self.SelectRegion(x[self.spectrum], y[self.spectrum])
        xmin, xmax = self.ExtractRegion(x[self.spectrum], xregion)

        xreduced = np.array([])
        yreduced = np.array([])

        for i, spectrum in enumerate(y):
            xtemp, ytemp = self.ReduceRegion(x[i], y[i], xmin, xmax)
            xreduced = data.VStack(i, xreduced, xtemp)
            yreduced = data.VStack(i, yreduced, ytemp)

        return xreduced, yreduced

    def RemoveAllMuons(self, x, y, prnt=False, **kwargs):
        """
        Removes muons from all spectra and approximates linearly
        in the muon region.

        Parameters
        ----------
        x : numpy.ndarray
            x-data of the selected spectrum.

        y : numpy.ndarray
            y-data that contains muons which should be removed.

        prnt : boolean
            Prints if muons were found in the spectrum of interest.

        **kwargs
            see method :func:`DetectMuonsWavelet() <spectrum.spectrum.DetectMuonsWavelet()>`

        Returns
        -------
        y : numpy.ndarray
            Muon-free y-data.
        """

        for i, spectrum in enumerate(y):
            y[i] = self.RemoveMuons(x[i], y[i], prnt=prnt, **kwargs)

        return y

    def SelectAllBaselines(self, x, y, color='b', degree=1):
        """
        Function that lets the user distinguish between the background
        and the signal. It runs the
        method :func:`PlotVerticalLines() <spectrum.spectrum.PlotVerticalLines()>`
        to select the regions that do not belong to the background and
        are therefore not used for background fit.

        Parameters
        ----------
        x : numpy.ndarray
            x-data of the selected spectrum.

        y : numpy.ndarray
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
        xregion = self.SelectBaseline(x[self.spectrum],
                                      y[self.spectrum],
                                      color=color, degree=degree)
        return xregion

    def FitAllBaselines(self, x, y, xregion, show=False, degree=1):
        """
        Fit of the baseline by using the
        `PolynomalModel()
        <https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.PolynomialModel>`_
        from lmfit.

        Parameters
        ----------
        x : numpy.ndarray
            x-values of spectrum which should be background-corrected.

        y : numpy.ndarray
            y-values of spectrum which should be background-corrected.

        show : boolean, default: False
            Decides whether the a window with the fitted baseline is opened
            or not.

        degree : int, default: 1
            Degree of the polynomial that describes the background.

        Returns
        -------
        baselines : numpy.ndarray
            Baseline of the input spectrum.
        """
        baselines = np.array([])
        for i, spectrum in enumerate(y):
            self.label = i
            baseline = self.FitBaseline(x[i], y[i], xregion, show=show, degree=degree)
            baselines = data.VStack(i, baselines, baseline)

        return baselines

    def WaveletSmoothAll(self, y, wavelet='sym8', level=2):
        """
        Smooth arrays by using wavelet transformation and soft threshold.

        Parameters
        ----------
        y : numpy.ndarray
            Array that should be denoised.

        wavelet : string, default : 'sym8'
            Wavelet for the transformation, see pywt documentation for
            different wavelets.

        level : int, default : 2
            Used to vary the coefficient-level. 1 is the highest level,
            2 the second highest, etc. Depends on the wavelet used.

        Returns
        -------
        ydenoised : numpy.ndarray
            Denoised array of the input array.
        """
        ydenoised = np.array([])
        for i, spectrum in enumerate(y):
            ytemp = self.WaveletSmooth(y[i], wavelet=wavelet, level=level)
            ydenoised = data.VStack(i, ydenoised, ytemp)

        return ydenoised

    def NormalizeAll(self, y, ymax=None):
        ynormed = np.array([])

        if type(ymax) == type(None):
            for i, spectrum in enumerate(y):
                ynormed_temp, ymax_temp = self.Normalize(y[i])
                ynormed = data.VStack(i, ynormed, ynormed_temp)
                ymax = data.VStack(i, ymax, ymax_temp)
        else:
            for i, spectrum in enumerate(y):
                ynormed_temp, ymax_temp = self.Normalize(y[i], ymax=ymax[i])
                ynormed = data.VStack(i, ynormed, ynormed_temp)

        return ynormed, ymax

    def SelectAllPeaks(self, x, y, peaks):
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

        x : numpy.ndarray
            x-values of the mapping

        y : numpy.ndarray
            y-values of the mapping
        """
        select = True
        if self.answer == 'y':
            answer = input('Do you want to select all peaks again? (y/n)\n')
            if answer == 'y':
                select = True
            else:
                select = False

        if select:
            for i, spectrum in enumerate(y):
                self.label = i
                self.SelectPeaks(x[i], y[i], peaks)

    def FitAllSpectra(self, x, y, peaks):
        results = []

        for i, spectrum in enumerate(y):
            self.label = i
            temp = self.FitSpectrum(x[i], y[i], peaks=peaks)
            results.append(temp)

        return results

    def PlotAllFits(self, x, y, ymax, fitresults, show=False):
        for i, spectrum in enumerate(y):
            print(self.label + ' plotted')
            self.label = i
            self.PlotFit(x[i], y[i], ymax[i], fitresults[i], show=show)

    def Save2nd(self, file, value, error):
        # get values and update them
        values, stderrs = np.genfromtxt(file, unpack = True)

        values[self.indices[self.spectrum]] = value
        stderrs[self.indices[self.spectrum]] = error

        with open(file, 'w') as f:
            for i in range(len(values)):
                f.write('{:>13.5f}'.format(values[i])
                        + '\t' + '{:>11.5f}'.format(stderrs[i])
                        + '\n')

    def SavePeak(self, ymax, peak, params):
        # iterate through all fit parameters
        for name in params.keys():
            # and find the current peak
            peakparameter = re.findall(peak, name)

            if peakparameter:
                # create file for each parameter
                file = self.get_file(dir=self.pardir_peak,
                                     prefix='', suffix='',
                                     datatype='dat', label=name)

                # get parameters for saving
                peakparameter = name.replace(peak, '')
                value = params[name].value
                error = params[name].stderr

                value, error = self.ScaleParameters(ymax, peakparameter,
                                                    value, error)

                if self.second_analysis == True:
                    self.Save2nd(file, value, error)
                else:
                    with open(file, 'a') as f:
                        f.write('{:>13.5f}'.format(value)
                                + '\t' + '{:>11.5f}'.format(error)
                                + '\n')

    def GenerateUsedPeaks(self, fitresults):
        # find all peaks that were fitted and generate a list
        allpeaks = []
        for i, fit in enumerate(fitresults):
            if fitresults[i] != None:
                allpeaks.extend(re.findall('prefix=\'(.*?)\'', fitresults[i].model.name))

        usedpeaks = list(set(allpeaks))

        return usedpeaks

    def SaveUnusedPeaks(self, peaks, usedpeaks, fitresults):
        # find all prefixes used in the current model
        modelpeaks = re.findall('prefix=\'(.*?)\'', fitresults.model.name)
        unusedpeaks = list(set(usedpeaks)-set(modelpeaks))

        # save default value for each parameter of unused peaks
        for peak in unusedpeaks:
            # get the peaktype and number of the peak
            number = int(re.findall('\d', peak)[0]) - 1
            peaktype = re.sub('_p.*_', '', peak)

            # create model with parameters as before
            model = self.ChoosePeakType(peaktype, number)
            model = StartingParameters(model, peaks)
            model.make_params()

            # go through all parameters and write missing values
            for parameter in model.param_names:
                peakfile = self.get_file(dir=self.pardir_peak,
                                         prefix='', suffix='',
                                         label=parameter, datatype='dat')

                # open file and write missing values
                if self.second_analysis == True:
                    self.Save2nd(peakfile, self.missingvalue, self.missingvalue)
                else:
                    with open(peakfile, 'a') as f:
                        f.write('{:>13.5f}'.format(self.missingvalue)
                                + '\t' + '{:>11.5f}'.format(self.missingvalue)
                                + '\n')

    def UpdatePeaklist(self, usedpeaks):
        # get all peaks that are used
        list_of_files, number_of_files = data.GetFolderContent(
                                            folder=self.pardir_peak,
                                            filetype='dat',
                                            quiet=True)
        peaklist = []
        for i, file in enumerate(list_of_files):
            peakfile = list_of_files[i].split('/')[-1]
            parameter = peakfile.split('_')[-1]
            peak = re.sub(parameter, '', peakfile)
            peaklist.append(peak)

        # make a set of all peaks used
        peaklist = list(set(peaklist))
        for peak in peaklist:
            usedpeaks.append(peak)

        usedpeaks = list(set(peaklist))

        return usedpeaks

    def SaveAllFitParams(self, ymax, fitresults, peaks):
        usedpeaks = self.GenerateUsedPeaks(fitresults)

        if self.second_analysis == True:
            usedpeaks = self.UpdatePeaklist(usedpeaks)

        for i, spectrum in enumerate(ymax):
            self.label = i
            self.SaveFuncParams(self.SaveSpec, ymax[i][0], fitresults[i], peaks)
            self.SaveFuncParams(self.SavePeak, ymax[i][0], fitresults[i], peaks)
            self.SaveUnusedPeaks(peaks, usedpeaks, fitresults[i])

    def ReduceDecimals(self, values):
        """
        Function that reduces the decimal places to one and returns the
        values and the corresponding exponent.
        """
        # look for zeros and replace them with negligible exponent
        values[values == 0] = 1e-256
        # get exponents of the values and round them to smallest integer
        exponents = np.log10(abs(values))
        exponents = np.floor(exponents)
        # check if there is any number smaller than 1
        if any(exponent < 0 for exponent in exponents):
            # not implemented yet
            pass
        # divide tickvalues by biggest exponent
        max_exp = np.max(exponents)
        divisor = np.power(10, max_exp)
        values = values / divisor

        return values, max_exp

    def LabelZ(self, plt, ax, label='Integrated Intensity\n', nbins=5,
               linear=False, unit='arb. u.'):
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
        clb.locator = tick_locator
        clb.update_ticks()

        # get tickvalues and reduce the decimals
        tickvalues = clb.get_ticks()
        tickvalues, max_exp = self.ReduceDecimals(tickvalues)

        clb.set_label(label
                      + '(10$^{{{:1.0f}}}$ '.format(max_exp)
                      + unit + ')', fontsize='small')
        clb.ax.set_yticklabels('{:1.2f}'.format(x) for x in tickvalues)
        clb.ax.tick_params(labelsize='small')

    def CreatePlotValues(self, maptype, y, **kwargs):
        """
        Create plot values accordingly to the type specified
        """
        plot_value = np.empty(y.shape[0])
        savefile = ''

        if maptype == 'raw':
            for i, spectrum in enumerate(y):
                selectedvalues = spectrum[(kwargs['x'][0] > kwargs['xmin'])
                                          & (kwargs['x'][0] < kwargs['xmax'])]
                plot_value[i] = sum(selectedvalues)

            savefile = self.pltdir + '/map_raw'
        elif maptype == 'params' or (maptype in mapoperators):
            plot_value = np.copy(y)
            savefile = self.pltdir + '/map_' + kwargs['name']
        elif maptype == 'errs':
            plot_value = np.copy(y)
            savefile = self.pltdir + '/err_' + kwargs['name']

        return plot_value, savefile

    def CorrectPlotValues(self, plot_value):
        """
        Check plot_values and replace them with mean value.
        """
        # check if any value in plot_value is a missing value or 1
        missingindices = [i for i, x in enumerate(plot_value)
                          if ((x == self.missingvalue) or (x == 1.0)
                                                      or (x == 0.0))]
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

        return plot_value, fitmean

    def CreatePlotMatrices(self, maptype, y, mapdims, **kwargs):

        plot_value, savefile = self.CreatePlotValues(maptype, y, **kwargs)

        plot_value, fitmean = self.CorrectPlotValues(plot_value)

        # create matrix for plotting
        plot_matrix = np.reshape(plot_value, mapdims)
        plot_matrix = np.flipud(plot_matrix)

        # create matrix with missing values
        missing_matrix = np.full_like(plot_matrix, False, dtype=bool)
        missing_matrix = (plot_matrix == fitmean)

        return plot_matrix, missing_matrix, savefile

    def CreatePatchMask(self, mapdims, fig, missing_matrix, size=1.0):
        xdim = mapdims[0]
        ydim = mapdims[1]
        # Create list for all the missing values as missing patches
        missingboxes = []
        facecolor = []

        # find all fields not containing signals and append to
        for iy in range(0,ydim):
            for ix in range(0,xdim):
                if missing_matrix[iy][ix]:
                    # calculate position correction for the patches
                    corr = size / 2.0
                    linecorr = matplotlib.rcParams['axes.linewidth']/fig.dpi/4
                    # create the missing patch and add to list
                    rect = matplotlib.patches.Rectangle((ix - corr + linecorr*4,
                                              iy - corr - linecorr), size, size)
                    missingboxes.append(rect)
                if size == 1.0:
                    facecolor.append((0.05, 0.05, 0.05, 1))
                else:
                    facecolor.append('white')

        # Create patch collection with specified colour/alpha
        pc = matplotlib.collections.PatchCollection(missingboxes,
                                                    facecolor=facecolor)
        return pc

    def ConfigureTicks(self, mapdims, step, xticker, plt):
        xdim = mapdims[0]
        ydim = mapdims[1]
        # create x and y ticks accordingly to the parameters of the mapping
        x_ticks = np.arange(step, step * (xdim + 1), step=xticker*step)
        y_ticks = np.arange(step, step * (ydim + 1), step=step)
        y_ticks = y_ticks[::-1]

        plt.xticks(np.arange(xdim, step=xticker), x_ticks, fontsize='small')
        plt.yticks(np.arange(ydim), y_ticks, fontsize='small')

    def ConfigurePlot(self, plt, ax, peak, **kwargs):
        # set title, label of x, y and z axis
        plt.title('Mapping of ' + self.folder + ' ' + peak, fontsize='small')
        plt.ylabel('y-Position ($\mathrm{\mu}$m)', fontsize='small')
        plt.xlabel('x-Position ($\mathrm{\mu}$m)', fontsize='small')
        self.LabelZ(plt, ax, **kwargs)

        # have a tight layout
        plt.tight_layout()

    def PlotMapping(self, maptype, y, mapdims, step,
                    xticker=1, colormap='Reds', alpha=1.0,
                    numbered=False, vmin=None, vmax=None, grid=False, **kwargs):
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
        xticker : int
        colormap : string
            Defines the coloring of the mapping according to the `matplotlib
            colormaps <https://matplotlib.org/users/colormaps.html>`_
        """
        plot_matrix, missing_matrix, savefile = self.CreatePlotMatrices(maptype,
                                                        y, mapdims[::-1], **kwargs)

        # create and configure figure for mapping
        matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
        fontsize_int = 20 + np.sqrt(mapdims[0] * mapdims[1])
        matplotlib.rcParams.update({'font.size': fontsize_int})

        def set_size(mapdims, ax=None):
            w = mapdims[0]
            h = mapdims[1]
            """ w, h: width, height in inches """
            if not ax: ax=plt.gca()
            left = ax.figure.subplotpars.left
            right = ax.figure.subplotpars.right
            top = ax.figure.subplotpars.top
            bot = ax.figure.subplotpars.bottom
            figw = float(w)/(right-left)
            figh = float(h)/(top-bot)
            # correct width and hight for non quadratic sizes
            dims = [figw, figh]
            dims.sort(reverse=True)
            correction = dims[0]/dims[1]/10
            figw = figw + correction*2
            figh = figh + correction
            ax.figure.set_size_inches(figw, figh)

        fig, ax = plt.subplots(figsize=mapdims)
        ax.set_aspect('equal')
        set_size(mapdims)
        self.ConfigureTicks(mapdims, step, xticker, plt)

        # plot mapping, create patch mask and plot it over map
        plt.imshow(plot_matrix, cmap=colormap, vmin=vmin, vmax=vmax)
        pc = self.CreatePatchMask(mapdims, fig, missing_matrix)
        ax.add_collection(pc)
        if grid:
            patches = self.CreatePatchMask(mapdims, fig, plot_matrix, size=0.75)
            ax.add_collection(patches)

        # number the patches if numbered == True
        def NumberMap(mapdims, ax):
            product = mapdims[0] * mapdims[1]
            for i in range(0, mapdims[0]):
                for j in range(0, mapdims[1]):
                    color = 'black'
                    if missing_matrix[j][mapdims[0] - i-1]:
                        color = 'white'
                    text = ax.text(mapdims[0] - i-1, j,
                                   product - (j * mapdims[0] + i),
                                   ha='center', va='center',
                                   color=color, fontsize=fontsize_int*0.8)

        if numbered:
            NumberMap(mapdims, ax)

        # configure, save and show the plot
        plotname = re.sub(self.folder + '/results/plot/', '', savefile)
        parameter = plotname.split('_')[-1]
        peaknumber = plotname.split('_')[-2]
        peakshape = 'raw'
        zlabel = modelparameters[parameter] + '\n'
        if parameter != 'raw':
            peakshape = plotname.split('_')[-3]
        if maptype == 'errs':
            zlabel = 'Relative error of\n' + zlabel
        elif maptype in mapoperators:
            parameters = plotname.split('_' + maptype + '_')
            shapeA = parameters[0].split('_')[-3]
            shapeB = parameters[1].split('_')[-3]
            parameterA = parameters[0].split('_')[-1]
            parameterB = parameters[1].split('_')[-1]
            zlabel = (shapeA + ' ' + modelparameters[parameterA] + ' '
                     + mapoperators[maptype] + '\n'
                     + shapeB + ' ' + modelparameters[parameterB] + '\n')
        self.ConfigurePlot(plt, ax,
                           peak = peakshape[0:4] + ' ' + peaknumber,
                           label = zlabel,
                           unit = modelunits[parameter])
        plt.savefig(savefile + '_' + colormap + '.pdf', format='pdf')
        plt.savefig(savefile + '_' + colormap + '.png')
        plt.close()

        # function removes color from image and sets it to transparent
        def setColorAlpha(filename, color, alpha):
            with Image.open(filename) as img:
                img = img.convert('RGBA')
                datas = img.getdata()

                newData = []
                for item in datas:
                    if (item[0] == color[0]
                    and item[1] == color[1]
                    and item[2] == color[2]):
                        newData.append((color[0], color[1], color[2], alpha))
                    else:
                        newData.append(item)

                img.putdata(newData)
                img.save(filename, 'png')

        # function that sets transparency of all colors but one
        def setAllToAlpha(filename, color, alpha):
            alpha = int(alpha * 255)
            with Image.open(filename) as img:
                img = img.convert('RGBA')
                datas = img.getdata()

                newData = []
                for item in datas:
                    if (item[0] == color[0]
                    and item[1] == color[1]
                    and item[2] == color[2]):
                        newData.append((color[0], color[1], color[2], 255))
                    elif item[3] == 255:
                        newData.append((item[0], item[1], item[2], alpha))
                    else:
                        newData.append(item)

                img.putdata(newData)
                img.save(filename, 'png')

        if grid:
            # set white to transparent
            setColorAlpha(savefile + '_' + colormap + '.png',
                         (255, 255, 255), 0)
            # set alpha of all but black
            setAllToAlpha(savefile + '_' + colormap + '.png',
                         (0, 0, 0), alpha)

        print(plotname + ' ' + colormap + ' plotted')

    def PlotAllColormaps(self, maptype, y, mapdims, step, **kwargs):
        """
        """
        for category in cmaps:
            print(category[0])
            for colormap in category[1]:
                self.PlotMapping(maptype=maptype, y=y, mapdims=mapdims, step=step,
                                 colormap=colormap, **kwargs)

    def CreatePeakList(self, peakFileList, filetype='dat'):
        """
        Function that creates a list of peaks from a list of file paths
        handed to it.
        """
        peakList = []
        for mapping in peakFileList:
            mapping = mapping.split('/')[-1]
            mapping = re.sub('.' + filetype, '', mapping)
            peakList.append(mapping)
        return peakList

    def ReplaceMissingValues(self, corrected, parameterArray):
        """
        Function that returns a corrected array, with missing indices
        taken from parameterArray.
        """
        missingvalue = self.missingvalue
        missingindices = [i for i, x in enumerate(parameterArray) if
                                                 (x == missingvalue)]
        for index in missingindices:
            corrected[index] = missingvalue
        return corrected

    def ModifyValues(self, first, second, operation='div'):
        """
        Function that modifies two arrays with the selected operation.
        It takes the missing values from both arrays and sets them as missing
        values of the resulting array.
        """
        if operation == 'div':
            result = np.divide(first, second)
        elif operation == 'mult':
            result = np.multiply(first, second)
        elif operation == 'add':
            result = np.add(first, second)
        elif operation == 'sub':
            result = np.subtract(first, second)
        result = self.ReplaceMissingValues(result, first)
        result = self.ReplaceMissingValues(result, second)
        return result
