import gc
import os
import numpy as np

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spectrum_analysis.spectrum import *
from spectrum_analysis.dictionaries import *
from spectrum_analysis import data

from PIL import Image
from skimage import io

import traceback

from svgpath2mpl import parse_path
from mpl_toolkits import axes_grid1


"""
This module contains the mapping class to work multiple x, y structured
data sets.
"""

class scatter():
    def __init__(self, x, y, ax, msize=1, area=None, **kwargs):
        marker_size = 50

        # change markers if area is defined.
        # solution taken from here:
        # https://stackoverflow.com/questions/52303660/iterating-markers-in-plots/52303895#52303895
        # create array for linewidth corresponding to intensity
        if area is not None:
            area_sub = (area - min(area))
            area_norm = area_sub/max(area_sub)
            marker_linewidth = 2 + 13 * (1 - area_norm)
        else:
            # create frame like marker
            marker_linewidth = 5 * np.ones_like(y)

        # create all paths
        frames = []
        for marker_lw in marker_linewidth:
            marker_rest = marker_size - 2 * marker_lw
            markerstring = (f'm 0,0 v 0 {marker_size} h {marker_size} '
                            f'v -{marker_size} '
                            f'z m {marker_lw},{marker_lw} '
                            f'h {marker_rest} v {marker_rest} '
                            f'h -{marker_rest} z')
            frame = parse_path(markerstring)
            frames.append(frame)

        self.n = len(x)
        self.ax = ax
        self.ax.figure.canvas.draw()
        self.size_data=msize
        self.size = msize
        self.sc = self.mscatter(x, y, s=self.size, m=frames, **kwargs)
        self._resize()
        self.cid = ax.figure.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self,event=None):
        ppd=72./self.ax.figure.dpi
        trans = self.ax.transData.transform
        s =  ((trans((1, self.size_data)) - trans((0, 0))) * ppd)[1]
        if s != self.size:
            self.sc.set_sizes(s**2 * np.ones(self.n))
            self.size = s
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.ax.figure.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.ax.figure.canvas.draw_idle())
        self.timer.start()

    def mscatter(self, x, y, ax=None, m=None, **kwargs):
        import matplotlib.markers as mmarkers
        if not ax: ax=plt.gca()
        scs = ax.scatter(x,y,**kwargs)
        if (m is not None):# and (len(m)==len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                            marker_obj.get_transform())
                paths.append(path)
            scs.set_paths(paths)
        return scs

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

    def __init__(self, foldername, plot=False, datatype='txt',
                 peaknames={}):
        self.folder = foldername
        self.second_analysis = False
        self.answer = 'n'
        self.listOfFiles, self.numberOfFiles = data.GetFolderContent(
                                                        self.folder,
                                                        datatype)
        if os.path.exists(os.path.join(self.folder, 'results')) and not plot:
            self.second_analysis = True
            self.listOfFiles, self.numberOfFiles, self.indices = self.Get2ndLabels()

        self.spectrum = 0
        self.spectra = []
        for spec in self.listOfFiles:
            self.spectra.append(spectrum(spec.split('.')[-2]))

        self.pardir_peak = os.path.join(self.pardir, 'peakwise')
        self.pardir_peak_bg = os.path.join(self.pardir, 'peakwise_bg')
        self.peaknames = peaknames

        if not os.path.exists(self.pardir_peak):
            os.makedirs(self.pardir_peak)
            os.makedirs(self.pardir_peak_bg)

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
    def rawdir(self):
        return self.spectra[self.spectrum].rawdir

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
    def rawname(self):
        return self.spectra[self.spectrum].rawname

    @property
    def missingvalue(self):
        return self.spectra[self.spectrum].missingvalue

    def SplitLabel(self, file):
        return file.split(os.sep)[-1].split('.')[-2]

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
                print(f'{self.SplitLabel(label)}')

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

    def PlotAllRawSpectra(self, x, y):
        for i, spectrum in enumerate(y):
            self.label = i
            self.PlotRawSpectrum(x[i], y[i])

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

    def EvaluateAllBaselines(self, x, baselinefits):
        baselines = np.array([])
        for i, spectrum in enumerate(x):
            self.label = i
            baseline = self.EvaluateBaseline(x[i], baselinefits[i][0])
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

    def SavePeak(self, ymax, peak, params, prefix='', directory=''):
        if directory == '':
            directory = self.pardir_peak
        # iterate through all fit parameters
        for name in params.keys():
            # and find the current peak
            peakparameter = re.findall(peak, name)
            # improvement possible here, but works for now
            if peak == 'c' and 'center' in name:
                peakparameter = []

            if peakparameter:
                # create file for each parameter
                file = self.get_file(directory=directory, prefix=prefix, suffix='',
                                     datatype='dat', label=name)

                # get parameters for saving
                if prefix == '':
                    peakparameter = name.replace(peak, '')
                else:
                    peakparameter = name
                value = params[name].value
                error = params[name].stderr

                value, error = self.ScaleParameters(ymax, peakparameter,
                                                    value, error)

                if self.second_analysis == True:
                    self.Save2nd(file, value, error)
                else:
                    with open(file, 'a') as f:
                        f.write(f'{value:>13.5f}\t{error:>11.5f}\n')

    def GenerateUsedPeaks(self, fitresults):
        # find all peaks that were fitted and generate a list
        allpeaks = []
        for i, fit in enumerate(fitresults):
            if fitresults[i] != None:
                allpeaks.extend(re.findall('prefix=\'(.*?)\'',
                                fitresults[i].model.name))

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
                peakfile = self.get_file(directory=self.pardir_peak,
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
            peakfile = list_of_files[i].split(os.sep)[-1]
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

    def SaveAllBackgrounds(self, bgfits, fits, ymax, peaks):
        prefix = 'background'
        for i, background in enumerate(bgfits):
            self.label = i
            # save background parameters
            self.SaveBackground(bgfits[i][0], fits[i], ymax[i][0])
            # for each parameter in polynomial Model
            for parameter in bgfits[i][0].params.keys():
                self.SavePeak(ymax[i][0], parameter, bgfits[i][0].params,
                              prefix=prefix, directory=self.pardir_peak_bg)
            # for constant from constant Model
            self.SavePeak(ymax[i][0], 'c', fits[i].params,
                          prefix=prefix, directory=self.pardir_peak_bg)

    def LabelZ(self, clb, label='Integrated Intensity\n', nbins=5,
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

            savefile = os.path.join(self.pltdir, 'map_raw')
        elif maptype == 'params' or (maptype in mapoperators):
            plot_value = np.copy(y)
            savefile = os.path.join(self.pltdir, f'map_{kwargs["name"]}')
        elif maptype == 'errs':
            plot_value = np.copy(y)
            savefile = os.path.join(self.pltdir, f'err_{kwargs["name"]}')

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

        return plot_matrix, missing_matrix, savefile, fitmean

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

    def ConfigureTicks(self, mapdims, step, xticker, plot, grid, remove=2):
        xdim = mapdims[0]
        ydim = mapdims[1]
        # create x and y ticks accordingly to the parameters of the mapping
        x_ticks = np.arange(step, step * (xdim + 1), step=xticker*step)
        y_ticks = np.arange(step, step * (ydim + 1), step=step)
        if not grid:
            y_ticks = y_ticks[::-1]


        plot.xticks(np.arange(xdim, step=xticker), x_ticks, fontsize='small')
        plot.yticks(np.arange(ydim), y_ticks, fontsize='small')

        ax = plot.gca()
        plot.setp(ax.xaxis.get_ticklabels()[1::remove], visible=False)
        if grid:
            plot.setp(ax.yaxis.get_ticklabels()[1::remove], visible=False)
        else:
            if mapdims[1] % 2 == 0:
                plot.setp(ax.yaxis.get_ticklabels()[0::remove], visible=False)
            else:
                plot.setp(ax.yaxis.get_ticklabels()[1::remove], visible=False)

    def ConfigurePlot(self, clb, plot, peak, **kwargs):
        # set title, label of x, y and z axis
        #plt.title('Mapping of ' + self.folder + ' ' + peak, fontsize='small')
        plot.ylabel('y-Position ($\mathrm{\mu}$m)', fontsize='small')
        plot.xlabel('x-Position ($\mathrm{\mu}$m)', fontsize='small')
        self.LabelZ(clb, **kwargs)

        # have a tight layout
        plot.tight_layout()

    def PlotMapping(self, maptype, y, mapdims, step, area=None,
                    xticker=1, colormap='Reds', alpha=1.0,
                    numbered=False, vmin=None, vmax=None, grid=False,
                    background='', msize=1, plot_missing=True, **kwargs):
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
        colormap : string or colormap
            Defines the coloring of the mapping according to the `matplotlib
            colormaps <https://matplotlib.org/users/colormaps.html>`_
        """
        plot_matrix, missing_matrix, savefile, fitmean = self.CreatePlotMatrices(maptype,
                                                        y, mapdims[::-1], **kwargs)

        # create and configure figure for mapping
        matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
        fontsize_int = 14 + 3 * np.sqrt(mapdims[0] * mapdims[1])
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
        self.ConfigureTicks(mapdims, step, xticker, plt, grid)

        # plot mapping, create patch mask and plot it over map
        if grid:
            # create data for plotting
            x = []
            y = []
            x_missing = []
            y_missing = []
            plot_matrix = np.flipud(plot_matrix)
            plot_vector = list(plot_matrix.flatten())
            missing_vector = np.full_like(plot_vector, False, dtype=bool)
            missing_vector = (plot_matrix == fitmean)
            missing_vector = missing_vector.flatten()
            if area is not None:
                area_corr = list(area.flatten())
            else:
                area_corr = None

            cor = 1.5
            for i in range(1, mapdims[1]+1):
                for j in range(1, mapdims[0]+1):
                    x.append(j-cor)
                    y.append(i-cor)

            ax.set_xlim(min(x), max(x)+1)
            ax.set_ylim(min(y), max(y)+1)

            deletelist = []
            for i, missing in enumerate(missing_vector):
                if missing:
                    deletelist.append(i)
            deleted = 0
            for i in deletelist:
                x_missing.append(x[i-deleted])
                y_missing.append(y[i-deleted])
                del(x[i-deleted])
                del(y[i-deleted])
                del(plot_vector[i-deleted])
                if area_corr is not None:
                    del(area_corr[i-deleted])
                deleted += 1

            try:
                img = io.imread(background)
                pos = cor - 2
                plt.imshow(img, zorder=0, cmap='Greys_r',
                           extent=[0+pos, mapdims[0]+pos,
                                   0+pos, mapdims[1]+pos])
                del img
            except ValueError:
                #traceback.print_exc()
                print('No background given.')

            if plot_missing:
                missng_col = scatter(x_missing, y_missing, ax, msize=msize,
                                     color='black', linewidth=0.5, alpha=alpha)

            sclb = scatter(x, y, ax, c=plot_vector, msize=msize, area=area_corr,
                           cmap=colormap, linewidth=0.5, alpha=alpha)
            im = sclb.sc
            del sclb
        else:
            im = plt.imshow(plot_matrix, cmap=colormap, vmin=vmin, vmax=vmax)

            pc = self.CreatePatchMask(mapdims, fig, missing_matrix)
            ax.add_collection(pc)

        def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
            """Add a vertical color bar to an image plot."""
            divider = axes_grid1.make_axes_locatable(im.axes)
            #width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
            #pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            current_ax = plt.gca()
            cax = divider.append_axes("right", size='5%', pad=0.05)
            plt.sca(current_ax)
            cbar = im.axes.figure.colorbar(im, cax=cax, **kwargs)
            return cbar

        clb = add_colorbar(im)

        # number the patches if numbered == True
        def NumberMap(mapdims, ax):
            product = mapdims[0] * mapdims[1]
            for i in range(0, mapdims[0]):
                for j in range(0, mapdims[1]):
                    color = 'black'
                    if missing_matrix[j][mapdims[0] - i-1]:
                        color = 'white'
                    ax.text(mapdims[0] - i-1, j,
                                   product - (j * mapdims[0] + i),
                                   ha='center', va='center',
                                   color=color, fontsize=fontsize_int*0.4)

        if numbered:
            NumberMap(mapdims, ax)

        # configure, save and show the plot
        plotname = re.sub(os.path.join(self.folder, 'results', 'plot', ''), '', savefile)
        try:
            # remove grid prefix
            if grid:
                gridname = kwargs['name'].split('_')[0]
                peakparameter = re.sub(gridname, '', kwargs['name'])[1:]
            else:
                peakparameter = kwargs['name']

            # remove scaled prefix
            if vmin is not None:
                scalename = peakparameter.split('_')[0]
                peakparameter = re.sub(scalename, '', peakparameter)[1:]

            # define peak shape and parameter
            peakshape = peakparameter
            peakparameter = peakparameter.split('_')[-1]
            peakshape = re.sub('_' + peakparameter, '', peakshape)
        except KeyError:
            peakshape = 'raw'
            peakparameter = peakshape

        zlabel = self.peaknames[peakshape][peakparameter]['name'] + '\n'
        unit = self.peaknames[peakshape][peakparameter]['unit']

        if maptype == 'errs':
            zlabel = 'Relative error of\n' + zlabel

        self.ConfigurePlot(clb, plt,
                           peak = peakshape,
                           label = zlabel,
                           unit = unit)
        try:
            if 'pca' in kwargs['name']:
                clb.set_label('')
                clb.set_ticks([])
                clb.set_label('Clusters')
        except KeyError:
            pass

        if isinstance(colormap, str):
            colormap_name = colormap
        else:
            colormap_name = colormap.name
        plt.savefig(f'{savefile}_{colormap_name}.pdf', format='pdf')
        plt.savefig(f'{savefile}_{colormap_name}.png')
        plt.clf()
        plt.close(fig)
        del im, clb
        gc.collect()

        print(f'{plotname} {colormap_name} plotted')

        return plot_matrix, plotname

    def PlotHistogram(self, plot_matrix, plotname, bins=5, rng=None):
        """
        Function that plots a histogram of plotdata handed to it.
        """
        matplotlib.rcParams.update({'font.size': 16})
        # flatten matrix to vector
        plot_values = plot_matrix.flatten()

        # generate peak name
        peakparameter = plotname.split('_')[-1]
        type = plotname.split('_')[0]
        peakshape = re.sub(type + '_', '', plotname)
        peakshape = re.sub('_' + peakparameter, '', peakshape)
        print(f'hist_{plotname}')

        # generate labels from peaknames
        label = self.peaknames[peakshape][peakparameter]['name']
        unit = self.peaknames[peakshape][peakparameter]['unit']

        fig, ax = plt.subplots()
        plt.hist(plot_values, bins=bins, range=rng)

        self.FormatxLabelAndTicks(plt, name=label, unit=unit)
        self.FormatyLabelAndTicks(plt, name='counts', unit='cts')

        plt.tight_layout()

        plt.savefig(os.path.join(self.folder, 'results', 'plot',
                                 f'hist_{plotname}.png'), dpi=300)
        fig.clf()
        plt.close()

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
            mapping = mapping.split(os.sep)[-1]
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
