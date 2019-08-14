import glob
import numpy as np

from spectrum_analysis.spectrum import *

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

    def __init__(self, foldername, datatype='txt'):
        self.folder = foldername
        self.second_analysis = False
        self.listOfFiles, self.numberOfFiles = self.GetFolderContent(
                                                        self.folder,
                                                        datatype)
        if os.path.exists(self.folder + '/results'):
            self.second_analysis = True
            self.listOfFiles, self.numberOfFiles = self.Get2ndLabels()

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
        answer = input('These spectra have been analyzed already.\n'
                       'Do you want to analyze all of them again? (y/n)\n')
        if answer == 'y':
            list_of_files = self.listOfFiles
            number_of_files = self.numberOfFiles
            pass
        elif answer == 'n':
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
                    print('Added ' + self.SplitLabel(self.listOfFiles[index[0]]))
                else:
                    print('This spectrum does not exist.')
            number_of_files = len(list_of_files)

        return list_of_files, number_of_files

    def Teller(self, number, kind, location='folder'):
        """
        Function that prints out how many instances there are in a
        location.

        Parameters
        ----------
        number : int
            Number of instances in a requested location.

        kind : string
            Instance that is analyzed.

        location : string, default : 'folder'
            Location where the instance can be found.
        """
        if number > 1:
            print('There are {} {}s in this {}.'.format(number, kind,
                                                        location))
            print()
        else:
            print('There is {} {} in this {}.'.format(number, kind,
                                                      location))
            print()

    def GetFolderContent(self, folder, filetype,
                         object='spectra', quiet=False):
        """
        Get a list of all files of a defined type from a folder

        Parameters
        ----------
        folder : string
            Name of the folder that should be analyzed.

        filetype : string
            Ending of the file types that should be analyzed.
            For example 'txt', 'csv' or 'dat'.

        object : string, default : 'spectra'
            Type of the objects that are analyzed.

        quiet : boolean, default : False
            Whether the command line should tell how many files are in
            the folder or not.

        Returns
        -------
        listOfFiles : sorted list of strings
            List of strings containing the names of all the files in
            the folder.

        numberOfFiles : int
            The number of files in the requested folder.
        """
        # generate list of files in requested folder
        files = folder + '/*.' + filetype
        listOfFiles = sorted(glob.glob(files))
        numberOfFiles = len(listOfFiles)
        # tell the number of files in the requested folder
        if not quiet:
            self.Teller(numberOfFiles, object)

        return listOfFiles, numberOfFiles

    def VStack(self, i, x, xtemp):
        """
        Stacks arrays
        """
        if i != 0:
            x = np.vstack((x, xtemp))
        else:
            x = np.array(xtemp)
            if len(x.shape) == 1:
                x = x.reshape(1,len(x))
            if len(x.shape) == 0:
                x = x.reshape(1,1)
        return x

    def GetAllData(self, measurement='', prnt=False):
        """
        Get data of the mapping.

        Parameters
        ----------
        measurement : string, default : ''
            Type of measurement. Supported are '' for simple x, y data
            and 'xps' for XPS-Data from BL11 at DELTA.

        prnt : boolean, default : False
            Print what data was read.

        Returns
        -------
        x : numpy.ndarray
            Read in data of the x-axis.

        y : numpy.ndarray
            Read in data of the y-axis.
        """

        # define arrays to hold data from the files
        x = np.array([])
        y = np.array([])

        for i, spec in enumerate(self.spectra):
            xtemp, ytemp = spec.GetData(measurement=measurement, prnt=prnt)
            x = self.VStack(i, x, xtemp)
            y = self.VStack(i, y, ytemp)

        return x, y

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
        xmin, xmax = self.SelectRegion(x[self.spectrum], y[self.spectrum])

        xreduced = np.array([])
        yreduced = np.array([])

        for i, spectrum in enumerate(y):
            xtemp, ytemp = self.ReduceRegion(x[i], y[i], xmin, xmax)
            xreduced = self.VStack(i, xreduced, xtemp)
            yreduced = self.VStack(i, yreduced, ytemp)

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
            baselines = self.VStack(i, baselines, baseline)

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
            ydenoised = self.VStack(i, ydenoised, ytemp)

        return ydenoised

    def NormalizeAll(self, y, ymax=None):
        ynormed = np.array([])

        if type(ymax) == type(None):
            for i, spectrum in enumerate(y):
                ynormed_temp, ymax_temp = self.Normalize(y[i])
                ynormed = self.VStack(i, ynormed, ynormed_temp)
                ymax = self.VStack(i, ymax, ymax_temp)
        else:
            for i, spectrum in enumerate(y):
                ynormed_temp, ymax_temp = self.Normalize(y[i], ymax=ymax[i])
                ynormed = self.VStack(i, ynormed, ynormed_temp)

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
            self.label = i
            self.PlotFit(x[i], y[i], ymax[i], fitresults[i], show=show)

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
                with open(file, 'a') as g:
                    # get parameters for saving
                    peakparameter = name.replace(peak, '')
                    value = params[name].value
                    error = params[name].stderr

                    value, error = self.ScaleParameters(ymax, peakparameter,
                                                        value, error)

                    g.write('{:>13.5f}'.format(value)
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
            model = ChoosePeakType(peaktype, number)
            model = StartingParameters(model, peaks)
            model.make_params()

            # go through all parameters and write missing values
            for parameter in model.param_names:
                peakfile = self.get_file(dir=self.pardir_peak,
                                         prefix='', suffix='',
                                         label=parameter, datatype='dat')

                # open file and write missing values
                with open(peakfile, 'a') as f:
                    f.write('{:>13.5f}'.format(self.missingvalue)
                            + '\t' + '{:>11.5f}'.format(self.missingvalue)
                            + '\n')

    def SaveAllFitParams(self, ymax, fitresults, peaks):
        usedpeaks = self.GenerateUsedPeaks(fitresults)

        for i, spectrum in enumerate(ymax):
            self.label = i
            self.SaveFuncParams(self.SaveSpec, ymax[i][0], fitresults[i], peaks)
            self.SaveFuncParams(self.SavePeak, ymax[i][0], fitresults[i], peaks)
            self.SaveUnusedPeaks(peaks, usedpeaks, fitresults[i])
