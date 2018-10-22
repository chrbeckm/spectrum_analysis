import numpy as np

import matplotlib.pyplot as plt

from uncertainties import correlated_values, ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds

from scipy.optimize import curve_fit
from scipy.special import wofz, erf

from lmfit import Model
from lmfit.models import ConstantModel
from lmfit.model import save_modelresult, load_modelresult
from lmfit.model import save_model, load_model

def voigtn(x, x0, sigma, gamma):
    # Voigt function as model for the peaks. Calculated numerically
    # with the complex errorfunction wofz
    # (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.wofz.html)

    z = ((x - x0) + 1j*gamma) / sigma / np.sqrt(2)
    return  np.real(wofz(z))

def poly(x, a):
    # Constant function as model for the background.

    return [a for i in x]

def initialize(data_file):
    # function that initializes data for evaluation
    x, y = np.genfromtxt(data_file, unpack = True)  # get data
    maxyvalue = np.max(y)                           # get max of y to
    y = y / maxyvalue                               # norm the intensity for
                                                    # faster fit
    return x, y, maxyvalue                          # return x and y and maxyvalue

def PlotVerticalLines(ymax, color, fig):
    # function that plots regions chosen by clicking into the plot
    xregion = []                            # variable to save chosen region

    def onclickbase(event):                 # choose region by clicking
        if event.button:                    # if clicked
            xregion.append(event.xdata)     # append data to region
            # plot vertical lines to mark chosen region
            plt.vlines(x = event.xdata,
                       color = color,
                       linestyle = '--',
                       ymin = 0, ymax = ymax)
            # fill selected region with transparent colorbar
            if(len(xregion) % 2 == 0 & len(xregion) != 1):
                barheight = np.array([ymax])                    # define bar height
                barwidth = np.array([xregion[-1] - xregion[-2]])# define bar width
                # fill region between vertical lines with prior defined bar
                plt.bar(xregion[-2],
                        height = barheight, width = barwidth,
                        facecolor = color,
                        alpha=0.2,
                        align = 'edge',
                        edgecolor='black',
                        linewidth = 5)
            fig.canvas.draw()

    # actual execution of the defined function onclickbase
    cid = fig.canvas.mpl_connect('button_press_event', onclickbase)
    figManager = plt.get_current_fig_manager()  # get current figure
    figManager.window.showMaximized()           # show it maximized

    return xregion

def SelectSpectrum(x, y, label):
    # Select the interesting region in the spectrum, by clicking on the plot

    # plot spectrum
    fig, ax = plt.subplots()        # create figure
    ax.plot(x, y)                   # plot data to figure
    ax.set_title('Select Spectrum') # define title
    ax.set_ylim(bottom = 0)         # set ylim as zero
    ymax = np.max(y)                # calculate max of y

    xregion = PlotVerticalLines(ymax, 'green', fig)

    plt.show()
    yreduced = y[(x > xregion[0]) & (x < xregion[-1])]
    xreduced = x[(x > xregion[0]) & (x < xregion[-1])]
    np.savetxt(label + '/spectrumborders_' + label + '.txt', np.array(xregion))

    return xreduced, yreduced

def SelectBaseline(x, y, label):
    # Function opens a window with the data,
    # you can select the regions that do not belong to background signal
    # by clicking in the plot

    # plot the reduced spectrum
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Baseline-Fit')
    ax.set_ylim(bottom = 0)
    ymax = np.max(y)

    # choose the region
    xregion = PlotVerticalLines(ymax, 'red', fig)
    plt.show()

    np.savetxt(label + '/baseline_'+ label + '.txt', np.array(xregion))

def PlotRawData(x, y, show = True, ax = None):
    # Creates a plot of the raw data.
    # show = True will show the plot,
    # show = False will return a matplotlib object

    if (ax != None):
        return ax.plot(x, y, 'kx', label = 'Messdaten', linewidth = 0.5)
    if(show == True):
        plt.plot(x, y, 'k-', label = 'Messdaten')
        plt.show()
    else:
        return plt.plot(x, y, 'bx', label = 'Messdaten', linewidth = 0.5)

def Fitbaseline(x, y, baselinefile, show = False):
    # Fit baseline

    # load the data from SelectBaseline
    bed = np.genfromtxt(baselinefile, unpack = True)
    #generate mask for baseline fit
    bgndx = (x <= bed[0])
    for i in range(1, len(bed) - 2, 2):
        bgndx = bgndx | ((x >= bed[i]) & (x <= bed[i + 1]))
    bgndx = bgndx | (x >= bed[-1])

    #FIT Baseline
    polyparams, cov = curve_fit(poly, x[bgndx], y[bgndx])
    if (show == True):
        PlotRawData(False)
        xplot = np.linspace(x[0], x[-1], 100)
        plt.plot(xplot, poly(xplot, *polyparams), 'r-')
        plt.show()
    base = polyparams[0]
    return correlated_values(polyparams, cov)

def PlotPeaks(fig):
    # function that allows you to select peaks for fitting
    xpeak = []  # x and
    ypeak = []  # y arrays for peak coordinates

    def onclickpeaks(event):                        # choose region by clicking
        if event.button:                            # if clicked
            xpeak.append(event.xdata)               # append x data and
            ypeak.append(event.ydata)               # append y data
            plt.plot(event.xdata, event.ydata, 'ko')# plot the selected peak
            fig.canvas.draw()                       # and show it

    # actual execution of the defined function oneclickpeaks
    cid = fig.canvas.mpl_connect('button_press_event', onclickpeaks)
    figManager = plt.get_current_fig_manager()  # get current figure
    figManager.window.showMaximized()           # show it maximized

    return xpeak, ypeak

def SelectPeaks(x, y, label):
    # Function that opens a Window with the data,
    # you can choose initial values for the peaks by clicking on the plot.

    # plot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    polyparams = Fitbaseline(x, y, label + '/baseline_'+ label + '.txt')
    ax.plot(x, poly(x, *noms(polyparams)), 'r-')

    xpeak, ypeak = PlotPeaks(fig)

    plt.show()
    # store the chosen initial values
    np.savetxt(label + '/locpeak_' + label + '.txt',
               np.transpose([np.array(xpeak), np.array(ypeak)]))
    peakfile = label + '/locpeak_' + label + '.txt'

def FitSpectrum(x, y, maxyvalue, label):
    # Fit Spectrum with initial values provided by SelectBaseline()
    # and SelectPeaks()

    polyparams = Fitbaseline(x, y, label + '/baseline_'+ label + '.txt')
    base = polyparams[0].n
    ramanmodel = ConstantModel()
    ramanmodel.set_param_hint('c', value = base, min = 0)
    globwidth = 1

    xpeak, ypeak = np.genfromtxt(label + '/locpeak_' + label + '.txt',
                                 unpack = True)
    if type(xpeak) == np.float64:
        xpeak = [xpeak]
        ypeak = [ypeak]

    for i in range(0, len(xpeak)):
        prefix = 'p' + str(i + 1)

        temp = Model(func = voigtn, prefix = prefix)
        temp.set_param_hint(prefix + 'x0',
                            value = xpeak[i],
                            min = 0)
        temp.set_param_hint(prefix + 'sigma',
                            value = globwidth,
                            min = 0)
        temp.set_param_hint(prefix + 'gamma',
                            value = globwidth,
                            min = 0)
        temp.set_param_hint(prefix + 'height',
                            value = ypeak[i],
                            expr = 'wofz(((0) + 1j*'+ prefix + 'gamma) / '+
                                    prefix + 'sigma / sqrt(2)).real')
        temp.set_param_hint(prefix + 'fwhm',
                            expr = '0.5346 * 2 *' + prefix +
                                   'gamma + sqrt(0.2166 * (2*' + prefix +
                                   'gamma)**2 + (2 * ' + prefix +
                                   'sigma * sqrt(2 * log(2) ) )**2  )')
        ramanmodel += temp

    pars = ramanmodel.make_params()
    fitresult = ramanmodel.fit(y, pars, x = x, scale_covar = True)

    print(fitresult.fit_report(min_correl=0.5))
    comps = fitresult.eval_components()
    xplot = np.linspace(x[0], x[-1], 1000)
    plt.plot(x, y * maxyvalue, 'rx')
    plt.plot(x, fitresult.best_fit * maxyvalue)
    for i in range(0, len(xpeak)):
        plt.plot(x, comps['p' + str(i+1)] * maxyvalue +
                 comps['constant'] * maxyvalue, 'k-')

    figManager = plt.get_current_fig_manager()  # get current figure
    figManager.window.showMaximized()           # show it maximized

    plt.savefig(label + '/rawplot_' + label + '.pdf')
    plt.show()
    return fitresult

def FitSpectrumInit(x, y, maxyvalue, oldlabel, label):
    # Fit the spectrum with the fit params of another spectrum
    # (given by label) as initial values. Useful when you fit big number
    # of similar spectra.

    borders = np.genfromtxt(oldlabel + '/spectrumborders_' + oldlabel + '.txt',
                            unpack = True)
    np.savetxt(label + '/spectrumborders_' + label + '.txt', borders)
    y = y[(x > borders[0])  &  (x < borders[-1])]
    x = x[(x > borders[0])  &  (x < borders[-1])]
    FitData =  np.load(oldlabel + '/fitparams_' + oldlabel + '.npz')
    baseline = FitData['c'] / maxyvalue
    ctr = FitData['x0']
    sigma = FitData['sigma']
    gamma = FitData['gamma']
    ramanmodel = ConstantModel()
    ramanmodel.set_param_hint('c', value = baseline[0], min = 0)

    for i in range(len(sigma)):
        prefix = 'p' + str(i + 1)
        temp = Model(func = voigtn, prefix = prefix)
        temp.set_param_hint(prefix + 'x0', value = ctr[i], min = 0)
        temp.set_param_hint(prefix + 'sigma', value = sigma[i], min = 0)
        temp.set_param_hint(prefix + 'gamma', value = gamma[i], min = 0)
        temp.set_param_hint(prefix + 'height', expr = 'wofz(((0) + 1j*'+
                            prefix + 'gamma) / '+ prefix +
                            'sigma / sqrt(2)).real')
        temp.set_param_hint(prefix + 'fwhm', expr = '0.5346 * 2 *' + prefix +
                            'gamma + sqrt(0.2166 * (2*' + prefix +
                            'gamma)**2 + (2 * ' + prefix +
                            'sigma * sqrt(2 * log(2) ) )**2  )')
        ramanmodel += temp

    pars = ramanmodel.make_params()
    fitresult = ramanmodel.fit(y, pars, x = x, scale_covar = True)

    plt.clf()
    comps = fitresult.eval_components()
    xplot = np.linspace(x[0], x[-1], 1000)
    plt.plot(x, y * maxyvalue, 'rx')
    plt.plot(x, fitresult.best_fit * maxyvalue)
    for i in range(0, len(sigma)):
        plt.plot(x, comps['p' + str(i+1)] * maxyvalue +
                 comps['constant'] * maxyvalue, 'k-')
    plt.savefig(label + '/rawplot_' + label + '.pdf')
    plt.clf()
    return fitresult

def SaveFitParams(x, y, maxyvalue, fitresult, label):
    #Save the Results of the fit in a .zip file using numpy.savez().

    fitparams = fitresult.params

    c, stdc, \
    x0, stdx0, \
    height, stdheight, \
    sigma, stdsigma, \
    gamma, stdgamma, \
    fwhm, stdfwhm = ([] for i in range(12))

    for name in list(fitparams.keys()):
        par = fitparams[name]
        param = ufloat(par.value, par.stderr)

        if ('c' in name):
            param = param * maxyvalue
            c.append(param.n)
            stdc.append(param.s)

        elif ('height' in name):
            param = param * maxyvalue
            height.append(param.n)
            stdheight.append(param.s)

        elif ('x0' in name):
            x0.append(param.n)
            stdx0.append(param.s)

        elif ('sigma' in name):
            sigma.append(param.n)
            stdsigma.append(param.s)

        elif ('gamma' in name):
            gamma.append(param.n)
            stdgamma.append(param.s)

        elif ('fwhm' in name):
            fwhm.append(param.n)
            stdfwhm.append(param.s)

    np.savez(label + '/fitparams_' + label , x0 = x0, stdx0 = stdx0,
             c = c, stdc = c, height = height, stdheight = stdheight,
             sigma = sigma, stdsigma = stdsigma, gamma = gamma,
             stdgamma = stdgamma, fwhm = fwhm, stdfwhm = stdfwhm)
