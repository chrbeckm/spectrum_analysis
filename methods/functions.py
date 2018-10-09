import numpy as np

import os, shutil

import matplotlib.pyplot as plt

from uncertainties import correlated_values, ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds

from scipy.optimize import curve_fit
from scipy.special import wofz, erf

from lmfit import Model
from lmfit.models import PolynomialModel, VoigtModel, ConstantModel
from lmfit.model import save_modelresult, load_modelresult
from lmfit.model import save_model, load_model


# function that initializes data for evaluation
def initialize(data_file):
    x, y = np.genfromtxt(data_file, unpack = True)  # get data
    maxyvalue = np.max(y)                           # get max of y to
    y = y / maxyvalue                               # norm the intensity for
                                                    # faster fit
    return x, y, maxyvalue                          # return x and y and maxyvalue


# function that plots regions chosen by clicking into the plot
def PlotVerticalLines(ymax, color, fig):
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


# Select the interesting region in the spectrum, by clicking on the plot
def SelectSpectrum(x, y, label):
    # plot spectrum
    fig, ax = plt.subplots()        # create figure
    ax.plot(x, y, 'b-', label = 'Data')     # plot data to figure
    ax.set_title('Select the part of the spectrum you wish to consider by clicking into the plot.') 
    ax.set_ylim(bottom = 0)         # set ylim as zero
    ymax = np.max(y)                # calculate max of y

    xregion = PlotVerticalLines(ymax, 'green', fig)

    plt.legend(loc='upper right')
    plt.show()
    yreduced = y[(x > xregion[0]) & (x < xregion[-1])]
    xreduced = x[(x > xregion[0]) & (x < xregion[-1])]
    np.savetxt(label + '/spectrumborders_' + label + '.txt', np.array(xregion))

    return xreduced, yreduced #arrays with data from the spectra


#function to select the data that is relevent for the background
def SelectBaseline(x, y, label):
    # plot the reduced spectrum
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b-', label = 'Data')
    ax.set_title('Normalized spectrum \n Select the area of the spectrum you wish to consider for the background by licking into the plot \n (3rd-degree polynomial assumed)')
    ax.set_ylim(bottom = 0)
    ymax = np.max(y)

    # choose the region
    xregion = PlotVerticalLines(ymax, 'red', fig)

    plt.legend(loc = 'upper right')
    plt.show()

    np.savetxt(label + '/baseline_'+ label + '.txt', np.array(xregion))

    #return the name of the baselinefile
    return label + '/baseline_'+ label + '.txt'


# Creates a plot of the raw data.
def PlotRawData(x, y, show = True, ax = None):    # show = True will show the plot, show = False will return a matplotlib object

    if (ax != None):
        return ax.plot(x, y, 'kx', label = 'Data', linewidth = 0.5)
    if(show == True):
        plt.plot(x, y, 'k-', label = 'Data')
        plt.show()
    else:
        return plt.plot(x, y, 'bx', label = 'Data', linewidth = 0.5)


def Fitbaseline(x, y, baselinefile, show = False):
    # Load the bounderies for the relevent data from SelectBaseline()
    bed = np.genfromtxt(baselinefile, unpack = True)
    #generate mask for the baseline fit, for relevent data relevant = True, else relevant = False
    
    relevant = (x <= bed[0]) #bed[0] is the lowest border
    for i in range(1, len(bed) - 2, 2): #upper borders i
        relevant = relevant | ((x >= bed[i]) & (x <= bed[i + 1])) #take only the data between the borders
    relevant = relevant | (x >= bed[-1]) #bed[-1] is the highest border

    # Third-degree polynomial to model the background
    background = PolynomialModel(degree = 3) 
    pars = background.guess(y[relevant], x = x[relevant])
    fitresult_background = background.fit(y[relevant], pars, x = x[relevant])

    if (show == True):
        PlotRawData(False)
        # plot the fitted function in the hole range
        xplot = np.linspace(x[0], x[-1], 100)
        plt.plot(xplot, background.eval(fitresult_background.params, x = xplot), 'r-')
        plt.show()

    return fitresult_background #return fit parameters


# function that plots the dots at the peaks you wish to fit
def PlotPeaks(fig):
    xpeak = []  # x and
    ypeak = []  # y arrays for peak coordinates

    def onclickpeaks(event):                        
        if event.button:                            
            xpeak.append(event.xdata)               # append x data and
            ypeak.append(event.ydata)               # append y data
            plt.plot(event.xdata, event.ydata, 'ko')# plot the selected peak
            fig.canvas.draw()                       # and show it

    # actual execution of the defined function oneclickpeaks
    cid = fig.canvas.mpl_connect('button_press_event', onclickpeaks)
    figManager = plt.get_current_fig_manager()  # get current figure
    figManager.window.showMaximized()           # show it maximized

    return xpeak, ypeak

# function that allows you to select peaks for fitting
def SelectPeaks(x, y, fitresult_background, label):

    #Load the background
    background = PolynomialModel(degree = 3) # Third-degree polynomial to model the background

    # plot
    fig, ax = plt.subplots()
    ax.plot(x, y - background.eval(fitresult_background.params, x = x), 'b-') #row-data
    ax.set_title('Background substracted, normalized spectrum \n Select the maxima of the peaks to fit.') #title

    xpeak, ypeak = PlotPeaks(fig) #xpeak and ypeak are arrays of initial values

    plt.legend(loc = 'upper right')
    plt.show()
    # store the chosen initial values
    peakfile = label + '/locpeak_' + label + '.txt'
    np.savetxt(peakfile,
               np.transpose([np.array(xpeak), np.array(ypeak)]))

def FitSpectrum(x, y, maxyvalue, fitresult_background, label):
    print('Function: FitSpectrum')

    background = PolynomialModel(degree = 3) # Third-degree polynomial to model the background
    y_fit = y - background.eval(fitresult_background.params, x = x) #substracted-data
    
    xpeak, ypeak = np.genfromtxt(label + '/locpeak_' + label + '.txt',
                                 unpack = True)
    
    if type(xpeak) == np.float64:
        xpeak = [xpeak]
        ypeak = [ypeak]
    #fit the first prak of a composed model    

    
    ramanmodel = ConstantModel()

    #make a model composed of all single voigt-peaks
    for i in range(0, len(xpeak)):
        prefix = 'p' + str(i + 1) + '_'

        temp = VoigtModel(prefix = prefix, nan_policy = 'omit')
        temp.set_param_hint('center', #staring value 'peak position' is not allowed to vary much
                            value = xpeak[i],
                            min = xpeak[i]-10,
                            max = xpeak[i]+10)
        temp.set_param_hint('sigma', #starting value gauß-width
                            value = 1,
                            min = 0,
                            max = 100)
        temp.set_param_hint('gamma', #starting value lorentzian-width (== gauß-width by default)
                            value = 1,
                            min = 0,
                            max = 100,
                            vary = True, expr = '') #vary gamma indedendently
        temp.set_param_hint('amplitude', # starting value amplitude ist approxamitaly 11*height (my guess)
                            value = ypeak[i]*11,
                            min = 0)
        # height through function evaluation
        temp.set_param_hint('height',
                            value = ypeak[i],
                            expr = 'wofz(((0) + 1j*'+ prefix + 'gamma) / '+
                                    prefix + 'sigma / sqrt(2)).real')
        #precise FWHM approximation by Olivero and Longbothum (doi:10.1016/0022-4073(77)90161-3)
        temp.set_param_hint('fwhm',
                            expr = '0.5346 * 2 *' + prefix +
                                   'gamma + sqrt(0.2166 * (2*' + prefix +
                                   'gamma)**2 + (2 * ' + prefix +
                                   'sigma * sqrt(2 * log(2) ) )**2  )')

        ramanmodel += temp #compose the models to 'ramanmodel'


    pars = ramanmodel.make_params() #create the fit parameters of the beackgound substracted fit
    #fitting method can be varied (https://lmfit.github.io/lmfit-py/fitting.html)
    fitresult_peaks = ramanmodel.fit(y_fit, pars, x = x, method = 'leastsq', scale_covar = True) #acutal fit

    #show fit report in terminal
    print(fitresult_peaks.fit_report(min_correl=0.5))
    comps = fitresult_peaks.eval_components()

    #Plot the sprectrum, the fitted data, and the background
    plt.plot(x, y * maxyvalue, 'bx', label = 'Data') #raw-data
    plt.plot(x, background.eval(fitresult_background.params, x = x) * maxyvalue, 'k-', label = 'Background') #background
    plt.plot(x, (ramanmodel.eval(fitresult_peaks.params, x = x) + background.eval(fitresult_background.params, x = x)) * maxyvalue, 'r-', label = 'Fit') #fit + background

    figManager = plt.get_current_fig_manager()  # get current figure
    figManager.window.showMaximized()           # show it maximized

    plt.legend(loc = 'upper right')
    plt.savefig(label + '/rawplot_' + label + '.pdf')
    plt.show()
    return fitresult_peaks

def FitSpectrumInit(x, y, maxyvalue, oldlabel, label, baselinefile):
    print('Function: FitSpectrumInit')
    # Fit the spectrum with the fit params of another spectrum
    # (given by label) as initial values. Useful when you fit several
    # similar spectra.

    #oldlabel: the label used before

    #copy the spectrum borders into the right folder
    borders = np.genfromtxt(oldlabel + '/spectrumborders_' + oldlabel + '.txt',
                            unpack = True)
    np.savetxt(label + '/spectrumborders_' + label + '.txt', borders)

    y = y[(x > borders[0])  &  (x < borders[-1])]
    x = x[(x > borders[0])  &  (x < borders[-1])]

    #take the fit data from the pervious spectrum
    FitData =  np.load(oldlabel + '/fitparams_' + oldlabel + '.npz')
    
    baseline = [FitData['c0'], FitData['c1'], FitData['c2'], FitData['c3']] / maxyvalue
    
    center = FitData['x0']
    sigma = FitData['sigma']
    gamma = FitData['gamma']
    height = FitData['height'] / maxyvalue


    #Fit Baseline (with starting values from the previous spectrum)
    # Same data range as the previous spectrum
    bed = np.genfromtxt(baselinefile, unpack = True)
    relevant = (x <= bed[0]) 
    for i in range(1, len(bed) - 2, 2): 
        relevant = relevant | ((x >= bed[i]) & (x <= bed[i + 1])) 
    relevant = relevant | (x >= bed[-1]) 

    background = PolynomialModel(degree = 3) # Third-degree polynomial to model the background
    #pars = background.guess(y[relevant], x = x[relevant])
    background.set_param_hint('c0', value = baseline[0])
    background.set_param_hint('c1', value = baseline[1])
    background.set_param_hint('c2', value = baseline[2])
    background.set_param_hint('c3', value = baseline[3])

    pars_background = background.make_params()

    fitresult_background = background.fit(y[relevant], pars_background, x = x[relevant])


    #Fit Peaks (with starting values from the previous sprectrum)
    ramanmodel = ConstantModel()

    #make a model composed of all single voigt-peaks
    for i in range(0, len(center)):
        prefix = 'p' + str(i + 1) + '_'

        temp = VoigtModel(prefix = prefix, nan_policy = 'omit')
        temp.set_param_hint('center', #staring value 'peak position' is not allowed to vary much
                            value = center[i],
                            min = center[i]-10,
                            max = center[i]+10)
        temp.set_param_hint('sigma', #starting value gauß-width
                            value = sigma[i], #starting value gauß-width
                            min = 0,
                            max = 100)
        temp.set_param_hint('gamma', #starting value lorentzian-width (== gauß-width by default)
                            value = gamma[i],
                            min = 0,
                            max = 100,
                            vary = True, expr = '') #vary gamma indedendently
        temp.set_param_hint('amplitude', # starting value amplitude ist approxamitaly 11*height (my guess)
                            value = height[i]*11,
                            min = 0)
        # height through function evaluation
        temp.set_param_hint('height',
                            value = height[i],
                            expr = 'wofz(((0) + 1j*'+ prefix + 'gamma) / '+
                                    prefix + 'sigma / sqrt(2)).real')
        #precise FWHM approximation by Olivero and Longbothum (doi:10.1016/0022-4073(77)90161-3)
        temp.set_param_hint('fwhm',
                            expr = '0.5346 * 2 *' + prefix +
                                   'gamma + sqrt(0.2166 * (2*' + prefix +
                                   'gamma)**2 + (2 * ' + prefix +
                                   'sigma * sqrt(2 * log(2) ) )**2  )')

        ramanmodel += temp #compose the models to 'ramanmodel'

    pars_peaks = ramanmodel.make_params() #create the fit parameters

    #fit only the peaks without the backgound:
    y_fit = y - background.eval(fitresult_background.params, x = x)
    #fitting method can be varied (https://lmfit.github.io/lmfit-py/fitting.html)
    fitresult_peaks = ramanmodel.fit(y_fit, pars_peaks, x = x, method = 'leastsq', scale_covar = True) #acutal fit

    #show fit report in terminal
    print(fitresult_peaks.fit_report(min_correl=0.5))

    #ATTETION (https://lmfit.github.io/lmfit-py/fitting.html)
    #In some cases, it may not be possible to estimate the errors and correlations. 
    #For example, if a variable actually has no practical effect on the fit, it will 
    #likely cause the covariance matrix to be singular, making standard errors impossible to estimate.
    #Placing bounds on varied Parameters makes it more likely that errors cannot be estimated, as 
    #being near the maximum or minimum value makes the covariance matrix singular. In these cases,
    #the errorbars attribute of the fit result (Minimizer object) will be False.

    plt.clf()


    plt.plot(x, y * maxyvalue, 'bx', label = 'Data') #raw-data
    plt.plot(x, background.eval(fitresult_background.params, x = x) * maxyvalue, 'k-', label = 'Background') #background
    plt.plot(x, (ramanmodel.eval(fitresult_peaks.params, x = x) + background.eval(fitresult_background.params, x = x)) * maxyvalue, 'r-', label = 'Fit') #fit + background
    plt.legend(loc = 'upper right')
    plt.savefig(label + '/rawplot_' + label + '.pdf')
    plt.clf()
#
    return fitresult_peaks, fitresult_background
    #return fitresult_peaks

def SaveFitParams(x, y, maxyvalue, fitresult_peaks, fitresult_background, label):
    print('Function: SaveFitParams')
    #Save the Results of the fit in a .zip file using numpy.savez().

    fitparams_back = fitresult_background.params #Fitparameter Background
    fitparams_peaks = fitresult_peaks.params #Fitparamter Peaks

    height, stdheight, \
    x0, stdx0, \
    sigma, stdsigma, \
    gamma, stdgamma, \
    fwhm, stdfwhm = ([] for i in range(10))

    for name in list(fitparams_peaks.keys()):
        par_peaks = fitparams_peaks[name]
        param_peaks = ufloat(float(par_peaks.value), float(par_peaks.stderr)) #error may occur in this line (par.stderr = infty?)

        if ('height' in name):
            param_peaks = param_peaks * maxyvalue #because the fitted spectrum is normalized
            height.append(param_peaks.n)
            stdheight.append(param_peaks.s)

        elif ('center' in name):
            x0.append(param_peaks.n)
            stdx0.append(param_peaks.s)

        elif ('sigma' in name):
            sigma.append(param_peaks.n)
            stdsigma.append(param_peaks.s)

        elif ('gamma' in name):
            gamma.append(param_peaks.n)
            stdgamma.append(param_peaks.s)

        elif ('fwhm' in name):
            fwhm.append(param_peaks.n)
            stdfwhm.append(param_peaks.s)

        elif ('c' in name):
            param_peaks = param_peaks * maxyvalue #because the fitted spectrum is normalized
            c = param_peaks.n
            stdc = param_peaks.s

    for name in list(fitparams_back.keys()):
        par_back = fitparams_back[name]
        param_back = ufloat(par_back.value, par_back.stderr) #error may occur in this line (par.stderr = infty?)

        if ('c0' in name):
            param_back = param_back * maxyvalue #because the fitted spectrum is normalized
            c0 = param_back.n
            stdc0 = param_back.s
            
        elif ('c1' in name):
            param_back = param_back * maxyvalue #because the fitted spectrum is normalized
            c1 = param_back.n
            stdc1 = param_back.s

        elif ('c2' in name):
            param = param_back * maxyvalue #because the fitted spectrum is normalized
            c2 = param_back.n
            stdc2 = param_back.s

        elif ('c3' in name):
            param_back = param_back * maxyvalue #because the fitted spectrum is normalized
            c3 = param_back.n
            stdc3 = param_back.s 

    np.savez(label + '/fitparams_' + label , x0 = x0, stdx0 = stdx0,            
        height = height, stdheight = stdheight,            
        sigma = sigma, stdsigma = stdsigma, gamma = gamma,
        stdgamma = stdgamma, fwhm = fwhm, stdfwhm = stdfwhm,
        c0 = c0, c1 = c1, c2=c2, c3=c3, stdc0 = stdc0,
        stdc1 = stdc1, stdc2 = stdc3, stdc3 = stdc3)

    
    # save fit parameter of single peakt in txt-files:
    #delete old folder that contains plot parameters
    dirpath = label + '/fitparams/'
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    #create new folder:
    os.makedirs(dirpath)
    #save the peak parameters
    for peak in range(0,len(x0)):
        f = open(dirpath + 'peak_' + str(peak + 1) + '.txt','a')
        f.write('Peak Position [cm^-1]: ' + str(x0[peak]) + ' +/- ' + str(stdx0[peak]) + '\n')
        f.write('Height [arb.u.]: ' + str(height[peak]) + ' +/- ' + str(stdheight[peak]) + '\n')
        f.write('Sigma (Gaussian) [cm^-1]: ' + str(sigma[peak]) + ' +/- ' + str(stdsigma[peak]) + '\n')
        f.write('Gamma (Lorentzin) [cm^-1]: ' + str(gamma[peak]) + ' +/- ' + str(stdgamma[peak]) + '\n')
        f.write('FWHM [cm^-1]: ' + str(fwhm[peak]) + ' +/- ' + str(stdfwhm[peak]) + '\n')
        f.write('FWHM, Gaussian [cm^-1]: ' + str(2*np.sqrt(2*np.log(2))*sigma[peak]) + ' +/- ' + str(2*np.sqrt(2*np.log(2))*stdsigma[peak]) + '\n')
        f.write('FWHM, Lorentzian [cm^-1]: ' + str(2*gamma[peak]) + ' +/- ' + str(2*stdgamma[peak]) + '\n')
        f.close()
    #save background parameters
    f = open(dirpath + 'background.txt','a')
    f.write('Third degree polynominal: c0 + c1*x + c2*x^2 + c3*x^3 \n')
    f.write('c0: ' + str(c+c0) + ' +/- ' + str(stdc0 + stdc) + '\n')
    f.write('c1: ' + str(c1) + ' +/- ' + str(stdc1) + '\n')
    f.write('c2: ' + str(c2) + ' +/- ' + str(stdc2) + '\n')
    f.write('c3: ' + str(c3) + ' +/- ' + str(stdc3) + '\n')
    f.close()




# function: delete temporary files
def DeleteTempFiles(label):
    os.remove(label + '/baseline_'+ label + '.txt')
    os.remove(label + '/locpeak_' + label + '.txt')
    os.remove(label + '/spectrumborders_' + label + '.txt')
    os.remove(label + '/fitparams_' + label + '.npz')