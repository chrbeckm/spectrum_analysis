import numpy as np

import os, shutil

import matplotlib.pyplot as plt

from uncertainties import correlated_values, ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds

from scipy.optimize import curve_fit
from scipy.special import wofz, erf

import lmfit
from lmfit import Model
from lmfit.models import PolynomialModel, ConstantModel
from lmfit.models import VoigtModel, BreitWignerModel, LorentzianModel, GaussianModel
from lmfit.model import save_modelresult, load_modelresult
from lmfit.model import save_model, load_model

from starting_params import *


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
    ax.set_title('Select the part of the spectrum you wish to consider\
    by clicking into the plot.')
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
    ax.set_title('Normalized spectrum \n Select the area of the spectrum \
    you wish to exclude from the background by licking into the plot \n \
    (3rd-degree polynomial assumed)')
    ymax = np.max(y)

    # choose the region
    xregion = PlotVerticalLines(ymax, 'red', fig)

    plt.legend(loc = 'upper right')
    plt.show()

    np.savetxt(label + '/baseline_'+ label + '.txt', np.array(xregion))

    #return the name of the baselinefile
    return label + '/baseline_'+ label + '.txt'

def SelectBaseline2(x, y, folder):
    # plot the reduced spectrum
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b-', label = 'Data')
    ax.set_title('Normalized spectrum \n Select the area of the spectrum \
    you wish to exclude from the background by licking into the plot \n \
    (3rd-degree polynomial assumed)')    #ax.set_ylim(bottom = 0)
    ymax = np.max(y)

    # choose the region
    xregion = PlotVerticalLines(ymax, 'red', fig)

    plt.legend(loc = 'upper right')
    plt.show()

    np.savetxt(folder + 'baseline.dat', np.array(xregion))

    #return the name of the baselinefile
    return folder + 'baseline.dat'

# Creates a plot of the raw data.
# show = True will show the plot, show = False will return a matplotlib object
def PlotRawData(x, y, show = True, ax = None):

    if (ax != None):
        return ax.plot(x, y, 'kx', label = 'Data', linewidth = 0.5)
    if(show == True):
        plt.plot(x, y, 'k-', label = 'Data')
        plt.show()
    else:
        return plt.plot(x, y, 'b.', label = 'Data', linewidth = 0.5)

# actual fit of the baseline
def FitBaseline(x, y, baselinefile, show = False):
    # Load the bounderies for the relevent data from SelectBaseline()
    bed = np.genfromtxt(baselinefile, unpack = True)

    # generate mask for the baseline fit, for relevent data relevant = True,
    # else relevant = False
    relevant = (x <= bed[0]) #bed[0] is the lowest border
    for i in range(1, len(bed) - 2, 2): #upper borders i
        # take only data between the borders
        relevant = relevant | ((x >= bed[i]) & (x <= bed[i + 1]))
    relevant = relevant | (x >= bed[-1]) #bed[-1] is the highest border

    # Third-degree polynomial to model the background
    background = PolynomialModel(degree = 3)
    pars = background.guess(y[relevant], x = x[relevant])
    fitresult_background = background.fit(y[relevant], pars, x = x[relevant])

    # plot the fitted function in the hole range
    if (show == True):
        PlotRawData(x, y, show=True)
        xplot = np.linspace(x[0], x[-1], 100)
        baseline = background.eval(fitresult_background.params, x = xplot)
        plt.plot(xplot, baseline , 'r-')
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

# function that allows you to select Voigt-, Fano-, Lorentzian-,
# and Gaussian-peaks for fitting
def SelectPeaks(x, y, fitresult_background, label, peaks):

    # Load the background
    background = PolynomialModel(degree = 3) # Third-degree polynomial

    # loop over all peaks and save the selected positions
    for peaktype in peaks:
        # create plot and baseline
        fig, ax = plt.subplots()
        baseline = background.eval(fitresult_background.params, x = x)
        # plot corrected data
        ax.plot(x, y - baseline, 'b-')
        ax.set_title('Background substracted, normalized spectrum \n\
                      Select the maxima of the ' + peaktype + '-PEAKS to fit.')
        xpeak, ypeak = PlotPeaks(fig) #arrays of initial values for the fits
        plt.show()
        # store the chosen initial values
        peakfile = label + '/locpeak_' + peaktype + '_' + label + '.txt'
        np.savetxt(peakfile,
                   np.transpose([np.array(xpeak), np.array(ypeak)]))

# Fit the Voigt-, Fano-, Lorentzian-, and Gaussian-Peaks
# for detailed describtions see:
# https://lmfit.github.io/lmfit-py/builtin_models.html
def FitSpectrum(x, y, maxyvalue, fitresult_background, label, peaks):

    # values from the background fit and the SelectPeak-funtion are used
    # in the following
    background = PolynomialModel(degree = 3) # Third-degree polynomial
    y_fit = y - background.eval(fitresult_background.params, x = x)

    # Create a composed model of a ConstantModel,
    # (possibly) multiple Voigt-, Fano-, Gaussian-, and Lorentzian-Models:
    ramanmodel = ConstantModel() # Add a constant for a better fit

    # go through all defined peaks
    for peaktype in peaks:
        peakfile = label + '/locpeak_' + peaktype + '_' + label + '.txt'
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
                temp = ConstantModel()
                temp = ChoosePeakType(peaktype, prefix)
                temp = StartingParameters(xpeak, ypeak, i, temp, peaks)

                ramanmodel += temp # add the models to 'ramanmodel'

    # create the fit parameters of the background substracted fit
    pars = ramanmodel.make_params()
    # fit the data to the created model
    fitresult_peaks = ramanmodel.fit(y_fit, pars, x = x, method = 'leastsq',
                                     scale_covar = True)
    # calculate confidence band
    dely = fitresult_peaks.eval_uncertainty(x = x, sigma=3)

    # show fit report in terminal
    print(fitresult_peaks.fit_report(min_correl=0.5))

    # Plot the raw sprectrum, the fitted data, and the background
    bg_line = background.eval(fitresult_background.params, x = x)
    fit_line = ramanmodel.eval(fitresult_peaks.params, x = x)
    plt.plot(x, y * maxyvalue, 'b.', label = 'Data',  markersize=2)
    plt.plot(x, bg_line * maxyvalue, 'k-', label = 'Background')
    plt.plot(x, (fit_line + bg_line) * maxyvalue, 'r-', label = 'Fit')
    # plot confidence band
    plt.fill_between(x, (fit_line + bg_line + dely) * maxyvalue,
                        (fit_line + bg_line - dely) * maxyvalue,
                        color = 'r', alpha = 0.5, label = '3$\sigma$')

    figManager = plt.get_current_fig_manager()  # get current figure
    figManager.window.showMaximized()           # show it maximized


    plt.legend(loc = 'upper right')
    plt.savefig('results_plot/rawplot_' + label + '.pdf')
    plt.show()

    return fitresult_peaks

# Fit the spectrum with the fit params of another spectrum
# (given by oldlabel, oldlabel = label from the previous spectrum)
# as initial values.
# Useful when you fit several similar spectra.
def FitSpectrumInit(x, y, maxyvalue, oldlabel, label, baselinefile):

    #copy the spectrum borders into the right folder
    borders = np.genfromtxt(oldlabel + '/spectrumborders_' + oldlabel + '.txt',
                            unpack = True)
    np.savetxt(label + '/spectrumborders_' + label + '.txt', borders)

    y = y[(x > borders[0])  &  (x < borders[-1])]
    x = x[(x > borders[0])  &  (x < borders[-1])]

    #take the fit-parameters from the pervious spectrum
    FitData =  np.load(oldlabel + '/fitparams_' + oldlabel + '.npz')
    baseline = [FitData['c0'], FitData['c1'], FitData['c2'], FitData['c3']] / maxyvalue

    center_voigt = FitData['x0_voigt']
    sigma_voigt = FitData['sigma_voigt']
    gamma_voigt = FitData['gamma_voigt']
    amplitude_voigt = FitData['amplitude_voigt'] / maxyvalue

    center_fano = FitData['x0_fano']
    sigma_fano = FitData['sigma_fano']
    q_fano = FitData['q_fano']
    amplitude_fano = FitData['amplitude_fano'] / maxyvalue

    center_lorentzian = FitData['x0_lorentzian']
    sigma_lorentzian = FitData['sigma_lorentzian']
    amplitude_lorentzian = FitData['amplitude_lorentzian'] / maxyvalue

    center_gaussian = FitData['x0_gaussian']
    sigma_gaussian = FitData['sigma_gaussian']
    amplitude_gaussian = FitData['amplitude_gaussian'] / maxyvalue

    #Fit Baseline (with starting values from the previous spectrum)
    # Same data range as the previous spectrum
    bed = np.genfromtxt(baselinefile, unpack = True)
    relevant = (x <= bed[0])
    for i in range(1, len(bed) - 2, 2):
        relevant = relevant | ((x >= bed[i]) & (x <= bed[i + 1]))
    relevant = relevant | (x >= bed[-1])

    background = PolynomialModel(degree = 3) # Third-degree polynomial to model the background
    background.set_param_hint('c0', value = baseline[0])
    background.set_param_hint('c1', value = baseline[1])
    background.set_param_hint('c2', value = baseline[2])
    background.set_param_hint('c3', value = baseline[3])

    pars_background = background.make_params()
    fitresult_background = background.fit(y[relevant], pars_background, x = x[relevant])


    #Fit Peaks (with starting values from the previous sprectrum)
    #detailled commentation see function FitSpectrum
    ramanmodel = ConstantModel()

    #make a model composed of all single voigt-peaks
    for i in range(0, len(center_voigt)):
        prefix = 'voigt_p' + str(i + 1) + '_'
        temp = VoigtModel(prefix = prefix, nan_policy = 'omit')
        temp.set_param_hint('center', #staring value 'peak position' is not allowed to vary much
                            value = center_voigt[i],
                            min = center_voigt[i]-20,
                            max = center_voigt[i]+20)
        temp.set_param_hint('sigma', #starting value gaussian-width
                            value = sigma_voigt[i], #starting value gaussian-width
                            min = 0,
                            max = 100)
        temp.set_param_hint('gamma_voigt', #starting value lorentzian-width (== gaussian-width by default)
                            value = gamma_voigt[i],
                            min = 0,
                            max = 100,
                            vary = True, expr = '') #vary gamma indedendently
        temp.set_param_hint('amplitude', # starting value amplitude is approxamitaly 11*height (my guess)
                            value = amplitude_voigt[i],
                            min = 0)
        temp.set_param_hint('height')
        #precise FWHM approximation by Olivero and Longbothum (doi:10.1016/0022-4073(77)90161-3)
        temp.set_param_hint('fwhm',
                            expr = '0.5346 * 2 *' + prefix +
                                   'gamma + sqrt(0.2166 * (2*' + prefix +
                                   'gamma)**2 + (2 * ' + prefix +
                                   'sigma * sqrt(2 * log(2) ) )**2  )')

        ramanmodel += temp #compose the models to 'ramanmodel'

    #FANO-PEAKS
    for i in range(0, len(center_fano)):
        prefix = 'fano_p' + str(i + 1) + '_'
        temp = BreitWignerModel(prefix = prefix, nan_policy = 'omit')
        temp.set_param_hint('center', #staring value 'peak position' is not allowed to vary much
                            value = center_fano[i],
                            min = center_fano[i]-20,
                            max = center_fano[i]+20)
        temp.set_param_hint('sigma', #starting value width
                            value = sigma_fano[i],
                            min = 0,
                            max = 150)
        temp.set_param_hint('q', #starting value q
                            value = q_fano[i],
                            min = -100,
                            max = 100)
        temp.set_param_hint('amplitude',
                            value = amplitude_fano[i],
                            min = 0)

        ramanmodel += temp #add the models to 'ramanmodel'

        #LORENTZIAN-PEAKS
        for i in range(0, len(center_lorentzian)):
            prefix = 'lorentzian_p' + str(i + 1) + '_'

            temp = LorentzianModel(prefix = prefix, nan_policy = 'omit')
            temp.set_param_hint('center', #staring value 'peak position' is not allowed to vary much
                                value = center_lorentzian[i],
                                min = center_lorentzian[i]-20,
                                max = center_lorentzian[i]+20)
            temp.set_param_hint('sigma', #starting value width
                                value = sigma_lorentzian[i],
                                min = 0,
                                max = 200)
            temp.set_param_hint('amplitude',
                                value = amplitude_lorentzian[i],
                                min = 0)
            temp.set_param_hint('height')
            temp.set_param_hint('fwhm')

            ramanmodel += temp #add the models to 'ramanmodel'

        #GAUSSIAN-PEAKS
        for i in range(0, len(center_gaussian)):
            prefix = 'gaussian_p' + str(i + 1) + '_'

            temp = GaussianModel(prefix = prefix, nan_policy = 'omit')
            temp.set_param_hint('center', #staring value 'peak position' is not allowed to vary much
                                value = center_gaussian[i],
                                min = center_gaussian[i]-20,
                                max = center_gaussian[i]+20)
            temp.set_param_hint('sigma', #starting value gaussian-width
                                value = sigma_gaussian[i],
                                min = 0,
                                max = 100)
            temp.set_param_hint('amplitude',
                                value = amplitude_gaussian[i],
                                min = 0)
            temp.set_param_hint('height')
            temp.set_param_hint('fwhm')

            ramanmodel += temp #add the models to 'ramanmodel'



    pars_peaks = ramanmodel.make_params() #create the fit parameters

    #fit only the peaks without the backgound:
    y_fit = y - background.eval(fitresult_background.params, x = x)
    #fitting method can be varied (https://lmfit.github.io/lmfit-py/fitting.html)
    fitresult_peaks = ramanmodel.fit(y_fit, pars_peaks, x = x, method = 'leastsq', scale_covar = True) #acutal fit

    #show fit report in terminal
    print(fitresult_peaks.fit_report(min_correl=0.5))

    #save plot
    plt.clf()
    plt.plot(x, y * maxyvalue, 'b.', label = 'Data',  markersize=2) #raw-data
    plt.plot(x, background.eval(fitresult_background.params, x = x) * maxyvalue, 'k-', label = 'Background') #background
    plt.plot(x, (ramanmodel.eval(fitresult_peaks.params, x = x) + background.eval(fitresult_background.params, x = x)) * maxyvalue, 'r-', label = 'Fit') #fit + background
    plt.legend(loc = 'upper right')
    plt.savefig('results_plot/rawplot_' + label + '.pdf')
    plt.clf()

    return fitresult_peaks, fitresult_background #return fitresult_peaks

#Save the Results of the fit in a .zip file using numpy.savez()
# and in txt-files (in folder results_fitparameter).
def SaveFitParams(x, y, maxyvalue, fitresult_peaks, fitresult_background, label):

    # get the data to be stored
    fitparams_back = fitresult_background.params #Fitparameter Background
    fitparams_peaks = fitresult_peaks.params #Fitparamter Peaks

    #fit-parameters of the composed ramanmodel (arrays in case of several peaks per spectrum)
    height_voigt, x0_voigt, sigma_voigt, gamma_voigt, fwhm_voigt, amplitude_voigt, \
    height_fano, amplitude_fano, x0_fano, sigma_fano, q_fano, fwhm_fano, \
    height_lorentzian, amplitude_lorentzian, x0_lorentzian, sigma_lorentzian, fwhm_lorentzian, \
    height_gaussian, amplitude_gaussian, x0_gaussian, sigma_gaussian, fwhm_gaussian, \
    = ([] for i in range(22))

    #put the fit-paramters into the right arrays
    for name in list(fitparams_peaks.keys()):
        par_peaks = fitparams_peaks[name]

        #voigt-peaks
        if('voigt' in name):
            if ('height' in name):
                height_voigt.append(par_peaks.value*maxyvalue) #because the fitted spectrum is normalized

            elif ('amplitude' in name):
                amplitude_voigt.append(par_peaks.value*maxyvalue) #because the fitted spectrum is normalized

            elif ('center' in name):
                x0_voigt.append(par_peaks.value)

            elif ('sigma' in name):
                sigma_voigt.append(par_peaks.value)

            elif ('gamma' in name):
                gamma_voigt.append(par_peaks.value)

            elif ('fwhm' in name):
                fwhm_voigt.append(par_peaks.value)

        #fano-peaks
        elif('fano' in name):
            if ('height' in name):
                height_fano.append(par_peaks.value*maxyvalue) #because the fitted spectrum is normalized

            elif ('amplitude' in name):
                amplitude_fano.append(par_peaks.value*maxyvalue) #because the fitted spectrum is normalized

            elif ('center' in name):
                x0_fano.append(par_peaks.value)

            elif ('sigma' in name):
                sigma_fano.append(par_peaks.value)

            elif ('fwhm' in name):
                fwhm_fano.append(par_peaks.value)

            elif ('q' in name):
                q_fano.append(par_peaks.value)

        #lorentzian-peaks
        elif('lorentzian' in name):
            if ('height' in name):
                height_lorentzian.append(par_peaks.value*maxyvalue) #because the fitted spectrum is normalized

            elif ('amplitude' in name):
                amplitude_lorentzian.append(par_peaks.value*maxyvalue) #because the fitted spectrum is normalized

            elif ('center' in name):
                x0_lorentzian.append(par_peaks.value)

            elif ('sigma' in name):
                sigma_lorentzian.append(par_peaks.value)

            elif ('fwhm' in name):
                fwhm_lorentzian.append(par_peaks.value)


        #gaussian-peaks
        elif('gaussian' in name):
            if ('height' in name):
                height_gaussian.append(par_peaks.value*maxyvalue) #because the fitted spectrum is normalized

            elif ('amplitude' in name):
                amplitude_gaussian.append(par_peaks.value*maxyvalue) #because the fitted spectrum is normalized

            elif ('center' in name):
                x0_gaussian.append(par_peaks.value)

            elif ('sigma' in name):
                sigma_gaussian.append(par_peaks.value)

            elif ('fwhm' in name):
                fwhm_gaussian.append(par_peaks.value)


        elif('c' in name):
            c = par_peaks.value * maxyvalue #because the fitted spectrum is normalized

    #background
    for name in list(fitparams_back.keys()):
        par_back = fitparams_back[name]

        if ('c0' in name):
            c0 = par_back.value * maxyvalue

        elif ('c1' in name):
            c1 = par_back.value * maxyvalue

        elif ('c2' in name):
            c2 = par_back.value * maxyvalue

        elif ('c3' in name):
            c3 = par_back.value * maxyvalue

    # save as .npz-file
    np.savez(label + '/fitparams_' + label , x0_voigt = x0_voigt,
        height_voigt = height_voigt, sigma_voigt = sigma_voigt, gamma_voigt = gamma_voigt,
        fwhm_voigt = fwhm_voigt, amplitude_voigt = amplitude_voigt,
        x0_fano = x0_fano, height_fano = height_fano,
        sigma_fano = sigma_fano, q_fano = q_fano, fwhm_fano = fwhm_fano, amplitude_fano = amplitude_fano,
        x0_lorentzian = x0_lorentzian, height_lorentzian = height_lorentzian,
        sigma_lorentzian = sigma_lorentzian, fwhm_lorentzian = fwhm_lorentzian, amplitude_lorentzian = amplitude_lorentzian,
        x0_gaussian = x0_gaussian, height_gaussian = height_gaussian,
        sigma_gaussian = sigma_gaussian, fwhm_gaussian = fwhm_gaussian, amplitude_gaussian = amplitude_gaussian,
        c0 = c0, c1 = c1, c2=c2, c3=c3 )


    # save fit parameter of single peakt in txt-files:
    # save the voigt-peak parameters
    for peak in range(0,len(x0_voigt)):
        f = open('results_fitparameter/' + label + '_voigt_' + str(peak + 1) + '.txt','a')
        f.write('Peak Position [cm^-1]: ' + str(x0_voigt[peak]) + ' +/- '  + '\n \n')
        f.write('Height [arb.u.]: ' + str(height_voigt[peak]) + ' +/- '  + '\n')
        f.write('Intensity [arb.u.]: ' + str(amplitude_voigt[peak]) + ' +/- '  + '\n \n')
        f.write('Sigma (Gaussian) [cm^-1]: ' + str(sigma_voigt[peak]) + ' +/- '  + '\n')
        f.write('Gamma (Lorentzin) [cm^-1]: ' + str(gamma_voigt[peak]) + ' +/- '  + '\n \n')
        f.write('FWHM [cm^-1]: ' + str(fwhm_voigt[peak]) + ' +/- '  + '\n')
        f.write('FWHM, Gaussian [cm^-1]: ' + str(2*np.sqrt(2*np.log(2))*sigma_voigt[peak]) + ' +/- ' + '\n')
        f.write('FWHM, Lorentzian [cm^-1]: ' + str(2*gamma_voigt[peak]) + ' +/- ' +  '\n')
        f.close()
    #save the fano-peak parameters
    for peak in range(0,len(x0_fano)):
        f = open('results_fitparameter/' + label + '_fano_' + str(peak + 1) + '.txt','a')
        f.write('Peak Position [cm^-1]: ' + str(x0_fano[peak]) + ' +/- ' + '\n \n')
        f.write('Height [arb.u.]: ' + str(height_fano[peak]) + ' +/- ' + '\n')
        f.write('Intensity [arb.u.]: ' + str(amplitude_fano[peak]) + ' +/- ' + '\n \n')
        f.write('Sigma [cm^-1]: ' + str(sigma_fano[peak]) + ' +/- ' + '\n')
        f.write('q : ' + str(q_fano[peak]) + ' +/- ' + '\n')
        f.write('FWHM [cm^-1]: ' + str(fwhm_fano[peak]) + ' +/- ' + '\n')
        f.close()
    #save the lorentzian-peak parameters
    for peak in range(0,len(x0_lorentzian)):
        f = open('results_fitparameter/' + label + '_lorentzian_' + str(peak + 1) + '.txt','a')
        f.write('Peak Position [cm^-1]: ' + str(x0_lorentzian[peak]) + ' +/- ' + '\n \n')
        f.write('Height [arb.u.]: ' + str(height_lorentzian[peak]) + ' +/- ' + '\n')
        f.write('Intensity [arb.u.]: ' + str(amplitude_lorentzian[peak]) + ' +/- ' + '\n \n')
        f.write('Sigma [cm^-1]: ' + str(sigma_lorentzian[peak]) + ' +/- ' + '\n')
        f.write('FWHM [cm^-1]: ' + str(fwhm_lorentzian[peak]) + ' +/- ' + '\n')
        f.close()
    #save the gaussian-peak parameters
    for peak in range(0,len(x0_gaussian)):
        f = open('results_fitparameter/' + label + '_gaussian_' + str(peak + 1) + '.txt','a')
        f.write('Peak Position [cm^-1]: ' + str(x0_gaussian[peak]) + ' +/- ' + '\n \n')
        f.write('Height [arb.u.]: ' + str(height_gaussian[peak]) + ' +/- ' + '\n')
        f.write('Intensity [arb.u.]: ' + str(amplitude_gaussian[peak]) + ' +/- ' + '\n \n')
        f.write('Sigma [cm^-1]: ' + str(sigma_gaussian[peak]) + ' +/- ' + '\n')
        f.write('FWHM [cm^-1]: ' + str(fwhm_gaussian[peak]) + ' +/- ' + '\n')
        f.close()
    #save background parameters
    f = open('results_fitparameter/' + label + '_background.txt','a')
    f.write('Third degree polynominal: c0 + c1*x + c2*x^2 + c3*x^3 \n')
    f.write('c0: ' + str(c+c0) + ' +/- '  + '\n')
    f.write('c1: ' + str(c1) + ' +/- '  + '\n')
    f.write('c2: ' + str(c2) + ' +/- '  + '\n')
    f.write('c3: ' + str(c3) + ' +/- '  + '\n')
    f.close()

# function: delete temporary files
def DeleteTempFiles(label):
    os.remove(label + '/baseline_'+ label + '.txt')
    os.remove(label + '/locpeak_voigt_' + label + '.txt')
    os.remove(label + '/locpeak_fano_' + label + '.txt')
    os.remove(label + '/locpeak_lorentzian_' + label + '.txt')
    os.remove(label + '/locpeak_gaussian_' + label + '.txt')
    os.remove(label + '/spectrumborders_' + label + '.txt')
    os.remove(label + '/fitparams_' + label + '.npz')
