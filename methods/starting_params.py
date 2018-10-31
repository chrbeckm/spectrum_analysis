import re

def StartingParameters(xpeak, ypeak, i, fitmodel, peaks):
    # starting position for the peak position is not allowed to vary much
    fitmodel.set_param_hint('center',
                             value = xpeak[i],
                             min = xpeak[i] - 50,
                             max = xpeak[i] + 50)
    # get model name1
    model = re.findall('\((.*?),', fitmodel.name)
    model = model[0]
    # search if model is in peak list
    if any(model in peak for peak in peaks):
        if model == 'voigt':
            print(model)
            fitmodel.set_param_hint('sigma', #starting value gauß-width
                                value = 1,
                                min = 0,
                                max = 100)
            fitmodel.set_param_hint('gamma', #starting value lorentzian-width (== gauß-width by default)
                                value = 1,
                                min = 0,
                                max = 100,
                                vary = True, expr = '') #vary gamma independently
            fitmodel.set_param_hint('amplitude', # starting value amplitude ist approxamitaly 11*height (my guess)
                                value = ypeak[i]*11,
                                min = 0)
            #parameters calculated based on the fit-parameters
            fitmodel.set_param_hint('height',
                                value = ypeak[i])
            # precise FWHM approximation by Olivero and Longbothum (doi:10.1016/0022-4073(77)90161-3)
            # it is not possible to take the fwhm form lmfit for an independently varying gamma
            fitmodel.set_param_hint('fwhm',
                                expr = '0.5346 * 2 *' + fitmodel.prefix +
                                       'gamma + sqrt(0.2166 * (2*' + fitmodel.prefix +
                                       'gamma)**2 + (2 * ' + fitmodel.prefix +
                                       'sigma * sqrt(2 * log(2) ) )**2  )')

        if model == 'breit_wigner': # should be BreitWignerModel!
            print(model)
            #fit-parameter
            fitmodel.set_param_hint('sigma', #starting value width
                                value = 100,
                                min = 0,
                                max = 200)
            fitmodel.set_param_hint('q', #starting value q
                                value = -5,
                                min = -100,
                                max = 100)
            fitmodel.set_param_hint('amplitude', # starting value amplitude is approxamitaly 11*height (my guess)
                                value = ypeak[i]/50,
                                min = 0)

        if model == 'lorentzian':
            print(model)
            fitmodel.set_param_hint('sigma', #starting value gaussian-width
                                value = 50,
                                min = 0,
                                max = 150)
            fitmodel.set_param_hint('amplitude', # starting value amplitude is approxamitaly 11*height (my guess)
                                value = 20,
                                min = 0)
            #parameters calculated based on the fit-parameters
            fitmodel.set_param_hint('height') # function evaluation
            fitmodel.set_param_hint('fwhm') # 2*sigma (see website lmfit)

        if model == 'gaussian':
            print(model)
            fitmodel.set_param_hint('sigma', #starting value gaussian-width
                                value = 1,
                                min = 0,
                                max = 150)
            fitmodel.set_param_hint('amplitude', # starting value amplitude is approxamitaly 11*height (my guess)
                                value = ypeak[i]*11,
                                min = 0)
            #parameters cacluated based on the fit parameters
            fitmodel.set_param_hint('height') #function evaluation
            fitmodel.set_param_hint('fwhm') #=2.3548*sigma (see website lmfit)
    else:
        print('Used ' + model + ' model is not in List')

    return fitmodel
