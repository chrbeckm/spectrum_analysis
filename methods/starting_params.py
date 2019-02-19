import re
import lmfit
from warnings import warn

class ParameterWarning(UserWarning):
    pass


def StartingParameters(fitmodel, peaks, xpeak=[0], ypeak=[0], i=0):
    """
    The initial values of the fit depend on the maxima of the peaks but also on their line shapes.
    They have to be chosen carefully.
    In addition the borders of the fit parameters are set in this function.
    Supplementary to the fit parameters and parameters calculated from them provided by
    `lmfit <https://lmfit.github.io/lmfit-py/builtin_models.html#>`_
    the FWHM of the voigt-profile as well as the height and intensity of the breit-wigner-fano-profile are given.


    Parameters
    ----------
    fitmodel : CLASS from lmfit ?
    peaks : list, default: ['breit_wigner', 'lorentzian']
        Possible line shapes of the peaks to fit are
        'breit_wigner', 'lorentzian', 'gaussian', and 'voigt'.
    xpeak array (float), default = 0
        Position of the peaks maxima (x-value).
    ypeak array (float), default = 0
        Position of the peaks maxima (y-value).
    i : int
        Integer between 0 and (N-1) to distinguish between N peaks of the same peaktype. It is used in the prefix.

    """
    # starting position for the peak position is not allowed to vary much
    fitmodel.set_param_hint('center',
                             value = xpeak[i],
                             min = xpeak[i] - 50,
                             max = xpeak[i] + 50)
    # get model name
    # search all letters between ( and ,
    model = re.findall('\((.*?),', fitmodel.name)
    model = model[0]
    # search if model is in peak list
    if any(model in peak for peak in peaks):
        if model == 'voigt':
            fitmodel.set_param_hint('sigma', #starting value gauß-width
                                value = 100,
                                min = 0,
                                max = 300)
            fitmodel.set_param_hint('gamma', #starting value lorentzian-width (== gauß-width by default)
                                value = 100,
                                min = 0,
                                max = 300,
                                vary = True, expr = '') #vary gamma independently
            fitmodel.set_param_hint('amplitude', # starting value amplitude ist approxamitaly 11*height (my guess)
                                value = ypeak[i]*11,
                                min = 0)
            #parameters calculated based on the fit-parameters
            fitmodel.set_param_hint('height',
                                value = ypeak[i])
            fitmodel.set_param_hint('fwhm_g',
                                expr = '2 *' + fitmodel.prefix + 'sigma * sqrt(2 * log(2))')
            fitmodel.set_param_hint('fwhm_l',
                                expr = '2 *' + fitmodel.prefix + 'gamma')
            # precise FWHM approximation by Olivero and Longbothum (doi:10.1016/0022-4073(77)90161-3)
            # it is not possible to take the fwhm form lmfit for an independently varying gamma
            fitmodel.set_param_hint('fwhm',
                                expr = '0.5346 *' + fitmodel.prefix +
                                       'fwhm_l + sqrt(0.2166 *' + fitmodel.prefix +
                                       'fwhm_l**2 +' + fitmodel.prefix +
                                       'fwhm_g**2 )')

        if model == 'breit_wigner': # should be BreitWignerModel!
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
            fitmodel.set_param_hint('height', # maximum calculated to be at A(q^2+1)
                                expr = fitmodel.prefix +'amplitude * ((' +fitmodel.prefix + 'q )**2+1)' )
            fitmodel.set_param_hint('intensity', # intensity is A*q^2 (compared to the used expression in the paper)
                                expr = fitmodel.prefix +'amplitude * (' +fitmodel.prefix + 'q )**2' )


        if model == 'lorentzian':
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

def ChoosePeakType(peaktype, i):
    """
    This function helps to create the `CompositeModel() <https://lmfit.github.io/lmfit-py/model.html#lmfit.model.CompositeModel>`_ .
    Implemented models are:
    `GaussianModel() <https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.GaussianModel>`_  ,
    `LorentzianModel() <https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.LorentzianModel>`_  ,
    `VoigtModel() <https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.VoigtModel>`_  ,
    `BreitWignerModel() <https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.BreitWignerModel>`_  .

    Parameters
    ----------
    peaktype : string
        Possible line shapes of the peaks to fit are
        'breit_wigner', 'lorentzian', 'gaussian', and 'voigt'.
    i : int
        Integer between 0 and (N-1) to distinguish between N peaks of the same peaktype. It is used in the prefix.

    """
    prefix = peaktype + '_p'+ str(i + 1) + '_'
    if peaktype == 'voigt':
        return lmfit.models.VoigtModel(prefix = prefix, nan_policy = 'omit')
    elif peaktype == 'breit_wigner':
        return lmfit.models.BreitWignerModel(prefix = prefix, nan_policy = 'omit')
    elif peaktype == 'lorentzian':
        return lmfit.models.LorentzianModel(prefix = prefix, nan_policy = 'omit')
    elif peaktype == 'gaussian':
        return lmfit.models.GaussianModel(prefix = prefix, nan_policy = 'omit')
