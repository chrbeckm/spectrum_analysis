"""Starting parameters for various functions."""
import re


def StartingParameters(fitmodel, peaks, xpeak=[0], ypeak=[0], i=0):
    """Define starting parameters for different functions.

    The initial values of the fit depend on the maxima of the peaks but
    also on their line shapes. They have to be chosen carefully.
    In addition the borders of the fit parameters are set in this function.
    Supplementary to the fit parameters and parameters calculated from
    them provided by
    `lmfit <https://lmfit.github.io/lmfit-py/builtin_models.html#>`_
    the FWHM of the voigt-profile as well as the height and intensity
    of the breit-wigner-fano-profile are given.

    Parameters
    ----------
    fitmodel : class
        Model chosen in :func:`~starting_params.ChoosePeakType`.
    peaks : list, default: ['breit_wigner', 'lorentzian']
        Possible line shapes of the peaks to fit are
        'breit_wigner', 'lorentzian', 'gaussian', and 'voigt'.
    xpeak array (float), default = 0
        Position of the peak's maxima (x-value).
    ypeak array (float), default = 0
        Height of the peak's maxima (y-value).
    i : int
        Integer between 0 and (N-1) to distinguish between N peaks of
        the same peaktype. It is used in the prefix.

    Returns
    -------
    fitmodel : class
        Model chosen in :func:`~starting_params.ChoosePeakType`
        including initial values for the fit (set_param_hint).
    """
    # starting position for the peak position is not allowed to vary much

    fitmodel.set_param_hint('center',
                            value=xpeak[i],
                            min=xpeak[i] - 50,
                            max=xpeak[i] + 50)
    # get model name
    # search all letters between ( and ,
    model = re.findall('\((.*?),', fitmodel.name)
    model = model[0]
    # search if model is in peak list
    if any(model in peak for peak in peaks):
        if model == 'voigt':
            fitmodel.set_param_hint('sigma',  # starting value gauß-width
                                    value=10,
                                    min=0,
                                    max=100)
            fitmodel.set_param_hint('gamma',  # starting value lorentzian-width
                                    value=5,   # (== gauß-width by default)
                                    min=0,
                                    max=100,
                                    vary=True, expr='')  # vary gamma indep.
            fitmodel.set_param_hint('amplitude',  # starting value amplitude is
                                    value=ypeak[i]*20,  # approx. 11*height
                                    min=0)             # (guess)
            # parameters calculated based on the fit-parameters
            fitmodel.set_param_hint('height',
                                    value=ypeak[i])
            fitmodel.set_param_hint('fwhm_g',
                                    expr=(f'2 * {fitmodel.prefix}sigma'
                                          '* sqrt(2 * log(2))'))
            fitmodel.set_param_hint('fwhm_l',
                                    expr=f'2 * {fitmodel.prefix}gamma')
            # precise FWHM approximation by Olivero and Longbothum
            # (doi:10.1016/0022-4073(77)90161-3)
            # it is not possible to take the fwhm form lmfit for an
            # independently varying gamma
            fitmodel.set_param_hint('fwhm',
                                    expr=(f'0.5346 * {fitmodel.prefix}fwhm_l'
                                          '+ sqrt(0.2166'
                                          f'* {fitmodel.prefix}fwhm_l**2'
                                          f'+ {fitmodel.prefix}fwhm_g**2 )'))

        if model == 'breit_wigner':  # should be BreitWignerModel!
            # fit-parameter
            fitmodel.set_param_hint('sigma',  # starting value width
                                    value=100,
                                    min=0,
                                    max=200)
            fitmodel.set_param_hint('q',  # starting value q
                                    value=-5,
                                    min=-100,
                                    max=100)
            fitmodel.set_param_hint('amplitude',  # starting value amplitude is
                                                  # approxamitaly 11*height
                                                  # (guess)
                                    value=ypeak[i]/50,
                                    min=0)
            fitmodel.set_param_hint('height',  # max calculated to A(q^2+1)
                                    expr=(f'{fitmodel.prefix}amplitude'
                                          f'* (({fitmodel.prefix}q )**2+1)'))
            fitmodel.set_param_hint('intensity',  # intensity is A*q^2
                                                  # (compared to the used
                                                  # expression in the paper)
                                    expr=(f'{fitmodel.prefix}amplitude'
                                          f'* ({fitmodel.prefix}q )**2'))

        if model == 'lorentzian':
            fitmodel.set_param_hint('sigma',  # starting value gaussian-width
                                    value=50,
                                    min=0,
                                    max=150)
            fitmodel.set_param_hint('amplitude',  # starting value amplitude is
                                    value=20,       # approxamitaly 11*height
                                    min=0)          # (guess)
            # parameters calculated based on the fit-parameters
            fitmodel.set_param_hint('height')  # function evaluation
            fitmodel.set_param_hint('fwhm')  # 2*sigma (see website lmfit)

        if model == 'gaussian':
            fitmodel.set_param_hint('sigma',  # starting value gaussian-width
                                    value=1,
                                    min=0,
                                    max=150)
            fitmodel.set_param_hint('amplitude',  # starting value amplitude is
                                    value=ypeak[i]*11,  # approx. 11*height
                                    min=0)             # (guess)
            # parameters cacluated based on the fit parameters
            fitmodel.set_param_hint('height')  # function evaluation
            fitmodel.set_param_hint('fwhm')  # =2.3548*sigma (see lmfit doc)
    else:
        print('Used ' + model + ' model is not in List')

    return fitmodel
