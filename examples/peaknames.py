lorentzian_p1 = {'amplitude': {'name': 'Area of peakfunction',
                               'unit': 'cts'},
                 'center': {'name': 'Raman shift',
                            'unit': 'cm$^{-1}$'},
                 'fwhm': {'name': 'FWHM',
                          'unit': 'cm$^{-1}$'},
                 'height': {'name': 'Intensity',
                            'unit': 'cts'},
                 'sigma': {'name': 'Standard deviation',
                           'unit': 'cm$^{-1}$'}}

lorentzian_p2 = {'amplitude': {'name': 'Area of peakfunction',
                               'unit': 'cts'},
                 'center': {'name': 'Raman shift',
                            'unit': 'cm$^{-1}$'},
                 'fwhm': {'name': 'FWHM',
                          'unit': 'cm$^{-1}$'},
                 'height': {'name': 'Intensity',
                            'unit': 'cts'},
                 'sigma': {'name': 'Standard deviation',
                           'unit': 'cm$^{-1}$'}}

breit_wigner_p1 = {'amplitude': {'name': 'Area of peakfunction',
                                 'unit': 'cts'},
                   'center': {'name': 'Raman shift',
                              'unit': 'cm$^{-1}$'},
                   'fwhm': {'name': 'FWHM',
                            'unit': 'cm$^{-1}$'},
                   'height': {'name': 'Intensity',
                              'unit': 'cts'},
                   'intensity': {'name': 'Intensity',
                                 'unit': 'cts'},
                   'q': {'name': 'Fano parameter',
                         'unit': 'cts'},
                   'sigma': {'name': 'Standard deviation',
                             'unit': 'cts'}}

l_p1_div_bw_p1 = {'height': {'name': 'Intensity Ratio',
                             'unit': 'cts/cts'}}
bw_p1_div_l_p1 = {'height': {'name': 'Intensity Ratio',
                             'unit': 'cts/cts'}}

raw = {'raw': {'name': 'Integrated raw data',
               'unit': 'cts'}}

peaknames = {'lorentzian_p1': lorentzian_p1,
             'lorentzian_p2': lorentzian_p2,
             'breit_wigner_p1': breit_wigner_p1,
             'raw': raw,
             'lorentzian_p1_div_breit_wigner_p1': l_p1_div_bw_p1,
             'breit_wigner_p1_div_lorentzian_p1': bw_p1_div_l_p1}
