lorentzian_p1 = {'amplitude': {'name': 'Area of peakfunction A',
                               'unit': 'cts'},
                 'center': {'name': 'Raman shift '
                                    '$\mathrm{\omega}$',
                            'unit': 'cm$^{-1}$'},
                 'fwhm': {'name': 'FWHM',
                          'unit': 'cm$^{-1}$'},
                 'height': {'name': 'Intensity I',
                            'unit': 'cts'},
                 'sigma': {'name': 'Standard deviation $\mathrm{\sigma}$',
                           'unit': 'cm$^{-1}$'}}

lorentzian_p2 = {'amplitude': {'name': 'Area of peakfunction A',
                               'unit': 'cts'},
                 'center': {'name': 'Raman shift '
                                    '$\mathrm{\omega}$',
                            'unit': 'cm$^{-1}$'},
                 'fwhm': {'name': 'FWHM',
                          'unit': 'cm$^{-1}$'},
                 'height': {'name': 'Intensity I',
                            'unit': 'cts'},
                 'sigma': {'name': 'Standard deviation $\mathrm{\sigma}$',
                           'unit': 'cm$^{-1}$'}}

breit_wigner_p1 = {'amplitude': {'name': 'Area of peakfunction A',
                                 'unit': 'cts'},
                   'center': {'name': 'Raman shift '
                                      '$\mathrm{\omega}$',
                              'unit': 'cm$^{-1}$'},
                   'fwhm': {'name': 'FWHM',
                            'unit': 'cm$^{-1}$'},
                   'height': {'name': 'Intensity I',
                              'unit': 'cts'},
                   'intensity': {'name': 'Intensity I',
                                 'unit': 'cts'},
                   'q': {'name': 'Fano parameter q',
                         'unit': 'cts'},
                   'sigma': {'name': 'Standard deviation $\mathrm{\sigma}$',
                             'unit': 'cts'}}

l_p1_div_bw_p1 = {'height': {'name': 'Intensity Ratio',
                             'unit': 'cts/cts'}}
bw_p1_div_l_p1 = {'height': {'name': 'Intensity Ratio',
                             'unit': 'cts/cts'}}
bw_p1_sub_l_p1 = {'center': {'name': 'Distance',
                             'unit': 'cm$^{-1}$'}}

raw = {'raw': {'name': 'Integrated raw data',
               'unit': 'cts'}}

bg_c = {'c': {'name': 'background c',
              'unit': 'cts'},
        'c0': {'name': 'background c$_0$',
               'unit': 'cts'},
        'c1': {'name': 'background c$_1$',
               'unit': 'cts'}}

peaknames = {'lorentzian_p1': lorentzian_p1,
             'lorentzian_p2': lorentzian_p2,
             'breit_wigner_p1': breit_wigner_p1,
             'raw': raw,
             'lorentzian_p1_div_breit_wigner_p1': l_p1_div_bw_p1,
             'breit_wigner_p1_div_lorentzian_p1': bw_p1_div_l_p1,
             'breit_wigner_p1_sub_lorentzian_p1': bw_p1_sub_l_p1,
             'background': bg_c}
