lorentzian_p1 = {'amplitude': {'name': 'Area of peakfunction A$\mathrm{_{l1}}$',
                               'unit': 'cts'},
                 'center': {'name': '$\mathrm{\omega_{l1}}$',
                            'unit': 'cm$^{-1}$'},
                 'fwhm': {'name': 'FWHM$\mathrm{_{l1}}$',
                          'unit': 'cm$^{-1}$'},
                 'height': {'name': 'Intensity I$\mathrm{_{l1}}$',
                            'unit': 'cts'},
                 'sigma': {'name': 'Standard deviation $\mathrm{\sigma_{l1}}$',
                           'unit': 'cm$^{-1}$'}}

lorentzian_p2 = {'amplitude': {'name': 'Area of peakfunction A$\mathrm{_{l2}}$',
                               'unit': 'cts'},
                 'center': {'name': '$\mathrm{\omega_{l2}}$',
                            'unit': 'cm$^{-1}$'},
                 'fwhm': {'name': 'FWHM$\mathrm{_{l2}}$',
                          'unit': 'cm$^{-1}$'},
                 'height': {'name': 'Intensity I$\mathrm{_{l2}}$',
                            'unit': 'cts'},
                 'sigma': {'name': 'Standard deviation $\mathrm{\sigma_{l2}}$',
                           'unit': 'cm$^{-1}$'}}

breit_wigner_p1 = {'amplitude': {'name': 'Area of peakfunction A$\mathrm{_{b1}}$',
                                 'unit': 'cts'},
                   'center': {'name': '$\mathrm{\omega_{b1}}$',
                              'unit': 'cm$^{-1}$'},
                   'fwhm': {'name': 'FWHM$\mathrm{_{b1}}$',
                            'unit': 'cm$^{-1}$'},
                   'height': {'name': 'Intensity I$\mathrm{_{b1}}$',
                              'unit': 'cts'},
                   'intensity': {'name': 'Intensity I$\mathrm{_{b1}}$',
                                 'unit': 'cts'},
                   'q': {'name': 'Fano parameter q$\mathrm{_{b1}}$',
                         'unit': 'cts'},
                   'sigma': {'name': 'Standard deviation $\mathrm{\sigma_{b1}}$',
                             'unit': 'cts'}}

l_p1_div_bw_p1 = {'height': {'name': 'Intensity Ratio',
                             'unit': 'cts/cts'}}
bw_p1_div_l_p1 = {'fwhm': {'name': 'FWHM Ratio',
                           'unit': 'cm$^{-1}$/cm$^{-1}$'},
                  'height': {'name': 'Intensity Ratio',
                             'unit': 'cts/cts'}}
bw_p1_sub_l_p1 = {'center': {'name': 'Distance',
                             'unit': 'cm$^{-1}$'}}

raw = {'raw': {'name': 'Integrated raw data',
               'unit': 'cts'}}

pca = {'pca': {'name': 'clustered data',
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
             'pca': pca,
             'lorentzian_p1_div_breit_wigner_p1': l_p1_div_bw_p1,
             'breit_wigner_p1_div_lorentzian_p1': bw_p1_div_l_p1,
             'breit_wigner_p1_sub_lorentzian_p1': bw_p1_sub_l_p1,
             'background': bg_c}
