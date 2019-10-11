modelparameters = {'raw': 'integrated raw data',
                   'amplitude': 'Area of peakfunction',
                   'center': 'Raman shift',
                   'fwhm': 'FWHM',
                   'height': 'Intensity',
                   'intensity': 'Intensity',
                   'sigma': 'Standard deviation',
                   'q': 'Fano parameter'}

modelunits = {'raw': 'cts',
              'amplitude': 'cts',
              'center': 'cm$^{-1}$',
              'fwhm': 'cm$^{-1}$',
              'height': 'cts',
              'intensity': 'cts',
              'sigma': 'cm$^{-1}$',
              'q': 'arb. u.'}

mapoperators = {'div': '/',
                'mult': '*',
                'add': '+',
                'sub': '-'}

cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]
