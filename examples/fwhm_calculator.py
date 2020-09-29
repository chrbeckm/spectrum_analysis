"""
Calculate FWHM of BWF function.

Calculations are taken from:
http://openafox.com/science/peak-function-derivations.html#breit-wigner-fano
Errors are calculated by simplified error propagation.
"""
import os

from spectrum_analysis import data
from spectrum_analysis import spectrum as sp
import numpy as np

folders = [os.path.join('testdata', '1'),
           os.path.join('testdata', '2'),
           os.path.join('testdata', '3')]

dirpath = os.path.join('results', 'fitparameter', 'peakwise')

for folder in folders:
    spec = sp.spectrum(os.path.join(folder, '0001'))

    sigma, sigma_err = data.GetData(
        os.path.join(folder,
                     dirpath,
                     'breit_wigner_p1_sigma.dat'))
    q, q_err = data.GetData(
        os.path.join(folder,
                     dirpath,
                     'breit_wigner_p1_q.dat'))

    fwhm = np.sqrt((q**2 + 2) * q**2 * sigma**2) / (q**2 - 2)
    ds_fwhm = np.sqrt(q**2 * (q**2+2))/(q**2 - 2)
    dq_fwhm = ((2*sigma * (1+q**2) * (2-q**2)**2
                - 2*q**2 * sigma * (2+q**2))
               / ((2-q**2)**3 * np.sqrt(q**2+2)))
    fwhm_err = np.sqrt((ds_fwhm * sigma_err)**2 + (dq_fwhm * q_err)**2)

    fwhm[q == spec.missingvalue] = spec.missingvalue
    fwhm[sigma == spec.missingvalue] = spec.missingvalue

    fwhm_err[q == spec.missingvalue] = spec.missingvalue
    fwhm_err[sigma == spec.missingvalue] = spec.missingvalue

    with open(
        os.path.join(folder,
                     dirpath,
                     'breit_wigner_p1_fwhm.dat'), 'w') as f:
        for i, element in enumerate(fwhm):
            # write to file
            f.write(f'{np.abs(element):>13.5f}\t{fwhm_err[i]:>11.5f}\n')
