import os
import shutil

from spectrum_analysis import mapping as mp
from spectrum_analysis import data

folder='testdata'

map = mp.mapping(foldername=folder, plot=True)

# get x- and y-data
x, y = data.GetAllData(map.spectra)

map.PlotMapping(x, y, 3, 3, 10, 1300, 1400)
