import os
import shutil

from spectrum_analysis import mapping as mp

folder='testdata'

if os.path.exists(folder + '/results'):
    shutil.rmtree(folder + '/results')
    shutil.rmtree(folder + '/temp')

map = mp.mapping(foldername=folder)

# get x- and y-data
x, y = map.GetAllData()

map.PlotMapping(x, y, 3, 3, 10, 1300, 1400)
