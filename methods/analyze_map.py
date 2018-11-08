from mapping import *

map = mapping('smallmap')
# plot mapping
# input values are
# xdim:     the number of Spectra in x direction
# ydim:     the number of Spectra in y direction
# stepsize: the interval at which the mapping was collected in µm
# xmin:     the lowest wavenumber to be used in the mapping
# xmax:     the highest wavenumber to be used in the mapping
map.PlotMapping(2, 2, 10, 1550, 1620)
