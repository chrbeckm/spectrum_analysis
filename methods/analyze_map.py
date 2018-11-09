from mapping import *

# xdim:     the number of Spectra in x direction
# ydim:     the number of Spectra in y direction
# stepsize: the interval at which the mapping was collected in µm
map = mapping('smallmap', 2, 2, 10)
# plot mapping
# xmin:     the lowest wavenumber to be used in the mapping
# xmax:     the highest wavenumber to be used in the mapping
map.PlotMapping(1550, 1650)
