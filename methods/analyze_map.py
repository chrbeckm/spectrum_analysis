from mapping import *

# xdim:     the number of Spectra in x direction
# ydim:     the number of Spectra in y direction
# stepsize: the interval at which the mapping was collected in µm
map = mapping('smallmap', 2, 2, 10, raw=False)

# plot mapping in different ways
# use PlotMapping(xmin, xmax) to integrate over fitted or raw data
# xmin:     the lowest wavenumber to be used in the mapping
# xmax:     the highest wavenumber to be used in the mapping

# use PlotMapping(maptype='peak_fit_value_file') to map the fitted parameter
#map.PlotMapping(maptype='lorentzian_p1_sigma')
# or use PlotAllMappings for all fit parameters to be mapped
map.PlotAllMappings()
