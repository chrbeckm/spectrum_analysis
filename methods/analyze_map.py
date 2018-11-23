from mapping import *

# colors for clustering (add more if you use more than 10 clusters)
colorlist = ['red', 'blue', 'green', 'orange', 'black', 'purple',
          'lightgreen', 'turquoise', 'lightblue', 'yellow']

# xdim:     the number of Spectra in x direction
# ydim:     the number of Spectra in y direction
# stepsize: the interval at which the mapping was collected in µm
map = mapping('smallmap', 2, 4, 10)

# plot mapping in different ways
# use PlotMapping(xmin, xmax) to integrate over fitted or raw data
# xmin:     the lowest wavenumber to be used in the mapping
# xmax:     the highest wavenumber to be used in the mapping
#map.PlotMapping(1550,1650)

# use PlotMapping(maptype='peak_fit_value_file') to map the fitted parameter
#map.PlotMapping(maptype='lorentzian_p1_sigma')

# or use PlotAllMappings for all fit parameters to be mapped
map.PlotAllMappings()

# use PlotMapping(top='file1', bot='file2') to plot a mapping of
# top/bot
map.PlotMapping(top='lorentzian_p1_height.dat', bot='breit_wigner_p1_height.dat')

# Use cluster algorithms to identify something in the mapping
map.PlotClusteredPCAMapping(colorlist=colorlist)
