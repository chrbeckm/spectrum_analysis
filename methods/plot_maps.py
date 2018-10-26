import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from functions_mapping import *
from denoise import *

from mpl_toolkits.axes_grid1 import make_axes_locatable
'''
folder = '02018-10-18-Stage-IV-mapping/'
xdim = 15
ydim = 15
name = 'dry'
'''
folder = '02018-10-12-Stage-III-mapping/'
xdim = 25
ydim = 10
name = 'oil'

save = True

# define parameters of the mapping
xticker = 2
stepsize = 10
integrationtime = 60
file = 'results_1330cm'
title = ''

colormap = 'RdYlGn'

# create list of all files and read data from mapping
listOfFiles = get_folder_content(folder, 'txt')
x, y = get_mono_data(listOfFiles)
ymax = np.max(y, axis=1)    # get maximum of each spectrum
ynormed = y/ymax[:,None]    # norm each spectrum to its maximum
ymuon = np.empty_like(y)    # array to save muon removed spectra
yrec = np.empty_like(y)     # array to save denoised spectra
ymuon_ren = np.empty_like(y)# array to save muon removed spectra renormalized
yrec_ren = np.empty_like(y) # array to save denoised spectra renormalized
ymuon_sized = np.empty_like(y) # array to save muon remoced spectra resized
yrec_sized = np.empty_like(y) # array to save denoised spectra resized

# generate all denoised spectra
iterator = 0
for spectrum in ynormed:
    print('spectrum ' + str(iterator + 1).zfill(4))
    # denoise spectrum
    muon, rec = DenoiseSpectrum(x[0], spectrum, thresh_mod=2)
    ymuon[iterator] = muon
    yrec[iterator] = rec
    ymuon_ren[iterator] = renormalize(muon, muon)
    yrec_ren[iterator] = renormalize(rec, muon)
    yrec_sized[iterator] = yrec[iterator] * ymax[iterator]
    ymuon_sized[iterator] = ymuon[iterator] * ymax[iterator]
    print('Maximum: %.2f at index: %4d and position: %4d' %
         (max(yrec_sized[iterator]),
          np.argmax(yrec_sized[iterator]),
          x[iterator][np.argmax(yrec_sized[iterator])]))

    # save the muons removed spectra
    np.savetxt(folder + 'muons-removed/' + str(iterator + 1).zfill(4) + '.dat',
               np.transpose([x[iterator], ymuon_sized[iterator]]),
               fmt='%3.3f')

    # save the denoised spectra
    np.savetxt(folder + 'denoised/' + str(iterator + 1).zfill(4) + '.dat',
               np.transpose([x[iterator], yrec_sized[iterator]]),
               fmt='%3.3f')

    # plot muonfree and reconstructed spectra
    #WaveletPlot(x[0], ymuon_sized[iterator],
    #                  yrec_sized[iterator],
    #                  title=('Spectrum ' + str(iterator + 1)))
    iterator += 1

WaveletPlot(x[0], ymuon_sized.sum(axis=0), yrec_sized.sum(axis=0),
            save=save, name=name + '_sum',
            title='Sum of ' + folder + ' (' + name + ' sample)')

# create x and y ticks accordingly to the parameters of the mapping
x_ticks = np.arange(stepsize, stepsize * (xdim + 1), step=xticker*stepsize)
y_ticks = np.arange(stepsize, stepsize * (ydim + 1), step=stepsize)
y_ticks = y_ticks[::-1]

# sum up each spectrum and create matrix
ysum = yrec_sized[:, 950:1050].sum(axis=1)
ysum_matrix = np.reshape(ysum, (ydim, xdim))
ysum_matrix = np.flipud(ysum_matrix)

# create figure
fig = plt.figure(figsize=(18,6))
plt.suptitle(title)

# set font and parameters
matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
matplotlib.rcParams.update({'font.size': 22})
tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)

# create the different subplots and modify them
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(ysum_matrix,
           cmap=colormap)
plt.xticks(np.arange(xdim, step=xticker), x_ticks)
plt.yticks(np.arange(ydim), y_ticks)
plt.ylabel('y-Position ($\mathrm{\mu}$m)')
plt.xlabel('x-Position ($\mathrm{\mu}$m)')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
clb = plt.colorbar(cax=cax)
clb.set_label('Integrated Intensity\n(arb. u.)')
clb.locator = tick_locator
clb.update_ticks()

# have a tight layout
plt.tight_layout()

# save everything and show the plot
#plt.savefig(file + '_short.png', format='png')
#plt.savefig(file + '_short.pdf', format='pdf')
#plt.savefig(file + '_short.svg', format='svg')

plt.show()
