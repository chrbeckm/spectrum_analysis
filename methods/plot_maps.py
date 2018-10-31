import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from functions_mapping import *
from functions_denoising import *
from functions_fitting import *

from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering

from lmfit.models import PolynomialModel

# define folder containting data from a mapping
folder = '02018-10-12-Stage-III-mapping/'
sample_name = '150C 10N 300000u'

# define parameters of the mapping
xdim = 25           # number of steps in x
ydim = 10           # number of steps in y
stepsize = 10       # size of each step
rmin = 950          # minimal index for integration
rmax = 1050         # maximal index for integration

# define number of kmeans cluster and
# if you want to see the kmeans labeled PCA data
cluster = 3            # need to add more colors if more than 10 clusters
plotLabeledPCA = True

# define if you want to display the denoised sum of all spectra and save it
# with sample_name in folder
display_sumwavelet = False
save_sumwavelet = False

# save the mapping
save_mapping = False

#############################################################
# fine tuning (most likely nothing has to be adjusted)
#############################################################

# define special parameters of the mapping
xticker = 2             # only every 2nd xtick is plotted
colormap = 'RdYlGn'     # which colormap do you want?

# colors for clustering (add more if you use more than 10 clusters)
colors = ['red', 'blue', 'green', 'orange', 'black', 'purple',
          'lightgreen', 'turquoise', 'lightblue', 'yellow']

# create list of all files and read data from mapping
listOfFiles = GetFolderContent(folder, 'txt')
x, y = GetMonoData(listOfFiles)

# prepare y for fitting
ymax = np.max(y, axis=1)    # get maximum of each spectrum
ynormed = y/ymax[:,None]    # norm each spectrum to its maximum

# create arrays to save data to
ymuon = np.empty_like(y) # save muon removed spectra rescaled
yden = np.empty_like(y) # save denoised spectra rescaled
yden_bgfree = np.empty_like(y) # save background free spectra

# Denoise the whole mapping
ymuon, yden = DenoiseMapping(x, ynormed, ymax, folder)

if display_sumwavelet:
    WaveletPlot(x[0], ymuon.sum(axis=0), yden.sum(axis=0),
            save=save_sumwavelet, name=folder + sample_name + '_sum',
            title='Sum of ' + folder + ' (' + sample_name + ' sample)')

# select region for baseline and save points to file
baselinefile = SelectBaseline2(x[0], yden.sum(axis=0), folder)

# fit and remove baseline from each denoised spectrum
yden_bgfree = RemoveBaselineMapping(x, yden, baselinefile)

# reduce data to two main principal components
pca = PCA(n_components=2)
#yden_bgfree_pca = pca.fit(yden).transform(yden) # -> seems to generate better results
yden_bgfree_pca = pca.fit(yden_bgfree).transform(yden_bgfree)
print('explained variance ratio (first two components): %s'
     % str(pca.explained_variance_ratio_))

# cluster the data with kmeans
kmeans = KMeans(init='k-means++', n_clusters=cluster)
kmeans.fit(yden_bgfree_pca)

# cluster with spectral clustering
#spectral = SpectralClustering(n_clusters=cluster)
#spectral.fit(yden_bgfree_pca)

# plot pca reduced data with kmeans labels
if plotLabeledPCA:
    PlotClusteredPCA(x, yden, folder, yden_bgfree_pca, kmeans, 'KMeans', cluster, colors)
#    PlotClusteredPCA(x, yden, folder, yden_bgfree_pca, spectral, 'spectral', cluster, colors)

# create x and y ticks accordingly to the parameters of the mapping
x_ticks = np.arange(stepsize, stepsize * (xdim + 1), step=xticker*stepsize)
y_ticks = np.arange(stepsize, stepsize * (ydim + 1), step=stepsize)
y_ticks = y_ticks[::-1]

# sum up each spectrum and create matrix
ysum = yden_bgfree[:, rmin:rmax].sum(axis=1)
ysum_matrix = np.reshape(ysum, (ydim, xdim))
ysum_matrix = np.flipud(ysum_matrix)

# create figure
fig = plt.figure(figsize=(18,6))
plt.suptitle('Mapping of ' + sample_name)

# set font and parameters
matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
matplotlib.rcParams.update({'font.size': 22})
tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)

# create the different subplots and modify them
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(ysum_matrix, cmap=colormap)
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
if save_mapping:
    plt.savefig(folder + 'mapping_bgfree.pdf', format='pdf')

plt.show()
