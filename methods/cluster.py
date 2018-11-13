import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from functions_mapping import *
from functions_denoising import *
from spectrum import *

from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering

from lmfit.models import PolynomialModel

# define folder containting data from a mapping
folder = 'smallmap'
sample_name = 'sample'

# define number of kmeans cluster and
# if you want to see the kmeans labeled PCA data
cluster = 3            # need to add more colors if more than 10 clusters
plotLabeledPCA = True

# possible peaks (breit_wigner == fano)
# implemented are: breit_wigner, lorentzian, gaussian, voigt
peaks = ['breit_wigner', 'lorentzian']

#############################################################
# fine tuning (most likely nothing has to be adjusted)
#############################################################

# colors for clustering (add more if you use more than 10 clusters)
colors = ['red', 'blue', 'green', 'orange', 'black', 'purple',
          'lightgreen', 'turquoise', 'lightblue', 'yellow']

# create list of all files and read data from mapping
spec = spectrum(folder)

# select region for baseline and save points to file
spec.SelectSpectrum()
spec.SelectBaseline()

# Denoise the whole mapping
spec.WaveletSmoothAllSpectra()

# fit and remove baseline from each denoised spectrum
spec.FitAllBaselines()
yden_bgfree = spec.y

# reduce data to two main principal components
pca = PCA(n_components=2)
#yden_bgfree_pca = pca.fit(yden).transform(yden) # -> seems to generate better results
yden_bgfree_pca = pca.fit(yden_bgfree).transform(yden_bgfree)
print('explained variance ratio (first two components): %s'
     % str(pca.explained_variance_ratio_))

# cluster the data with kmeans
kmeans = KMeans(init='k-means++', n_clusters=cluster)
kmeans.fit(yden_bgfree_pca)
SpectraPerCluster = np.bincount(kmeans.labels_)

# plot pca reduced data with kmeans labels
cluster_sum = PlotClusteredPCA(spec.x, spec.ydenoised, folder, yden_bgfree_pca,
                               kmeans, 'KMeans',
                               cluster, colors)
plt.show()
'''
# iterate over all clusters and generate cluster specific fit models
for clust in range(0, cluster):
    if True:
        print('In Cluster ' + str(clust + 1) + ' are ' +
              str(SpectraPerCluster[clust]) + ' Spectra.')
    # get a baseline for the current cluster
    baselinefile = SelectBaseline2(x[0], cluster_sum[clust], folder,
                                   label='cluster' + str(clust),
                                   color=colors[clust])
    # get the indices of the current cluster
    indices = [i for i, x in enumerate(kmeans.labels_) if x == clust]

    # select the peaks for the current cluster and fit them each by itself
    for index in indices:
        print(index)
        baseline_fit = FitBaseline(x[index], y[index], baselinefile)
        SelectPeaks(x[index], y[index], baseline_fit,
                    folder + '/temp', str(index).zfill(4), peaks)
        fitresult = FitSpectrum(x[index], y[index], ymax[index],
                                baseline_fit, folder + '/temp', str(index).zfill(4), peaks)
        SaveFitParams(x[index], y[index], ymax[index], fitresult, baseline_fit,
                      str(index).zfill(4), peaks)
'''
