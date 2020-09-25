"""Methods ofr PCA analysis of mappings."""
import os

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, CSS4_COLORS
from matplotlib.ticker import MaxNLocator

from sklearn import preprocessing
from sklearn.cluster import OPTICS, SpectralClustering

from spectrum_analysis import data
from peaknames import peaknames


def addPoint(scat, new_point, c='k'):
    """Add point to scatter plot."""
    old_off = scat.get_offsets()
    new_off = np.concatenate([old_off, np.array(new_point, ndmin=2)])
    old_c = scat.get_facecolors()
    new_c = np.concatenate([old_c, np.array(to_rgba(c), ndmin=2)])

    scat.set_offsets(new_off)
    scat.set_facecolors(new_c)

    scat.axes.autoscale_view()
    scat.axes.figure.canvas.draw_idle()


def scaleParameters(params):
    """Scale parameters to [0, 1]."""
    scaled_params = np.zeros_like(params)
    for idx, param in enumerate(params):
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_param = min_max_scaler.fit_transform(param.reshape(-1, 1))
        scaled_params[idx] = scaled_param.reshape(1, -1)
    return scaled_params


def createCluster(method, principal_components, n_clust=3, min_samples=5):
    """Create cluster and plot the corresponding scatter plot."""
    if method == 'SpectralClustering':
        clust = SpectralClustering(n_clusters=n_clust)
        clust.fit(principal_components)
        scat = plt.scatter(-100, -100, zorder=2)
    elif method == 'OPTICS':
        clust = OPTICS(min_samples=min_samples)
        clust.fit(principal_components)
        scat = plt.scatter(principal_components[clust.labels_ == -1, 0],
                           principal_components[clust.labels_ == -1, 1], c='k')
    return clust, scat, principal_components


def get_image(pltdir, label, img_size):
    """Get image."""
    img_name = f'{pltdir}{os.sep}fitplot_{label}.png'
    img = Image.open(img_name)
    img.thumbnail(img_size, Image.ANTIALIAS)  # resize the image
    return img


def printPCAresults(pc_ana, param_list, print_components=False):
    """Print results of PCA analysis to command line."""
    print(f'explained variance ratio '
          f'({pc_ana.components_.shape[0]} components): '
          f'{sum(pc_ana.explained_variance_ratio_):2.2f} '
          f'({pc_ana.explained_variance_ratio_.round(2)})')
    if print_components:
        for j, principal_component in enumerate(pc_ana.components_):
            print(f'Principal component {j+1}')
            for idx, lbl in enumerate(param_list):
                print(f'{principal_component[idx]: 2.4f} * {lbl}')
            print()


def plotCluster(axes, cl_labels, cl_tupel, specList, colors, prnt=True):
    """Plot a mean spectrum of a complete cluster from a PCA analysis."""
    spec_mask = [cl_labels == cl_tupel[1]]
    cl_spectra = [name for j, name in enumerate(specList) if spec_mask[0][j]]
    cl_x, cl_y = data.GetAllData(cl_spectra)
    if prnt:
        print(f'Cluster {cl_tupel[1]}, containing {cl_tupel[0]} spectra.')
    axes.plot(cl_x[0], sum(cl_y)/len(cl_y), color=colors[cl_tupel[1]])

    return spec_mask[0]


def plotHistInCluster(axes, clst, spectra_mask, hist_params, param_list,
                      params, nbins, missing):
    """Plot histogrammed fwhm and position of each cluster into plot."""
    axs_twin = axes.twinx()
    for param in hist_params:
        param_idx = param_list.index(param)
        color = CSS4_COLORS[list(CSS4_COLORS)[param_idx+20]]
        hist_params = params[param_idx][spectra_mask]
        peakname = '_'.join(param.split('_')[:-1])
        parametername = param.split("_")[-1]
        label = peaknames[peakname][parametername]['name'].split(' ')[-1]
        axs_twin.hist(hist_params[hist_params != missing], label=label,
                      bins=nbins, histtype='step', color=color)
        axs_twin.yaxis.tick_left()
        axs_twin.tick_params(axis='y', labelsize=7)

    axes.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=3))
    axs_twin.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=3,
                                                 integer=True))
    axs_twin.set_ylabel('Spectra')
    axs_twin.yaxis.set_label_position('left')
    axs_twin.text(0.05, 0.85, f'C{clst[1]}, {clst[0]} S',
                  transform=axes.transAxes, fontsize=8,
                  bbox=dict(color='white', alpha=0.75, pad=3))
    __, y_max_val = axs_twin.get_ylim()
    if y_max_val < 1:
        axs_twin.yaxis.set_ticks([])
        axs_twin.set_ylabel('')

    return axs_twin


def selectSpecType(mappng, plt_clust=False):
    """Select .dat-type or .txt-type spectra."""
    if plt_clust:
        spectra, __ = data.GetFolderContent(mappng.fitdir, 'dat', quiet=True)
    else:
        spectra, __ = data.GetFolderContent(mappng.folder, 'txt', quiet=True)
    return spectra


def plotClusterOverview(spectra, ax_main, ax_arr, rank_clust, clust_lbl,
                        colors, hist_params, param_list, params, nbins,
                        missing):
    """Plot overview of current cluster analysis."""
    print('The clusters are')
    minimum = min(len(ax_arr), len(rank_clust))
    for i in range(0, minimum):
        spec_mask = plotCluster(ax_arr[i], clust_lbl, rank_clust[i],
                                spectra, colors)
        ax_twin = plotHistInCluster(ax_arr[i], rank_clust[i], spec_mask,
                                    hist_params, param_list, params, nbins,
                                    missing)
        if i == 1:
            hnd, lbl = ax_twin.get_legend_handles_labels()
            ax_main.legend(hnd, lbl, ncol=len(hist_params),
                           bbox_to_anchor=(0, 1.01),  # legend on top pca plot
                           loc='lower left', prop={'size': 6},
                           borderaxespad=0.)
