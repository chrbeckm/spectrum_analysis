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


def addPoint(scat, new_point, c='k', white=0.0):
    """Add point to scatter plot."""
    old_off = scat.get_offsets()
    new_off = np.concatenate([old_off, np.array(new_point, ndmin=2)])
    old_c = scat.get_facecolors()
    colormix = get_color(c, white)
    new_c = np.concatenate([old_c, np.array(colormix, ndmin=2)])

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


def createCluster(method, principal_components, n_clust=3, min_samples=5,
                  pointsize=None, xi=0.05, min_cluster_size=None):
    """Create cluster and plot the corresponding scatter plot."""
    if method == 'SpectralClustering':
        clust = SpectralClustering(n_clusters=n_clust)
        clust.fit(principal_components)
        scat = plt.scatter(-100, -100, zorder=2, s=pointsize)
    elif method == 'OPTICS':
        clust = OPTICS(min_samples=min_samples, xi=xi,
                       min_cluster_size=min_cluster_size)
        clust.fit(principal_components)
        scat = plt.scatter(-100, -100, zorder=2, s=pointsize)
    return clust, scat, principal_components


def createRankedClusters(PC, clusterlabels):
    """Create cluster colors from labels."""
    clustertypes = set(clusterlabels)
    PC_ranked = []
    c_and_s = []
    for klass in clustertypes:
        PC_k = PC[clusterlabels == klass]
        PC_ranked.append(PC_k)
        c_and_s.append((klass, len(PC_k)))

    PC_ranked = sorted(PC_ranked, key=len, reverse=True)
    c_and_s = sorted(c_and_s, reverse=True,
                     key=lambda c_and_s: c_and_s[1])

    # find unclustered data points and move them to the end of the lists
    try:
        sorted_cluster = [item[0] for item in c_and_s]
        idx = sorted_cluster.index(-1)
        c_and_s.append(c_and_s.pop(idx))
        PC_ranked.append(PC_ranked.pop(idx))
    except ValueError:
        pass

    shift = 10000
    newlabels = clusterlabels + shift
    for item in c_and_s:
        newlabels[clusterlabels == item[0]] = c_and_s.index(item)

    return PC_ranked, newlabels


def get_color(color, white):
    """Get color from color and white content."""
    ccolor = tuple((1-white) * x for x in to_rgba(color))
    wcolor = tuple(white * x for x in to_rgba('w'))
    colormix = tuple(np.sum([cc, wc]) for cc, wc in zip(ccolor, wcolor))

    return colormix


def get_image(pltdir, label, img_size):
    """Get image."""
    img_name = f'{pltdir}{os.sep}fitplot_{label}.png'
    img = Image.open(img_name)
    img.thumbnail(img_size, Image.ANTIALIAS)  # resize the image
    return img


def get_index(PC, point):
    """Get index in principal component analysis of a specific point."""
    idxlist = []
    for element in PC:
        idxlist.append(np.allclose(element, point))
    idx = idxlist.index(True)
    return idx


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


def fixRanges(cl_x, cl_y):
    """Fix x ranges of the clusters."""
    x_fix = np.zeros_like(cl_x)
    y_fix = np.zeros_like(cl_y)
    for i, spectrum in enumerate(cl_x):
        if np.ptp(cl_x[0]) != np.ptp(cl_x[i]):
            x_fix[i] = cl_x[0]
            y_fix[i] = np.interp(cl_x[0], cl_x[i], cl_y[i])
        else:
            x_fix[i] = cl_x[i]
            y_fix[i] = cl_y[i]
    return x_fix, y_fix


def plotCluster(axes, cl_labels, number, specList, colors, prnt=True):
    """Plot a mean spectrum of a complete cluster from a PCA analysis."""
    spec_mask = [cl_labels == number]
    elements, counts = np.unique(spec_mask, return_counts=True)
    cl_spectra = [name for j, name in enumerate(specList) if spec_mask[0][j]]
    cl_x, cl_y = data.GetAllData(cl_spectra)
    x_fixed, y_fixed = fixRanges(cl_x, cl_y)
    if prnt:
        print(f'Cluster {number}, containing {counts[1]} spectra.')
    if number == list(set(cl_labels))[-1]:
        axes.plot(x_fixed[0], sum(y_fixed)/len(y_fixed), color='grey')
    else:
        axes.plot(x_fixed[0], sum(y_fixed)/len(y_fixed), color=colors[number])

    return spec_mask[0]


def plotHistInCluster(axes, clst, spectra_mask, number, hist_params,
                      param_list, params, nbins, missing):
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
    axs_twin.text(0.05, 0.85, f'C {number}, {len(clst)} S',
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
    print('The biggest clusters are')
    minimum = min(len(ax_arr), len(rank_clust))
    for i in range(0, minimum):
        spec_mask = plotCluster(ax_arr[i], clust_lbl, i,
                                spectra, colors)

        ax_twin = plotHistInCluster(ax_arr[i], rank_clust[i], spec_mask, i,
                                    hist_params, param_list, params, nbins,
                                    missing)

        if i == 1:
            hnd, lbl = ax_twin.get_legend_handles_labels()
            ax_main.legend(hnd, lbl, ncol=len(hist_params),
                           bbox_to_anchor=(0, 1.01),  # legend on top pca plot
                           loc='lower left', prop={'size': 6},
                           borderaxespad=0.)


def plotReachability(ax, cluster, labels, colors, pointsize=None):
    """Plot the reachabilit of an OPTICS clustering."""
    # plot reachability
    space = np.arange(len(cluster.reachability_))
    reachability = cluster.reachability_[cluster.ordering_]
    labels = labels[cluster.ordering_]
    label_set = list(set(labels))

    # Reachability plot
    for klass, color in zip(label_set[:-1], colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax.scatter(Xk, Rk, c=color, s=pointsize)
    ax.scatter(space[labels == label_set[-1]],
               reachability[labels == label_set[-1]],
               c='grey', s=pointsize)
    ax.set_ylabel('Reachability (epsilon distance)')
    ax.set_xlabel('Data points')
