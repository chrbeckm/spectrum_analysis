"""PCA analysis of mapping parameters.

Perform a PCA analysis of a fitted mapping.
By hovering over the data points, the corresponding spectra
show up.
"""
import gc
import os
import shutil

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import to_rgba, CSS4_COLORS, ListedColormap
from matplotlib import rcdefaults

from sklearn.decomposition import PCA

from spectrum_analysis import mapping as mp
from spectrum_analysis import data
from peaknames import peaknames
from pca_methods import addPoint, scaleParameters, createCluster, get_image, \
                        printPCAresults, plotCluster, selectSpecType, \
                        plotClusterOverview, plotHistInCluster

# define all data to plot mappings
# 'mapfolder': The dir-path of the mapping to be plotted
# 'dims': The dimensions of the mapping
# 'stepsize': The space between two points of measurement in µm
# 'background': Name of the background image (best png, jpg works as wel)
#               images need to be in the defined 'mapfolder'
# 'n_clusters': Number of clusters to be used in SpectralClustering PCA
mappings = {
    '001': {'mapfolder': os.path.join('testdata', '1'),
            'dims': (4, 4),
            'stepsize': 10,
            'background': 'bg_test.png',
            'markersize': 1.04,
            'n_clusters': 4},
    '002': {'mapfolder': os.path.join('testdata', '2'),
            'dims': (8, 2),
            'stepsize': 10,
            'background': 'bg_test.jpg',
            'markersize': 1.04,
            'n_clusters': 3},
    }

components = 3    # number of PCA components
component_x = 0   # component to plot on x axis
component_y = 1   # component to plot on x axis

show_hover_plot = True  # set True if interactive plot should be displayed
display_parameter_values = True    # show fitting values at hovering
print_PCA_results = True           # print PCA results to command line
print_PC_components = False        # print the principal components
plot_parameter_directions = True   # plot direction of parameters in PC space
plot_only_dirs = {'fwhm': {'linestyle': '-',
                           'plot_label': True},
                  'center': {'linestyle': '--',
                             'plot_label': True},
                  'height': {'linestyle': '-.',
                             'plot_label': True},
                  }
clustering = 'SpectralClustering'  # SpectralClustering (or OPTICS)

additional_fitplot_folder = 'testdata/2/results/plot'  # additional fit data
show_both_images = False   # True to display both fits in hovering plot
shift_second_image = [0.8, 0]

numberOfSamples = 2   # minimal number of samples (needed for OPTICS)
brim = 0.25           # minimal brim around plotted data

imagesize = (150, 150)   # size of hovering image
imageshift = (100, -50)  # shift of hovering image

plot_clustered_fitlines = False  # plot summed raw data if False
plot_histogramm_parameters = True  # plot histogramm_parameters in cluster plot
histogramm_parameters = ['breit_wigner_p1_fwhm', 'breit_wigner_p1_center',
                         'lorentzian_p1_fwhm', 'lorentzian_p1_center']
bins = 10  # number of bins for histogrammed parameters

linebreaker = '============================================================'

if not os.path.exists(clustering):
    os.makedirs(clustering)

if not os.path.exists(f'{clustering}{os.sep}allclusters'):
    os.makedirs(f'{clustering}{os.sep}allclusters')
else:
    shutil.rmtree(f'{clustering}{os.sep}allclusters')
    os.makedirs(f'{clustering}{os.sep}allclusters')

if plot_parameter_directions:
    if not os.path.exists(os.path.join(clustering, 'directions')):
        os.makedirs(os.path.join(clustering, 'directions'))

print(linebreaker + '\n' + linebreaker)

for key in mappings.keys():
    folder = mappings[key]['mapfolder']
    print(f'Mapping {key} of {len(mappings)}\n')
    print(f'{folder} mappings are plotted now.')

    mapp = mp.mapping(foldername=folder, plot=True, peaknames=peaknames)
    if not os.path.exists(f'{mapp.pltdir}{os.sep}clusters'):
        os.makedirs(f'{mapp.pltdir}{os.sep}clusters')
    else:
        shutil.rmtree(f'{mapp.pltdir}{os.sep}clusters')
        os.makedirs(f'{mapp.pltdir}{os.sep}clusters')

    # get fit data
    peakFileList, numberOfPeakFiles = data.GetFolderContent(
        mapp.pardir_peak,
        filetype='dat',
        objects='peakparameter')
    parameters, errors = data.GetAllData(peakFileList)
    parameterList = mapp.CreatePeakList(peakFileList)
    spectraList = selectSpecType(mapp, plot_clustered_fitlines)
    # preprocessing data
    # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
    # scale all data to [0,1]
    scaled_parameters = scaleParameters(parameters)

    # transpose data, so all parameters of one spectrum are in one array
    transposed = np.transpose(scaled_parameters)

    # perform PCA analysis and print results to commmand line
    pca = PCA(n_components=components)
    analyzed = pca.fit(transposed).transform(transposed)
    if print_PCA_results:
        printPCAresults(pca, parameterList,
                        print_components=print_PC_components)

    # plot everything and annotate each datapoint
    fig = plt.figure()
    ax_sum = [plt.subplot2grid((4, 3), (0, 2)),
              plt.subplot2grid((4, 3), (1, 2)),
              plt.subplot2grid((4, 3), (2, 2)),
              plt.subplot2grid((4, 3), (3, 2))]
    ax = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=4)

    x = analyzed[:, component_x]
    y = analyzed[:, component_y]
    PC = np.vstack((x, y)).transpose()
    cluster, sc, PC = createCluster(clustering, PC,
                                    mappings[key]['n_clusters'])

    # plot clustered data
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    clustertypes = set(cluster.labels_)
    clustersizes = []
    for klass, color in zip(range(0, len(clustertypes)), colors):
        PC_k = PC[cluster.labels_ == klass]
        clustersizes.append(len(PC_k))
        for point in PC_k:
            addPoint(sc, point, color)

    # get four biggest clusters and plot the sum spectrum
    rankedcluster = sorted(zip(clustersizes, clustertypes), reverse=True)
    plotClusterOverview(spectraList,
                        ax, ax_sum,
                        rankedcluster, cluster.labels_, colors,
                        histogramm_parameters,
                        parameterList, parameters, bins, mapp.missingvalue)

    # create labels and delete empty plots
    ax_copy = ax_sum.copy()
    for axs in ax_copy:
        if axs.lines:
            axs.tick_params(axis='y', labelsize=7)
            axs.tick_params(axis='x', labelsize=7)
            axs.yaxis.tick_right()
            axs.yaxis.set_label_position('right')
            axs.set_ylabel('Intensity')
        else:
            fig.delaxes(axs)
            ax_sum.remove(axs)
    for axs in ax_sum[0:-1]:
        axs.get_xaxis().set_ticks([])
    ax_sum[0].set_title('Cluster mean spectra')
    ax_sum[-1].set_xlabel('Wavenumber (cm$^{-1}$)')

    # set center, min and max of the plot
    xmin, xmax, ymin, ymax = [min(x)-brim, max(x)+brim,
                              min(y)-brim, max(y)+brim]
    xcenter, ycenter = [(xmax-abs(xmin))/2, (ymax-abs(ymin))/2]
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.axhline(0, linestyle='-', color=CSS4_COLORS['lightgrey'],
               zorder=1)
    ax.axvline(0, linestyle='-', color=CSS4_COLORS['lightgrey'],
               zorder=1)

    # add composing directions
    if plot_parameter_directions:
        pcx = pca.components_[component_x]
        pcy = pca.components_[component_y]
        plot_lines = []
        for i, parameter in enumerate(parameterList):
            peaknumber = int(parameter.split('_')[-2][-1])
            peaktype = parameter[0][0]
            param = parameter.split('_')[-1]
            if param in plot_only_dirs.keys():
                if peaktype == 'b':
                    color = 'b'
                elif peaktype == 'l':
                    color = 'r'
                elif peaktype == 'g':
                    color = 'g'
                elif peaktype == 'v':
                    color = 'c'
                color = (np.array(to_rgba(color))
                         - np.array((0, 0, 0, 0.3 * (peaknumber-1))))
                line = ax.plot((0, pcx[i]), (0, pcy[i]), color=color, zorder=3,
                               linestyle=plot_only_dirs[param]['linestyle'])
                plot_lines.append(line)
                if plot_only_dirs[param]['plot_label']:
                    ax.text(pcx[i]*1.15, pcy[i]*1.15,
                            f'{peaktype}{peaknumber}_{param[0:3]}',
                            ha='center', va='center', fontsize=7, zorder=4)

    # create annotation text
    annot = ax.annotate('', xy=(0, 0), xytext=(20, 20),
                        textcoords='offset points',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', fc='w'),
                        arrowprops=dict(arrowstyle='->'))
    annot.set_visible(False)

    # create annotation image
    label = mapp.listOfFiles[0].split(os.sep)[-1].split('.')[0]
    image = get_image(mapp.pltdir, label, imagesize)
    imagebox = OffsetImage(image)
    ab = AnnotationBbox(imagebox, (0, 0), xybox=imageshift,
                        boxcoords="offset points")
    if show_both_images:
        additional_image = get_image(additional_fitplot_folder, label,
                                     imagesize)
        additional_imagebox = OffsetImage(additional_image)
        ac = AnnotationBbox(additional_imagebox, (0, 0), xybox=imageshift,
                            boxcoords="offset points")

    # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
    def update_annot(ind):
        """Update annotation and image."""
        # update text annotation
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        idxlist = []
        for element in PC:
            idxlist.append(np.allclose(element, pos))
        idx = idxlist.index(True)
        annotation_string = f'{idx + 1}\n'
        if display_parameter_values:
            for i, label in enumerate(parameterList):
                annotation_string += (f'{parameters[i, idx]: 10.2f} '
                                      f'+/- {errors[i, idx]:8.2f} '
                                      f'({label})\n')
        annot.set_text(annotation_string[:-1])
        annot.get_bbox_patch().set_alpha(0.4)

        # update immage annotation
        label = mapp.listOfFiles[idx].split(os.sep)[-1].split('.')[0]
        image = get_image(mapp.pltdir, label, imagesize)
        ab.xy = pos
        ab.offsetbox = OffsetImage(image)
        ax.add_artist(ab)
        if show_both_images:
            additional_image = get_image(additional_fitplot_folder, label,
                                         imagesize)
            ac.xy = pos + shift_second_image
            ac.offsetbox = OffsetImage(additional_image)
            ax.add_artist(ac)

    def hover(event):
        """Hovering behavoir."""
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    # annot.set_visible(False)
                    # ab = AnnotationBbox(None, (0,0))
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    figManager = plt.get_current_fig_manager()  # get current figure
    figManager.full_screen_toggle()             # show it maximized

    ax.set_xlabel(f'principal component {component_x + 1}')
    ax.set_ylabel(f'principal component {component_y + 1}')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.001)

    if plot_parameter_directions:
        plt.savefig(
            (f'{clustering}{os.sep}directions{os.sep}'
             f'{mapp.folder.replace(os.sep, "_")}'
             f'_pc{component_x}_pc{component_y}_dirs.png'),
            dpi=300)
        plt.savefig(f'{mapp.pltdir}{os.sep}pca_analysis'
                    f'_pc{component_x}_pc{component_y}_dirs.png', dpi=300)
        plt.savefig(f'{mapp.pltdir}{os.sep}pca_analysis'
                    f'_pc{component_x}_pc{component_y}_dirs.pdf')
    else:
        plt.savefig(
            (f'{clustering}{os.sep}{mapp.folder.replace(os.sep, "_")}'
             f'_pc{component_x}_pc{component_y}.png'),
            dpi=300)
        plt.savefig(f'{mapp.pltdir}{os.sep}pca_analysis'
                    f'_pc{component_x}_pc{component_y}.png', dpi=300)
        plt.savefig(f'{mapp.pltdir}{os.sep}pca_analysis'
                    f'_pc{component_x}_pc{component_y}.pdf')
    ax.set_title(f'PCA Analysis of {folder}')

    if show_hover_plot:
        plt.show()

    plt.close()

    for i, clust in enumerate(rankedcluster):
        fig, ax = plt.subplots()
        spec_mask = plotCluster(ax, cluster.labels_, clust, spectraList,
                                colors, prnt=False)
        ax_twin = plotHistInCluster(ax, clust, spec_mask,
                                    histogramm_parameters, parameterList,
                                    parameters, bins, mapp.missingvalue)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax.set_ylabel('Intensity')
        ax_twin.yaxis.set_label_position('left')
        ax_twin.tick_params(axis='y', labelsize=12)
        ax_twin.legend(ncol=len(histogramm_parameters),
                       bbox_to_anchor=(0, 1.01), loc='lower left',
                       borderaxespad=0., prop={'size': 7})
        fig.tight_layout()
        plt.savefig(
            (f'{clustering}{os.sep}allclusters{os.sep}'
             f'{mapp.folder.replace(os.sep, "_")}'
             f'_pc{component_x}_pc{component_y}_'
             f'S{clust[1]:03}_C{clust[0]}.png'),
            dpi=300)
        plt.savefig(
            (f'{mapp.pltdir}{os.sep}clusters{os.sep}'
             f'{mapp.folder.replace(os.sep, "_")}'
             f'_pc{component_x}_pc{component_y}_'
             f'S{clust[1]:03}_C{clust[0]}.png'),
            dpi=300)
        # plt.show()
        plt.close()

    cmap = ListedColormap(colors[0:max(cluster.labels_)+1])
    mapp.PlotMapping('params', cluster.labels_+2, mappings[key]['dims'],
                     mappings[key]['stepsize'], name='pca',
                     colormap=cmap)
    mapp.PlotMapping('params', cluster.labels_+2, mappings[key]['dims'],
                     mappings[key]['stepsize'],
                     name='grid_pca', alpha=0.75, grid=True,
                     background=(f'{folder}{os.sep}'
                                 f'{mappings[key]["background"]}'),
                     msize=mappings[key]['markersize'],
                     plot_missing=False, area=None, colormap=cmap)
    # set rcParams to default, as PlotMapping modifies it
    rcdefaults()

    print(linebreaker + '\n' + linebreaker)
    gc.collect()
