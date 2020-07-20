"""PCA analysis of mapping parameters.

Perform a PCA analysis of a fitted mapping.
By hovering over the data points, the corresponding spectra
show up.
"""
import os

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import to_rgba

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, SpectralClustering

from spectrum_analysis import mapping as mp
from spectrum_analysis import data
from peaknames import peaknames

mapFolderList = [
    os.path.join('testdata', '1'),
    os.path.join('testdata', '2'),
    ]

# number of clusters (needed for SpectralClustering)
n_clusters = [
    4,
    3,
    ]

components = 3    # number of PCA components
component_x = 0   # component to plot on x axis
component_y = 1   # component to plot on x axis

show_hover_plot = True  # set True if interactive plot should be displayed
display_parameter_values = True    # show fitting values at hovering
print_PCA_results = True           # print PCA results to command line
plot_parameter_directions = True   # plot direction of parameters in PC space
clustering = 'SpectralClustering'  # SpectralClustering or OPTICS

numberOfSamples = 2   # minimal number of samples (needed for OPTICS)
brim = 0.25           # minimal brim around plotted data

imagesize = (150, 150)   # size of hovering image
imageshift = (100, -50)  # shift of hovering image


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
        scaled_param = min_max_scaler.fit_transform(
            param.reshape(-1, 1))
        scaled_params[idx] = scaled_param.reshape(1, -1)
    return scaled_params


def createCluster(method, n_clust=3, min_samples=5):
    """Create cluster and plot the corresponding scatter plot."""
    if method == 'SpectralClustering':
        clust = SpectralClustering(n_clusters=n_clust)
        clust.fit(PC)
        scat = plt.scatter(-10, -10)
    elif method == 'OPTICS':
        clust = OPTICS(min_samples=min_samples)
        clust.fit(PC)
        scat = plt.scatter(PC[clust.labels_ == -1, 0],
                           PC[clust.labels_ == -1, 1], c='k')
    return clust, scat


def printPCAresults(pc_ana, param_list):
    """Print results of PCA analysis to command line."""
    print(f'explained variance ratio'
          f'({pc_ana.components_.shape[0]} components): '
          f'{sum(pc_ana.explained_variance_ratio_):2.2f} '
          f'({pc_ana.explained_variance_ratio_.round(2)})')
    for j, principal_component in enumerate(pc_ana.components_):
        print(f'Principal component {j+1}')
        for idx, lbl in enumerate(param_list):
            print(f'{principal_component[idx]: 2.4f} * {lbl}')
        print()


if not os.path.exists(clustering):
    os.makedirs(clustering)

for folder in mapFolderList:
    index = mapFolderList.index(folder)
    print(f'Mapping {index + 1} of {len(mapFolderList)}\n')
    print(f'{folder} mappings are plotted now.')

    mapp = mp.mapping(foldername=folder, plot=True, peaknames=peaknames)

    # get fit data
    peakFileList, numberOfPeakFiles = data.GetFolderContent(
        mapp.pardir_peak,
        filetype='dat',
        objects='peakparameter')
    parameters, errors = data.GetAllData(peakFileList)
    parameterList = mapp.CreatePeakList(peakFileList)

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
        printPCAresults(pca, parameterList)

    # plot everything and annotate each datapoint
    fig, ax = plt.subplots()

    x = analyzed[:, component_x]
    y = analyzed[:, component_y]
    PC = np.vstack((x, y)).transpose()
    cluster, sc = createCluster(clustering, n_clusters[index])

    # plot clustered data
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for klass, color in zip(range(0, len(set(cluster.labels_))), colors):
        PC_k = PC[cluster.labels_ == klass]
        for point in PC_k:
            addPoint(sc, point, color)

    # set center, min and max of the plot
    xmin, xmax, ymin, ymax = [min(x)-brim, max(x)+brim,
                              min(y)-brim, max(y)+brim]
    xcenter, ycenter = [(xmax-abs(xmin))/2, (ymax-abs(ymin))/2]
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))

    # add composing directions
    if plot_parameter_directions:
        pcx = pca.components_[component_x]
        pcy = pca.components_[component_y]
        for i, parameter in enumerate(parameterList):
            peaknumber = int(parameter.split('_')[-2][-1])
            peaktype = parameter[0][0]
            param = parameter.split('_')[-1]
            if peaktype == 'b':
                color = colors[0]
            elif peaktype == 'l':
                color = colors[1]
            elif peaktype == 'g':
                color = colors[2]
            elif peaktype == 'v':
                color = colors[3]
            color = (np.array(to_rgba(color))
                     - np.array((0, 0, 0, 0.2 * (peaknumber-1))))
            plt.arrow(xcenter, ycenter, pcx[i], pcy[i], color=color, alpha=0.5)
            plt.text(xcenter + pcx[i]*1.15, ycenter + pcy[i]*1.15,
                     f'{peaktype}_{param[0:3]}', ha='center', va='center')

    # create annotation text
    annot = ax.annotate('', xy=(0, 0), xytext=(20, 20),
                        textcoords='offset points',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', fc='w'),
                        arrowprops=dict(arrowstyle='->'))
    annot.set_visible(False)

    # create annotation image
    label = mapp.listOfFiles[0].split(os.sep)[-1].split('.')[0]

    imagename = f'{mapp.pltdir}{os.sep}fitplot_{label}.png'
    image = Image.open(imagename)
    image.thumbnail(imagesize, Image.ANTIALIAS)  # resize the image
    imagebox = OffsetImage(image)
    ab = AnnotationBbox(imagebox, (0, 0), xybox=imageshift,
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
        imagename = f'{mapp.pltdir}{os.sep}fitplot_{label}.png'
        image = Image.open(imagename)
        image.thumbnail(imagesize, Image.ANTIALIAS)  # resize the image
        ab.xy = pos
        ab.offsetbox = OffsetImage(image)
        ax.add_artist(ab)

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

    plt.xlabel(f'PC {component_x + 1}')
    plt.ylabel(f'PC {component_y + 1}')
    if plot_parameter_directions:
        plt.savefig(
            (f'{clustering}{os.sep}{mapp.folder.replace(os.sep, "_")}'
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
    plt.title(f'PCA Analysis of {folder}')

    if show_hover_plot:
        plt.show()

    plt.close()
