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
    # os.path.join('testdata', '2'),
    ]

components = 3    # number of PCA components
component_x = 0   # component to plot on x axis
component_y = 1   # component to plot on x axis

clustering = 'SpectralClustering'  # SpectralClustering or OPTICS
n_clusters = 4        # number of clusters (needed for SpectralClustering)
numberOfSamples = 2   # minimal number of samples (needed for OPTICS)
brim = 0.25           # minimal brim around plotted data

imagesize = (150, 150)         # size of hovering image
imageshift = (100, -50)  # shift of hovering image


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
    scaled_parameters = np.zeros_like(parameters)
    for i, parameter in enumerate(parameters):
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_parameter = min_max_scaler.fit_transform(
            parameter.reshape(-1, 1))
        scaled_parameters[i] = scaled_parameter.reshape(1, -1)

    # transpose data, so all parameters of one spectrum are in one array
    transposed = np.transpose(scaled_parameters)

    # perform PCA analysis
    pca = PCA(n_components=components)
    analyzed = pca.fit(transposed).transform(transposed)
    print(f'explained variance ratio ({components} components):'
          f'{pca.explained_variance_ratio_}')

    # plot everything and annotate each datapoint
    x = analyzed[:, component_x]
    y = analyzed[:, component_y]

    # clustering of dataset
    fig, ax = plt.subplots()

    PC = np.vstack((x, y)).transpose()
    if clustering == 'SpectralClustering':
        cluster = SpectralClustering(n_clusters=n_clusters)
        cluster.fit(PC)
        sc = plt.scatter(-10, -10)
    elif clustering == 'OPTICS':
        cluster = OPTICS(min_samples=numberOfSamples)
        cluster.fit(PC)
        sc = plt.scatter(PC[cluster.labels_ == -1, 0],
                         PC[cluster.labels_ == -1, 1], c='k')

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

    # plot clustered data
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for klass, color in zip(range(0, len(set(cluster.labels_))), colors):
        PC_k = PC[cluster.labels_ == klass]
        for point in PC_k:
            addPoint(sc, point, color)

    plt.xlim((min(x)-brim, max(x)+brim))
    plt.ylim((min(y)-brim, max(y)+brim))

    # create annotation text
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
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
        annot.set_text(idx+1)
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

    plt.title(f'PCA Analysis of {folder}')
    plt.xlabel(f'PC {component_x + 1}')
    plt.ylabel(f'PC {component_y + 1}')

    plt.show()
