"""PCA analysis of mapping parameters.

Perform a PCA analysis of a fitted mapping.
By hovering over the data points, the corresponding spectra
show up.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn import preprocessing
from sklearn.decomposition import PCA

from spectrum_analysis import mapping as mp
from spectrum_analysis import data
from peaknames import peaknames

mapFolderList = [
    os.path.join('testdata', '1'),
    #os.path.join('testdata', '2'),
    ]

components = 3    # number of PCA components
component_x = 0   # component to plot on x axis
component_y = 1   # component to plot on x axis

imagesize = 0.08
imageshift = (100, -50)


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

    fig, ax = plt.subplots()
    sc = plt.scatter(x, y)

    # create annotation text
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    # create annotation image
    label = mapp.listOfFiles[0].split(os.sep)[-1].split('.')[0]
    imagename = f'{mapp.pltdir}{os.sep}fitplot_{label}.png'
    image = mpimg.imread(imagename)
    imagebox = OffsetImage(image, zoom=imagesize)
    ab = AnnotationBbox(imagebox, (0, 0), xybox=imageshift,
                        boxcoords="offset points")

    names = list(range(1, len(transposed) + 1))

    # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
    def update_annot(ind):
        """Update annotation and image."""
        # update text annotation
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = [names[n] for n in ind["ind"]][0]
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

        # update immage annotation
        label = mapp.listOfFiles[text-1].split(os.sep)[-1].split('.')[0]
        imagename = f'{mapp.pltdir}{os.sep}fitplot_{label}.png'
        image = mpimg.imread(imagename)
        ab.xy = pos
        ab.offsetbox = OffsetImage(image, zoom=imagesize)
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
