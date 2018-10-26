import os
import glob
import shutil

def create_structure():
    # Method to create a folder structure for the spectra.
    # Later you can store the results for every spectrum in the folder
    # named by the file with the original data.

    # take all files in the current folder with *.txt ending
    labels = sorted(glob.glob('*.txt'))

    # create folders and rename data files
    for i in labels:
        os.makedirs(i.split('.')[0] )
        os.rename(i, i.split('.')[0] + '/data_' + i)

    # create labels
    for i in range(len(labels)):
        labels[i] = labels[i].split('.')[0]

    # save list of labels so that you can easy iterate all spectra later
    with open('labels.txt', 'w') as w:
        for i in labels:
            w.write(i + '\n')

    #create folders for results
    os.makedirs('results_plot/')
    os.makedirs('results_fitparameter/')

if __name__ == '__main__':
    create_structure()
