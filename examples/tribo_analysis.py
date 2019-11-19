import os

import matplotlib.pyplot as plt
import numpy as np

from spectrum_analysis.data import GetData, GetFolderContent

datafiles, datasets = GetFolderContent('testdata/tribo', 'txt')
running_in = 100
running_out = 10
datapoints = 100

linebreaker ='============================================================'

start = []
stop = []

folder = 'testdata/tribo/results/tribo'
if not os.path.exists(folder):
    os.makedirs(folder)

startfile = f'{folder}/start_mu.dat'
stopfile = f'{folder}/stop_mu.dat'

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def save_mu(file, namelist, valuelist):
    with open(file, 'w') as f:
        for i, value in enumerate(valuelist):
            savename = namelist[i].split('/')[-1]
            savename = savename.split('.')[0]
            f.write(f'{savename}\t{value}\n')

print(f'{linebreaker}\n{linebreaker}')

for datafile in datafiles:
    index = datafiles.index(datafile)
    print(f'Data set {index + 1} of {datasets} is analyzed.')
    x, y = GetData(datafile,
                   measurement='tribo',
                   xdata='Time',
                   ydata='µ')

    laps, cycle = GetData(datafile,
                          measurement='tribo',
                          xdata='laps',
                          ydata='Cycle')

    # calculate nearest index to running_in/_out laps
    # and get means of first and last number of datapoints
    f_idx = find_nearest_index(laps, running_in)
    first = np.mean(y[f_idx : (f_idx + datapoints)])
    start.append(first)

    l_idx = find_nearest_index(abs(laps - laps[-1]), running_out)
    last = np.mean(y[-l_idx : -running_out])
    stop.append(last)

    print(f'Median of the first {datapoints} points after running in '
          f'for {laps[f_idx]} laps:\t{first}')
    print(f'Median of the last {datapoints} points before running out '
          f'for {laps[-1]-laps[l_idx]} laps:\t{last}')
    print(linebreaker + '\n' + linebreaker)

    # plot spectrum
    #fig, ax = plt.subplots()
    #ax.plot(x, y, 'b-', label='Data')
    #
    #plt.show()

save_mu(startfile, datafiles, start)
save_mu(stopfile, datafiles, stop)
