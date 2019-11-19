import os

import matplotlib.pyplot as plt
import numpy as np

from spectrum_analysis.data import GetData, GetFolderContent

tribofolder = 'testdata/tribo'
datafiles, datasets = GetFolderContent(tribofolder, 'txt')
running_in = 100
running_out = 10
datapoints = 100

linebreaker ='============================================================'

start = {}
stop = {}

resfolder = f'{tribofolder}/results/tribo'
if not os.path.exists(resfolder):
    os.makedirs(resfolder)

startfile = f'{resfolder}/start_mu.dat'
stopfile = f'{resfolder}/stop_mu.dat'

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def update_dict(dict, key, array):
    mean = np.mean(array)
    err = np.std(array)
    content = {key : {'value' : mean, 'error': err}}
    dict.update(content)

def save_mu(file, dict):
    with open(file, 'w') as f:
        for key in dict.keys():
            value = dict[key]['value']
            error = dict[key]['error']
            f.write(f'{key}\t{value:13.5f} +/- {error:11.5f}\n')

print(f'{linebreaker}\n{linebreaker}')

for datafile in datafiles:
    index = datafiles.index(datafile)
    savename = datafile.split('/')[-1]
    savename = savename.split('.')[0]

    print(f'Data set {index + 1} of {datasets} is analyzed.')
    print(f'{datafile} is analyzed.')
    x, y = GetData(datafile,
                   measurement='tribo',
                   xdata='Distance',
                   ydata='µ')

    laps, cycle = GetData(datafile,
                          measurement='tribo',
                          xdata='laps',
                          ydata='Cycle')

    # calculate nearest index to running_in/_out laps
    f_idx = find_nearest_index(laps, running_in)
    y_temp = y[f_idx : (f_idx + datapoints)]
    update_dict(start, savename, y_temp)

    l_idx = find_nearest_index(abs(laps - laps[-1]), running_out)
    y_temp = y[(l_idx - datapoints) : l_idx]
    update_dict(stop, savename, y_temp)

    print(f'Median of the first {datapoints} points after running in '
          f'for {laps[f_idx]} laps:\t{start[savename]["value"]:.5f}')
    print(f'Median of the last {datapoints} points before running out '
          f'for {laps[-1]-laps[l_idx]} laps:\t{stop[savename]["value"]:.5f}')
    print(linebreaker + '\n' + linebreaker)

    # plot and save spectrum
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b-', label='Data')

    fig.legend(loc = 'lower right')
    plt.ylabel('µ (arb. u.)')
    plt.xlabel('Distance (m)')

    # save figures
    fig.savefig(f'{resfolder}/{savename}_plot.pdf')
    fig.savefig(f'{resfolder}/{savename}_plot.png', dpi=300)
    plt.close()

save_mu(startfile, start)
save_mu(stopfile, stop)
