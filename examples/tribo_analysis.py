import matplotlib.pyplot as plt
import numpy as np

from spectrum_analysis.data import GetData

datafile = 'testdata/tribo_test.txt'
running_in = 100
running_out = 10
datapoints = 100

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

x, y = GetData(datafile,
               measurement='tribo',
               xdata='Time',
               ydata='µ')

laps, cycle = GetData(datafile,
                      measurement='tribo',
                      xdata='laps',
                      ydata='Cycle')

# calculate nearest index to running_in laps
index = find_nearest_index(laps, running_in)
first_100 = np.mean(y[index : (index + datapoints)])

last_100 = np.mean(y[-(datapoints +  running_out) : -(running_out)])

print(f'Median of the first {datapoints} points after running in '
      f'for {laps[index]} laps:\t{first_100}')
print(f'Median of the last {datapoints} points before running out '
      f'for {laps[-1]-laps[-running_out]} laps:\t{last_100}')

# plot spectrum
#fig, ax = plt.subplots()
#ax.plot(x, cycle, 'b-', label='Data')
#
#plt.show()
