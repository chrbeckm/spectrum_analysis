# spectrum_analysis
Python class to fit spectra interactively.

## Needed packages

uncertainties
matplotlib
uncertainties
scipy
lmfit
statsmodels
PyWavelets (pywt) 1.0.3.

## How to use the methods for spectrum fit:

**step 1**: put all the spectra you want to analyze in a folder in the methods folder

**step 2**: modify analyze.py and write the folder to analyze into the file. If you want to analyze a mapping you should specify the parameters xdim, ydim, stepsize, xmin and xmax of the mapping.

**step 3a**: run analyze.py.
```
python analyze.py
```

**step 3b**: A plot window opens. Select the spectral region you want to analyze with the fit. Just click on the plot at two or more positions. You may choose a region wider than the peaks of interest, as you will need to substract background later. The selected region will be marked green. Than close the window.

**step 3c**: A new window opens showing the before selected region. Now select the data that should not be used in a background fit. Here you should choose the peaks of interest. Again click on the plot to select them in the same way as before. The regions will be marked in red. Close the window.

**step 3d**: Again a new window opens and shows you the background line that will be substracted from your spectral region. Now select the maxima of peaks you want to be fitted. Again this is done by clicking on the spectrum by clicking. By closing the window, you will start the fitting process. Now the programm runs and you got to wait some time. When the programm is finished it will show you a new plot window with the selected spectral region and the fit plotted. Now you should find everything (data, results and plots) inside the recently created folder.

**step 4**: if you want to analyze several, similar spectra, this is done the same way, you have to follow the steps 3a - 3d for the first spectrum, for the other spectra you only have to do 3d to get the proper starting parameters.
