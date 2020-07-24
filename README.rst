==========
Spectrum analysis
==========

Python package to analyze spectra interactively.

==========
Dependencies
==========

The software is tested and developed with python 3.7.3

Needed packages are

* lmfit>=0.9.11
* matplotlib>=3.1.1
* numpy>=1.16.4
* pandas>=0.24.0
* Pillow>=7.0.0
* pytest>=5.3.5
* PyWavelets>=1.0.1
* scikit-image>=0.16.2
* scipy>=1.3.0
* statsmodels>=0.9.0
* svgpath2mpl>=0.2.1

==========
Installing
==========

The software is installed via
::
  sudo python setup.py install

==========
Documentation
==========
Fitting
----------
1. Put all the spectra you want to analyze in a folder (or take the testdata
   folder from the examples folder).

2. Decide if you want to analyze a single spectrum (``spec_tester.py``) or a
   mapping (``map_tester.py``).

3. Modify the ``*_tester.py`` accordingly to your data.
   For a single spectrum it should read

::

  spec = sp.spectrum(os.path.join('testdata', '1', '0005'))

|   For a mapping it should read
::

  mapp = mp.mapping(foldername=os.path.join('testdata', '1'))

4. Define which peaks you want to use in your analysis.
   Implemented are ``gaussian``, ``lorentzian``, ``breit_wigner`` and
   ``voigt``

::

  peaks = ['gaussian', 'lorentzian']

5. run ``*_tester.py`` in your console (replace * with ``spec`` or ``map``)

::

  python *_tester.py

6. A plot window opens. Select the spectral region you want to analyze.
   Just click on the plot at two positions. You may choose a region wider than
   the peaks of interest, as you will need to substract background later.
   The selected region will be marked green. Than close the window (to fasten
   up your analysis, you can close the window with ``Ctrl`` + ``W``).

7. A new window opens showing the before selected region. Now select the data
   that should not be used in a background fit. Here you should choose the
   peaks of interest. Again click on the plot to select them in the same way
   as before, you may select more than one region. The regions will be marked
   in red. Close the window.

8. If you selected to analyse a single spectrum (``spec_tester.py``),
   a new window opens and shows you the FFT of the spectrum.
   You can select one frequency you want to remove. If you don't want to do so,
   just close the window.

9. A new window opens. Now you are asked to select the maxima of the peaks you
   want to be analyzed. Again this is done by clicking on the maxima of
   interest. First you will be asked to select all ``gaussian`` peaks.
   Close the window. A new window opens and you will be asked to select all
   ``lorentzian`` peaks. Close the window when you are finished with your
   selection.

10. If you selected ``map``, you will be asked to repeat the aforementioned step
    for each spectrum in the complete mapping.

11. Closing the window of the last peak selection will start the fitting
    process. Now the program fits all the spectra you selected.
    You have to wait some time. When the programm is finished you should find
    everything (data, results and plots) inside the newly created ``results``
    inside your data folder.

12. If you selected ``breit_wigner`` as one of your fitting functions, you
    might want to analyze its ``fwhm``. The automatic calculation as been
    removed     from recent versions of ``lmfit`` (see `lmfit google-Group
    <https://groups.google.com/forum/#!topic/lmfit-py/IctDA3DEjrE>`_).
    You can calculate the ``fwhm`` using the ``fwhm_calculator.py``. Just
    change the ``folders`` array accordingly to your fit folder and run it.

Plot Mappings
----------

If you fitted a mapping you can now plot it using

::

  python -u map_plot_tester.py 2>&1 | tee log.txt

The ``-u`` option forces python to print the output streams unbuffered.
Using ``tee log.txt`` a log file will be written to keep your console output.
Using ``2>&1`` also prints errors to the log file.
This might be important to debugging if you scale multiple mappings to
the same color scale. **The list of the origins of the color scale minima and
maxima is not printed elsewhere!**


The first lines of ``map_plot_tester.py`` should read

::

  mapFolderList = [os.path.join('testdata', '1'),
                   os.path.join('testdata', '2')
                   ]

  dims = [(4, 4),
          (8, 2)
          ]

  stepsize = [10,
             10
              ]

  # plot ratios
  top = 'lorentzian_p1_height'
  bot = 'breit_wigner_p1_height'
  opt = 'div'

In case you want to analyze multiple mappings, just add more lines to
``mapFolderList`` (folder with fitted mapping data), ``dims`` (x and y
dimensions of the corresponding mapping) and ``stepsize`` (step size of
your xy pattern).

You can also adjust which two peak parameters (``top`` and ``bot``) should
be linked by a specified operation (``opt``). Operations possible are
``'div'``, ``'mult'``, ``'add'`` and ``'sub'``. You can take any of the
peakparameters found in ``results/fitparameter/peakwise/`` of your mapping.

**Caution**
If you want to analyse different parameter operations than presented in the
example file or use more than one ``breit_wigner`` and two ``lorentzian``
peaks, you need to modify the ``peaknames.py`` accordingly to your wishes.
The dictionary is rather self-explanatory.

Principal Component Analysis
----------

If you fitted and plotted a mapping you can perform an interactive principal
component analysis (pca) with additional cluster analysis.

::

  python -u pca_analysis.py 2>&1 | tee log.txt

An interactive plot opens, displaying multiple panels.
In the left panel the first two principal components (PC) are plotted, as well
as the projections of the fitting parameters in the two dimensional PC space.
Each dot here is linked to the fitting data and to the corresponding fit plot.
By hovering over a dot, the data will show up.

In the right panel, the results of the ``SpectralClustering`` method of
``scikit-learn`` are shown.
Here the mean spectra of the clusters with the most spectra are presented.
The colors of the spectra correspond to the colors in the PC plot in order to
easen the analysis of the data.

Additionally to the clusters' mean spectra, it might be interesting to
plot histogrammed fitting parameters of a cluster as well. The parameters are
plotted in the same plot as the mean spectra, to get a fast overview.
Here for example, the ``center`` and the ``fwhm`` of selected peak functions
are plotted.

The script can be tuned to the analysts' needs. Details on the different
tuning options are described in the script.

In case of using the ``plot_parameter_directions = True`` option, the plots
are saved to the ``SpectralClustering/directions`` directory. Setting the
option to ``False`` the plots are saved in the ``SpectralClustering`` directory.
The plots are saved in the corresponding mapping folder as well.

Additionally all mean spectra of the clusters are saved to the
``SpectralClustering/allclusters`` folder. Each file name contains information
on the principal components (pc), the number of spectra (S) and the number of
cluster of the corresponding plot.
