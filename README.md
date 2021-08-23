Overview
=====
This is the source code that generates mocks, reduce high-resolution spectra and bootstrap QMLE results. It further helps reading and plotting QMLE outputs. Please cite papers Karaçaylı et al. (2020) and Karaçaylı et al. (submitted to MNRAS).

+ Karaçaylı N. G., Font-Ribera A., Padmanabhan N., 2020, [MNRAS](https://doi.org/10.1093/mnras/staa2331), 497, 4742
+ Karaçaylı N. G., et al., 2021, MNRAS, submitted

## Programs
Scripts under `bin` folder are executable. Pass `--help` for arguments. The most important ones are these four scripts:

+ [genKXUMocks.py](bin/genKXUMocks.py) reduces KODIAQ, SQUAD and XQ-100 spectra as described in the paper.
+ [genDESILiteMocks.py](bin/genDESILiteMocks.py) generates DESI-lite mock spectra as described in the paper. Can also generate with quickquasars and picca file formats.
+ [estP1D-FFT.py](bin/estP1D-FFT.py) performs a rough FFT estimate for a given QMLE config file. This should be used only for a rough cross check.
+ [bootstrapQMLE.py](bin/bootstrapQMLE.py) calculates bootstrap realizations for QMLE results. Individual estimates has to be saved and converted to a FITS file using [convertFpbin2FITS.py](bin/convertFpbin2FITS.py). This can consume substantial memory and CPU time.

The other scripts are also helpful.

+ [checkConfigQMLE.py](bin/checkConfigQMLE.py) does a surface consistency check for a given QMLE config file. Testing config files with this can save a lot of time debugging. Note, passing this script does not guarantee perfect functioning of QMLE.
+ [computeLognMeanFlux.py](bin/computeLognMeanFlux.py) and [computeLognPower.py](bin/computeLognPower.py) compute true mean flux and power spectrum for lognormal mocks.
+ [convertBQ2QQFits.py](bin/convertBQ2QQFits.py) (outdated) converts BinaryQSO files to quickquasars files.
+ [convertFlux2Fluctuations.py](bin/convertFlux2Fluctuations.py) converts flux to flux fluctuations back and forth. Can also add simplified continuum errors.
+ [convertFpbin2FITS.py](bin/convertFpbin2FITS.py) converts individuals files saved by QMLE to a combined FITS file.
+ [maskSpectra.py](bin/maskSpectra.py) masks BinaryQSO spectrum files for random DLAs.

## Source
+ [fiducial.py](py/qsotools/fiducial.py) contains transition lines, spectrogpraph functions, power spectrum and mean flux fitting functions and their fitters.
+ [io.py](py/qsotools/io.py) handles input and output for various spectrum file formats (Binary, KODIAQ, SQUAD, XQ-100, picca and quickquasars) under an umbrella `Spectrum` class.
+ [mocklib.py](py/qsotools/mocklib.py) generates the mocks. Analytical functions describing the lognormal mocks are here.
+ [plotter.py](py/qsotools/plotter.py) helps plotting QMLE results.
+ [specops.py](py/qsotools/specops.py) does various spectrum operations such as chunking, resampling and binning pixel statistics.
+ [kodiaqviewer.py](py/qsotools/kodiaqviewer.py) is a beta jupyter notebook class for viewing KODIAQ spectra.

## Data
Please cite respective works when using anything related to data.

+ Keck Observatory Database of Ionized Absorption toward Quasars (KODIAQ) Data Release 2 (DR2) can be found on their [website](https://koa.ipac.caltech.edu/workspace/TMP_939bFW_53591/kodiaq53591.html) (Lehner et al. 2014; O’Meara et al. 2015, 2017). A summary [table](py/qsotools/tables/kodiaq_asu.tsv) from VizieR is part of this package. A [master file](py/qsotools/tables/master_kodiaq_table.csv) is constructed to gather as much information as possible into a single file. Visually inspected DLAs are listed in [here](py/qsotools/tables/kodiaq_vi_dlas.csv). However, this is constructed in order to roughly mask affected regions, and is not accurate enough for precise scientific studies. It can also contain duplicate regions.
+ The Spectral Quasar Absorption Database (SQUAD) DR1 (Murphy et al. 2019) is [here](https://archive.eso.org/cms/eso-archive-news/the-uves-spectral-quasar-absorption-database--squad--data-releas.html). The provided [table](py/qsotools/tables/uves_squad_dr1_quasars_master.csv) is part of this package. An updated [DLA table](py/qsotools/tables/squad_vi_dlas.csv) comes with caveats above.
+ XQ-100 is [here](http://telbib.eso.org/detail.php?bibcode=2016A%26A...594A..91L). A DLA catalog is provided by Sánchez-Ramírez et al. (2016). The relevant part of it is provided [here](py/qsotools/tables/xq100_dla_table_sanchez-ramirez_2016.csv). Similarly, visually identified DLA (with same caveats) are [here](py/qsotools/tables/xq100_vi_dlas.csv).

Installation
=====
Conda installation is recommended as running setup.py can break down and/or complicate removal of this package.

## Conda installation
Change the name of the environment to your taste in [requirements.yml](requirements.yml). Then create, activate and install by running

    conda env create -f requirements.yml
    conda activate [ENVNAME]
    pip install .

This should work even for clusters.

## Manual home installation
**Warning**: running `python setup.py install` is **NOT** recommended.

Install requirements `pip install -r requirements.txt`.

Home install is recommended. Create `bin/` and `lib/python` directories in your `$HOME`. Add these to your `PATH` and `PYTHONPATH` in  `.bashrc` (or `.bash_profile`, `.zshrc`, etc.), and restart your terminal.

    export PATH="$HOME/bin:$PATH"
    export PYTHONPATH="$HOME/lib/python:$PYTHONPATH"

Then, home installation is simply `python setup.py install --home=$HOME`.

References
=====
+ Lehner N., O’Meara J. M., Fox A. J., Howk J. C., Prochaska J. X., Burns V., Armstrong A. A., 2014, ApJ, 788, 119
+ Murphy M. T., Kacprzak G. G., Savorgnan G. A. D., Carswell R. F., 2019, MNRAS, 482, 3458
+ O’Meara J. M., et al., 2015, AJ, 150, 111
+ O’Meara J. M., Lehner N., Howk J. C., Prochaska J. X., Fox A. J., Peeples M. S., Tumlinson J., O’Shea B. W., 2017, AJ, 154, 114
+ Sánchez-Ramírez R., et al., 2016, MNRAS, 456, 4488


