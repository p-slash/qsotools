Overview
=====
This is the source code behind papers.

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

Installation
=====
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
