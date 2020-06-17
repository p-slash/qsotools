# Do not use distutils try setuptools
# from distutils.core import setup
import os
binscripts = [os.path.join("bin", f) for f in os.listdir("bin") if f.endswith(".py")]

setup(name = "qsotools",
    version = "1.0",
    description = ("Python scripts to generate mock Lyman-alpha forest,"
    " read & reduce quasar spectra from KODIAQ, XQ-100 & UVES."),
    author = "Naim Goksel Karacayli",
    author_email = "naimgoksel.karacayli@yale.edu",
    packages=['qsotools'],
    scripts=binscripts,
    package_dir={'qsotools': 'py/qsotools'},
    package_data={'qsotools': ['tables/*']},
    )