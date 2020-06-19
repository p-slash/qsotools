from setuptools import setup, find_packages
import os
binscripts = [os.path.join("bin", f) for f in os.listdir("bin") if f.endswith(".py")]

setup(
    name="qsotools",
    version="1.0",
    packages=find_packages(),
    scripts=binscripts,

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    # install_requires=["docutils>=0.3"],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
    },

    # metadata to display on PyPI
    author = "Naim Goksel Karacayli",
    author_email = "naimgoksel.karacayli@yale.edu",
    description=("Python scripts to generate mock Lyman-alpha forest,"
    " read & reduce quasar spectra from KODIAQ, XQ-100 & UVES."),
    url="https://bitbucket.org/naimgk/qsotools",   # project home page, if any

    # could also include long_description, download_url, etc.
)
