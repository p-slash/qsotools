from setuptools import setup
import os
binscripts = [os.path.join("bin", f) for f in os.listdir("bin") if f.endswith(".py")]

setup(
    name="qsotools",
    version="1.1",
    packages=['qsotools'],
    package_dir={'': 'py/'},
    package_data={"qsotools": ["tables/*"]},
    include_package_data=True,
    scripts=binscripts,

    # install_requires=["docutils>=0.3"],

    # metadata to display on PyPI
    author = "Naim Goksel Karacayli",
    author_email = "naimgoksel.karacayli@yale.edu",
    description=("Python scripts to generate mock Lyman-alpha forest,"
    " read & reduce quasar spectra from KODIAQ, XQ-100 & SQUAD."),
    url="https://bitbucket.org/naimgk/qsotools",   # project home page, if any

    # could also include long_description, download_url, etc.
)
