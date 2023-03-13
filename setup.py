from setuptools import setup, find_namespace_packages
import os

binscripts = [os.path.join("bin", f) for f in os.listdir("bin")
              if not f.startswith(".")]
with open("requirements.txt") as file_reqs:
    requirements = file_reqs.read().splitlines()

setup(
    name="qsotools",
    version="1.2",
    packages=find_namespace_packages(where='py'),
    package_dir={'': 'py/'},
    scripts=binscripts,
    install_requires=requirements,
    package_data={"qsotools": ["tables/*"]},
    include_package_data=True,
    author="Naim Goksel Karacayli",
    author_email="ngokselk@gmail.com",
    description=(
        "Python scripts to generate mock Lyman-alpha forest,"
        " read & reduce quasar spectra from KODIAQ, XQ-100 & SQUAD."),
    url="https://bitbucket.org/naimgk/qsotools"
)
