
### ~~~
## ~~~ From https://github.com/maet3608/minimal-setup-py/blob/master/setup.py
### ~~~ 

from setuptools import setup, find_packages

setup(
    name = 'bnns',
    version = '0.7.0',
    author = 'Thomas Winckelman',
    author_email = 'winckelman@tamu.edu',
    description = 'Package intended for testing (but not optimized for deploying) varions BNN algorithms',
    packages = find_packages(),    
    install_requires = [
        "fiona",    # ~~~ for loading data in .shp format
        "quality_of_life @ git+https://github.com/ThomasLastName/quality-of-life.git"
    ]
)