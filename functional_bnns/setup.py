### ~~~
## ~~~ From https://github.com/maet3608/minimal-setup-py/blob/master/setup.py
### ~~~

from setuptools import setup, find_packages


#
# ~~ Load the a .txt file, into a list of strings (each line is a string in the list)
def txt_to_list(filepath):
    with open(filepath, "r") as f:
        return [line.strip() for line in f]


#
# ~~~ Install
setup(
    name="bnns",
    version="0.7.0",
    author="Thomas Winckelman",
    author_email="winckelman@tamu.edu",
    description="Package intended for testing (but not optimized for deploying) varions BNN algorithms",
    packages=find_packages(),
    install_requires=txt_to_list(
        "requirements.txt"
    ),  # ~~~ assuming, of course, that "requirements.txt" is in the same directory as this file
)
