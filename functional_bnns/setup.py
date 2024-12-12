
### ~~~
## ~~~ From https://github.com/maet3608/minimal-setup-py/blob/master/setup.py
### ~~~ 

import warnings
import platform
import os
from setuptools import setup, find_packages



### ~~~
## ~~ Load the a .txt file into a list of strings (each line is a string in the list)
### ~~~

#
# ~~ Load the a .txt file, into a list of strings (each line is a string in the list)
def txt_to_list(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f]

requirements = txt_to_list("requirements.txt")



### ~~~
## ~~~ Prompt the user about pytorch
### ~~~

#
# ~~~ Set up colored printing in the console
if platform.system()=="Windows":    # platform.system() returns the OS python is running o0n | see https://stackoverflow.com/q/1854/11595884
    os.system("color")              # Makes ANSI codes work | see Szabolcs' comment on https://stackoverflow.com/a/15170325/11595884

class bcolors:                      # https://stackoverflow.com/a/287944/11595884
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#
# ~~~ Prompt the user to install pytorch, if the requirent is not already installed
awaiting_response = True
while awaiting_response:
    try:
        import torch
        awaiting_response = False
    except ModuleNotFoundError:
        #
        # ~~~ If pytorch is not already installed, ask the user whether they're ok with a non-CUDA version being installed automatically on their behalf?
        Y = bcolors.OKGREEN + "Y" + bcolors.HEADER
        N = bcolors.FAIL    + "N" + bcolors.HEADER
        response = input( bcolors.HEADER + f"Proceed with automatic installation of non-CUDA PyTorch? (please type {Y}/{N})\n    {Y}: yes, install non-CUDA pytorch.\n    {N}: no, I will install my preferred version of pytorch manually.\n" + bcolors.ENDC )
        if response == "Y":
            requirements.append("torch")
            awaiting_response = False
        elif response == "N":
            awaiting_response = False
        else:
            print(f"Invalid response: {response}. Please type {Y}/{N}.")

#
# ~~~ Install
setup(
    name = 'bnns',
    version = '0.7.0',
    author = 'Thomas Winckelman',
    author_email = 'winckelman@tamu.edu',
    description = 'Package intended for testing (but not optimized for deploying) varions BNN algorithms',
    packages = find_packages(),    
    install_requires = requirements  # ~~~ assuming, of course, that "requirements.txt" is in the same directory as this file
)

if response=="N":
    warnings.warn("In order to use this package, please install PyTorch manually. See https://pytorch.org/get-started/locally/ for instructions.")