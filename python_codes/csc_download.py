"""Python program to download CSC spectra."""


import os
import subprocess
import numpy as np


def get_src_obsid_fromtable(csc_table, skiprows=80):
