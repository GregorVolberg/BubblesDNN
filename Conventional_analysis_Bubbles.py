#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:24:15 2023

@author: vog20246
"""

# additional imports
import os
from pathlib import Path
import pandas as pd

# read protocol file
rawpath    = Path('./dsetComposite/')
fname      = 'BubblesProtocolComposite.txt'
allfiles   = pd.read_csv(os.path.join(rawpath, fname), sep="\t", header=None)

len(pd.Series.unique(allfiles.iloc[:,1]))

# Idee: RDM for all versus all participants, with svm 