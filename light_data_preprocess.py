# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:30:01 2018

@author: mark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("CSV/light/light2017.csv")
df = df.dropna(axis=1)
ts = df.any()
df = df[ts[ts.values].index]
