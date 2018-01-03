# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:30:01 2018

@author: mark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def remove_na_zero(df):
    df = df.dropna(axis=1)
    ts = df.any()
    df = df[ts[ts.values].index]
    return df

df = pd.read_csv("CSV/light/light2017.csv")
df = remove_na_zero(df)

features = ["V", "A", "PF", "W"]
plot_num = len(features) * 100 + 10
for i in range(len(features)):
    train_series = pd.Series(df[features[i]].values, name=features[i], 
                             index=pd.to_datetime(df['REPORTTIME'])).sort_index()
    plt.subplot(plot_num + i + 1)
    plt.title(features[i])
    plt.plot(train_series['2017-03-30':'2017-05-01'])
plt.tight_layout()
plt.show()

plot_acf(train_series)
