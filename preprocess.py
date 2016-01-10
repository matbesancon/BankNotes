# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:20:02 2015

@author: MATHIEU
"""

# source : https://archive.ics.uci.edu/ml/datasets/banknote+authentication

# Variables 
#1. variance of Wavelet Transformed image (continuous)
#2. skewness of Wavelet Transformed image (continuous)
#3. kurtosis of Wavelet Transformed image (continuous)
#4. entropy of image (continuous)
#5. class (integer)


# Loading librairies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ggplot
import scipy.stats as stats
import scipy.spatial.distance
from collections import Counter
import urllib3

# Importing data from the UCI repository

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
#http = urllib3.PoolManager()
#r = http.request('GET',url)
#
#with open("data_banknote_authentication.txt",'wb') as f:
#    f.write(r.data)
## disconnect
#r.release_conn()

# import data 
data0 = pd.read_csv("data_banknote_authentication.txt",
                    names=["vari","skew","kurtosis","entropy","class"])