# import and install
import pip

def import_or_install(packages):
  for package in packages:
      try:
          __import__(package)
      except ImportError:
          pip.main(['install', package]) 
PACKAGES = ['os', 'blitz-bayesian-pytorch']
import_or_install(PACKAGES)

# dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import time
import blitz
from blitz.utils import variational_estimator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

### globals

dic_country = \
{'AUS': 'Australia',
 'BRA': 'Brazil',
 'CAN': 'Canada',
 'CHN': 'China',
 'GBR': 'United Kingdom',
 'IND': 'India',
 'JPN': 'Japan',
 'SGP': 'Singapore',
 'USA': 'United States'}
 
### Load Data
def get_sheet_by_methods(data, method_num, verbose = 0):
  sheet_names = data.sheet_names
  sheet_imputation_map = pd.DataFrame([[s, s[:3].strip(),s[3:]] for s in sheet_names], columns = ['sheet_name', 'country_code', 'imputation method'])
  methods = sheet_imputation_map['imputation method'].unique()
  if verbose:
    print('methods tried:',methods, len(methods))
  return list(sheet_imputation_map.loc[sheet_imputation_map['imputation method'] == methods[method_num]]['sheet_name'].values)