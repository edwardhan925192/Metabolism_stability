# -*- coding: utf-8 -*-
"""data_prep_main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19NeFY4LdUE0JCQGcMN8ukYgiZvp6yS8x
"""

from mol_features import data_prep
from maccs_features import generate_and_concatenate_MACCS_keys
import pandas as pd

train = pd.read_csv('/train.csv')
test = pd.read_csv('/test.csv')

train.loc[2796, 'AlogP'] = train.loc[2796, 'LogD']
train.loc[3387, 'AlogP'] = train.loc[3387, 'LogD']
test.loc[10, 'AlogP'] = test.loc[10, 'LogD']

trainf = data_prep(train)
testf = data_prep(test)

generate_and_concatenate_MACCS_keys(trainf, drop_columns=['HLM'])
generate_and_concatenate_MACCS_keys(testf)