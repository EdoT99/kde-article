import pandas as pd

import numpy as np

import os

import argparse

import glob

from definitive_cv_method_10 import dataset_parser_def, occurrence_def

from collections import Counter

import math





def bs_score(
        freq:list
        )-> float:
    
    suma = sum(freq)
    base = len(freq)
    prob = [el/suma for el in freq]
    entropy = - sum([(p*(math.log(p,base))) for p in prob ])
    balance_score = 1-entropy

    return round(balance_score,3)


fodler_datasets = os.path.join('datasets/','*')
print(fodler_datasets)
results_df = []

max_class = 7

for dataset in glob.glob(fodler_datasets):
    print(dataset)
    name = os.path.basename(dataset)

    title = name[:-4]
            #table_name = name[name.find('_')+1:-4]
    table_name = title
    print('─' * 100)
    print("NEW DATASET: ",title)
    dataset = pd.read_csv(dataset)
         
    x,y,count = dataset_parser_def(dataset)
    classes = len(np.unique(y))
    print(classes)
    d = Counter(y)

    freq = []
    lista, ir, max_value = occurrence_def(y)
    for item in lista:
        k = item[0]
        if k in d.keys():
            val = d[k]
            freq.append(val)
    #print(ir)
    bs = bs_score(freq)
    print(bs)
    if len(freq) < max_class:
        diff = max_class - len(freq)

    for i in range(diff):
        freq.append('-')
    print(freq)

    imb_ratios = ""
    imb_ratios = " : ".join(str(item[1]) for item in ir)
    print(imb_ratios)

    row = pd.Series({
                    "Name":title,
                    "Attributes": count,
                    "A": freq[0],
                    "B":freq[1],
                    "C":freq[2],
                    "D":freq[3],
                    "E":freq[4],
                    "F":freq[5],
                    "G":freq[6],
                    "Imbalance Ratio": imb_ratios,
                    "BS": bs
                }).to_frame().T

    results_df.append(row)


dataframe = pd.concat(results_df)
print(dataframe)

output_details = os.path.join('article_related/','datasets_details.csv')
dataframe.to_csv(output_details, index=False, sep=',', decimal='.')
