#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import argcomplete

parser = argparse.ArgumentParser()
parser.add_argument("--patient", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--corr_root", default="correlations/", type=str)

argcomplete.autocomplete(parser)
args = parser.parse_args()

import os
import json
import scipy.stats
import numpy as np
import pandas as pd
import pathlib

gene_order_dict = {
        "ALB": 1,
        "SERPINA1": 2,
        "APOA1": 3,
        "APOC1": 4,
        "IGKC": 5,
        "APOA2": 6,
        "FGB": 7,
        "FGA": 8,
        "C1S": 9,
        "CD74": 10,
        "MT2A": 11
    }

corr_json = f"{args.corr_root}{args.patient}_{args.model}-model_correlations.json"

for gene in gene_order_dict.keys():
    print(gene)