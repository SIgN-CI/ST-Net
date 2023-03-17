#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import argcomplete

parser = argparse.ArgumentParser()
parser.add_argument("--patient", type=str)
parser.add_argument("--genes", nargs="+", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--inference_type", type=str)
parser.add_argument("--corr_root", default="correlations/", type=str)

argcomplete.autocomplete(parser)
args = parser.parse_args()

import os
import json
import scipy.stats
import numpy as np
import pathlib

root = f"output_inference/{args.inference_type}/{args.model}_model"
npz = f"{root}/{args.patient}_1.npz"
data = np.load(npz)
gene_names = data["gene_names"]

count = []
pred = []
count.append(data["counts"])
pred.append(data["predictions"])

corr = {}
for gene in args.genes:

    index = np.argmax(gene_names == gene)
    # print(f"count=\n{count}")
    # print(f"{index = }")
    # print(f"{type(index) = }")
    c = count[0][:, index]
    p = pred[0][:, index]

    spearmanr, pval = scipy.stats.spearmanr(c,p)
    spearmanr = np.round(spearmanr, 5)
    if pval < 0.001:
        pval = "<0.001"
    else:
        pval = np.round(pval, 3)

    corr[gene] = [spearmanr, pval]

pathlib.Path(os.path.dirname(f"{args.corr_root}")).mkdir(parents=True, exist_ok=True)
with open(f"{args.corr_root}{args.patient}_{args.model}-model_correlations.json", "w") as f:
    json.dump(corr, f, indent=4)
