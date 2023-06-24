#!/usr/bin/env python3
"""
Calculate Spearman's correlation for ST-Net predictions
"""
import os
import argparse
import argcomplete
parser = argparse.ArgumentParser()
parser.add_argument("output_dir", type=str, nargs="+")
parser.add_argument("--epoch", type=str, default="cv")
parser.add_argument("--plotname", type=str, default="boxplot")
parser.add_argument("--plotlims", type=float, nargs=2, default=[0.0,0.0])
argcomplete.autocomplete(parser)
args = parser.parse_args()
import sys
sys.path.append(".")
import stnet
import pathlib
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.ticker import MultipleLocator

if __name__ == "__main__":

    # print(f"{args.epoch = }")
    npzs = glob(f"{args.output_dir[0]}/*_{args.epoch}.npz",recursive=True)
    # Sort npzs found by glob, so that boxplot is logical and pleasant
    npzs.sort(key=lambda x: int(os.path.basename(x).split(f"_{args.epoch}.")[0][2:]))
    # print(f"{args.output_dir[0] = }")
    # print(f"{npzs = }")
    # print(f"{len(npzs) = }")
    patient_map = {"BC30001":"COVID_HCC1",
                   "BC30002":"COVID_HCC2",
                   "BC30003":"COVID_HCC3",
                   "BC30004":"COVID_HCC4",
                   "BC30005":"COVID_HCC1",
                   "BC30006":"COVID_HCC2",
                   "BC30007":"COVID_HCC3",
                   "BC30010":"TCGA_LIHC0",
                   "BC30011":"TCGA_LIHC1",
                   "BC30012":"TCHA_LIHC2",
                   "BC30013":"TCGA_LIHC3",
                   "BC30014":"TCGA_LIHC4",
                   "BC30015":"TCGA_LIHC5",
                   'BC30020': 'TCGA_LIHC0_rgb', 'BC30021': 'TCGA_LIHC1_rgb', 'BC30022': 'TCGA_LIHC2_macenko', 'BC30023': 'TCGA_LIHC3_reinhard', 'BC30024': 'TCGA_LIHC4_rgb', 'BC30025': 'TCGA_LIHC5_rgb', 'BC30030': 'TCGA_LIHC0_rgb', 'BC30031': 'TCGA_LIHC1_rgb', 'BC30032': 'TCGA_LIHC2_macenko', 'BC30033': 'TCGA_LIHC3_reinhard', 'BC30034': 'TCGA_LIHC4_rgb', 'BC30035': 'TCGA_LIHC5_rgb', 'BC30040': 'TCGA_LIHC0_rgb', 'BC30041': 'TCGA_LIHC1_rgb', 'BC30042': 'TCGA_LIHC2_macenko', 'BC30043': 'TCGA_LIHC3_reinhard', 'BC30044': 'TCGA_LIHC4_rgb', 'BC30045': 'TCGA_LIHC5_rgb',
                   "BC50111":"NCC001Post_NCC011Post",
                   "BC51218":"NCC012Post_NCC018Post",
                   "BC51517":"NCC015Post_NCC017Post",
                   "BC52337":"NCC023Post_NCC037Post",
                   "BC42334":"NCC023Pre_NCC034Pre"  ,
                   "BC50027":"NCC027Post"           ,
                   "BC42740":"NCC027Pre_NCC040A2Pre",
                   "BC43740":"NCC037Pre_NCC040A1Pre",
                   "BC53934":"NCC039Post_NCC034Post",
                   "BC50040":"NCC040Post"           }

    vals = {}
    for npz in npzs:
        
        data = np.load(npz,allow_pickle=True)
        patient     = patient_map[data["patient"][0]]
        counts      = data["counts"]
        predictions = data["predictions"]
        gene_names  = data["gene_names"]
        this_npz_vals = []

        for gene in gene_names:
            index = np.argmax(gene_names == gene)
            # print(f"{index = }")
            c = counts[:,index]
            p = predictions[:,index]
            spearmanr, pval = scipy.stats.spearmanr(c,p)
            
            # Only append 5% significant correlations
            if pval < 0.05:
                this_npz_vals.append(spearmanr)

        vals[patient] = this_npz_vals

    # print(vals)
    # print(len(vals))
    # print(len(vals[0]))
    # print(len(vals[1]))

    title_size = 20
    label_size = 16
    fig,ax = plt.subplots(1,1,figsize=(12,8))
    # print(f"{vals = }")
    ax.boxplot(vals.values(),
               labels=vals.keys(),
               patch_artist=True)
    model_testpatient = patient_map[args.plotname.split('_')[0]]
    ax.set_title(f"Spearman's Correlations for inference using\n{model_testpatient}-trained model\n",fontsize=title_size)
    ax.set_xlabel(f"\nPatients",fontsize=label_size)
    ax.set_xticklabels(vals.keys(),rotation=45,ha='right')
    ax.set_ylabel(f"Spearman's Correlation\n",fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=label_size)
    ax.minorticks_on()
    ax.grid(which="minor",axis='y',linewidth=0.3,linestyle='--')
    ax.grid(which="major",axis='y',linewidth=1.2)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.axhline(y=0,color='r')
    
    if not args.plotlims == [0.0,0.0]:
        print(f"{args.plotlims = }")
        lower = args.plotlims[0]
        upper = args.plotlims[1]
        ax.set_ybound(lower=lower,upper=upper)

    # plt.tight_layout()

    # figroot = args.output_dir[0].split('/')[:-3]
    # figroot = '/'.join(figroot)
    # figroot = f"{figroot}/fig/visualize/"
    pathlib.Path(os.path.dirname(args.output_dir[0])).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{args.output_dir[0]}/{args.plotname}.png",dpi=300, bbox_inches="tight")
    print(f"Saved to {args.output_dir[0]}/{args.plotname}.png")
