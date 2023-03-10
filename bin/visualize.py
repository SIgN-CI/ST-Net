#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import argcomplete

parser = argparse.ArgumentParser()
parser.add_argument("filenames", type=str, nargs="+")
parser.add_argument("--gene", type=str, nargs="?", default="FASN")
parser.add_argument("--figroot", default="fig/visualize/", type=str)
parser.add_argument("--output_extension", default="pdf", type=str)
parser.add_argument("--figure_spot_size", default=88, type=int)
parser.add_argument("--title_font_size", default=14, type=int)

argcomplete.autocomplete(parser)
args = parser.parse_args()

import sys
sys.path.append(".")
import stnet
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import pathlib
import scipy.stats
from sklearn.decomposition import PCA

stnet.utils.util.latexify()
cmap = plt.get_cmap("viridis")
pathlib.Path(os.path.dirname(args.figroot)).mkdir(parents=True, exist_ok=True)
dataset = stnet.datasets.Spatial()

"""(1) Generates legend figure"""
fig = plt.figure(figsize=(1.00, 0.50))
cmap_bin = plt.get_cmap("binary")
plt.scatter([-1], [-1], color=cmap_bin(1 - 1e-5), label="Tumor",  s=10, linewidth=0.2, edgecolors="k")
plt.scatter([-1], [-1], color=cmap_bin(1e-5),     label="Normal", s=10, linewidth=0.2, edgecolors="k")
plt.axis([0, 1, 0, 1])
plt.gca().axis("off")
plt.gca().set_aspect("equal")
plt.xticks([])
plt.yticks([])
plt.legend(loc="center")
plt.tight_layout()
# fig.savefig(args.figroot + "tumor_legend.pdf")
plt.close(fig)

fig = plt.figure(figsize=(2.25, 0.75))
norm = matplotlib.colors.Normalize(vmin=-3, vmax=3)
cb = matplotlib.colorbar.ColorbarBase(plt.gca(), cmap=cmap,
                                      norm=norm,
                                      orientation='horizontal')
cb.set_ticks(range(-3, 4))
cb.set_label("Standard deviations from mean")
plt.tight_layout()
# fig.savefig(args.figroot + "colorbar.pdf")
plt.close(fig)

task = None
gene_names = None
patient = []
section = []
tumor = []
count = []
pred = []
pixel = []

# print(f"{args.filenames = }")

for f in args.filenames:
    data = np.load(f)

    if task is not None:
        assert(task == data["task"])
    task = data["task"]
    # print(f"{task = }")

    if gene_names is not None:
        assert((gene_names == data["gene_names"]).all())
    gene_names = data["gene_names"]
    # print(f"{len(gene_names) = }")

    patient.append(data["patient"])
    section.append(data["section"])
    tumor.append(data["tumor"])
    count.append(data["counts"])
    pred.append(data["predictions"])
    pixel.append(data["pixel"])

patient = np.concatenate(patient)
section = np.concatenate(section)
ps = list(zip(patient, section))
tumor = np.concatenate(tumor)
count = np.concatenate(count)
pred = np.concatenate(pred)
pixel = np.concatenate(pixel)

if pred.shape[0] == count.shape[0] and pred.shape[1] != count.shape[1]:
    # For total count prediction
    count = np.sum(count, 1, keepdims=True)
    gene_names = ["count"]

if task == "gene":
    # pca = PCA(n_components=3)
    # pca.fit(count)
    # print("PCA Explained Variance: {}".format(pca.explained_variance_ratio_))
    # print(count.shape)
    # c = pca.transform(count)
    # p = pca.transform(pred)
    index = np.argmax(gene_names == args.gene)
    c = count[:, index]
    p = pred[:, index]
    this_gene = gene_names[index]
    print(f"{this_gene = }")

    # with open(f"/home/tester/bokleong/ST-Net_custom/custom_fig/genes.txt","w") as f:
    #     f.write(f"{len(gene_names) = }\n")
    #     f.write(f"{gene_names}\n")

    if not this_gene == args.gene:
        print(f"Gene name mismatch. Skipping {args.gene}...")
        raise SystemExit(0)

for (patient, section) in sorted(set(ps)):
    tol = 1e-5
    mask = np.array([((p == patient) and (s == section)) for (p, s) in ps])
    margin = 112 + 50
    xmin, xmax, ymin, ymax = min(pixel[mask, 0]), max(pixel[mask, 0]), min(pixel[mask, 1]), max(pixel[mask, 1])

    # TODO: check aspect ratio
    # TODO: remove box/axis 
    print(patient, section, flush=True)
    # image_filename = os.path.join(stnet.config.SPATIAL_RAW_ROOT, dataset.subtype[patient], patient, "{}_{}.tif".format(patient, section))
    image_filename = os.path.join(stnet.config.SPATIAL_RAW_ROOT, "HE_{}_{}.tif".format(patient, section))
    image = plt.imread(image_filename)

    xsize = xmax - xmin + 2 * margin
    ysize = ymax - ymin + 2 * margin
    # ~
    # figsize = (0.00017 * xsize, 0.00017 * ysize)
    figsize = (0.0017 * xsize, 0.0017 * ysize)

    fig = plt.figure(figsize=figsize)
    plt.imshow(image, aspect="equal", interpolation="nearest")
    fig.patch.set_visible(False)
    plt.gca().axis("off")
    plt.gca().set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    x = xmin + 300
    y = ymax - 300
    plt.gca().annotate("1 mm", (x + 743, y - 200), fontsize=5, ha="center")
    plt.plot([x, x + 1486],        [y, y],             color="black", linewidth=0.3)
    plt.plot([x, x],               [y - 100, y + 100], color="black", linewidth=0.3)
    plt.plot([x + 1486, x + 1486], [y - 100, y + 100], color="black", linewidth=0.3)
    plt.axis([xmin - margin, xmax + margin, ymin - margin, ymax + margin])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # ~
    # fig.savefig("{}{}_{}.jpg".format(args.figroot, patient, section), quality=100, dpi=600)
    # fig.savefig("{}{}_{}.pdf".format(args.figroot, patient, section), quality=100, dpi=600)
    # fig.savefig("{}{}_{}.jpg".format(args.figroot, patient, section), dpi=600)
    # fig.savefig("{}{}_{}.pdf".format(args.figroot, patient, section), dpi=600)
    plt.close(fig)

    assert(task == "gene")

    # coef, _ = scipy.stats.pearsonr(c[mask], p[mask])
    # print(coef)
    # coef, _ = scipy.stats.pearsonr(c, p)
    # print(coef)
    # print("tumor: ", sum(tumor[mask, 0]) / sum(mask))
    
    # Calculating spearman's correlation
    spearmanr, pval = scipy.stats.spearmanr(c,p)
    if pval < 0.00001:
        pval = "<0.00001"

    # ~
    # visualize_spot_size = 16
    # visualize_spot_size = 18
    # visualize_spot_size = 88
    # visualize_spot_size = 500
    visualize_spot_size = args.figure_spot_size

    # ~
    # title_font_size = 14
    title_font_size = args.title_font_size

    hcc_patients = {"BC30001":"HCC1",
                    "BC30002":"HCC2",
                    "BC30003":"HCC3",
                    "BC30004":"HCC4"}

    y90_patients = {"BC50027":"NCC027Post"}

    if patient in hcc_patients.keys():
        gt_title = f"{hcc_patients[patient]} {args.gene} Ground Truth"
        pred_title = f"{hcc_patients[patient]} {args.gene} Prediction"
        visualize_spot_size = 18
    elif patient in y90_patients.keys():
        gt_title = f"{y90_patients[patient]} {args.gene} Ground Truth"
        pred_title = f"{y90_patients[patient]} {args.gene} Prediction"
        visualize_spot_size = 88
    else:
        gt_title = f"{patient} {section} {args.gene} Ground Truth"
        pred_title = f"{patient} {section} {args.gene} Prediction"

    """Ground Truth"""
    value = c[mask]
    value = ((value - value.mean(0)) / (3 * value.std(0) + tol) + 0.5).clip(tol, 1 - tol)
    fig = plt.figure(figsize=figsize)
    # ~
    # plt.scatter(pixel[mask, 0], pixel[mask, 1], color=list(map(cmap, value)), s=2, linewidth=0, edgecolors="none")
    # plt.scatter(pixel[mask, 0], pixel[mask, 1], color=list(map(cmap, value)), s=10, linewidth=0, edgecolors="none")
    plt.scatter(pixel[mask, 0], pixel[mask, 1], color=list(map(cmap, value)), s=visualize_spot_size, linewidth=0, edgecolors="none")
    plt.gca().axis("off")
    plt.gca().set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    plt.axis([xmin - margin, xmax + margin, ymin - margin, ymax + margin])
    plt.gca().invert_yaxis()
    
    plt.title(f"{gt_title}\n",fontdict={'fontsize':title_font_size})
    plt.tight_layout()
    # fig.savefig("{}{}_{}_{}.pdf".format(args.figroot, patient, section, args.gene))
    fig.savefig(f"{args.figroot}{patient}_{args.gene}.{args.output_extension}")
    print(f"Saved \"{gt_title}\".")
    plt.close(fig)

    """Prediction"""
    value = p[mask]
    value = ((value - value.mean(0)) / (3 * value.std(0) + tol) + 0.5).clip(tol, 1 - tol)
    fig = plt.figure(figsize=figsize)
    # ~
    # plt.scatter(pixel[mask, 0], pixel[mask, 1], color=list(map(cmap, value)), s=2, linewidth=0, edgecolors="none")
    # plt.scatter(pixel[mask, 0], pixel[mask, 1], color=list(map(cmap, value)), s=10, linewidth=0, edgecolors="none")
    plt.scatter(pixel[mask, 0], pixel[mask, 1], color=list(map(cmap, value)), s=visualize_spot_size, linewidth=0, edgecolors="none")
    plt.gca().axis("off")
    plt.gca().set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    plt.axis([xmin - margin, xmax + margin, ymin - margin, ymax + margin])
    plt.gca().invert_yaxis()
    try:
        plt.title(f"{pred_title}\nSpearman's Correlation = {np.round(spearmanr,5)} [{np.round(pval,5)}]",fontdict={'fontsize':title_font_size})
    except Exception as e:
        print(e)
        plt.title(f"{pred_title}\nSpearman's Correlation = {np.round(spearmanr,5)} [{pval}]",fontdict={'fontsize':title_font_size})
    plt.tight_layout()
    # fig.savefig("{}{}_{}_{}_{}.pdf".format(args.figroot, patient, section, args.gene, "pred"))
    # fig.savefig("{}{}_{}_{}.pdf".format(args.figroot, patient, args.gene, "pred"))
    fig.savefig(f"{args.figroot}{patient}_{args.gene}_pred.{args.output_extension}")
    print(f"Saved \"{pred_title}\".")
    plt.close(fig)

    """Tumor annotation"""
    value = tumor[mask, 0].clip(tol, 1 - tol)
    fig = plt.figure(figsize=figsize)
    cmap_bin = plt.get_cmap("binary")
    plt.scatter(pixel[mask, 0], pixel[mask, 1], color=list(map(cmap_bin, value)), s=10, linewidth=0.2, edgecolors="k")
    plt.gca().axis("off")
    plt.gca().set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    plt.axis([xmin - margin, xmax + margin, ymin - margin, ymax + margin])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # fig.savefig("{}{}_{}_{}.pdf".format(args.figroot, patient, section, "tumor"))
    plt.close(fig)
