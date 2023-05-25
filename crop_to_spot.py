"""
PURPOSE: Generate individual images to feed to ST-Net, when full size TIF image is too large to load into openslide.
"""

import os
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1544877984
from tqdm import tqdm
from time import time

if __name__ == "__main__":

    # patient = "HCC1"
    # patient = "HCC2"
    # patient = "HCC3"
    # patient = "HCC4"
    patient = "NCC001Post_NCC011Post"

    czi_scalefactor = {
        "HCC1": 9.995,
        "HCC2": 9.994,
        "HCC3": 9.998,
        "HCC4": 9.997,
        "NCC001Post_NCC011Post": 4.0,
    }

    spot_diameter = {
        "HCC1": 296,
        "HCC2": 296,
        "HCC3": 296,
        "HCC4": 296,
        "NCC001Post_NCC011Post": 296
    }

    patient_map = {
        "HCC1":"BC30001",
        "HCC2":"BC30002",
        "HCC3":"BC30003",
        "HCC4":"BC30004",
        "NCC001Post_NCC011Post":"BC50111",
        "NCC012Post_NCC018Post":"BC51218",
        "NCC015Post_NCC017Post":"BC51517",
        "NCC023Post_NCC037Post":"BC52337",
        "NCC023Pre_NCC034Pre"  :"BC42334",
        "NCC027Post"           :"BC50027",
        "NCC027Pre_NCC040A2Pre":"BC42740",
        "NCC037Pre_NCC040A1Pre":"BC43740",
        "NCC039Post_NCC034Post":"BC53934",
        "NCC040Post"           :"BC50040",
        "BC50111": "BC50111"
    }

    if patient[:3] == "HCC" or patient[:3] == "BC3":
        patient_type = "old_hcc"
    elif patient[:3] == "NCC" or patient[:3] == "BC5" or patient[:3] == "BC4":
        patient_type = "new_hcc"

    # for patient in tqdm(["HCC1", "HCC2", "HCC3", "HCC4"], "Patients: "):
    for patient in tqdm(["NCC001Post_NCC011Post"], "Patients: "):

        input_folder  = f"input/{patient_type}/"
        output_folder = f"output/{patient_type}/{patient_map[patient]}_cropped_images/"

        img = Image.open(f"{input_folder}{patient}_czi.tif")

        # Open tissue_positions_list.csv
        df = pd.read_csv(f"{input_folder}{patient}_tissue_positions_list.csv", header=None, index_col=0)
        # Filter only spots in tissue
        # Refer to https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/spatial, search for "in_tissue" for more info
        df = df[df[1] == 1]
        # print(df)

        # Create individual lists for coordinates and pixel coordinates
        x = np.array(df[df.columns[2]].to_list())
        y = np.array(df[df.columns[1]].to_list())
        x_px = np.array(df[df.columns[4]].to_list())
        y_px = np.array(df[df.columns[3]].to_list())

        # Scale pixel coordinates by scalefactor
        x_px = x_px * czi_scalefactor[patient]
        y_px = y_px * czi_scalefactor[patient]
        
        # Get spot size
        window = spot_diameter[patient]
        half_window = round(window/2, 3)

        # Verify number of spots
        # print(f"{len(x) = }")
        # print(f"{len(y) = }")
        # print(f"{len(x_px) = }")
        # print(f"{len(y_px) = }")
        # print(f"{x = }")

        per_spot = 0

        # Loop over all spots
        for i in tqdm(range(len(x)), "Spots: "):

            s = time()

            this_x = x[i]
            this_y = y[i]
            this_x_px = x_px[i]
            this_y_px = y_px[i]

            # (left, upper, right, lower)
            left  = this_x_px - half_window
            upper = this_y_px - half_window
            right = this_x_px + half_window
            lower = this_y_px + half_window

            cropped = img.crop((left, upper, right, lower))
            pathlib.Path(os.path.join(os.getcwd(), "output", patient_type, f"{patient_map[patient]}_cropped_images")).mkdir(parents=True, exist_ok=True)
            cropped.save(f"{output_folder}{patient_map[patient]}_{this_x}_{this_y}.tif")

            duration = time() - s
            per_spot += duration

        print(f"Took {per_spot / len(x):.3f}s per spot on average.")

        # # npz = np.load(f"{folder}C1_11_11.npz", allow_pickle=True)
        # # npz = np.load(f"{folder}C1_11_11_nonCZI.npz", allow_pickle=True)
        # npz = np.load(f"{folder}C1_12_8_nonCZI.npz", allow_pickle=True)
        # print(npz["index"])
        # print(npz["pixel"])
        # print(npz["patient"])