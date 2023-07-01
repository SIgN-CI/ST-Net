```convert_ome_to_tif.py```
---------------------------
Part of workflow to extract full-res TIF file from CZI image. 

**Input:** OME-TIFF image

**Output:** TIF image

Full workflow:
1. Open CZI in QuPath and select relevant scene.
2. Export scene as OME-TIFF (File > Export images > OME TIFF).
3. Run `convert_ome_to_tif.py` to convert OME-TIFF image to TIF image.
4. Use TIF image as desired (E.g., tile into individual spots using `crop_to_spot.py` below.)

```crop_to_spot.py```
---------------------------
Tiles TIF image to individual spots

**Inputs:**
1. TIF image (OME-TIFF doesn't work as far as I can tell)
2. `tissue_positions_list.csv` in Visium data format (CTRL-F 'tissue_positions.csv' in https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/spatial to view data format.)

**Outputs:** Individual TIF images corresponding to each in-tissue spot inside new directory corresponding to patient name. Patient name may have to be added to patient map dictionary.