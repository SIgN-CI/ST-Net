from time import time
import numpy as np
import tifffile

# scene = 1
# scene = 2
# scene = 3
# scene = 4

patient = "BC50111"

s = time()
# Load the OME-TIFF image
ome_image = tifffile.imread(f"{patient}.ome.tif")
e = time()
d = e - s
print(f"Took {d:.3f}s to open OME-TIFF image.")

# print(type(ome_image))
# print(ome_image.shape)

s = time()
# Save the 2D TIFF image
tifffile.imwrite(f"{patient}.tif", ome_image)
e = time()
d = e - s
print(f"Took {d:.3f}s to save as TIF image.")