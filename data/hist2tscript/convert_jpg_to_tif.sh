for i in ./HE_BC*.jpg;
do
    echo ${i}
    convert ${i} -define tiff:tile-geometry=256x256 ${i%.jpg}.tif
done