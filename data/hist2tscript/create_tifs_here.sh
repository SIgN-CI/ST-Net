root="data/hist2tscript"

# for i in ./*.jpg;
# for i in ./HE_BC4*.jpg;
# for i in ${root}/HE_BC3*.jpg;
for i in ./HE_BC*.jpg;
do
    echo ${i}
    convert ${i} -define tiff:tile-geometry=256x256 ${i%.jpg}.tif
done