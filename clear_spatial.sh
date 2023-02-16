echo "Removing hist2tscript-patch directory..."
rm -rf data/hist2tscript-patch

for file in data/hist2tscript/*.pkl
do
    echo "Removing ${file}..."
    rm ${file}
done

echo ""
for file in data/hist2tscript
do
    echo ${file}
done