# patch_size=30
patch_size=74
# patch_size=224

# gene=C1S
# gene=CD74
# gene=MT2A
# gene=MT-CO2
# gene=HLA-A
# gene=HLA-B

# for patient in "BC50027"
for patient in "BC50040"
# for patient in "BC30001" "BC30002" "BC30003" "BC30004"
do
    for gene in "C1S" "CD74" "MT2A" "MT-CO2" "HLA-A" "HLA-B" "ALB" "SERPINA1" "APOA1" "APOC1" "IGKC" "APOA2" "FGB" "FGA"
    # for gene in "ALB"
    do
        bin/visualize.py output/densenet121_${patch_size}/top_250/${patient}_cv.npz --gene ${gene}
    done
done