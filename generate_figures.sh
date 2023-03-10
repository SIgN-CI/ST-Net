# patch_size=30
# patch_size=74
# patch_size=224
patch_size=296

## Select the testpatient of the model you wish to use
# model_testpatient=BC50027
# model_testpatient=BC50040
# model_testpatient=BC50111
# model_testpatient=BC51218
# model_testpatient=BC51517
# model_testpatient=BC52337
model_testpatient=BC53934

# gene=C1S
# gene=CD74
# gene=MT2A
# gene=MT-CO2
# gene=HLA-A
# gene=HLA-B

# for patient in "BC50027"
# for patient in "BC50040"
# for patient in "BC30001" "BC30002" "BC30003" "BC30004"
for patient in "BC42740"
do
    for gene in "C1S" "CD74" "MT2A" "ALB" "SERPINA1" "APOA1" "APOC1" "IGKC" "APOA2" "FGB" "FGA"
    # for gene in "ALB"
    do
        bin/visualize.py output_inference/densenet121_${patch_size}/top_250/${model_testpatient}_model/${patient}_1.npz --gene ${gene} --figure_spot_size 1500 --title_font_size 80
    done
done