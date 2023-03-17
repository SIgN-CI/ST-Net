## [1] SET THIS
## Select the inference patient type
# inference_type="COVID_HCC"
inference_type="Y90Pre"
# inference_type="Y90Post"

## [2] SET THIS
## Select the model used for inference
# model=BC50027
# model=BC50040
# model=BC50111
# model=BC51218
# model=BC51517
# model=BC52337
model=BC53934
# model=HCC12
# model=HCC1234

## Set automatically
dir="output_inference/$inference_type/${model}_model"

# for patient in "BC50027"
# for patient in "BC50040"
# for patient in "BC30001" "BC30002" "BC30003" "BC30004"
# for patient in "BC42740"
for patient in "BC42334" "BC43740"
do
    for gene in "C1S" "CD74" "MT2A" "ALB" "SERPINA1" "APOA1" "APOC1" "IGKC" "APOA2" "FGB" "FGA"
    # for gene in "ALB" "C1S" "CD74" "MT2A"
    do
        bin/visualize.py $dir/${patient}_1.npz --gene ${gene} --figure_spot_size 1500 --title_font_size 80 --output_extension png --figroot $dir/${patient}_visualize/
    done
done