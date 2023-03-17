#!/bin/bash

## [1] SET THIS
## Select the inference patient type
# inference_on="COVID_HCC"
# inference_on="Y90Pre"
inference_on="Y90Post"

## [2] SET THIS
## Select the model used for inference
# model=BC50027
# model=BC50040
# model=BC50111
# model=BC51218
# model=BC51517
# model=BC52337
# model=BC53934
model=HCC12
# model=HCC1234

## Set automatically
dir="output_inference/$inference_on/${model}_model"

# for patient in "BC50027"
# for patient in "BC50040"
for patient in "BC51517"
# for patient in "BC30001" "BC30002" "BC30003" "BC30004"
# for patient in "BC42740"
# for patient in "BC42334" "BC42740" "BC43740"
# for patient in "BC50027" "BC50040" "BC50111" "BC51218" "BC51517" "BC52337" "BC53934"
do

    case $patient in

          "BC30001" | "BC30002" | "BC30003" | "BC30004")
               figure_spot_size=18
               title_font_size=14
               ;;

          "BC42334" | "BC42740" | "BC43740")
               figure_spot_size=1500
               title_font_size=80
               ;;

          "BC50027" | "BC50040" | "BC50111" | "BC51218" | "BC51517" | "BC52337" | "BC53934")
               figure_spot_size=88
               title_font_size=20
               ;;

          *)
               figure_spot_size="unknown"
               title_font_size="unknown"
               ;;
    esac

    for gene in "C1S" "CD74" "MT2A" "ALB" "SERPINA1" "APOA1" "APOC1" "IGKC" "APOA2" "FGB" "FGA"
    # for gene in "ALB" "C1S" "CD74" "MT2A"
    # for gene in "ALB"
    do
        bin/visualize.py $dir/${patient}_1.npz --gene ${gene} --figure_spot_size $figure_spot_size --title_font_size $title_font_size --output_extension png --figroot $dir/${patient}_visualize/

        bin/combine_figures.py --patient $patient --gene $gene --model $model --inference_type $inference_on
    done
done