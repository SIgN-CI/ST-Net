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
# model=HCC12
model=HCC1234

## Set automatically
dir="output_inference/$inference_on/${model}_model"
genes="C1S CD74 MT2A ALB SERPINA1 APOA1 APOC1 IGKC APOA2 FGB FGA"

for patient in "BC50027"
# for patient in "BC50027" "BC50040" "BC50111" "BC51218" "BC51517" "BC52337" "BC53934"
do
    # bin/calculate_correlation.py --patient $patient --genes $genes --model $model --inference_type $inference_on

    bin/generate_correlation_table.py --patient $patient --model $model
done