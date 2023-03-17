#!/bin/bash

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
output_dir="$PWD/output_inference/$inference_type/${model}_model"
epoch=1

# bin/plot_boxplots.py ${output_dir} --epoch ${epoch} --plotname ${model}_model_inference --plotlims -0.5 0.5
bin/plot_boxplots.py ${output_dir} --epoch ${epoch} --plotname ${model}_model_inference