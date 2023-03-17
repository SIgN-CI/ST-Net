#!/bin/bash

ngenes=250

model=densenet121

## Set the patchsize corresponding to the pathsize for dataset that you wish to conduct inference on
# window=10
# window=30
window=74
# window=224
# window=296

## Fixed at 1, since we are only borrowing the training pipeline for inference
epochs=1

## Select the GPU you wish to use
# GPU=0
GPU=1
# GPU=2

## Select the model you wish to use
# model_to_load="BC50027_100"
# model_to_load="BC50040_75"
# model_to_load="BC50111_94"
# model_to_load="BC51218_99"
# model_to_load="BC51517_95"
# model_to_load="BC52337_95"
# model_to_load="BC53934_56"
# model_to_load="HCC12_100"
model_to_load="HCC1234_100"

model_testpatient=${model_to_load%_*}
model_epoch=${model_to_load##*_}

trained_on="BC30001 BC30002 BC30003 BC30004"
# trained_on="BC50027 BC50040 BC50111 BC51218 BC51517 BC52337 BC53934"

first_loop=1

# for patient in "BC30001"
# for patient in "BC30002"
# for patient in "BC30003"
# for patient in "BC30004"
# for patient in "BC30005"
# for patient in "BC50027"
for patient in "BC50027" "BC50040" "BC50111" "BC51218" "BC51517" "BC52337" "BC53934"
# for patient in BC42334 BC42740 BC43740
# for patient in BC42740
# for patient in BC43740
do
     echo "You want to do inference on ${patient}"
     echo "Model : ${model}"
     echo "Window: ${window}px"
     echo "GPU   : ${GPU}"
     echo ""

     if [[ $first_loop == 1 ]]
     then
          echo "Please check all parameters carefully."
          echo "This confirmation only displays for the first patient in the loop."
          echo "Continue? [y/n]"
          read user_confirmation
     fi

     if [[ $user_confirmation == "y" || $user_confirmation == "Y" || $first_loop == 0 ]]
     then
          first_loop=0
          CUDA_VISIBLE_DEVICES=${GPU} bin/cross_validate.py output_inference/${model}_${window}/top_${ngenes}/${model_testpatient}_model/${patient}_ ${patient} --lr 1e-6 --window ${window} --pretrain --average --batch 32 --workers 7 --gene_n ${ngenes} --norm --model ${model} --epochs ${epochs} --load ${PWD}/models/${model_testpatient}_epoch_${model_epoch}.pt --trained_on ${trained_on}
     elif [[ $user_confirmation == "exit" ]]
     then
          echo "Exiting..."
          break
     else
          echo "No user confirmation received. Aborting..."
          break
     fi
done