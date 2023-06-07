#!/bin/bash

## (1) Set this
## Select the GPU you wish to use
GPU=0
# GPU=1
# GPU=2

## (2) Set this
## Select the model you wish to use
# model_to_load="BC50027_100"
# model_to_load="BC50040_75"
# model_to_load="BC50111_94"
# model_to_load="BC51218_99"
# model_to_load="BC51517_95"
# model_to_load="BC52337_95"
# model_to_load="BC53934_56"
# model_to_load="HCC12_100"
model_to_load="BC30004_100"

## (3) Set this
## Set the patients that the model was trained on
# trained_on="BC30001 BC30002"
trained_on="BC30001 BC30002 BC30003"
# trained_on="BC50027 BC50040 BC50111 BC51218 BC51517 BC52337 BC53934"
# trained_on="BC50027 BC50040 BC50111 BC51218 BC51517 BC52337"

## These are set automatically / fixed
model_testpatient=${model_to_load%_*}
model_epoch=${model_to_load##*_}
first_loop=1
ngenes=250
model=densenet121
## Fixed at 1, since we are only borrowing the training pipeline for inference
epochs=1

# for patient in "BC30001"
# for patient in "BC30002"
# for patient in "BC30003"
for patient in "BC30004"
# for patient in "BC30005"
# for patient in "BC50027"
# for patient in "BC50027" "BC50040" "BC50111" "BC51218" "BC51517" "BC52337" "BC53934"
# for patient in BC42334 BC42740 BC43740
# for patient in BC42740
# for patient in BC43740
do
     case $patient in

          "BC30001" | "BC30002" | "BC30003" | "BC30004")
               data="COVID_HCC"
               window=30
               ;;

          "BC42334" | "BC42740" | "BC43740")
               data="Y90Pre"
               window=296
               ;;

          "BC50027" | "BC50040" | "BC50111" | "BC51218" | "BC51517" | "BC52337" | "BC53934")
               data="Y90Post"
               window=74
               ;;

          *)
               data="unknown"
               window="unknown"
               ;;
     esac

     echo "You want to do inference on $patient ($data)"
     echo "Model     : $model_to_load"
     echo "Trained on: $trained_on"
     # echo "Window    : ${window}px"
     echo "GPU       : $GPU"
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
          CUDA_VISIBLE_DEVICES=${GPU} bin/cross_validate.py output_inference/${data}/${model_testpatient}_model/${patient}_ ${patient} --lr 1e-6 --window ${window} --pretrain --average --batch 32 --workers 1 --gene_n ${ngenes} --norm --model ${model} --epochs ${epochs} --load ${PWD}/models/${model_testpatient}_epoch_${model_epoch}.pt --trained_on ${trained_on}

     elif [[ $user_confirmation == "exit" ]]
     then
          echo "Exiting..."
          break
          
     else
          echo "No user confirmation received. Aborting..."
          break
     fi
done
