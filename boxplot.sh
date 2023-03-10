## Select the testpatient of the model you wish to use
# model_testpatient=BC50027
model_testpatient=BC50040

output_dir="/home/tester/bokleong/ST-Net6/output_inference/densenet121_296/top_250/${model_testpatient}_model"

epoch=1

bin/plot_boxplots.py ${output_dir} --epoch ${epoch} --plotname ${model_testpatient}_model_inferenceon_other_datasets --plotlims -0.5 0.5