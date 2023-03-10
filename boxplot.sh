## Select the testpatient of the model you wish to use
# model_testpatient=BC50027
# model_testpatient=BC50040
# model_testpatient=BC50111
# model_testpatient=BC51218
# model_testpatient=BC51517
# model_testpatient=BC52337
model_testpatient=BC53934

output_dir="${PWD}/output_inference/densenet121_296/top_250/${model_testpatient}_model"

epoch=1

bin/plot_boxplots.py ${output_dir} --epoch ${epoch} --plotname ${model_testpatient}_model_inferenceon_other_datasets --plotlims -0.5 0.5