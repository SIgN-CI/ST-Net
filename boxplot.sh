## Select the testpatient of the model you wish to use
# model_testpatient=BC50027
# model_testpatient=BC50040
# model_testpatient=BC50111
# model_testpatient=BC51218
# model_testpatient=BC51517
# model_testpatient=BC52337
# model_testpatient=BC53934
# model_testpatient=HCC12
model_testpatient=HCC1234

## Set the window size
window=74
# window=296

output_dir="${PWD}/output_inference/densenet121_${window}/top_250/${model_testpatient}_model"

epoch=1

# bin/plot_boxplots.py ${output_dir} --epoch ${epoch} --plotname ${model_testpatient}_model_inference --plotlims -0.5 0.5
bin/plot_boxplots.py ${output_dir} --epoch ${epoch} --plotname ${model_testpatient}_model_inference