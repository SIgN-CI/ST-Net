ngenes=250

model=densenet121

# window=10
window=30
# window=74
# window=75
# window=224

# epochs=1
# epochs=5
# epochs=20
# epochs=50
epochs=100

GPU=0
# GPU=1
# GPU=2

training_label="COVIDHCC"

echo "Training Label: ${training_label}"
echo "Model : ${model}"
echo "Epochs: ${epochs}"
echo "Window: ${window}px"
echo "GPU   : ${GPU}"

echo "Continue [y/n]?"
read user_confirmation

if [[ $user_confirmation == "y" || $user_confirmation == "Y" ]]
then
    CUDA_VISIBLE_DEVICES=${GPU} bin/cross_validate.py output/${model}_${window}/top_${ngenes}/${training_label}_ ${training_label} ${epochs} --lr 1e-6 --window ${window} --model ${model} --pretrain --average --batch 32 --workers 7 --gene_n ${ngenes} --norm --epochs ${epochs} 
else
    echo "Confirmation not received. Aborting..."
fi
