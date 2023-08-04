Integrating spatial gene expression and breast tumour morphology via deep learning
----------------------------------------------------------------------------------

ST-Net is a machine learning model for predicting spatial transcriptomics measurements from haematoxylin-and-eosin-stained pathology slides.
For more details, see the acompanying paper,

> [**Integrating spatial gene expression and breast tumour morphology via deep learning**](https://rdcu.be/b46sX)<br/>
  by Bryan He, Ludvig Bergenstråhle, Linnea Stenbeck, Abubakar Abid, Alma Andersson, Åke Borg, Jonas Maaskola, Joakim Lundeberg & James Zou.<br/>
  <i>Nature Biomedical Engineering</i> (2020).



RUN ON IN-HOUSE DATASET
-----------------------

key folders, bash scripts and git branches
------
branches:
```git checkout <insert branch name>```
sam_individual_window (training) 
sam_inference (inference) 

script folders:
data/hist2tscript
generate_figures.sh
stnet

output folders:
(for training) ~/ST-Net/output/train_COVIDHCC_test_BC300xx/
- BC300xx_visualize (figures generated)
- BC30007_model.pt (saved model)

(for inference) ~/ST-Net/output_inference/COVID_HCC/BC300xx_model
- BC300xx_visualize (figures generated)


bash scripts:
inference.sh
training.sh
boxplot.sh (in sam_individual_window)
abstract/generate_figures.sh




****** activating the environment ******
------
conda activate stnet 


Downloading Dataset and Configuring Paths
-----------------------------------------
By default, the raw data must be downloaded from [here](https://entuedu-my.sharepoint.com/personal/bchua024_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fbchua024%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FGoodNotes%20Backup%2Faa%20Y4S2%2FSGH%20Biomedical%20Internship%2FST%2DNet&ct=1690862134954&or=Teams%2DHL&ga=1) and placed at `data/hist2tscript/`.
The processed files will then be written to `data/hist2tscript-patch/`.


Preparing Spatial Data
----------------------
This code assumes that the raw data has been extracted into SPATIAL_RAW_ROOT as specified by the config file.
```
bash  prepare_spatial.sh  # caches the counts and tumor labels into npz files
```

Training models
---------------

The models for the main results can be trained by running:

```
git checkout sam_individual_window
cd data/hist2tscript
#change the start and the end to the corresponding samples that you want to train on. e.g. BC30020 to BC30022 will change 
vim rename.py 
python rename.py
vim training.sh
```

Change the patients based on the names that you would like to train on
example: 
```
for patient in "BC30001" "BC30002" <other sample names>
```
The detected data to use for training would be the based on having a .tif file in the data/hist2tscript with a corresponding name

```
bash training.sh
```
Relevant arguments can be changed to accomodate customisation, see example below:

**important to note** if the tile size changes, you must change the window size correspondingly. For example, if a tile in your data set is 224, then the window must be 224, else training will not happen properly. 
```
vim training.sh
```

```
ngenes=250
model=densenet121
window=224
for patient in `python3 -m stnet patients`
do
    bin/cross_validate.py output/${model}_${window}/top_${ngenes}/${patient}_ 4 50 ${patient} --lr 1e-6 --window ${window} --model ${model} --pretrain --average --batch 32 --workers 7 --gene_n ${ngenes} --norm
done
```
After training, weights and models will be saved to the output dir, for example ~/ST-Net/output/train_COVIDHCC_test_BC30007



Preparing TCGA data (ignore if working on server)
------
Access the TCGA data and download it to from [here](https://portal.gdc.cancer.gov/repository?facetTab=files&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-LIHC%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.access%22%2C%22value%22%3A%5B%22open%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22tsv%22%5D%7D%7D%2C%7B%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22RNA-Seq%22%5D%7D%2C%22op%22%3A%22in%22%7D%5D%7D&searchTableTab=files) you will have to add all to cart and download it 

unzip it, put the list of tsvs into ~/ST-Net/TCGA_preprocess/TCGA_map/tsvs
and the metadata into ~/ST-Net/TCGA_preprocess/TCGA_map

Preparing TCGA data for input into inference pipeline 
------
```
cd ~/ST-Net/TCGA_preprocess/TCGA_map 
python map_names.py  
```

1) Map the the downloaded raw counts info to TCGA images and extract the top and bottom x configurable in script
Based on selected images paired with raw counts data, generate 10% of the tiles as well as dummy data, renaming it to {gene}_tiled folder (e.g. ALB_tiled).
```
cd ~/ST-Net/TCGA_preprocess 
python histolab_tile.py
```

2) Renaming of tiles and creation of other required data files
```
cd ~/ST-Net/TCGA_preprocess 
python create_dummy.py 
```
Move the data files to data/hist2tscript
python move_data.py 




Inference 
---------------
```
git checkout sam_inference
```

1) cp path/to/trained_weights ~/ST-Net/models/model_epoch_100
example: ~/ST-Net/output/train_COVIDHCC_test_BC30007/BC30007_model.pt ~/ST-Net/models/**BC30007_epoch_100.pt** 
important to follow the model naming conventions. 

```
cd ~/ST-Net/data/hist2tscript
```
2) edit the inference.sh file according to the model that you are using

change model_to_load = "BC30007_100" for example if weight name is BC30007_epoch_100.pt
trained_on = "BC300090" #change it to any data that has a .tif in the hist2tscript, doesn't matter

3) change the start and end variables for a sequence of same numbers and comment out the bottom two lines. Else, to specify specific numbers, follow the examples listed in the file/
start=53
end=55
#for i in $(seq $start $end);
    #patient="BC300$i" 



```
bash inference.sh
```


Generate_figures 
---------------

Figures can be generated by running:

```
cd ~/ST-Net
bash generate_figures.sh
```
Before running this, change the gene types:
```
    for gene in "ALB" "C1S" "CD74"
```

2) Also change the piping  ("BC30001" | "BC30002" | "BC30003" | "BC30004" | "BC30005" | "BC30006" | "BC30007")


The folders with all the labels and post inference information, can be found in BC30007, for example: 

~/ST-Net/output_inference/COVID_HCC/BC30007_model/BC30007_visualize
 

Generate TCGA_figures 
---------------
Change the start and end to reflect the TCGA
    start = 90 
    end = 95
```
cd abstract 
python create_figs.py
```

Generate boxplot of spearman's correlation (under sam_individual_window) 
---------------
git checkout sam_individual_window

```
git checkout sam_individual_window
bash boxplot.sh 
```




ORIGINAL PAPER INSTRUCTIONS
---------------------------

Downloading Dataset and Configuring Paths
-----------------------------------------
By default, the raw data must be downloaded from [here](https://data.mendeley.com/datasets/29ntw7sh4r) and placed at `data/hist2tscript/`.
The processed files will then be written to `data/hist2tscript-patch/`.
These locations  can be changed by creating a config file (the priority for the filename is `stnet.cfg`, `.stnet.cfg`, `~/stnet.cfg`, `~/.stnet.cfg`).
An example config file is given as `example.cfg`.

Preparing Spatial Data
----------------------
This code assumes that the raw data has been extracted into SPATIAL_RAW_ROOT as specified by the config file.
```
python3 -m stnet prepare spatial  # caches the counts and tumor labels into npz files
bin/create_tifs.sh                  # converts jpegs into tiled tif files
```

Training models
---------------

The models for the main results can be trained by running:
```
ngenes=250
model=densenet121
window=224
for patient in `python3 -m stnet patients`
do
    bin/cross_validate.py output/${model}_${window}/top_${ngenes}/${patient}_ 4 50 ${patient} --lr 1e-6 --window ${window} --model ${model} --pretrain --average --batch 32 --workers 7 --gene_n ${ngenes} --norm
done
```

To run the comparison for different window sizes:
```
ngenes=250
model=densenet121
for window in 128 299 512
do
    for patient in `python3 -m stnet patients`
    do
        bin/cross_validate.py output/${model}_${window}/top_${ngenes}/${patient}_ 4 50 ${patient} --lr 1e-6 --window ${window} --model ${model} --pretrain --average --batch 32 --workers 7 --gene_n ${ngenes} --norm
    done
done
```

To run the comparison for different magnifications:
```
ngenes=250
window=224
model=densenet121
for downsample in 2 4
do
    for patient in `python3 -m stnet patients`
    do
        bin/cross_validate.py output/${model}_${window}/top_${ngenes}_downsample_${downsample}/${patient}_ 4 50 ${patient} --lr 1e-6 --window ${window} --model ${model} --pretrain --average --batch 32 --workers 4 --gene_n ${ngenes} --norm --downsample ${downsample}
    done
done
```

To run the comparison against random initialization:
```
ngenes=250
model=densenet121
window=224
for patient in `python3 -m stnet patients`
do
    bin/cross_validate.py output/${model}_${window}/top_${ngenes}_rand/${patient}_ 4 50 ${patient} --lr 1e-6 --window ${window} --model ${model} --average --batch 32 --workers 7 --gene_n ${ngenes} --norm
done
```

To run the comparison against individual training of genes:
```
ngenes=250
model=densenet121
window=224
for i in `seq 10`
do
    ensg=`python3 -m stnet ensg ${i}`
    for patient in `python3 -m stnet patients`
    do
        bin/cross_validate.py output/${model}_${window}/top_${ngenes}_singletask_${i}/${patient}_ 4 50 ${patient} --lr 1e-6 --window ${window} --model ${model} --pretrain --average --batch 32 --workers 7 --gene_list ${ensg} --norm
    done
done
```

To run the comparison against hand-crafted features:
```
ngenes=250
window=224
model=rf
for patient in `python3 -m stnet patients`
do
    root=output/${model}_${window}/top_${ngenes}/${patient}_ 
    python3 -m stnet run_spatial --gene --logfile ${root}gene.log --epochs 1 --pred_root ${root} --testpatients ${patient} --window ${window} --model ${model} --batch 32 --workers 7 --gene_n ${ngenes} --norm --cpu
done
```

Analysis
--------

The main results can be generated by running:
```
bin/generate_figs.py output/densenet121_224/top_250/ cv
```

The corresponding results for the comparisons are generated by running:
```
for i in output/densenet121_224/*; do bin/generate_figs.py $i cv; done
```

Generating Figures
------------------

The following blocks of code are used for generating several of the figures.

Visualization of prediction across whole slide:
```
bin/visualize.py output/sherlock/BC23450_cv.npz --gene FASN
bin/visualize.py output/sherlock/BC23903_cv.npz --gene FASN
```

UMAP Clustering:
```
bin/cluster.py
```
