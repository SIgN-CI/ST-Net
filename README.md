Integrating spatial gene expression and breast tumour morphology via deep learning
----------------------------------------------------------------------------------

ST-Net is a machine learning model for predicting spatial transcriptomics measurements from haematoxylin-and-eosin-stained pathology slides.
For more details, see the acompanying paper,

> [**Integrating spatial gene expression and breast tumour morphology via deep learning**](https://rdcu.be/b46sX)<br/>
  by Bryan He, Ludvig Bergenstråhle, Linnea Stenbeck, Abubakar Abid, Alma Andersson, Åke Borg, Jonas Maaskola, Joakim Lundeberg & James Zou.<br/>
  <i>Nature Biomedical Engineering</i> (2020).

RUN ON IN-HOUSE DATASET
-----------------------

Creating the environment
------

Downloading Dataset and Configuring Paths
-----------------------------------------
By default, the raw data must be downloaded from [here](https://entuedu-my.sharepoint.com/personal/bchua024_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fbchua024%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FGoodNotes%20Backup%2Faa%20Y4S2%2FSGH%20Biomedical%20Internship%2FST%2DNet&ct=1690862134954&or=Teams%2DHL&ga=1) and placed at `data/hist2tscript/`.
The processed files will then be written to `data/hist2tscript-patch/`.
These locations  can be changed by creating a config file (the priority for the filename is `stnet.cfg`, `.stnet.cfg`, `~/stnet.cfg`, `~/.stnet.cfg`).
An example config file is given as `example.cfg`.


Preparing Spatial Data
----------------------
This code assumes that the raw data has been extracted into SPATIAL_RAW_ROOT as specified by the config file.
```
bash  prepare spatial  # caches the counts and tumor labels into npz files
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
