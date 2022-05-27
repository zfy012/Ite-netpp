# General purpose medical segmentation
This program can segment liver tissue for 2D CT slices and 3D MR volumes using the deep learning technique.
This method based on UNet++ architecture integrates the probabilistic map into the encoder of network.
The network is trained in an iterative fashion, since the probabilistic map comes from the previous classifier.


## data augmentation
augmentation.py
dataset.py

## model
Net.py
loss.py

## running
train.py

## data
The data folder includes five MR volumes(imgx.nii.gz) and their corresponding ground truth(labelx.nii.gz).

 



