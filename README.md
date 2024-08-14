# 2023-Hai-River-Flood-Mapping
This repository contains data and codes used in manuscript "Reconstruction of Three-dimensional Inundation Structure Reveals China's Flood Control Empowered by Flood Detention Areas"

## Table of Contents
- [1. Pre-processing to SAR amplitude images](#1-pre-processing-to-sar-amplitude-images)
- [2. Deep learning framework](#2-Deep-learning-framework)
    - [2.1 Training samples](#21-training-samples)
    - [2.2 DeepLabv3+ Model](#22-deeplabv3-model)








## 1. Pre-processing to SAR amplitude images.
**You can run `pre_process_batch.py` to perform pre-processing on the original SAR amplitude images.**

It involves four key steps:

*  **Step 1**: Mosaic the VV and VH images of the same track and same day separately.

We store the original VV and VH images in a folder. Images from the same date are placed into separate sub-folders. The structure of these sub-folders is shown below:
```sh
- date (e.g. August 1st)
    -- Scene-1
        --- VV.tif
        --- VH.tif
    -- Scene-2
        --- VV.tif
        --- VH.tif
    -- ...
```



*  **Step 2**: Clip the mosaic VV and VH images to the extent of the Hai River Basin.

> The entire Hai River Basin (HRB) is covered by four orbit path of Sentinel-1. We create shapefiles to represent the four orbits's extent and clip them to the extent of the HRB. Then we use them to clip the mosaicked images, thereby narrowing the processing area and saving processing time. The four orbits files are uploaded to `Pre-processing/orbits`. 

*  **Step 3**: Composite three bands. 

*  **Step 4**: Stretch
> The clip percentage is set as 0.5%.

## 2. Deep learning framework
### 2.1 Training samples
**The code for processing the ground truth shapefiles into training and label images can be adapted from https://github.com/YilingLin0610/Multi-temporal-RTS-mapping/tree/main/Codes/Pre-processing.**

> *  The training samples for the deep learning model are sourced from both **Beijing and Caofeidian, Hebei**. 
> *  We also provide two ground truth shapefiles for evaluating the accuracy of the trained model: one from **Gaofen-1 images of Xiaoqinghe on August 1st**, and another from **Sentinel-1 images of Dongdian on August 5th**. 
> *  These ground truth and nagetive shapefiles are available in the `Deep-learning/Ground_truth/` .
### 2.2 DeepLabv3+ Model
**You can start the DeepLabv3+ Model by run `deeplabv3-plus-pytorch-main/main_batch.py`.** It encapsulated the model functionality (training, prediction, and post-processing) for batch processing of the multi-temporal images,

The core components of the DeepLabv3+ model were adapted from
https://github.com/bubbliiiing/Semantic-Segmentation/tree/master/deeplab_Mobile.

#### 2.2.1 Environment configuration
For environment configuration, refer to file `env.yaml`
> conda env create -f env.yaml
#### 2.2.2 Train
**You can simply run `main_batch.py` to train the model.**

We have adopted MobileNetV2 as the backbone, which can be found in the `deeplabv3-plus-pytorch-main/model_data`.

> **Note:**
> *  Make sure to annotate the prediction and post-processing functions within the `process_all` function when training.
> *  Before starting the training process, run `voc_annotation.py` to split the dataset into training, validation, and testing groups.
> *  Remember to update the dataset paths in the following files:
>      * utils/dataloader.py: Edit lines 32 and 33.
>      * utils/callbacks.py: Edit lines 195 and 205.
#### 2.2.3 Prediction and post-processing.
Here we offer two checkpoints to predict the flood extent. 
*  `logs/0712_Beijing_with_negative_st/best_epoch_weights.pth` for predicting natural water bodies.
*  `logs/0712_Beijing_with_negative_st_0601_plus_0712_V2_more_negative/best_epoch_weights.pth` for predicting manual water bodies on coastal areas.

The structure of testing images are organized as below:
```sh
- date1 (e.g. 0601) 
    -- tifs (clipped tifs)
    -- jpgs (clipped jpgs)
- date2
    -- tifs (clipped tifs)
    -- jpgs (clipped jpgs)
-.....
```
### 2.3 Results
*  The mapped flood extent shapefiles for each day are availabel via https://disk.pku.edu.cn/link/AA07665B4F35C04846AB53BF0DDAD40A46
*  The final maximum flood extent is stored in the `results/Maximum_flood_extent`. Note that polygons smaller than 2000 m² have been removed from this shapefile using a script `Delete_small_polygons.py`.
## 3. Flood depth estimation
### 3.1 Core code
The flood depth estimation's core code is modified from https://github.com/csdms-contrib/fwdet/tree/master/qgis_port. Here we made some slight modifications and the modified version can be found in `Flood-depth-estimation/fwdet_21.py`

**This script is a Plugin in QGis**
*  In the QGIS `Processing Toolbox`, select the python icon drop down, and `Add Script to Toolbox...` then point to the downloaded script.
*  This should load new algorithms to the `Scripts/FwDET` group on the Processing Toolbox.
### 3.2 Batch processing
To fulfill the batch-processing, we write the script 
## 4. Drainage duration estimation
## 5. Figure drawing
