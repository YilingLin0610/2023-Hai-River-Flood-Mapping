# 2023-Hai-River-Flood-Mapping
This repository contains data and codes used in manuscript "Reconstruction of Three-dimensional Inundation Structure Reveals China's Flood Control Empowered by Flood Detention Areas"
## 1. Pre-processing to SAR amplitude images.
The pre-processing to SAR amplitude images involves four key steps:

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
*  The training samples for the deep learning model are sourced from both **Beijing and Caofeidian, Hebei**. 
*  We also provide two ground truth shapefiles for evaluating the accuracy of the trained model: one from **Gaofen-1 images of Xiaoqinghe on August 1st**, and another from **Sentinel-1 images of Dongdian on August 5th**. 
*  These ground truth and nagetive shapefiles are available in the `Deep-learning/Ground_truth/` .
*  The code for processing the ground truth shapefiles into training and label images can be adapted from https://github.com/YilingLin0610/Multi-temporal-RTS-mapping/tree/main/Codes/Pre-processing.
### 2.2 DeepLabv3+ Model
We utilized the DeepLabv3+ model to map multi-temporal flood extents. The core components of the DeepLabv3+ model were adapted from
https://github.com/bubbliiiing/Semantic-Segmentation/tree/master/deeplab_Mobile.
For batch processing of the multi-temporal images, we have encapsulated the model functionality (training, prediction, and post-processing) into functions and provided a script in `deeplabv3-plus-pytorch-main/main_batch.py`.
#### 2.2.1 Environment configuration
For environment configuration, refer to file `env.yaml`
> conda env create -f env.yaml
#### 2.2.2 Train
In a configured environment, you can simply run main_batch.py to train the model. We have adopted MobileNetV2 as the backbone, which can be found in the `deeplabv3-plus-pytorch-main/model_data` directory.
> *  Make sure to annotate the prediction and post-processing functions within the `process_all` function when training.
> *  Before starting the training process, run `voc_annotation.py` to split the dataset into training, validation, and testing groups.
> *  Remember to update the dataset paths in the following files:
>     * utils/dataloader.py: Edit lines 32 and 33.
>     * callbacks.py: Edit lines 195 and 205.
#### 2.2.3 Prediction and post-processing.
Here we offer two checkpoints to . 


