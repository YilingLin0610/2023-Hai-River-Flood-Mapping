# 2023-Hai-River-Flood-Mapping
This repository contains data and codes used in manuscript "Reconstruction of three-dimensional inundation structure reveals China's flood control empowered by flood retention areas"
## 1. Pre-processing to SAR amplitude images.
The pre-processing to SAR amplitude images involves four key steps:
> **Step 1**: Mosaic the VV and VH images of the same track and same day separately.
> 
> **Step 2**: Clip the mosaic VV and VH images to the extent of the Hai River Basin.
> 
> **Step 3**: Composite three bands.
> 
> **Step 4**: Stretch

We warp these steps in function `Mosaic_all` in  `pre_process_batch.py`. The function need four input parameter:
> filepath: The filepath of the images.
>
> shpfile: The shapefiles (orbits extent) used to clip images into HRB extent.
>
> mosaic_path: The filepath storing the final result images.
>
>  images_target_name: The target images as references of histogram matching
### 1.1 Orbits
