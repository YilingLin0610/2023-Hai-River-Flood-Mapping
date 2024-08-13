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

We warp these steps in function `Mosaic_all` in  `pre_process_batch.py`.  Two issues about the input data of the function must be clarified here.
### 1.1 Orbits file
The entire Hai River Basin (HRB) is covered by four orbit path of Sentinel-1. We clip these orbit paths to the extent of the HRB and use them to clip the mosaicked images, thereby narrowing the processing area and saving processing time. The orbits files are uploaded to `Pre-processing\orbits`

### 1.1 Orbits
