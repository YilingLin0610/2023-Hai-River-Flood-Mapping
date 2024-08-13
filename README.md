# 2023-Hai-River-Flood-Mapping
This repository contains data and codes used in manuscript "Reconstruction of three-dimensional inundation structure reveals China's flood control empowered by flood retention areas"
## 1. Pre-processing to SAR amplitude images.
The Pre-processing to SAR amplitude images contains four steps. We warp these steps in `pre_process_batch.py`
> Step 1: Mosaic the VV and VH images of the same track and same day separately. \n
> Step 2: Clip the mosaic VV and VH images to the extent of the Hai River Basin.
> 
> Step 3: Composite three bands.
> 
> Step 4: Stretch
### 1.1 Orbits
