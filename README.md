# 2023-Hai-River-Flood-Mapping
This repository contains data and codes used in manuscript "Reconstruction of three-dimensional inundation structure reveals China's flood control empowered by flood retention areas"
## 1. Pre-processing to SAR amplitude images.
The pre-processing to SAR amplitude images involves four key steps:

**Step 1**: Mosaic the VV and VH images of the same track and same day separately.
We can store the original VV and VH images in a folder.
```sh
- ## date-A (e.g. August 1st)
  -- Scene-A1
   --- VV.tif
   --- VH.tif
  -- Scene-A2
   --- VV.tif
   --- VH.tif
  -- ...
- ## date-B (e.g. August 13th)
  -- Scene-B1
   --- VV.tif
   --- VH.tif
  -- Scene-B2
   --- VV.tif
   --- VH.tif
  -- ...
- ...
```



**Step 2**: Clip the mosaic VV and VH images to the extent of the Hai River Basin.

> The entire Hai River Basin (HRB) is covered by four orbit path of Sentinel-1. We create shapefiles to represent the four orbits's extent and clip them to the extent of the HRB. Then we use them to clip the mosaicked images, thereby narrowing the processing area and saving processing time. The four orbits files are uploaded to `Pre-processing/orbits`. 

**Step 3**: Composite three bands. 

**Step 4**: Stretch


