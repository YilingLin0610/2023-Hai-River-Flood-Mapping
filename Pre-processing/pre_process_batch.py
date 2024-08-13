#coding=gbk
from osgeo import gdal
import os
from tqdm import tqdm

from stretch import all
import shutil
from utils import Mosaic,Clip
from band_composite import band_composite
from histogram_matching import bands_matching

os.environ['PROJ_LIB'] = r"C:\Users\xiehu\anaconda3\envs\threshold\Lib\site-packages\osgeo\data\proj"



def Mosaic_all(filepath,shpfile,mosaic_path,images_target_name):
    """
    Pre-processing to SAR amplitude images:
    Step 1: Mosaic the VV and VH images of the same track and same day separately.
    Step 2: Clip the mosaic VV and VH images to the extent of the Hai River Basin.
    Step 3: Composite three bands.
    Step 4: Stretch
    ——————————————————————————
    @param：
        filepath: The filepath of the images.
        shpfile: The shapefiles (orbits extent) used to clip images into HRB extent.
        mosaic_path: The filepath storing the final result images.
        images_target_name: The target images as references of histogram matching

    @return:
        None
    """

    in_files=os.listdir(filepath)
    outpath=os.path.join(r"E:\mosaic_sar_images",os.path.basename(filepath))
    if(os.path.exists(outpath)):
        print("exist")
    else:
        os.mkdir(outpath)
    VH_files=[]
    VV_files = []
    # Collect the path of VV and VH images and store their names in lists separately.
    for i in range(len(in_files)):
        #print(in_files[i])
        in_files[i]=os.path.join(filepath,in_files[i],in_files[i])
        VH_file=[file for file in os.listdir(in_files[i]) if file.endswith("_VH.tif")]
        VV_file=[file for file in os.listdir(in_files[i]) if file.endswith("_VV.tif")]
        VH_files.append(os.path.join(in_files[i],VH_file[0]))
        VV_files.append(os.path.join(in_files[i], VV_file[0]))


    shpfile=os.path.join(shpfile,str(VH_file[0][16:19])+"_cut.shp")
    # Mosaic the VV and VH images of the same track and same day separately.
    tif_file_VH=Mosaic(VH_files,outpath,"VH")
    tif_file_VV=Mosaic(VV_files, outpath, "VV")

    tif_file_clip_VH=tif_file_VH[0:-4]+"_clip.tif"
    tif_file_clip_VV = tif_file_VV[0:-4] + "_clip.tif"
    # Clip the mosaic VV and VH images to the extent of the Hai River Basin.
    Clip(shpfile, tif_file_VH, tif_file_clip_VH)
    Clip(shpfile, tif_file_VV, tif_file_clip_VV)
    # Composite three bands.
    composite_path=os.path.join(mosaic_path,os.path.basename(filepath)+".tif")
    band_composite(tif_file_clip_VV, tif_file_clip_VH, composite_path)
    # Histogram matching
    matched_name=composite_path[0:-4]+"_matched.tif"
    bands_matching(composite_path, images_target_name, matched_name)
    # Percent clip stretch
    stretch_name=matched_name[0:-4]+"_stretch.tif"
    all(matched_name, stretch_name, 0.5)
    # Delete the matched and composite images to free up the storage room.
    os.remove(matched_name)
    os.remove(composite_path)
    # Delete the mosaiced images to free up the storage room.
    try:
        shutil.rmtree(outpath)
        print(f"Folder '{outpath}' has been successfully deleted")
    except OSError as e:
        print(f"Folder '{outpath}' hasn't been successfully deleted: {e}")







if __name__ == "__main__":
    # The shapefiles (orbits extent) used to clip images into HRB extent.
    shp_file=r"C:\Users\xiehu\Desktop\Beijing_flood\shapefiles\orbits_haihe"
    # The filepath storing the final result images.
    mosaic_path=r"F:\mosaic_images_haihe"
    # The target images as references of histogram matching
    images_target_name=r"E:\Beijing_flood\mosaic_rgb\original\0712.tif"
    source_path=r"F:\SAR_images"
    paths=os.listdir(source_path)
    for i in tqdm(range(len(paths))):

        if(os.path.exists(os.path.join(mosaic_path,paths[i]+"_matched_stretch.tif"))):
            print("skip")
        else:
        #if(paths[i]!=paths[0] and paths[i]!=paths[1] and paths[i]!=paths[2] and paths[i]!=paths[3] and paths[i]!=paths[4]and paths[i]!=paths[5] and paths[i]!=paths[7]):
        #if (paths[i] == paths[0]):
            paths[i] = os.path.join(source_path, paths[i])
            Mosaic_all(paths[i], shp_file, mosaic_path, images_target_name)


