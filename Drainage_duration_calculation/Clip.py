"""
Generate a all-zero TIFF file covering the Hai River Basin.
Author: Yiling Lin
"""
import arcpy
import os


def Clip(input_tif,extent_shp,out_tif_path):
    arcpy.env.parallelProcessingFactor = 0
    arcpy.Clip_management(in_raster=input_tif,
                          out_raster=out_tif_path,
                          in_template_dataset=extent_shp, nodata_value="-3.402823e+38",
                          clipping_geometry="ClippingGeometry", maintain_clipping_extent="NO_MAINTAIN_EXTENT")



if __name__=="__main__":
    in_files_path=r"F:\retreat_time_batch\clips_haihe_basin\fishnets"
    all_clip=[filename for filename in os.listdir(in_files_path) if filename.endswith(".shp")]
    in_files_path_2 = r"F:\retreat_time_batch\haihe_basin_inundated_date"
    for clip in all_clip:
        print("now begin",clip)
        out_path=os.path.join(r"F:\retreat_time_batch\clips_haihe_basin","interval_results_0706",clip[0:-4])
        if(os.path.exists(out_path)):
            print("exist")
        else:
            os.mkdir(out_path)
        extent_shp=os.path.join(in_files_path,clip)
        files_names = [filename for filename in os.listdir(in_files_path_2) if
                       filename.endswith("calculated.tif")]
        for filename in files_names:
            file_path = os.path.join(in_files_path_2, filename)
            outname = os.path.join(out_path, filename[0:4] + ".tif")
            Clip(file_path, extent_shp, outname)
    #input_tif = r"F:\retreat_time_batch\clips_haihe_basin\mosaic_retreat_time.tif"
    #filedir = r"F:\retreat_time_batch\continent_retreattime_results\continent"
    #out_dir = r"F:\retreat_time_batch\continent_retreattime_results\retreat_tif"
    #filenames = [filename for filename in os.listdir(filedir) if filename.endswith(".shp")]
    #for filename in filenames:
        # print(filename)
        # out_tif_path = os.path.join(out_dir, filename[0:-4] + ".tif")
        # extent_shp = os.path.join(filedir, filename)
        # Clip(input_tif, extent_shp, out_tif_path)









    # out_path=r"F:\retreat_time_batch\clips_xuzhihongqu\dongdian"
    # in_files_path = r"F:\retreat_time_batch\haihe_basin_inundated_date"
    # extent_shp = r"F:\haihe_batch\xuzhihongqu\¶«µíÐîÖÍºéÇø.shp"
    # files_names = [filename for filename in os.listdir(in_files_path) if
    #               filename.endswith("calculated.tif")]
    # for filename in files_names:
    #     print(filename)
    #     file_path=os.path.join(in_files_path,filename)
    #     outname=os.path.join(out_path,filename[0:4]+".tif")
    #     Clip(file_path, extent_shp, outname)
    #
    #
    #
    #
    #
    # # input_tif=r"F:\retreat_time_batch\haihe_basin_inundated_date\0805_calculated.tif"
    # # extent_shp=r"F:\haihe_batch\xuzhihongqu\¶«µíÐîÖÍºéÇø.shp"
    # # out_tif_path=r"F:\retreat_time_batch\clips_xuzhihongqu\dongdian\0805_1.tif"
    # # Clip(input_tif, extent_shp, out_tif_path)