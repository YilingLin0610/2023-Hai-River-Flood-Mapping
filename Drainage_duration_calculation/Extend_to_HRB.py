"""
Extend the flood extent TIFF files to cover the entire HRB using the all-zero TIFF file
Author: Yiling Lin
"""
import arcpy
import os
def add_to_all_extent(in_tif_file1,in_tiffile_all_0,out_tif_file):
    '''
    Extend the flood extent TIFF files to cover the entire HRB using the all-zero TIFF file
    ！！！！！！！！！！！！！！！！！！！！！！！！！！
    @param��
        in_tif_file1: The original TIFF file.
        in_tiffile_all_0: The all-zero TIFF file covering the Hai River Basin.
        out_tif_file: The filepath that store the out put extended file.

    @return:
        None
    '''
    arcpy.env.snapRaster = in_tiffile_all_0
    arcpy.env.extent=in_tiffile_all_0
    expression='"'+in_tif_file1+'"+"'+in_tiffile_all_0+'"'
    arcpy.gp.RasterCalculator_sa(expression,
                                 out_tif_file)

if __name__=="__main__":
    in_file_path=r"F:\retreat_time_batch\date_tifs_1_2\\"
    in_tiffile_all_0=r"F:\retreat_time_batch\mosaic_all_0.tif"
    in_tif_files = [filename for filename in os.listdir(in_file_path) if
                 filename.endswith("final_pro.tif")]
    for in_tif_file in in_tif_files:
        shp_first = in_tif_file.split("\\")[-1][0:4]
        outname=os.path.join(r"F:\retreat_time_batch\haihe_basin_inundated_date",shp_first+"_calculated.tif")
        add_to_all_extent(os.path.join(in_file_path,in_tif_file),in_tiffile_all_0,outname)