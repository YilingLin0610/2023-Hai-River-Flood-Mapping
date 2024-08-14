"""
Generate a all-zero TIFF file covering the Hai River Basin.
Author: Yiling Lin
"""

from osgeo import gdal
import numpy as np

def writeTiff(im_data, im_geotrans, im_proj, path):
    '''
    Write an array data into a raster TIFF file.
    ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
    @param£º
        im_data: The array data to be written.
        im_geotrans: The geotrans information of the TIFF image dta.
        im_proj: The projection information of the TIFF image data.
        path: The filepath of the created TIFF file will be stored.

    @return:
        None
    '''
    gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'GEOKEYS')
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    print(im_data.shape)

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # Create the file
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset



def generate_all_extent_zero(out_tif_name,npy_name,npy_name_geotrans,txt_name):
    '''
    Generate a all zero TIFF file covering HRB.
    ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
    @param£º
        out_tif_name: The file path that stores the output all zero TIFF file.
        npy_name: A NPY file that store an array with the shape of HRB.
        npy_name_geotrans: A NPY file that store the geotrans information
        txt_name: A TXT file that store the projection information
    @return:
        None
    '''
    array=np.load(npy_name)
    # Load the geotrans information
    geotrans=np.load(npy_name_geotrans)
    # Read the projection information
    with open(txt_name, 'r') as file:
        proj = file.read()

    shape=np.shape(array)

    array_nan=np.full((shape[0], shape[1]),0)
    writeTiff(array_nan, geotrans, proj, out_tif_name)





if __name__=="__main__":

    npy_name=r"F:\retreat_time_batch\mosaic_all_npy.npy"
    npy_name_geotrans = r"F:\retreat_time_batch\mosaic_all_geotrans.npy"
    txt_name = r"F:\retreat_time_batch\mosaic_all_proj.txt"
    haihe_all_nan_name=r"F:\retreat_time_batch\mosaic_all_0.tif"
    generate_all_extent_zero(haihe_all_nan_name,npy_name,npy_name_geotrans,txt_name)