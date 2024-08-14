"""
Post-processing
Use predicted PNG files from a deep learning model to generate flood extent shapefiles and
 further calculate accuracy of the models.
Author: Yiling Lin
"""
import os
import numpy as np
from osgeo import gdal
import os
import numpy as np
import pandas as pd
from tqdm import *
import warnings
import logging
from tqdm import *
from osgeo import gdal, ogr, osr
import math
import cv2
os.environ['PROJ_LIB'] = r'~/miniconda3/envs/pytorch/share/proj'



from final_result import area,create_filtered_shapefile,Union,final_result




def readTif(fileName):
    """
    Read a TIFF file.
    ——————————————————————————
    @param：
        fileName (str): Name of the TIFF file.

    @return:
        dataset: Data contained in the TIFF file.
    """
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "file cannot be opened")
    return dataset


def writeTiff(im_data, im_geotrans, im_proj, path):
    """
    Save a TIFF file

    Parameters:
        im_data: An array containing the TIFF file data
        im_geotrans: The geographic location of the TIFF file
        im_proj: The projection information of the TIFF file
        path: The file path where the processed output will be saved.

    Returns:
        None
    """
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def raster2vector(raster_path, vecter_path, field_name="class", ignore_values=None):
    """
    Convert a raster TIFF file into vector shapefile images

    Parameters:
        raster_path: The file path of rater TIFF file
        vecter_path: The file path of vector shapefile
        field_name: Create a field in generated shapefile to store the pixel value of original TIFF file
        ignore_values: When converting, we delete the non-RTS pixels. The pixels' value is ignore_values

    Returns:
        None
    """
    raster = gdal.Open(raster_path)
    band = raster.GetRasterBand(1)

    # Read the projection information
    prj = osr.SpatialReference()
    prj.ImportFromWkt(raster.GetProjection())

    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(vecter_path):
        drv.DeleteDataSource(vecter_path)

    # Creata a shapefile
    polygon = drv.CreateDataSource(vecter_path)
    poly_layer = polygon.CreateLayer(vecter_path[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)
    # Add one field in newly created shapefile to store the pixels' value
    field = ogr.FieldDefn(field_name, ogr.OFTReal)
    poly_layer.CreateField(field)

    gdal.FPolygonize(band, None, poly_layer, 0)

    # Delete the features containing non-RTSs
    if ignore_values is not None:
        for feature in poly_layer:
            class_value = feature.GetField('class')
            for ignore_value in ignore_values:
                if class_value == ignore_value:
                    # ?¨??FID????????
                    poly_layer.DeleteFeature(feature.GetFID())
                    break

    polygon.SyncToDisk()
    polygon = None


def GetExtent(in_fn):
    """
     Get the geographic extent of input TIFF file

     Parameters:
         in_fn: The file path of rater TIFF file

     Returns:
         min_x: minimum location in the x-direction
         max_y??maximum location in the y-direction
         max_x??maximum location in the x-direction
         min_y: minimum location in the y-direction
     """

    ds = gdal.Open(in_fn)
    geotrans = list(ds.GetGeoTransform())
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x = geotrans[0]
    max_y = geotrans[3]
    max_x = geotrans[0] + xsize * geotrans[1]
    min_y = geotrans[3] + ysize * geotrans[5]
    ds = None

    return min_x, max_y, max_x, min_y


def Mosaic_all (path1,path2,out_path):
    """
     Mosaic all png_file into one TIFF label file

     Parameters:
         path1: The filepath where the reference TIF files are stored. These TIFF files are utilized to
                offer geographic location and projection information to corresponding PNG files sharing same
                filename with them.
         path2: The filepath where the PNG files are stored
         out_path: The filepath where the generated TIFF label file will be stored

     Returns:
         None
     """

    os.listdir(path1)
    gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'GEOKEYS')
    in_files = [z for z in os.listdir(path1) if z.endswith("tif")]
    #in_files = os.listdir(path1)
    for i in tqdm(range(len(in_files))):
        in_files[i] = path1 + "/" + in_files[i]
    in_files_png=[]
    for i in tqdm(range(len(in_files))):
        in_files_png.append(path2+ "/" + in_files[i])

    in_fn = in_files[0]
    # Getting the TIFF label image's extent
    min_x, max_y, max_x, min_y = GetExtent(in_fn)
    for in_fn in tqdm(in_files[1:]):
        # in_fn=path+"/"+in_fn
        minx, maxy, maxx, miny = GetExtent(in_fn)
        min_x = min(min_x, minx)
        min_y = min(min_y, miny)
        max_x = max(max_x, maxx)
        max_y = max(max_y, maxy)

    # Calculating the columns and rows of TIFF label images
    in_ds = gdal.Open(in_files[0])
    geotrans = list(in_ds.GetGeoTransform())
    width = geotrans[1]
    height = geotrans[5]

    columns = math.ceil((max_x - min_x) / width)
    rows = math.ceil((max_y - min_y) / (-height))
    in_band = in_ds.GetRasterBand(1)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_path, columns, rows, 1, in_band.DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)
    out_band = out_ds.GetRasterBand(1)
    outs = out_band.ReadAsArray()

    inv_geotrans = gdal.InvGeoTransform(geotrans)

    # Write the PNG files' values into newly created TIFF files
    for in_fn in tqdm(in_files):
        s=in_fn.split("/")[-1]
        s=path2 + s[0:-4]+".png"
        in_ds_tif = gdal.Open(in_fn)
        in_gt = in_ds_tif.GetGeoTransform()

        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
        x, y = map(int, offset)
        trans = gdal.Transformer(in_ds_tif, out_ds, [])
        success, xyz = trans.TransformPoint(False, 0, 0)
        x, y, z = map(int, xyz)
        data=cv2.imread(s)
        data = data[:,:,0]
        data=np.int64(data > 0)
        outs_300=outs[y:(y + 300), x:(x + 300)]
        d=np.array((outs_300,data))
        outs[y:(y + 300), x:(x + 300)]=d.max(axis=0)

    out_band.WriteArray(outs)
    del in_ds, out_band, out_ds
    return 0



def post_processing(RS_image_path,FilePath,num_gt,shapefile2):
    """
    Post-processing workflow functions

     Parameters:
         RS_image_path: The filepath where the reference TIF files are stored. These TIFF files are utilized to
                        offer geographic location and projection information to corresponding PNG files sharing same
                        filename with them.
         FilePath: The work dictionary where the intermedia and final results will be stored
         num_gt: The number of RTSs in ground-truth shapefiles
         shapefile2: The groundtruth shapefile
     Returns:
         None
     """
    prediction_name=FilePath+"/prediction/"
    jpg_files = os.listdir(prediction_name)
    print("Mosaic the PNGs")

    try:
        os.mkdir(FilePath+r'/merge_prediction_tifs')
    except:
        pass
    output_file = FilePath+r'/merge_prediction_tifs/'+ "prediction.tif"
    Mosaic_all(RS_image_path,prediction_name,output_file)
    try:
        os.mkdir(FilePath+r'/merge_prediction_shp')
    except:
        pass
    vecter_path = FilePath+r'/merge_prediction_shp/'+ "prediction.shp"
    field_name = "class"
    ignore_values = [0]
    print("Convert to vector")
    raster2vector(output_file, vecter_path, field_name=field_name, ignore_values=ignore_values)
    print("Calculate the area")
    area(vecter_path)
    print("Delete minuscule polygons")
    create_filtered_shapefile(vecter_path,
                             FilePath+'/merge_prediction_shp/prediction_filted.shp')
    shapefile1 = FilePath+'/merge_prediction_shp/prediction_filted.shp'
    out_path = FilePath+"/union/"

    out_excel =  FilePath+"/result.xlsx"
    ious, num_prediction = Union(shapefile1, shapefile2, out_path, num_gt)
    final_result(ious, out_excel, num_gt, num_prediction)

