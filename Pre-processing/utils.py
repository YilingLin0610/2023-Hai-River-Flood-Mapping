#coding=gbk
"""
Some basic functions used to process raster TIFF images.
    1. GetExtent: Calculate the geographic or projected coordinates of the image's corner points.
    2. Mosaic: Mosaic several RS images
    3. Clip: Clip raster images by shapefile.
    4. read_raster_as_array: Read raster TIFF file as numpy array.
    5. readTif: Read the basic information of a raster TIFF file.
    6. writeTiff: Write an array data into a raster TIFF file.
Author: Yiling Lin
"""
from osgeo import gdal,ogr,gdalconst
import os
from tqdm import tqdm
import math
import numpy as np

def GetExtent(in_fn,target_projection):
    '''
    Calculate the geographic or projected coordinates of the image's corner points.
    Before calculating, the function project the image into the target projection.
    ——————————————————————————
    @param：
        in_fn: The path of images
        target_projection: The target projection of the image.
    @return:
        min_x： Minimum value in the x-direction.
        max_y： Maximum value in the y-direction.
        max_x： Maximum value in the x-direction.
        min_y:  Minimum value in the y-direction.
    '''

    ds = gdal.Open(in_fn)

    if(ds.GetProjection()== target_projection):
        geotrans = list(ds.GetGeoTransform())
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        min_x = geotrans[0]
        max_y = geotrans[3]
        max_x = geotrans[0] + xsize * geotrans[1]
        min_y = geotrans[3] + ysize * geotrans[5]
        change=0
        print("nochange")
    else:

        print("change")
        change=1
        gdal.Warp(in_fn[0:-4]+"_pro.tif", in_fn,
                  dstSRS=target_projection,
                  # cutlineDSName=r'F:\Parper2\Shp_Mask\RectangleOutline.shp',
                  resampleAlg=gdalconst.GRA_NearestNeighbour,

                  )

        ds = gdal.Open(in_fn[0:-4]+"_pro.tif")
        geotrans = list(ds.GetGeoTransform())
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        min_x = geotrans[0]
        max_y = geotrans[3]
        max_x = min_x + xsize * geotrans[1]
        min_y = max_y + ysize * geotrans[5]


        geotrans = ds.GetGeoTransform()




    #ds = None

    return min_x, max_y, max_x, min_y,ds,change

def Mosaic(in_files ,out_file_path ,polarization):
    '''
    Mosaic several RS images

    ——————————————————————————
    @param：
        in_files: The path to the images to be mosaiced is stored.
        out_file_path: The folder where the mosaiced image will be stored.
        polarization: The polarication of images to be mosaiced. (VV or VH)
    @return:
        out_name：The complete file path of mosaiced image will be stored.
    '''

    in_fn = in_files[0]
    # target_projection = in_fn.GetProjection()
    in_files_new = []
    # Get the extent of mosaiced images
    ds_small = gdal.Open(r"E:\mosaic_rgb\0912_matched_stretch.tif")
    target_projection = ds_small.GetProjection()
    min_x, max_y, max_x, min_y ,ds ,change = GetExtent(in_fn ,target_projection)

    # target_projection="PROJCS["WGS 84 / UTM zone 50N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",117],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32650"]]"
    if change==1:
        in_files_new.append(in_files[0][0:-4 ] +"_pro.tif")
    else:
        in_files_new.append(in_files[0])


    for in_fn in tqdm(in_files[1:]):

        minx, maxy, maxx, miny ,ds ,change = GetExtent(in_fn ,target_projection)
        if change == 1:
            in_files_new.append(in_fn[0:-4] + "_pro.tif")
        else:
            in_files_new.append(in_fn)
        min_x = min(min_x, minx)
        min_y = min(min_y, miny)
        max_x = max(max_x, maxx)
        max_y = max(max_y, maxy)
    in_ds = gdal.Open(in_files_new[0])
    geotrans = list(in_ds.GetGeoTransform())
    width = geotrans[1]
    height = geotrans[5]

    columns = math.ceil((max_x - min_x) / width)
    rows = math.ceil((max_y - min_y) / (-height))
    in_band = in_ds.GetRasterBand(1)

    driver = gdal.GetDriverByName('GTiff')

    if polarization=="VV":
        out_ds = driver.Create(os.path.join(out_file_path ,'mosaiced_VV.tif'), columns, rows, 1, in_band.DataType)
        out_name =os.path.join(out_file_path ,'mosaiced_VV.tif')
    elif polarization=="VH":
        out_ds = driver.Create(os.path.join(out_file_path, 'mosaiced_VH.tif'), columns, rows, 1, in_band.DataType)
        out_name = os.path.join(out_file_path, 'mosaiced_VH.tif')
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x
    geotrans[3] = max_y
    # print(geotrans)
    out_ds.SetGeoTransform(geotrans)
    out_band = out_ds.GetRasterBand(1)
    outs = out_band.ReadAsArray()


    inv_geotrans = gdal.InvGeoTransform(geotrans)

    # Write the data into the final TIFF image.
    for in_fn in tqdm(in_files_new):
        in_ds = gdal.Open(in_fn)
        in_gt = in_ds.GetGeoTransform()
        trans = gdal.Transformer(in_ds, out_ds, [])
        success, xyz = trans.TransformPoint(False, 0, 0)
        x, y, z = map(int, xyz)
        data = in_ds.GetRasterBand(1).ReadAsArray()
        outs = out_band.ReadAsArray()
        shape_data =np.shape(data)
        data = np.where(data == s, outs[y:y+shape_data[0],x:x+shape_data[1]], data)
        out_band.WriteArray(data,x,y)
    del in_ds, out_band, out_ds
    return out_name

def Clip(shp_file,tif_file,tif_file_clip):
    '''
    Clip raster images by shapefile.
    ——————————————————————————
    @param：
        shp_file: The shapefile used to clip the raster images.
        tif_file: The path of raster TIFF file to be clipped.
       tif_file_clip: The path of clipped raster images.
    @return:
        None
    '''

    # Open the raster TIFF file
    input_ds = gdal.Open(tif_file)

    input_srs = input_ds.GetProjection()

    shapefile_ds = ogr.Open(shp_file)
    layer = shapefile_ds.GetLayer()

    gdal.Warp(tif_file_clip,
                   tif_file,
                   format='GTiff',
                   cutlineDSName=shp_file,
              cropToCutline=True,
              dstNodata=None
                 )
    return 0

def read_raster_as_array(file_path):
    '''
    Read raster TIFF file as numpy array.
    ——————————————————————————
    @param：
        file_path: The file path of TIFF file.

    @return:
        raster_out: The array that stores the image data.
        proj: The projection information of the TIFF image data.
        geotrans: The geotrans information of the TIFF image data.
    '''

    dataset = gdal.Open(file_path)
    if dataset is None:
        raise Exception("Cannot open the raster, check the path.")


    # Read the rows and columns of the rater file.
    rows = dataset.RasterYSize
    cols = dataset.RasterXSize
    bands_num=dataset.RasterCount
    raster_out=np.zeros((rows,cols,bands_num))

    # Read the raster file as array.
    for i in range(bands_num):
        band = dataset.GetRasterBand(i+1)
        proj = dataset.GetProjection()
        geotrans = dataset.GetGeoTransform()
        raster_out[:,:,i] = band.ReadAsArray(0, 0, cols, rows)

    dataset=None

    return raster_out,proj,geotrans

def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    '''
    Read the basic information of a raster TIFF file.
    ——————————————————————————
    @param：
        fileName: The file path of TIFF file.
        xoff: The number of pixels to skip horizontally from the left edge of the raster
               image before starting to read the data. It essentially defines the starting column for the data extraction.
        yoff: The number of pixels to skip vertically from the upper edge of the raster
               image before starting to read the data. It essentially defines the starting column for the data extraction.
        data_width: The width of the TIFF image.
        data_height: The height of the TIFF image.

    @return:
        width: The width of the TIFF image.
        height: The height of the TIFF image.
        bands: The band count of the TIFF image.
        data: The array data of the raster image.
        proj: The projection information of the TIFF image data.
        geotrans: The geotrans information of the TIFF image data.
    '''
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if (data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

def writeTiff(im_data, im_geotrans, im_proj, path):
    '''
    Write an array data into a raster TIFF file.
    ——————————————————————————
    @param：
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

