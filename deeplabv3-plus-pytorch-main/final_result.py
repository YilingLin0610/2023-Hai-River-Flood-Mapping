"""
Post-processing functions used to calculate deep-learning models' accuracy
Author: Yiling Lin
"""
from osgeo import gdal, ogr, osr
import geopandas as gpd
from shapely.geometry import Polygon
from tqdm import *
import numpy as np
import pandas as pd
from fiona.crs import from_epsg
import math

def area(shpPath):
    """
     Calculate the area (ha) of each predicted RTSs and add one field in generated RTS prediction shapefiles to
     store these values

     Parameters:
         shpPath: The predicted RTS shapefile
     Returns:
         None
     """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shpPath, 1)
    layer = dataSource.GetLayer()


    new_field = ogr.FieldDefn("Area", ogr.OFTReal)
    new_field.SetWidth(32)
    new_field.SetPrecision(16)
    layer.CreateField(new_field)
    for feature in layer:
        geom = feature.GetGeometryRef()
        area_in_sq_m = geom.GetArea()
        area_in_sq_ha = area_in_sq_m / 10000

        feature.SetField("Area", area_in_sq_ha)
        layer.SetFeature(feature)



def create_filtered_shapefile(in_shapefile, out_shapefile):
    """
     Delete features with areas smaller than certain threshold.

     Parameters:
         in_shapefile: The original shapefile
         out_shapefile: The filtered shapefile
     Returns:
         None
     """
    gdf = gpd.read_file(in_shapefile)
    gdf = gdf[gdf.Area > 0.1] # Here set the threshold as 0.1

    gdf.to_file(out_shapefile)

def Union(shapefile1,shapefile2,num_GT):
    """
     Calculate the IoU of each RTS ground truth

     Parameters:
         shapefile1: predicted RTS shapefile
         shapefile2: RTS ground truth shapefile
         num_GT: Number of RTS ground truth
     Returns:
         ious: The IoU dictionary of each RTS ground truth
         length: The number of prediction RTSs
     """

    gdf = gpd.read_file(shapefile1)
    gdf.set_crs("EPSG:32646")
    gdf.to_file(shapefile1)
    gdf = gpd.read_file(shapefile1)
    gdf_gt=gpd.read_file(shapefile2)
    ious = {i: 0 for i in range(num_GT)} # The dictionary used to store the IoU of each RTS ground truth

    for i in tqdm(range(len(gdf))):
        Union = 0
        Inter = 0
        row_to_export=gdf.iloc[i]
        row_gdf = gpd.GeoDataFrame([row_to_export])
        row_gdf = row_gdf.set_crs(epsg=32646)
        result=gpd.overlay(gdf_gt,row_gdf,how="union")
        if(len(result)>(num_GT+1)):
            area_not_nan = np.where(~np.isnan(result['Area']))[0]
            gt_values=result.iloc[area_not_nan]["FID_gt"]
            RTS_id=np.array(gt_values[~(np.isnan(np.array(gt_values)))])
            for n in range(len(result)):
                if((result["FID_gt"][n] in RTS_id) and (result["Area"][n] > 0) ):
                    Inter = Inter + result.iloc[n].geometry.area
                elif((result["FID_gt"][n] in RTS_id) and (np.isnan(result["Area"][n])) ):
                    Union = Union + result.iloc[n].geometry.area
                elif((result["FID_gt"][n] not in RTS_id) and (result["Area"][n] > 0)  ):
                    Union = Union + result.iloc[n].geometry.area
            iou=float(Inter/(Inter+Union))
            for n in range(len(RTS_id)):
                if(ious[RTS_id[n]]<iou):
                    ious[RTS_id[n]]=iou
    print(ious)
    length=len(gdf)
    return ious,length





def final_result(ious,out_exel,num_gt,num_prediction):
    """
     Calculate the F1 of the deep-learning model

     Parameters:
         ious: The IoU of each RTS ground truth
         out_exel: The file path of the Excel storing the F1 accuracy
         num_gt: Number of RTS ground truth
         num_prediction:The number of prediction RTSs
     Returns:
         None
     """

    TP = 0
    FP = 0
    result={}
    result["recall"]=[]
    result["precision"] = []
    result["F1"]=[]
    result["prediction_num"]=[]
    thresholds=[]
    for i in range(10):
        thresholds.append(i*0.1)
    s=[]
    for i in range(len(ious)):
        s.append(ious[i])
    s=np.array(s)

    for threshold in thresholds:

        TP = np.sum(s > threshold)
        recall=float(TP/num_gt)
        print(recall)
        precision=float(TP/num_prediction)
        print(precision)
        if((recall+precision)!=0):
            f1=float(2*(recall*precision)/(recall+precision))
        else:
            f1=0
        result["recall"].append(recall)
        result["precision"].append(precision)
        result["F1"].append(f1)
        result["prediction_num"].append(num_prediction)

    df = pd.DataFrame(result)
    df.index=[thresholds]
    df=df.T
    df.to_excel(out_exel, index=True)
    print(result)


