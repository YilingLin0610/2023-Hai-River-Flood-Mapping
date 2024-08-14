import os
import arcpy
arcpy.env.parallelProcessingFactor = 0
DB_dir=r"F:\haihe_batch\detention_basins_areas"
DEM_dir=r"F:\haihe_batch\DEM_xuzhihongqu"

DBs=[os.path.join(DB_dir, filename) for filename in os.listdir(DB_dir)]
DBs=[r"F:\haihe_batch\detention_basins_areas\langouwa"]
for DB in DBs:
    dates=[os.path.join(DB, filename) for filename in os.listdir(DB) if
                  filename.endswith(".shp")]
    if(len(dates)>0):
        name = dates[0].split("\\")[3]
        dem=os.path.join(DEM_dir,name+".tif")
    for i in range(len(dates)):
        print(dates[i])
        dates_eliminate = 'F:/haihe_batch/volume_time_series/' + name + '/holes_eliminate//' + dates[i].split("\\")[-1][0:-4] + "_eliminate.shp"
        dates_fix='F:/haihe_batch/volume_time_series/'+name+'/fix//'+dates[i].split("\\")[-1][0:-4]+"_fix.shp"
        try:
            os.mkdir('F:/haihe_batch/volume_time_series/'+name)
        except:
            pass
        try:
            os.mkdir('F:/haihe_batch/volume_time_series/'+name+'/fix//')
        except:
            pass
        try:
            os.mkdir('F:/haihe_batch/volume_time_series/'+name+'/holes_eliminate//')
        except:
            pass
        try:
            os.mkdir('F:/haihe_batch/volume_time_series/'+name+'/volume//')
        except:
            pass
        try:
            os.mkdir('F:/haihe_batch/volume_time_series/'+name+'/focal//')
        except:
            pass
        try:
            os.mkdir('F:/haihe_batch/volume_time_series/'+name+'/focal_clip//')
        except:
            pass

        print(dates_fix)
        out_path=os.path.join('F:/haihe_batch/volume_time_series/'+name+'/volume',dates[i].split("\\")[-1][0:-4]+".tif")
        print(out_path)
        out_path_filter = os.path.join('F:/haihe_batch/volume_time_series/'+name+'/volume',dates[i].split("\\")[-1][0:-4]+"_filter.tif")
        out_path_focal=os.path.join('F:/haihe_batch/volume_time_series/'+name+'/focal',dates[i].split("\\")[-1][0:-4]+"_focal.tif")
        out_path_clip = os.path.join('F:/haihe_batch/volume_time_series/' + name + '/focal_clip',
                                      dates[i].split("\\")[-1][0:-4] + "_final.tif")
        print(out_path_filter)
        # Replace a layer/table view name with a path to a dataset (which can be a layer file) or create the layer/table view within the script
        # The following inputs are layers or table views: "0801_filter.tif"
        if(os.path.exists(out_path_focal)):
            pass
        else:
            arcpy.gp.FocalStatistics_sa(out_path_filter, out_path_focal,
                                        "Circle 100 CELL", "MEAN", "DATA")

        # Replace a layer/table view name with a path to a dataset (which can be a layer file) or create the layer/table view within the script
        # The following inputs are layers or table views: "0801_filter.tif", "0801_fix"
        if(os.path.exists(out_path_clip)):
            pass
        else:
            arcpy.Clip_management(in_raster=out_path_focal,

                                  out_raster=out_path_clip,
                                  in_template_dataset=dates_fix, nodata_value="-3.402823e+38",
                                  clipping_geometry="ClippingGeometry", maintain_clipping_extent="NO_MAINTAIN_EXTENT")

        # if (os.path.exists(dates_eliminate)):
        #     pass
        # else:
        #     processing.run("native:deleteholes",
        #                    {'INPUT': dates[i], 'MIN_AREA': 50000,
        #                     'OUTPUT': dates_eliminate})
        #
        # if(os.path.exists(dates_fix)):
        #     pass
        # else:
        #     processing.run("native:fixgeometries", {'INPUT': dates_eliminate,
        #                                             'OUTPUT': dates_fix})
        # if(os.path.exists(out_path)):
        #     pass
        # else:
        #     processing.run("script:fwdet_v21", {'INPUT_DEM': dem,
        #                                         'INUN_VLAY': dates_fix,
        #                                         'numIterations': 100, 'slopeTH': 0, 'grow_metric': 'euclidean',
        #                                         'boundary': 'TEMPORARY_OUTPUT',
        #                                         'water_depth': out_path,
        #                                         'water_depth_filtered': out_path_filter})
        #
        #
        #
        #
        #
        #




#print(DBs)

