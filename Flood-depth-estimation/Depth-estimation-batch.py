"""
Estimate the flood depth in batches.
Author: Yiling Lin.
"""

import os

DB_dir=r"F:\haihe_batch\detention_basins_areas"
DEM_dir=r"F:\haihe_batch\DEM_xuzhihongqu"

DBs=[os.path.join(DB_dir, filename) for filename in os.listdir(DB_dir)]
for DB in DBs:
    # Read the flood extent shapefile
    dates=[os.path.join(DB, filename) for filename in os.listdir(DB) if
                  filename.endswith(".shp")]


    # Read the corresponding DEM TIFF file
    if(len(dates)>0):
        name = dates[0].split("\\")[3]
        dem=os.path.join(DEM_dir,name+".tif")
    for i in range(len(dates)):
        out_path = os.path.join('F:/haihe_batch/volume_time_series/' + name + '/volume',
                                dates[i].split("\\")[-1][0:-4] + ".tif")
        out_path_filter = os.path.join('F:/haihe_batch/volume_time_series/' + name + '/volume',
                                       dates[i].split("\\")[-1][0:-4] + "_filter.tif")
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

        # Eliminate the holes
        if (os.path.exists(dates_eliminate)):
            pass
        else:
            processing.run("native:deleteholes",
                           {'INPUT': dates[i], 'MIN_AREA': 50000,
                            'OUTPUT': dates_eliminate})

        # Fix the geometries
        if(os.path.exists(dates_fix)):
            pass
        else:
            processing.run("native:fixgeometries", {'INPUT': dates_eliminate,
                                                    'OUTPUT': dates_fix})
        # Estimate the flood depth
        if(os.path.exists(out_path)):
            pass
        else:
            processing.run("script:fwdet_v21", {'INPUT_DEM': dem,
                                                'INUN_VLAY': dates_fix,
                                                'numIterations': 100, 'slopeTH': 0, 'grow_metric': 'euclidean',
                                                'boundary': 'TEMPORARY_OUTPUT',
                                                'water_depth': out_path,
                                                'water_depth_filtered': out_path_filter})





