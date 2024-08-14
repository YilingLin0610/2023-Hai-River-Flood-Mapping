"""
Delete small polygons in maximum flood extent shapefile.
I used Python to process it because the large number of polygons often causes ArcMap to crash.
"""
import geopandas as gpd

gdf=gpd.read_file(r"F:\haihe_batch\erased_shapefile\all_erased_0609_final.shp")
print("read all")
gdf=gdf[gdf.Area>2000]
gdf.to_file(r"F:\haihe_batch\erased_shapefile\all_erased_0609_final_filter.shp")