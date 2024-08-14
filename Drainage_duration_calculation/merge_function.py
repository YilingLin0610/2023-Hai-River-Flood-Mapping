"""
Calculate the drainage duration by regions
Author: Yiling Lin
"""
import numpy as np
from datetime import datetime
from utils import read_raster_as_array,writeTiff
import os
from tqdm import tqdm
from skimage.transform import resize


def get_date_series(in_files_path):
    '''
    Get the date string array.
    ——————————————————————————
    @param：
        in_files_path: The file path where the multi-temporal flood extent TIFFs are stored (e.g., with filenames such as 0601.tif).
    @return:
        dates_str: The date string array.
    '''

    files_name= [filename for filename in os.listdir(in_files_path) if
                    filename.endswith("tif")]
    dates_str=[filename[0:4] for filename in files_name]
    return dates_str

def date_calculate(begin_date_str,end_date_str):
    '''
    Calculate the number of days between two dates
    ——————————————————————————
    @param：
        begin_date_str: The earlier day: '0729' (str)
        end_date_str: The later day: '0729' (str)
    @return:
        interval_days: The number of days
    '''

    date1 = datetime.strptime(begin_date_str, "%m%d")
    date2 = datetime.strptime(end_date_str, "%m%d")
    delta = date2 - date1
    interval_days = delta.days

    return interval_days



def next_index_find(start_index,array):
    next_index=np.where(~np.isnan(array[start_index+1:]))[0]

    if (next_index.size>0):
        next_index=next_index[0]+start_index+1
    else:
        next_index=start_index+1
    print(next_index)
    return next_index


def calculate_interval_days(in_files_path,out_tif):
    files_name = [os.path.join(in_files_path,filename) for filename in os.listdir(in_files_path) if
                  filename.endswith(".tif")]
    print(files_name[0])
    array,proj,geotrans=read_raster_as_array(files_name[0])
    array_ref=array
    print(np.shape(array_ref))
    for i in range(len(files_name)):
        if(i>0):
            array_1,proj,geotrans=read_raster_as_array(files_name[i])
            #print(array_1)
            #print(np.shape(array_1))
            if(np.shape(array_1)!=np.shape(array_ref)):
                print(files_name[i])
                array_1=resize(array_1,np.shape(array_ref))



            array = np.concatenate((array,array_1), axis=2)



    interval_array=np.zeros((np.shape(array)[0],np.shape(array)[1]))
    dates_str=get_date_series(in_files_path)
    dates_str = np.array(dates_str)


    for i in tqdm(range(np.shape(array)[0])):
        for n in range(np.shape(array)[1]):
            #print(array[i,n,:])
            pixel_array=array[i,n,:]
            #提取出该像素所有有值的数
            condition=(pixel_array ==1) | (pixel_array ==2)

            pixel_array_index = np.where(condition)[0]  # 所有非nan的数值的index
            #pixel_array_index_2=np.where(pixel_array ==2)[0]
            #pixel_array_index=np.concatenate(pixel_array_index_1,pixel_array_index_2)
            #np.sort(pixel_array_index)
            pixel_array_new=pixel_array[pixel_array_index]
            dates_str_new = dates_str[pixel_array_index]


            result_index = np.where(pixel_array_new == 2)[0] #提取淹没像素的位置

            if(len(result_index)==0):
                interval_array[i,n]=np.nan #未被淹没
            elif(len(result_index)==1):
                if(result_index[-1]+1==len(dates_str_new)): #在最后一天被淹没
                    interval_array[i, n] = np.nan
                elif(len(dates_str_new)==1): #中间某一天淹没了一次，其余无数据
                    interval_array[i, n] = np.nan
                else:
                    interval_array[i, n] = date_calculate(dates_str_new[result_index[0]],
                                                          dates_str_new[result_index[-1]+1])

            else:
                if (result_index[-1] + 1 == len(dates_str_new)):  # 在最后一天被淹没
                    interval_array[i, n] = date_calculate(dates_str_new[result_index[0]], dates_str_new[result_index[-1]])
                else:
                    interval_array[i, n] = date_calculate(dates_str_new[result_index[0]],
                                                          dates_str_new[result_index[-1]+1])




    writeTiff(interval_array, geotrans, proj, out_tif)













if __name__=="__main__":
    file_paths=r"F:\retreat_time_batch\clips_haihe_basin\interval_results_0706"
    in_files_paths=[filepath for filepath in os.listdir(file_paths)]
    print(in_files_paths)

    for in_files_path in in_files_paths:
        #if(in_files_path=="0"):
        #print(in_files_path)

        out_tif=os.path.join(r"F:\retreat_time_batch\clips_haihe_basin\results_clip_0706",in_files_path+".tif")
        in_files_path=os.path.join(file_paths,in_files_path)
        if(os.path.exists(out_tif)):
            print(out_tif)
        else:
            calculate_interval_days(in_files_path, out_tif)


