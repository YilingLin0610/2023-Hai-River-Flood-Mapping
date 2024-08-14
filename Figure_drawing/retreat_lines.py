#coding=gbk
from utils import read_raster_as_array
import numpy as np
import os
import math
from tqdm import tqdm
from scipy.interpolate import Akima1DInterpolator

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


# 设置全局字体为 "Times New Roman"
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Arial'
from sympy import symbols, Eq, solve

def solve_problem(a,b,c):
    result=np.log(((1/0.8)-a)/b)/c
    return  result



# 定义非线性函数模型
def nonlinear_model(x, a,b,c):
    return 1/(a+b*np.exp(c*x))
    #return 1 / (1+(np.exp(a*x*x+b*x+c)))


# 定义非线性拟合函数
def perform_nonlinear_fit(x_data, y_data, initial_guess=(1.0, 1.0, 1.0)):
    #print(x_data,y_data)
    param_bounds = ([-np.inf,-np.inf,-0.8],[np.inf,np.inf,np.inf])
    params, covariance = curve_fit(nonlinear_model, x_data, y_data,maxfev = 1000000,bounds=param_bounds)
    a,b,c= params
    x_smooth=range(60)
    y_fit = nonlinear_model(x_smooth, a,b,c)

    return x_smooth,y_fit,a,b,c



def interpolation(x,y):
    bci1 = Akima1DInterpolator(x, y)
    x_smooth=range(60)
    y_smooth=bci1(x_smooth)
    return x_smooth,y_smooth
def threshold_mask(arr,arr2,threshold):
    diff = np.empty(arr.shape)
    diff[0] = np.inf  # always retain the 1st element
    diff[1:] = np.diff(arr)
    mask = diff > threshold

    new_arr = arr[mask]
    new_arr2 = arr2[mask]
    return new_arr,new_arr2
def percentage_array(in_tif_name):
    array, proj, geotrans = read_raster_as_array(in_tif_name)
    percentage_array=[]
    #计算array中大于某个数的像素百分比
    all_values=np.unique(array)
    total_inundated_num=np.sum(array>0)
    #print(total_inundated_num)
    non_inundated_num = np.sum(array == 0)
    boundary_num=np.sum(array==-1)
    all_values_2=[]
    #print("total",total_inundated_num)
    # print("淹没总面积：",np.sum(array>0))
    # print("总面积：",np.sum(array>-1))
    # print("淹没比例",np.sum(array>0)/np.sum(array>-1))
    sum_area=np.sum(array>0)
    ratio=np.sum(array>0)/np.sum(array>-1)
    for i in range(len(all_values)):
        if(all_values[i]<52 and all_values[i]>=0):
            percentage=(np.sum(array<=all_values[i])-non_inundated_num-boundary_num)/(np.sum(array>0))
            percentage_array.append(
                    (np.sum(array <= all_values[i]) - non_inundated_num - boundary_num) / (np.sum(array > 0)))
            all_values_2.append(all_values[i])

    #percentage_array,all_values_2=threshold_mask( np.array(percentage_array), np.array(all_values_2),0.005)

    #print(percentage_array)
    # print(all_values)
    # print(percentage_array)
    # all_values=all_values[0:-1]
    # percentage_array=percentage_array[0:-1]
    all_values=all_values_2
    #print(all_values)
    x=list(range(1,70))
    #all_values,percentage_array=inte rpolation(all_values,percentage_array)
    x_smooth,y_fit,a,b,c=perform_nonlinear_fit(all_values, percentage_array)
    # print(a,b,c)
    # print("time series")
    # print(percentage_array)
    # print(all_values)
    # print("end")
    return percentage_array,all_values,total_inundated_num,x_smooth,y_fit,a,b,c,ratio,sum_area
    #plt.plot(x,percentage_array)



if __name__=="__main__":
    #file_path=r"F:\retreat_time_batch\continent_retreattime_results\retreat_tif"
    file_path=r"F:\retreat_time_batch\xuzhihongqu_retreattime_results_2"

    legends=[]
    in_tif_names_old=[os.path.join(file_path,filename) for filename in os.listdir(file_path) if filename.endswith(".tif")]
    in_tif_names=[]
    for tif_name in in_tif_names_old:
        if(tif_name.endswith("_0.tif")):
            in_tif_names.append(tif_name)
        else:
            pass
            #in_tif_names.append(tif_name)

    #print(in_tif_names)
    num=0
    a_all={}
    area_all={}
    b_all = {}
    ratio_all={}
    sum_areas={}
    colors=["#765A3C","#C87763","#7b9db8","#4498E5","#626262","#7D9F9B","#F8A662","#9FAE74"]
    order = 0
    for tif_name in tqdm(in_tif_names):

        #print(tif_name)
        legend=tif_name.split("\\")[-1][0:-11]


        array,x,total_inundated_num,x_smooth,y_fit,a,b,c,ratio,sum_area=percentage_array(tif_name)
        #expression="1/y={}+{}*(1/x)".format("{:.2f}".format(a),"{:.2f}".format(b))
        retreat_time=solve_problem(a, b, c)
        #expression="a="+str("{:.2f}".format(a))+"b="+str("{:.2f}".format(b))+"c="+str("{:.2f}".format(c))
        expression="time="+str("{:.2f}".format(retreat_time))
        #print(total_inundated_num)
        a_all[legend]=retreat_time
        ratio_all[legend]=ratio
        sum_areas[legend]=sum_area





        if(total_inundated_num>100):
            num=num+1
            plt.figure()
            plt.scatter(x,array,label="Real data",color=colors[order],s=200)
            order = order + 1

            legend=legend.capitalize()
            print(legend)
            legends.append(legend)
            plt.title(legend, fontsize=20)
            #plt.title(legend,fontsize=14)
            plt.plot(x_smooth, y_fit,label="Fit data",color="black",linewidth=3)
            xxx = [0, retreat_time]
            yyy = [0.8, 0.8]
            plt.plot(xxx, yyy, linewidth=2, linestyle='--',color="black")
            xxx = [retreat_time, retreat_time]
            yyy = [0, 0.8]
            plt.plot(xxx, yyy, linewidth=2, linestyle='--', color="black")
            plt.xlabel("Days",fontsize=30)
            plt.ylabel("Percentage of Flood recession",fontsize=30)
            ax = plt.gca()
            #plt.text(0, 0.6,expression,fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.ylim([0,1.2])
            plt.xlim([0, 60])
            plt.yticks([0,0.5,0.8,1])
            plt.xticks([0,20, 40, 60])
            plt.text(retreat_time+1,0.025,round(retreat_time,2),fontsize=40)
            plt.tick_params(axis='x', labelsize=40)
            plt.tick_params(axis='y', labelsize=40)
            plt.legend(fontsize=25,loc="lower right")
            #plt.legend(legends)
            #plt.show()
            plt.savefig(r"F:\retreat_time_batch\continent_retreattime_results\images\\"+legend+"_threshold.svg")


    # 将字典转换为数据帧
    df1 = pd.DataFrame(a_all,index=[0])
    df2=pd.DataFrame(ratio_all,index=[0])
    df3=pd.DataFrame(sum_areas,index=[0])
    #df2 = pd.DataFrame(b_all,index=[0])

    # 合并数据帧
    merged_df = pd.concat([df1,df2,df3])
    #merged_df=merged_df.transpose()

    # 输出到Excel文件
    merged_df.to_excel(r"F:\retreat_time_batch\retreat_speed_2.xlsx", index=False)

        #plt.plot(x_smooth, y_fit)

