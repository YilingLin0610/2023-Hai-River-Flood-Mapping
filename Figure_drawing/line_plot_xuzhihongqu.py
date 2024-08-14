# coding=gbk

from pylab import *
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.interpolate import Akima1DInterpolator,PchipInterpolator
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import matplotlib.dates as mdates
from scipy.interpolate import splprep, splev
from fontTools.ttLib import TTFont

def transform_time(date_string):
    print(date_string)
    date_string_new=[]
    for i in range(len(date_string)):
        date_format = "%Y%m%d"

        date_string_new.append(datetime.strptime("20230"+str(date_string[i]), date_format))
    return date_string_new


def interpolate(date_data,values):
    # 将datetime对象转换为浮点数
    x = np.array([d.timestamp() for d in date_data])
    y = np.array(values)
    print("max:",max(y))

    # 找到最大值点
    max_index = np.argmax(y)
    x_max = x[max_index]
    print(x_max)


    # 创建Barycentric Interpolator
    #interpolator = BarycentricInterpolator(x, y)
    bci1 = PchipInterpolator(x, y)
    date_format = "%Y%m%d"
    x_smooth = np.linspace((datetime.strptime("20230728", date_format)).timestamp(), (datetime.strptime("20230928", date_format)).timestamp(), 1465)
    print(x_smooth)
    x_smooth = np.sort(np.append(x_smooth, x_max))
    y_smooth=bci1(x_smooth)
    max_index = np.argmax(y)



    print(max(y_smooth))
    x_smooth_new=[]
    for i in range(len(x_smooth)):
        x_smooth_new.append(datetime.fromtimestamp(int(x_smooth[i])))
    return x_smooth_new,y_smooth





print(matplotlib.get_cachedir())
font = TTFont(
    r'E:\Beijing_flood\0013Helvetica全套经典黑体英文粗体字体包PS字体下载AI素材LOGO字体\Helvetica 34款 (经典版)/Helvetica.ttf')
mpl.rcParams["font.sans-serif"] = ["Arial"]
# plt.rcParams['font.family'] = 'Helvetica'
# 读取Excel文件
csv_file = r'E:\Beijing_flood\jingjinji_batch\excels\volume_all.csv'
dict = {"day1": [0, 1, 2, 3], "day2": [4, 5, 6, 7, 8, 9], "day3": [10, 11, 12], "day4": [13, 14]}
font_color_list = ['#4292C3', '#4292C3', '#4292C3', '#00615D','#00615D',
                 '#DFC27E', '#DFC27E', '#533006']

df = pd.read_csv(csv_file,encoding='utf-8')
print(df)
#print(df)
date=df.keys()[1:]
print("date",date)
date=transform_time(list(date))
print(date)
colors = [ '#053061', '#2166AD', '#4292C3', '#666666', '#31948C',
           '#DFC27E', '#BE802D',  '#8D0C24']


fig = plt.figure(figsize=(8,6))
countries=np.array(df.iloc[:,0])
ax_objs = []
df = df.set_index(pd.Index(countries))
print(df)
x_smooths=[]
y_smooths=[]
values=[]
x_orginals=[]
y_orgianls=[]
for i in range(len(countries)):
    data=df.loc[countries[i]]
    print(data)
    x=data.index
    print(x)
    x = transform_time(x[1:])
    value=data.values
    x=np.array(x)
    value=np.array(value[1:])


    x=x[~(isnan(value))]
    value = value[~(isnan(value))]
    print(countries[i], value)
    print(countries[i],max(value),x[np.argmax(value)])
    np.argmax(value)
    x_smooth,y_smooth=interpolate(x, value)
    x_smooths.append(x_smooth)
    y_smooths.append(y_smooth)
    x_orginals.append(x)
    y_orgianls.append(value)
gs = grid_spec.GridSpec(len(countries), 1)
i = 0
names_xuzhihongqu=["Daluze","Ningjinpo","Xiaoqinghe","Gongquxi","Langouwa","Xianxian","Yongdinghe","Dongdian"]
for country in countries:
    ax_objs.append(fig.add_subplot(gs[i:i + 1, 0:]))
    #if(country=="霸州市"):


    ax_objs[-1].plot(x_smooths[i], y_smooths[i], color="black", lw=1)
    ax_objs[-1].fill_between(x_smooths[i], y_smooths[i], alpha=0.7,color=colors[i])
    #ax_objs[-1].scatter(x_orginals[i], y_orgianls[i])

    # setting uniform x and y lims
    ax_objs[-1].set_xlim(datetime.strptime("20230728","%Y%m%d" ),datetime.strptime("20230928","%Y%m%d" ))
    ax_objs[-1].set_ylim(0,150000)
    ax_objs[-1].set_xticks([])
    ax_objs[-1].set_yticks([])

    # make background transparent
    rect = ax_objs[-1].patch
    rect.set_alpha(0)

    # remove borders, axis ticks, and labels
    ax_objs[-1].set_yticklabels([])

    if i == len(countries) - 1:
        print(i)

        ax_objs[-1].set_xlabel("Date", fontsize=15)
        x_smooths_str=[]
        for n in range(len(x_smooths[i])):
            x_smooths_str.append( x_smooths[i][n].strftime("%m-%d"))
        ax_objs[-1].set_xticks(x_smooths[i][::96])
        ax_objs[-1].set_xticklabels(x_smooths_str[::96], fontsize=15, rotation=40)
        #ax_objs[-1].xaxis.xticks(ind[::40], x_smooths[i][::40], rotation=40)

        #print(x_smooths[i])
        # date_format = mdates.DateFormatter('%m-%d')
        # ax_objs[-1].xaxis.set_major_formatter(date_format)


    else:
        ax_objs[-1].set_xticklabels([])

    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        ax_objs[-1].spines[s].set_visible(False)

    ax_objs[-1].text(datetime.strptime("20230728","%Y%m%d" ), 0,names_xuzhihongqu[i], fontsize=15, ha="right",
                color=font_color_list[i])

    i += 1



ax_objs[5].spines["right"].set_visible(True)
ax_objs[5].yaxis.tick_right()
ax_objs[5].yaxis.set_label_position("right")
# ax_objs[7].set_ylabel("Iundated areas (ha)",fontweight="bold",fontsize=14)
ax_objs[5].text(4.2, 10, " 淹没面积 (公顷)", fontsize=15, ha="right",  rotation='vertical')
ax_objs[5].set_yticks([0, 50000, 100000,150000])
ax_objs[5].set_yticklabels([0, 50000, 100000,150000],  fontsize=15)
gs.update(hspace=-0.75)



plt.tight_layout()   #xlable坐标轴显示不全
plt.show()
#plt.savefig(r"E:\Beijing_flood\pictures_shanxi\xuzhihongqu_scale_2.svg")