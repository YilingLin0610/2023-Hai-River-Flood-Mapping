"""
A multi-line plot show times series
Author: Yiling Lin
"""

from pylab import *
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from fontTools.ttLib import TTFont

def transform_time(date_string):
    '''
    Transform time string into datetime format.
    ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
    @param£º
        date_string: a list storing date strings.
    @return:
         date_string_new: a list storing datetime format string.
    '''
    print(date_string)
    date_string_new=[]
    for i in range(len(date_string)):
        date_format = "%Y%m%d"
        date_string_new.append(datetime.strptime("20230"+str(date_string[i]), date_format))
    return date_string_new


def interpolate(date_data,values):
    '''
    Interpolate: Here we ensure that the maximum and minimum values of the original data match
    the maximum and minimum values of the interpolated results.
    ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
    @param£º
        date_data: Date (x-axis).
        values: Detained floodwater volume (y-axis).
    @return:
         x_smooth_new: Interpolated date.
         y_smooth: Interpolated detained floodwater volume.
    '''
    x = np.array([d.timestamp() for d in date_data])
    y = np.array(values)
    max_index = np.argmax(y)
    x_max = x[max_index]

    bci1 = PchipInterpolator(x, y)
    date_format = "%Y%m%d"
    x_smooth = np.linspace((datetime.strptime("20230728", date_format)).timestamp(), (datetime.strptime("20230928", date_format)).timestamp(), 1465)
    x_smooth = np.sort(np.append(x_smooth, x_max))
    y_smooth=bci1(x_smooth)
    x_smooth_new=[]
    for i in range(len(x_smooth)):
        x_smooth_new.append(datetime.fromtimestamp(int(x_smooth[i])))
    return x_smooth_new,y_smooth

mpl.rcParams["font.sans-serif"] = ["Arial"]
# Read the excel
csv_file = r'E:\Beijing_flood\jingjinji_batch\excels\volume_all.csv'
dict = {"day1": [0, 1, 2, 3], "day2": [4, 5, 6, 7, 8, 9], "day3": [10, 11, 12], "day4": [13, 14]}
font_color_list = ['#4292C3', '#4292C3', '#4292C3', '#00615D','#00615D',
                 '#DFC27E', '#DFC27E', '#533006']

df = pd.read_csv(csv_file,encoding='utf-8')
date=df.keys()[1:]
date=transform_time(list(date))
colors = [ '#053061', '#2166AD', '#4292C3', '#666666', '#31948C',
           '#DFC27E', '#BE802D',  '#8D0C24']

fig = plt.figure(figsize=(8,6))
countries=np.array(df.iloc[:,0])
ax_objs = []
df = df.set_index(pd.Index(countries))
x_smooths=[]
y_smooths=[]
values=[]
x_orginals=[]
y_orgianls=[]
for i in range(len(countries)):
    data=df.loc[countries[i]]
    x=data.index
    x = transform_time(x[1:])
    value=data.values
    x=np.array(x)
    value=np.array(value[1:])
    x=x[~(isnan(value))]
    value = value[~(isnan(value))]
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
    ax_objs[-1].plot(x_smooths[i], y_smooths[i], color="black", lw=1)
    ax_objs[-1].fill_between(x_smooths[i], y_smooths[i], alpha=0.7,color=colors[i])
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

        ax_objs[-1].set_xlabel("Date", fontsize=15)
        x_smooths_str=[]
        for n in range(len(x_smooths[i])):
            x_smooths_str.append( x_smooths[i][n].strftime("%m-%d"))
        ax_objs[-1].set_xticks(x_smooths[i][::96])
        ax_objs[-1].set_xticklabels(x_smooths_str[::96], fontsize=15, rotation=40)
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
ax_objs[5].text(4.2, 10, " Inundated area (ha)", fontsize=15, ha="right",  rotation='vertical')
ax_objs[5].set_yticks([0, 50000, 100000,150000])
ax_objs[5].set_yticklabels([0, 50000, 100000,150000],  fontsize=15)
gs.update(hspace=-0.75)



plt.tight_layout()
plt.show()
#plt.savefig(r"E:\Beijing_flood\pictures_shanxi\xuzhihongqu_scale_2.svg")