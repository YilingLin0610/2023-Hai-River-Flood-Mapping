"""
Draw the pie charts of GDP and population ( Figure 1B)
Author: Yiling Lin
"""
from utils import read_raster_as_array
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Helvetica'


def pie_chart(data,title,area_name):
    '''
    Draw a pie charts with inner and outer circles.
    ！！！！！！！！！！！！！！！！！！！！！！！！！！
    @param��
        point: The coordinate of the point
        line_point1: The first point on the line.
        line_point2: The second point on the line.
    @return:
        distance: The distance.
    '''

    plt.style.use('ggplot')

    x1=[300]
    colors1 = ['#E05658',"#BAB0AC",  'white']

    plt.pie(x=x1, colors=["black"], wedgeprops={'width': 0.02, 'edgecolor': 'black'})  # Outer circle
    plt.pie(x=data, colors=colors1, radius=0.98, wedgeprops={'width': 0.3, 'edgecolor': 'w'},
            textprops={'color': 'w'},startangle=220,counterclock=False)  # Inner circle
    plt.text(-0.2,0.1,title,fontsize=25)
    plt.text(-0.2, -0.2, "Thousand \n Pop", fontsize=10)
    #plt.savefig("E:\Beijing_flood\pictures_shanxi\pictures\\"+area_name+".svg")
    plt.show()





import matplotlib
# 譜崔畠蕉忖悶葎 "Times New Roman"
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Helvetica'
Population={"xianxian":118000,"xiaoqinghe":952000,"ningjinpo":695000,"yongdinghe":270000,"langouwa":189000,"dongdian":163000,"daluze":383000,"gongquxi":5900}

dict={0:"nan",1:"water",2:"Trees",4:"Flooded vegetation",5:"Crops",7:"Built Area",8:"Bare ground",9:"Snow/Ice",10:"Clouds",11:"Rangeland"}
if __name__=="__main__":
    filepath=r"E:\haihe_data\landuse\landuse_xzhq"
    tifs_names = [os.path.join(filepath, filename) for filename in os.listdir(filepath) if
                  filename.endswith("gongquxi.tif")]

    for filename in tifs_names:
        raster_out, proj, geotrans = read_raster_as_array(filename)
        unique_values=np.unique(raster_out)
        propotion={}


        for value in unique_values:
            strings=dict[value]
            propotion[strings]=round(np.sum(raster_out[raster_out==value])/np.sum(raster_out[raster_out>0])*100,2)
        ratio=[propotion["Crops"],propotion["Built Area"],100-propotion["Built Area"]-propotion["Crops"]]
        #print(propotion)
        print(ratio)
        area_name = filename.split("\\")[-1]
        area_name = area_name.split(".")[0]
        print(area_name)
        pop=int(Population[area_name]/1000)
        pie_chart(ratio,pop,area_name)


        #print(np.unique(raster_out))


    print(tifs_names)
    #read_series(filepath)