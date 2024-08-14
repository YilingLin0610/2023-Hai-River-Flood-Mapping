"""
Draw a barsh figure of GDP ( Figure 1C)
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from sklearn.linear_model import LinearRegression
# Set the font family as "Times New Roman"
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Arial'


def get_distance_from_point_to_line(point, line_point1, line_point2):
    '''
    Calculate the distance from point to line
    ！！！！！！！！！！！！！！！！！！！！！！！！！！
    @param��
        point: The coordinate of the point
        line_point1: The first point on the line.
        line_point2: The second point on the line.
    @return:
        distance: The distance.
    '''

    # When the coordinates of the two points are the same, return the distance between the points.
    if line_point1 == line_point2:
        point_array = np.array(point)
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array - point1_array)
    # Calculate three basic parameters of the line.
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    # Calculate the distance from point to line
    distance = (A * point[0] + B * point[1] + C) / (np.sqrt(A ** 2 + B ** 2))
    return distance

names=["daluze","dongdian","langouwa","ningjinpo","xianxian","xiaoqinghe","yongdinghe","gongquxi"]
# The duration of draining 80% of the flood extent.
days=[7.763024894,34.09997074,9.93760916,13.39142858,8.680007488,7.637323767,6.771490507,5.29]
# The maximum flood ratio of each FDA.
percent=[0.138429244,0.551579434,0.415140888,0.097063091,0.190579081,0.08850843,0.158634358,0.698]
plt.figure(figsize=(12,9))
# Build a linear fitting model.
days_2=np.array(days)
days_2=days_2.reshape(-1,1)
percent_2=np.array(percent)
percent_2=percent_2.reshape(-1,1)
model = LinearRegression(fit_intercept=False)
model.fit(days_2,percent_2)
x=np.array(range(0,40))
x=x.reshape(-1,1)
prediction=model.predict(x)

distances=[]
for i in range(len(days_2)):
    distance=get_distance_from_point_to_line([days_2[i][0],percent_2[i][0]], [x[0][0],prediction[0][0]], [x[-1][0],prediction[-1][0]])
    distances.append(distance)


sizes=[]


cbar=plt.scatter(days,percent,s=1100,c=distances,cmap='RdBu_r',alpha=0.8,edgecolors="grey")
plt.xlabel("Retreat days",fontsize=20,color="#4b4b4b")
plt.ylabel("Maximum inundated percentage",fontsize=20,color="#4b4b4b")
plt.ylim([0,0.85])
plt.xlim([0,40])

cbar.set_clim(-0.3, 0.2)
plt.colorbar()
s=np.array(range(0,9,1))
s=np.round(s*0.1,1)
ax=plt.gca()
ax.set_xticklabels(range(0,40,5),fontsize=20,color="#4b4b4b")
ax.set_yticklabels(s,fontsize=20,color="#4b4b4b")


plt.show()
#plt.savefig(r"E:\Beijing_flood\pictures_shanxi\pictures\resillience_0719.svg")
