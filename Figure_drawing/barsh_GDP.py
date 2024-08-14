"""
Draw a barsh figure of GDP ( Figure 1C)
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# Set the font family as "Times New Roman"
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Arial'
# Sort the data
GDP={"xianxian":540,"xiaoqinghe":3478,"ningjinpo":2040,"yongdinghe":2792,"langouwa":810,"dongdian":1709,"daluze":1125,"gongquxi":96}
GDP=sorted(GDP.items(),key=lambda x:x[1],reverse=False)

gdp=[]
names=[]
for i in range(len(GDP)):
    gdp.append(GDP[i][1]/100)
    names.append(GDP[i][0])


y = np.arange(len(gdp))
plt.figure(figsize=(10,5))

plt.barh(range(len(gdp)), gdp, height=0.95, color="#BAB0AC") # ¥”œ¬Õ˘…œª≠
plt.yticks(range(len(names)),names,fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel("Detention Basin",fontsize=25)
plt.xlabel("GDP (Billion Yuan)",fontsize=25)

#plt.savefig("E:\Beijing_flood\pictures_shanxi\pictures\\GDP.svg")
plt.show()