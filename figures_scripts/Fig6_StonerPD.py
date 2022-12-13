#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import csv

path = "C:/Users/amnedic/Documents/GitHub/RPA_SrCo2As2/results/"

width = 1.5*3.375
height = width/1.618
x_min = 0
x_max = 0.5
y_min = -0.4
y_max = 0.4000000000001
deltax = 0.01
deltay = 0.1

#Phase diagram - y-axis mu(T)
table=csv.reader(open(path+'Phases_numba_phasediagram_A_JUpoints=50_MGXPNG_N=25.csv','r'))
n=[]
for row in table:
    interm = []
    for column in row:
        num = float(column)
        interm.append(num)
    n.append(interm)

xlist=np.linspace(0,50)*0.01
ylist = np.genfromtxt(path+'mulistT.dat')
ylist[0]=0.4
ylist[-1]=-0.4
X, Y = np.meshgrid(xlist,ylist)
Z = n

# J values for the entire phase diagram (Fig. 5e) for mu(T=0)
table=csv.reader(open(path+'JvaluesPD.csv','r'))
n1=[]
for row in table:
    interm = []
    for column in row:
        num = float(column)
        interm.append(num)
    n1.append(interm)
    
xyz_list = []
for i in range(len(n)):
    for j in range(len(n[0])):
        xyz_list.append([n1[i][j],ylist[i],n[i][j]])
xyz_array=np.array(xyz_list)

# Ucr for entire phase diagram
Ucr1=[]
with open(path+'Ucr_numba_phasediagram_A_JUpoints=50_MGXPNG_N=25.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        Ucr1inter = []
        for column in row:
            Ucr1inter.append(float(column))
        Ucr1.append(Ucr1inter)
        
        
#Cuts in PD U_cr for fixed J/Uc
Ucrtransp = np.transpose(Ucr1)
list_Ucr_01 =Ucrtransp[int(0.1*49/0.5)] #J/Uc=0.1
list_Ucr_025 =Ucrtransp[int(0.25*49/0.5)] #J/Uc=0.25
list_Ucr_04 = Ucrtransp[int(0.4*49/0.5)] #J/Uc=0.4
list_mu=ylist

fig = plt.figure(1, figsize = [width,height])
ax = fig.add_subplot(111)
plt.gca().xaxis.tick_bottom()
plt.hlines(y=0., xmin=0.0, xmax=1.2, color='w', linestyle='dashed', alpha=0.2)
plt.scatter(xyz_array[:,0], xyz_array[:,1], c=xyz_array[:,2],cmap=cm.twilight_shifted, vmin=0, vmax=61,  s=0)

ax.set_xticklabels(["$0.0$", "$0.2$", "$0.4$", "$0.6$", "$0.8$","$1.$","$1.2$"], color="k", size=12)
ax.set_yticklabels(["$-0.4$", "$-0.3$", "$-0.2$", "$-0.1$", "$0.0$", "$0.1$", "$0.2$", "$0.3$", "$0.4$"], color="k", size=12)
fig.tight_layout()
ax.set_ylim(-0.4, 0.4)
ax.set_xlim(0,1.21)
plt.scatter(list_Ucr_01*0.1, list_mu, color='orange',s=10)
plt.plot(list_Ucr_01*0.1, list_mu, color='orange', ls='-')
plt.scatter(list_Ucr_025*0.25, list_mu, color='r', linestyle='-',s=8)
plt.plot(list_Ucr_025*0.25, list_mu, color='r', ls='-')
plt.scatter(list_Ucr_04*0.4, list_mu,  color='magenta', linestyle='-', s=8)
plt.plot(list_Ucr_04*0.4, list_mu, color='magenta', ls='-')

plt.show()

width = 1.5*3.375
height = width/1.618

Ucr_cut = plt.figure(1, figsize = [width,height])
ax = Ucr_cut.add_subplot(111)
plt.axvline(x=0, color='gray', linestyle='dashed', alpha=0.5)
plt.axhline(y=list_Ucr_01[20], color='gray', linestyle='dotted', markersize=0.01)
plt.axhline(y=list_Ucr_025[20], color='gray', linestyle='dotted', markersize=0.01)
plt.axhline(y=list_Ucr_04[20], color='gray', linestyle='dotted', markersize=0.01)
plt.xlim(-0.4, 0.4)
plt.ylim(1.3,4.2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.plot(list_mu, list_Ucr_01, 'o', color='orange', markersize=4, linestyle='-')
plt.plot(list_mu, list_Ucr_025, 'o', color='r', markersize=4, linestyle='-')
plt.plot(list_mu, list_Ucr_04, 'o', color='magenta', markersize=4, linestyle='-')
 
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        
plt.show()