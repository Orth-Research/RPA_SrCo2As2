#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as cb
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import csv


# In[34]:


width = 1.5*3.375
height = width/1.618
x_min = 0
x_max = 0.5
y_min = -0.4
y_max = 0.4000000000001
deltax = 0.01
deltay = 0.1


# In[25]:


doping_list = np.linspace(start=13.4, stop=12.6, num=41)
QP = 61 #number of q-points in which bare susceptibility is calculated on cluster
JUdiscretization = 50 #discretization of x-axis
N=25


# ### Phase diagram (Fig. 5a)

# In[39]:


#Phase diagram - no interactions on As
path = "C:/Users/amnedic/Documents/GitHub/RPA_SrCo2As2/results/"
table=csv.reader(open(path+'Phases_numba_phasediagram_A_JUpoints=50_MGXPNG_N=25.csv','r'))
n=[]
for row in table:
    interm = []
    for column in row:
        num = float(column)
        interm.append(num)
    n.append(interm)


# In[40]:


# #Phase diagram - interactions on As = Co
# table=csv.reader(open(path+'Phases_numba_phasediagram_includingAs=1_A_JUpoints=50_MGXPNG_N=25.csv','r'))
# nAs=[]
# for row in table:
#     interm = []
#     for column in row:
#         num = float(column)
#         interm.append(num)
#     nAs.append(interm)


# In[41]:


#Phase diagram - y-axis mu(T=0)

fig = plt.figure(1, figsize = [width,height])
ax = fig.add_subplot(111)
cax = ax.matshow(n, extent=[x_min,x_max,y_min,y_max], aspect='auto', cmap=cm.twilight_shifted, vmin=0, vmax=61) #cmap=cm.twilight_shifted_r)
plt.gca().xaxis.tick_bottom()
plt.hlines(y=0., xmin=0.0, xmax=0.5, color='w', linestyle='dashed', alpha=0.2)
plt.show()


# In[42]:


#fig.savefig('Phases_numba_phasediagram_A_JUpoints=50_MGXPNG_N=25.pdf', bbox_inches='tight', dpi=2000)


# In[43]:


#fig.savefig('Phases_numba_phasediagram_As_A_JUpoints=50_MGXPNG_N=25.pdf', bbox_inches='tight', dpi=2000)


# In[44]:


#Phase diagram - y-axis mu(T)

fig = plt.figure(1, figsize = [width,height])
ax = fig.add_subplot(111)
xlist=np.linspace(0,50)*0.01
ylist = np.genfromtxt(path+'mulistT.dat')
ylist[0]=0.4
ylist[-1]=-0.4
X, Y = np.meshgrid(xlist,ylist)
Z = n
plt.pcolor(X, Y, n, cmap=cm.twilight_shifted,linewidth=0,rasterized=True,vmin=0, vmax=61) #cmap=cm.twilight_shifted_r)
plt.gca().xaxis.tick_bottom()
plt.hlines(y=0., xmin=0.0, xmax=0.5, color='w', linestyle='dashed', alpha=0.2)

#ticks
grid_x_ticks = np.arange(x_min, x_max)
grid_y_ticks = np.arange(y_min, y_max, deltay)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
        
plt.show()


# ### Ucr for entire phase diagram (Fig. 5c)

# In[46]:


# Ucr for entire phase diagram (Fig. 5c) for mu(T=0)
Ucr1=[]
with open(path+'Ucr_numba_phasediagram_A_JUpoints=50_MGXPNG_N=25.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        Ucr1inter = []
        for column in row:
            Ucr1inter.append(float(column))
        Ucr1.append(Ucr1inter)


# In[47]:


fig = plt.figure(1, figsize = [width,height])
ax = fig.add_subplot(111)
plt.pcolor(X, Y, Ucr1, cmap=cm.gist_earth_r,linewidth=0,rasterized=True) #cmap=cm.twilight_shifted_r)
plt.gca().xaxis.tick_bottom()
plt.hlines(y=0., xmin=0.0, xmax=0.5, color='w', linestyle='dashed', alpha=0.2)

#ticks
grid_x_ticks = np.arange(x_min, x_max)
grid_y_ticks = np.arange(y_min, y_max, deltay)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.plot(np.linspace(0.1, 0.1, 31), np.linspace(-0.4, 0.4, 31), color='orange', marker='o', markersize=2, linestyle='-', linewidth=1)
plt.plot(np.linspace(0.25, 0.25, 31), np.linspace(-0.4, 0.4, 31), color='r', marker='o', markersize=2, linestyle='-', linewidth=1)
plt.plot(np.linspace(0.4, 0.4, 31), np.linspace(-0.4, 0.4, 31), color='magenta', marker='o', markersize=2, linestyle='-', linewidth=1)

plt.show()


# In[48]:


minval = np.amin(Ucr1)
maxval = np.amax(Ucr1)
#print(minval, maxval)


# In[92]:


#fig.savefig('Ucr_numba_phasediagram_A_JUpoints=50_MGXPNG_N=25.pdf', bbox_inches='tight', dpi=2000)


# ### Ucr cuts for fixed J/Ucr (Fig. 5d)

# In[52]:


#Cuts in PD U_cr for fixed J/Uc
Ucrtransp = np.transpose(Ucr1)
list_Ucr_01 =Ucrtransp[int(0.1*49/0.5)] #J/Uc=0.1
list_Ucr_025 =Ucrtransp[int(0.25*49/0.5)] #J/Uc=0.25
list_Ucr_04 = Ucrtransp[int(0.4*49/0.5)] #J/Uc=0.4
list_mu=ylist


# In[53]:


width = 1.5*3.375
height = width/1.618

Ucr_cut = plt.figure(1, figsize = [width,height])
ax = Ucr_cut.add_subplot(111)
plt.axvline(x=0.001, color='gray', linestyle='dashed', alpha=0.5)
plt.axvline(x=0.25, color='orange', linestyle='dashed', alpha=0.5)
plt.axhline(y=list_Ucr_01[22], color='gray', linestyle='dotted', markersize=0.01)
plt.axhline(y=list_Ucr_025[22], color='gray', linestyle='dotted', markersize=0.01)
plt.axhline(y=list_Ucr_04[22], color='gray', linestyle='dotted', markersize=0.01)
plt.xlim(-0.4, 0.4)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.plot(list_mu, list_Ucr_01, 'o', color='orange', markersize=4, linestyle='-')
plt.plot(list_mu, list_Ucr_025, 'o', color='r', markersize=4, linestyle='-')
plt.plot(list_mu, list_Ucr_04, 'o', color='magenta', markersize=4, linestyle='-')
 
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        
plt.show()


# In[54]:


#Ucr_cut.savefig('Ucr_cut.pdf', bbox_inches='tight', dpi=2000)


# ### Frustration (Fig. 5f)

# In[56]:


sublead = np.loadtxt(path+'Ucr_sub_numba_phasediagram_A_JUpoints=50_MGXPNG_N=25.txt')
lead = np.loadtxt(path+'Ucr_numba_phasediagram_A_JUpoints=50_MGXPNG_N=25.txt')


# In[57]:


# Frustratrtion phase diagram (Fig. 5f) for mu(T=0)
subl = (sublead-lead)/lead

from matplotlib.colors import ListedColormap,LinearSegmentedColormap
fig = plt.figure(1, figsize = [width,height])
ax = fig.add_subplot(111)
plt.pcolor(X, Y, subl, cmap=cm.gist_earth_r,linewidth=0,rasterized=True) #cmap=cm.twilight_shifted_r)
plt.gca().xaxis.tick_bottom()
plt.hlines(y=0., xmin=0.0, xmax=0.5, color='w', linestyle='dashed', alpha=0.2)

#ticks
grid_x_ticks = np.arange(x_min, x_max)
grid_y_ticks = np.arange(y_min, y_max, deltay)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# In[113]:


#fig.savefig('subl_plot.pdf', bbox_inches='tight', dpi=2000)


# ### J values for the entire PD (Fig. 5e)

# In[68]:


# J values for the entire phase diagram (Fig. 5e) for mu(T=0)

import numpy as np
table=csv.reader(open(path+'JvaluesPD.csv','r'))
n1=[]
for row in table:
    interm = []
    for column in row:
        num = float(column)
        interm.append(num)
    n1.append(interm)


# In[69]:


# #colorbar
fig = plt.figure(1, figsize = [width,height])
ax = fig.add_subplot(111)
plt.pcolor(X, Y, n1, cmap=cm.twilight_shifted,linewidth=0,rasterized=True) #cmap=cm.twilight_shifted_r)
#plt.gca().xaxis.tick_bottom()
#plt.hlines(y=0., xmin=0.0, xmax=0.5, color='w', linestyle='dashed', alpha=0.2)

#ticks
#grid_x_ticks = np.arange(x_min, x_max)
#grid_y_ticks = np.arange(y_min, y_max, deltay)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
        
plt.show()


# In[70]:


#fig.savefig("Jvalues.pdf", format='pdf', dpi = 600, bbox_inches="tight", transparent=True)


# In[ ]:





# ### Phase diagram - J as x-axis (Fig. 5b)

# In[71]:


xyz_list = []
for i in range(len(n)):
    for j in range(len(n[0])):
        xyz_list.append([n1[i][j],ylist[i],n[i][j]])
xyz_array=np.array(xyz_list)


# In[74]:


#Phase diagram - J as x-axis (Fig. 5b)

fig = plt.figure(1, figsize = [width,height])
ax = fig.add_subplot(111)
plt.gca().xaxis.tick_bottom()
plt.hlines(y=0., xmin=0.0, xmax=1.2, color='w', linestyle='dashed', alpha=0.2)

plt.scatter(xyz_array[:,0], xyz_array[:,1], c=xyz_array[:,2],cmap=cm.twilight_shifted, vmin=0, vmax=61,  s=30)

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


# In[75]:


#fig.savefig("PD_J.pdf", format='pdf', dpi = 600, bbox_inches="tight", transparent=True)


# In[ ]:




