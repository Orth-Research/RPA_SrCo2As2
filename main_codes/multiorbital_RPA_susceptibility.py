#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
import csv

doping_list = np.linspace(13.4, 12.6, num=41)
QP = 61 #number of q-points in which bare susceptibility is calculated on cluster #13+11
JUdiscretization = 50 #discretization of x-axis
bandstructure = 'A'

def inv_phys_susc(U, JUratio):
    J = JUratio*U
    U1 = U-2*J
    J1 = J
    o=5
    inter=np.zeros((o**2,o**2), dtype=float)
    lis = []
    for i in range(o**2):
        inter[i,i] = U1
    for i in range(o):
        param = o*i+i
        lis.append(param)
        inter[param,param]=U
    for k in lis:
        for m in lis:
            if m !=k:
                inter[k,m] = J
    for k in range(o):
        for m in range(o):
            A = o*k+m
            B = o*m+k
            if m !=k:
                inter[A,B] = J1
    inter5=inter
    
    Co = np.zeros((80,80))
    Co[0:5,0:5] = inter5[0:5,0:5]
    Co[16:21,16:21] = inter5[5:10,5:10]
    Co[32:37,32:37] = inter5[10:15,10:15]
    Co[48:53,48:53] = inter5[15:20,15:20]
    Co[64:69,64:69] = inter5[20:25,20:25]
    Co[0:5,16:21] = inter5[0:5,5:10]
    Co[0:5,32:37] = inter5[0:5,10:15]
    Co[0:5,48:53] = inter5[0:5,15:20]
    Co[0:5,64:69] = inter5[0:5,20:25]
    Co[16:21,0:5] = inter5[5:10,0:5]
    Co[16:21,32:37] = inter5[5:10,10:15]
    Co[16:21,48:53] = inter5[5:10,15:20]
    Co[16:21,64:69] = inter5[5:10,20:25]
    Co[32:37,0:5] = inter5[10:15,0:5]
    Co[32:37,16:21] = inter5[10:15,5:10]
    Co[32:37,48:53] = inter5[10:15,15:20]
    Co[32:37,64:69] = inter5[10:15,20:25]
    Co[48:53,0:5] = inter5[15:20,0:5]
    Co[48:53,16:21] = inter5[15:20,5:10]
    Co[48:53,32:37] = inter5[15:20,10:15]
    Co[48:53,64:69] = inter5[15:20,20:25]
    Co[64:69,0:5] = inter5[20:25,0:5]
    Co[64:69,16:21] = inter5[20:25,5:10]
    Co[64:69,32:37] = inter5[20:25,10:15]
    Co[64:69,48:53] = inter5[20:25,15:20]
    
    As = np.zeros((48,48))

    INT_inter = np.zeros((269,269))
    INT=np.zeros((256,256))
    INT_inter[0:80,0:80] = Co
    INT_inter[85:165,85:165] = Co
    INT_inter[170:218,170:218] = As
    INT_inter[221:269,221:269] = As
    INT = INT_inter[0:256,0:256]

    o=16
    lis = []
    for i in range(o):
        lis.append(o*i+i)
    lis256x256 = []
    for i in lis:
        for k in lis:
            lis256x256.append(i*o**2+k)
    susc_resh=np.reshape(np.transpose(suscall), (o**2,o**2))
    inver = np.dot(np.linalg.inv(np.identity(o**2) - np.dot((susc_resh), INT)), susc_resh)
    eigenValues = la.eigh(inver)[0] #selecting eigenvalues only
    eigenValues = np.real(eigenValues)
    inveigenValues = 1/eigenValues
    
    idx = inveigenValues.argsort() #sorting eigenvalues
    inveigenValues = inveigenValues[idx]
    
    return abs(inveigenValues[0])

#automatic - full phase diagram
enum = 0 #enumerator
Ucrphasediagram = [] #Ucr on phase diagram
phases = [] #phases on phase diagram
t0=time.time()
for mu in doping_list:
    mucutsUcr = []
    mucutsphase = []
    enum += 1
    #reding in all files
    suscalllist = []
    for qpoint in range(30,41):
        suscall = np.loadtxt('GXPnumba_N=25_fil='+str(mu)+'_q='+str(qpoint)+'.dat', dtype=complex)
        suscalllist.append(suscall)
    for t in range(0, JUdiscretization):
        JUratio = 0.5*t/(JUdiscretization-1) #covering the range [0,0.5]
        Ucrqpointlist=[]
        for qpoint in range(11):
            suscall = suscalllist[qpoint]
            res = minimize(inv_phys_susc, 1, args=(JUratio), method='nelder-mead', options={'maxiter': 100, 'xatol': 1e-8})
            Ucrqpointlist.append(res.x[0])
        Ucr=min(Ucrqpointlist) #min Ucr among all q-values will determine GS
        index=Ucrqpointlist.index(min(Ucrqpointlist)) #the phase of GS
        mucutsUcr.append(Ucr)
        mucutsphase.append(index+30)
        if t==0 and mu==doping_list[0]:
            estimation = time.time()-t0
            print("Estimated time for full PD:", estimation*len(doping_list)*JUdiscretization/60, "mins = ", estimation*len(doping_list)*JUdiscretization/3600, "hrs")
    passed = time.time()-t0
    print("Progres:", enum/(len(doping_list))*100,"%, time=", passed/60, "mins")
    Ucrphasediagram.append(mucutsUcr)
    phases.append(mucutsphase)
    print(phases)

with open('XP_Ucr_numba_phasediagram_'+bandstructure+'_JUpoints='+str(JUdiscretization)+'_MGXPNG_N=25.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(np.array(Ucrphasediagram))
    
with open('XP_Phases_numba_phasediagram_'+bandstructure+'_JUpoints='+str(JUdiscretization)+'_MGXPNG_N=25.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(np.array(phases))

doping_list = np.linspace(12.84, 12.6, num=13)

#automatic - full phase diagram
enum = 0 #enumerator
Ucrphasediagram = [] #Ucr on phase diagram
phases = [] #phases on phase diagram
t0=time.time()
for mu in doping_list:
    mucutsUcr = []
    mucutsphase = []
    enum += 1
    #reding in all files
    suscalllist = []
    for qpoint in range(21):
        suscall = np.loadtxt('MGnumba_N=25_fil='+str(mu)+'_q='+str(qpoint)+'.dat', dtype=complex)
        suscalllist.append(suscall)
    for t in range(0, JUdiscretization):
        JUratio = 0.5*t/(JUdiscretization-1) #covering the range [0,0.5]
        Ucrqpointlist=[]
        for qpoint in range(21):
            suscall = suscalllist[qpoint]
            res = minimize(inv_phys_susc, 1, args=(JUratio), method='nelder-mead', options={'maxiter': 100, 'xatol': 1e-8})
            Ucrqpointlist.append(res.x[0])
        Ucr=min(Ucrqpointlist) #min Ucr among all q-values will determine GS
        index=Ucrqpointlist.index(min(Ucrqpointlist)) #the phase of GS
        mucutsUcr.append(Ucr)
        mucutsphase.append(index)
        if t==0 and mu==doping_list[0]:
            estimation = time.time()-t0
            print("Estimated time for full PD:", estimation*len(doping_list)*JUdiscretization/60, "mins = ", estimation*len(doping_list)*JUdiscretization/3600, "hrs")
    passed = time.time()-t0
    print("Progres:", enum/(len(doping_list))*100,"%, time=", passed/60, "mins")
    Ucrphasediagram.append(mucutsUcr)
    phases.append(mucutsphase)
    print(phases)

with open('MG_Ucr_numba_phasediagram_'+bandstructure+'_JUpoints='+str(JUdiscretization)+'_MGXPNG_N=25.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(np.array(Ucrphasediagram))
    
with open('MG_Phases_numba_phasediagram_'+bandstructure+'_JUpoints='+str(JUdiscretization)+'_MGXPNG_N=25.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(np.array(phases))

#LEGENDS for phases

#0-20: [M-G]
#20-40: [G-X-P]
#40-60: [P-N-G]

