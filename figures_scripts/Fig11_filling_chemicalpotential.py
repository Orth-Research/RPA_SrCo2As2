#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

path = "C:/Users/amnedic/Documents/GitHub/RPA_SrCo2As2/results/"

fill_mu = np.transpose(np.genfromtxt(path+'filling_chemicalpotential.dat'))
fill_mu_T = np.transpose(np.genfromtxt(path+'filling_chemicalpotential_T.dat'))
width = 1.5*3.375
height = width/1.618
x_min = -0.4
x_max = 0.4
#deltax = 0.02
eF = 6.2693

fig1 = plt.figure(1,figsize = [width,height])
ax2 = plt.subplot(1,1,1)
plt.axhline(y=0, color='darkgray', linestyle='--', markersize=0.01)
plt.axvline(x=0, color='darkgray', linestyle='--', markersize=0.01)
ax2.plot(fill_mu[0], fill_mu[1], marker = 'o', markersize = 6, linestyle = '-', linewidth = 1.,  color = 'purple', label = 'bitebr.py')
ax2.plot(fill_mu_T[0], fill_mu_T[1]-6.2693, marker = 'o', markersize = 6, linestyle = '-', linewidth = 1.,  color = 'r', label = 'bitebr.py')

#ax1.set_xticks(ticks = np.linspace(0, omega_max, num = int(omega_max)*2 + 1, endpoint = True), minor = True)
ax2.grid(which = 'major', linestyle = 'dashed', linewidth = 1., alpha = 0.4, color='darkgray')
ax2.grid(which = 'minor', linestyle = ':', linewidth = 1., alpha = 0.4)
plt.tight_layout()

#ticks
grid_x_ticks = np.arange(x_min, x_max)
ax2.set_xticks(grid_x_ticks , minor=True)
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlabel('$x$',fontsize=14)
plt.ylabel('$\mu - \mu_F(T=0)$ [eV]',fontsize=14)
plt.show()

#fig1.savefig("filling_chemicalpotential.png", format='png', dpi = 600, bbox_inches="tight")

