# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import division, unicode_literals, print_function

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pims
import trackpy as tp
from uncertainties import ufloat

T = ufloat(293,1)
eta = 1/1000
a = ufloat(0.00000099, 0.00000003)




frames = pims.TiffStack('C:/Users/Ewout van der Velde/Downloads/Untitled4.tif', as_grey=False)

f = tp.locate(frames[-250:], 21, invert = True, minmass=1000)

f.head()

plt.figure()
tp.annotate(f, frames[0])


fig,ax = plt.subplots()
ax.hist(f['mass'], bins=40)

f = tp.batch(frames[-100:], 21, minmass = 1000, invert=True)
t = tp.link_df(f, 5, memory=3)
t.head()

t1 = tp.filter_stubs(t, 25)

print('before:', t['particle'].nunique())
print('after:', t1['particle'].nunique())

plt.figure()
tp.plot_traj(t1)


d = tp.compute_drift(t1)
d.plot()
plt.show()
plt.savefig("1.pdf")

tm = tp.subtract_drift(t1.copy(),d)

ax = tp.plot_traj(tm)
plt.show()
plt.savefig("2.pdf")


em = tp.emsd(tm, 0.00000119029/1, 30) # microns per pixel = 100/285., frames per second = 24

fig, ax = plt.subplots()
ax.plot(em.index, em, 'o')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
ax.set(ylim=(1e-2, 100));
plt.savefig("3.pdf")

plt.figure()
plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
plt.xlabel('lag time $t$');
atre = tp.utils.fit_powerlaw(em)  # performs linear best fit in log space, plots]
plt.savefig("4.pdf")
print(atre)

atre_matrix = atre.as_matrix()


Kb = ( (atre_matrix[0,1]) * (6*np.pi*eta*a) ) / (4*T)

print(Kb)
