# -*- coding: utf-8 -*-
#
# Calculate a dynamic background image from Gaussian blurring of a seleted
# pattern with a selected sigma. Then background subtraction can be viewed interactively.
# This is useful to determine the optimal value of sigma for the dataset.

import numpy as np
import hyperspy.api as hs
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import scipy.ndimage as scn

hs.preferences.General.nb_progressbar = False
hs.preferences.General.parallel = True

# Parse input parameters
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('file', help='Full path of original file')
parser.add_argument('--lazy', dest='lazy', default=False, action='store_true',
                    help='Whether to read/write lazy or not')
parser.add_argument('--sigma', dest='sigma', default=8,
                    help='Sigma of the Gaussian blur.')
# Parse
arguments = parser.parse_args()
# Set data directory, filename and file extension
file = arguments.file
lazy = arguments.lazy
sigma = int(arguments.sigma)

# Set data directory, filename and file extension
datadir, fname = os.path.split(file)
fname, ext = os.path.splitext(fname)

# Read data
print('* Read data from file')
s = hs.load(file, lazy=lazy)
s.change_dtype('int16')

def update_imgs(val):
    
    th_i = int(txt_i.text)
    th_j = int(txt_j.text)
    
    sigma = slider_sigma.val
    
    mark_ij.set_data([th_i], [th_j])
    
    s_ij = s.inav[th_i,th_j].data
    img5.set_data(s_ij)
    cbar5.set_clim(vmin = np.min(s_ij), vmax = np.max(s_ij))
    cbar5.draw_all()
    ax5.set_title('Signal at i='+str(th_i)+' j='+str(th_j), fontsize=12)
    
    bkg_img = scn.gaussian_filter(s_ij, sigma=sigma)
    img4.set_data(bkg_img)
    cbar4.set_clim(vmin = np.min(bkg_img), vmax = np.max(bkg_img))
    cbar4.draw_all()
    ax4.set_title('Background sigma='+str(sigma), fontsize=12)
    
    s_ij_bkg = s_ij - bkg_img
    img6.set_data(s_ij_bkg)
    cbar6.set_clim(vmin = np.min(s_ij_bkg), vmax = np.max(s_ij_bkg))
    cbar6.draw_all()
    
    fig.canvas.draw_idle()

fig, ((ax4, ax5, ax6), (ax1, ax2, ax3)) = plt.subplots(nrows=2, ncols=3,
     sharex=False, sharey=False)


img_th = s.sum(s.axes_manager.signal_axes).T
if lazy: 
    img_th.compute()

i,j = 0,0
img1 = ax1.imshow(img_th.data)
fig.colorbar(img1, ax=ax1)
mark_ij, = ax1.plot(i,j, marker='x',color='blue')
ax1.axis('off')
ax1.set_title('Navigation', fontsize=12)

s_ij = s.inav[i,j].data
img5 = ax5.imshow(s_ij)
cbar5 = fig.colorbar(img5, ax=ax5)
ax5.axis('off')
ax5.set_title('Signal at i='+str(i)+' j='+str(j), fontsize=12)

bkg_img = scn.gaussian_filter(s_ij, sigma=sigma)
img4 = ax4.imshow(bkg_img)
cbar4 = fig.colorbar(img4, ax=ax4)
ax4.axis('off')
ax4.set_title('Background'+r'$\sigma$'+str(sigma), fontsize=12)

s_ij_bkg = s_ij - bkg_img
img6 = ax6.imshow(s_ij_bkg)
cbar6 = fig.colorbar(img6, ax=ax6)
ax6.axis('off')
ax6.set_title('Signal - Background', fontsize=12)

ax2.remove()
ax3.remove()

#Axes : [x from left, y from bottom, length in x, length in y]
ax_i = plt.axes([0.55, 0.4, 0.05, 0.03])
ax_j = plt.axes([0.65, 0.4, 0.05, 0.03])
ax_ij = plt.axes([0.55, 0.35, 0.15, 0.03])
txt_i = TextBox(ax_i, 'i', initial='0')
txt_j = TextBox(ax_j, 'j', initial='0')
btn_ij = Button(ax_ij, 'Set i and j')
ax_sigma = plt.axes([0.55, 0.25, 0.25, 0.03])
slider_sigma = Slider(ax_sigma, '$\\sigma$', 1, 20, 
                      valinit=int(0.05*np.min(np.shape(bkg_img))), 
                      valstep = 1, valfmt="%1.0f")
slider_sigma.on_changed(update_imgs)
btn_ij.on_clicked(update_imgs)

plt.show()
