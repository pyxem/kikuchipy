# -*- coding: utf-8 -*-
#
# Calculate a static background image from calculating the average intensty from
# patterns in a selected region of interest of the data. This is useful when some
# regions contain strong lines that might be present in the average background if
# all patterns are used. The selected region of interest is defined by thresholding
# an input image (optionally; if not inputed, this is performed on the navigation image
# of the signal). The thresholding and background subtraction can be viewed interactively.

import numpy as np
import hyperspy.api as hs
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

hs.preferences.General.nb_progressbar = False
hs.preferences.General.parallel = True

# Parse input parameters
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('file', help='Full path of original file')
parser.add_argument('--lazy', dest='lazy', default=False, action='store_true',
                    help='Whether to read/write lazy or not')
parser.add_argument('--img_to_threshold', dest='img_to_threshold', 
                    help='Optional: File name of image to threshold to create a mask for the average background. Assumes it lies in the same directory as the dataset.')
parser.add_argument('--method', dest='method', default='threshold-image',
                    help='Full path of original file')
parser.add_argument('--threshold_min', dest='threshold_min', default=0.00, 
                    help='Minimum threshold.')
parser.add_argument('--threshold_max', dest='threshold_max', default=1.01, 
                    help='Maximum threshold.')
# Parse
arguments = parser.parse_args()
# Set data directory, filename and file extension
file = arguments.file
lazy = arguments.lazy
img_to_threshold = arguments.img_to_threshold
method = arguments.method
threshold_min = arguments.threshold_min
threshold_max = arguments.threshold_max

# Set data directory, filename and file extension
datadir, fname = os.path.split(file)
fname, ext = os.path.splitext(fname)

# Read data
print('* Read data from file')
s = hs.load(file, lazy=lazy)

if method == 'threshold-image':    
        
    def update_imgs(val):
        
        th_i = int(txt_i.text)
        th_j = int(txt_j.text)
        
        th_min = slider_th_min.val
        th_max = slider_th_max.val
        
        mark_ij.set_data([th_i], [th_j])
        
        mask = np.choose(a = (img_th >= th_min), choices = [0,1])   
        mask = np.choose(a = (img_th < th_max), choices = [0,mask])
        img2.set_data(mask)
        
        img_masked = img_th.data*mask
        img3.set_data(img_masked)
        cbar3.set_clim(vmin = np.min(img_masked.data[img_masked.data != 0.]), 
                       vmax = np.max(img_masked.data[img_masked.data != 1.]))
        cbar3.draw_all()

        if lazy: 
            print('Background image calculation for a lazy signal is not yet implemented.')
        else:
            bkg_img = np.average(s.data[np.where(mask.data)], axis = 0)
        img4.set_data(bkg_img)
        cbar4.set_clim(vmin = np.min(bkg_img), vmax = np.max(bkg_img))
        cbar4.draw_all()
        
        s_ij = s.inav[th_i,th_j].data
        img5.set_data(s_ij)
        cbar5.set_clim(vmin = np.min(s_ij), vmax = np.max(s_ij))
        cbar5.draw_all()
        ax5.set_title('Signal at i='+str(th_i)+' j='+str(th_j), fontsize=12)
        
        s_ij_bkg = s_ij - bkg_img
        img6.set_data(s_ij_bkg)
        cbar6.set_clim(vmin = np.min(s_ij_bkg), vmax = np.max(s_ij_bkg))
        cbar6.draw_all()
        
        fig.canvas.draw_idle()
    
    def save_bkg_img(val):
        s.save(os.path.join(datadir, fname + '_bkg_img.hdf5'))
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
         sharex=False, sharey=False)
    
    if img_to_threshold:
        img_th = hs.load(os.path.join(datadir, img_to_threshold))
    else:
        img_th = s.sum(s.axes_manager.signal_axes).T
        if lazy: 
            img_th.compute()
    if img_th.min(axis = (0,1)).data[0] != 0.0:
        img_th = img_th.copy() - img_th.min(axis = (0,1)).data[0]
    if img_th.max(axis = (0,1)).data[0] != 1.0:
        img_th = img_th.copy()/img_th.max(axis = (0,1)).data[0]
    
    i,j = 0,0
    img1 = ax1.imshow(img_th.data)
    fig.colorbar(img1, ax=ax1)
    mark_ij, = ax1.plot(i,j, marker='x',color='blue')
    ax1.axis('off')
    ax1.set_title('Image to threshold', fontsize=12)
    
    mask = np.choose(a = (img_th >= threshold_min), choices = [0,1])   
    mask = np.choose(a = (img_th < threshold_max), choices = [0,mask])
    img2 = ax2.imshow(mask)
    cbar2 = fig.colorbar(img2, ax=ax2)
    cbar2.set_clim(vmin = 0, vmax = 1)
    ax2.axis('off')
    ax2.set_title('Mask', fontsize=12)
    
    img_masked = img_th.data*mask
    img3 = ax3.imshow(img_masked, vmin = img_th.min(axis=(0,1)).data[0])
    cbar3 = fig.colorbar(img3, ax=ax3)
    img3.cmap.set_under('k')
    ax3.axis('off')
    ax3.set_title('Image x Mask', fontsize=12)
    
    #bkg_img = np.average(s.data[np.where(mask.data)], axis = (0,1))
    if lazy: 
        pass
    else:
        bkg_img = np.average(s.data[np.where(mask.data)], axis = 0)
    img4 = ax4.imshow(bkg_img)
    cbar4 = fig.colorbar(bkg_img, ax=ax4)
    ax4.axis('off')
    ax4.set_title('Background', fontsize=12)
    
    s_ij = s.inav[i,j].data
    img5 = ax5.imshow(s_ij)
    cbar5 = fig.colorbar(s_ij, ax=ax5)
    ax5.axis('off')
    ax5.set_title('Signal at i='+str(i)+' j='+str(j), fontsize=12)
    
    s_ij_bkg = s_ij - bkg_img
    img6 = ax6.imshow(s_ij_bkg)
    cbar6 = fig.colorbar(img6, ax=ax6)
    ax6.axis('off')
    ax6.set_title('Signal - Background', fontsize=12)
    
    #Axes : [x from left, y from bottom, length in x, length in y]
    ax_i = plt.axes([0.65, 0.06, 0.04, 0.03])
    ax_j = plt.axes([0.70, 0.06, 0.04, 0.03])
    ax_ij = plt.axes([0.75, 0.06, 0.1, 0.03])
    txt_i = TextBox(ax_i, 'i', initial='0')
    txt_j = TextBox(ax_j, 'j', initial='0')
    btn_ij = Button(ax_ij, 'Set i and j')
    ax_save = plt.axes([0.75, 0.02, 0.2, 0.03])
    btn_save_bkg_img = Button(ax_save, 'Save background image')
        
    ax_th_min = plt.axes([0.25, 0.02, 0.25, 0.03])
    ax_th_max = plt.axes([0.25, 0.06, 0.25, 0.03])
    slider_th_min = Slider(ax_th_min, 'threshold_min', -0.01, 1.01, valinit=0.00)
    slider_th_max = Slider(ax_th_max, 'threshold_max', -0.01, 1.01, valinit=1.01)
    
    slider_th_min.on_changed(update_imgs)
    slider_th_max.on_changed(update_imgs)
    btn_ij.on_clicked(update_imgs)
    btn_save_bkg_img.on_clicked(save_bkg_img)
    
    plt.show()

else:
    # Equivalent to no mask..
    bkg_img = np.ones((s.axes_manager.signal_axes[0].size,
                        s.axes_manager.signal_axes[1].size))
    bkg_img = np.sum(s.data[np.where(mask)], axis = (0,1))
    plt.imshow(bkg_img)