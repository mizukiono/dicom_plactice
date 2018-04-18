
# coding: utf-8

# In[ ]:

# DICOM Processing and Segmentation in Python (Radiology Data Quest)
# https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
# によるチュートリアル


# In[ ]:

get_ipython().magic('matplotlib inline')

import numpy as np
import dicom
import os
import matplotlib.pyplot as plt
from glob import glob


# In[ ]:

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# In[ ]:

import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans


# In[ ]:

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[ ]:

from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *


# In[ ]:

init_notebook_mode(connected=True) 


# In[ ]:

data_path = "./ct/"
output_path = working_path = "./20180417/"
g = glob(data_path + "/*.dcm")


# In[ ]:

#globできてるかチェック
print("Total of %d DICOM images.\nFirst 5 filnames are:"% len(g))
print('\n'.join(g[:5]))


# In[ ]:

# Loop over the image files and store everything into a list.

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    
    for s in slices:
        s.SliceThickness = slice_thickness
    
    return slices


# In[ ]:

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16
    # should be possible as values should always be low enough(<32k)
    image = image.astype(np.int16)
    
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # ここ何やってんのかわかんない……
    
    # Convert to Hounsfield units(HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    
    image = image + np.int16(intercept)
    
    return np.array(image, dtype = np.int16)


# In[ ]:

id = 0
patient = load_scan(data_path)


# In[ ]:

imgs = get_pixels_hu(patient)


# In[ ]:

np.save(output_path + "fullimages_%d.npy"%(id), imgs)


# In[ ]:

file_used = output_path + "fullimages_%d.npy"% id
imgs_to_process = np.load(file_used).astype(np.float64)


# In[ ]:

plt.hist(imgs_to_process.flatten(), bins = 50, color = 'c')
plt.xlabel("Hounsfield Units(HU)")
plt.ylabel("Frequency")
plt.show()


# In[ ]:

id = 0
imgs_to_process = np.load(output_path + 'fullimages_{}.npy'.format(id))


# In[ ]:

def sample_stack(stack, rows = 6, cols = 6, start_with = 10, show_every = 3):
    fig,ax = plt.subplots(rows, cols, figsize = [12, 12])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[int(i/rows), int(i % rows)].set_title('slice %d'%ind)
        ax[int(i/rows), int(i % rows)].imshow(stack[ind], cmap = 'gray')
        ax[int(i/rows), int(i % rows)].axis('off')
    plt.show()


# In[ ]:

sample_stack(imgs_to_process)
#現状ではCT領域の外側がHU = -2000になっており、この部分が真っ黒。Airの部分は-1000であり、
#グレーに表示されてしまっている


# In[ ]:

print("Slice Thickness: %f"% patient[0].SliceThickness)
print("Pixel Spacing(row, col): (%f, %f)"% (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))


# In[ ]:

id = 0
imgs_to_process = np.load(output_path + 'fullimages_{}.npy'.format(id))


# In[ ]:

def resample(image, scan, new_spacing = [1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing


# In[ ]:

print("Shape before resampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1, 1, 1])
print("Shape after resampling\t", imgs_after_resamp.shape)


# In[ ]:

# 3d plotting
def make_mesh(image, threshold = 300, step_size = 1):
    
    print("Transposing surface")
    p = image.transpose(2, 1, 0)
    
    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size = step_size, allow_degenerate = True)
    return verts, faces


# In[ ]:

def plotly_3d(verts, faces):
    x, y, z = zip(*verts)
    
    print("Drawing...")
    
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']
    
    fig = FF.create_trisurf(x = x,
                           y = y,
                           z = z,
                           plot_edges = False,
                           colormap = colormap, 
                           simplices = faces,
                           backgroundcolor = 'rgb(64, 64, 64)',
                           title = "Interactive Visualization")
    iplot(fig)


# In[ ]:

def plt_3d(verts, faces):
    print("Drawing...")
    x, y, z = zip(*verts)
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = '3d')
    
    mesh = Poly3DCollection(verts[faces], linewidths = 0.05, alpha = 1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    plt.show()


# In[ ]:

v, f = make_mesh(imgs_after_resamp, 350)


# In[ ]:

plt_3d(v, f)


# In[ ]:

v, f = make_mesh(imgs_after_resamp, 350, 2)


# In[ ]:



