
# coding: utf-8

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
init_notebook_mode(connected=True)


# In[ ]:

data_path = "./ct/"
output_path = working_path = "./20180417/"


# In[ ]:

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

id = 0
patient = load_scan(data_path)
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

# ここから今日書いたコード

# Standardize the pixel values
def make_lungmask(img, display = False):
    row_size = img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size/5*4), int(row_size/5):int(row_size/5*4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the underflow and overflow on the pixel spectrum.
    img[img == max] = mean
    img[img == min] = mean
    
    # Using Kmeans to separate foreground(soft tissue/bone) and background (lung/air)
    
    kmeans = KMeans(n_clusters = 2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)
    
    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # まちがって肺を切り取っちゃいたくはないよね
    
    eroded = morphology.erosion(thresh_img, np.ones([3,3]))
    dilation = morphology.dilation(eroded, np.ones([8,8]))
    
    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype = np.int8)
    mask[:] = 0
    
    # 肺だけが残ったら別のlarge dilationを行って肺のマスクを埋める
    
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))
    #最後にちょっと膨張させる
    
    if (display):
        fig, ax = plt.subplots(3, 2, figsize = [12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap = 'gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap = 'gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap = 'gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap ='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img       


# In[ ]:

img = imgs_after_resamp[100]
make_lungmask(img, display = True)


# In[ ]:

#こぴぺ

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()


# In[ ]:

masked_lung = []

for img in imgs_after_resamp:
    masked_lung.append(make_lungmask(img))

sample_stack(masked_lung, show_every = 10)


# In[ ]:

np.save(output_path + "maskedimages_%d.npy"%(id), imgs_after_resamp)


# In[ ]:



