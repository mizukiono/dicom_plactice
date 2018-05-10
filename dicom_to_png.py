
# coding: utf-8

# In[ ]:


import os
import png
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


# In[ ]:


def get_pixels_hu(scan):
    # scanは2次元配列であってほしい
    image = np.array(scan)
    display(image)
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    display(image)
    
    # Convert to Hounsfield units (HU)
    dicomread = pydicom.read_file(ct_file_path)
    intercept = dicomread.RescaleIntercept
    slope = dicomread.RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# In[ ]:


def ct_to_png(ct_file_path, png_file_path):
    plan = pydicom.read_file(ct_file_path)
    shape = plan.pixel_array.shape
    image_2d = []
    max_val = 0
    
    for row in plan.pixel_array:
        pixels = []
        for col in row:
            pixels.append(col)
            if col < max_val:
                max_val = col
        image_2d.append(pixels)
    
    image_2d_HU = get_pixels_hu(image_2d)

    fig = plt.figure(figsize=(10, 10))
    
    ax = plt.subplot()
    ax.tick_params(labelbottom= False, bottom=False) # x軸の削除
    ax.tick_params(labelleft=False,left=False) # y軸の削除
    
    plt.imshow(image_2d_HU, cmap = 'gray')
    plt.savefig("testtest.png", bbox_inches="tight", pad_inches=0.0)
    
    
    
    '''
    # Rescalling greyscale between 0-255
    image_2d_scaled = []
    for row in image_2d:
        row_scaled = []
        for col in row:
            col_scaled = max(0, int((float(col)/float(max_val))*255.0))
            row_scaled.append(col_scaled)
        image_2d_scaled.append(row_scaled)
 
    # Writing the PNG file   
    f = open(png_file_path, 'wb')
    w = png.Writer(shape[0], shape[1], greyscale=True)
    w.write(f, image_2d_scaled)
    f.close()
    
    '''


# In[ ]:


ct_file_path = 'test.dcm'
png_file_path = 'test.png'
ct_to_png(ct_file_path, png_file_path)

