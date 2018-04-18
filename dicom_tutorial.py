
# coding: utf-8

# In[ ]:

import dicom
import os
import numpy as np
from matplotlib import pyplot, cm


# In[ ]:

PathDicom = "./ct/"
lstFilesDCM = [] #create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():
            lstFilesDCM.append(os.path.join(dirName, filename))


# In[ ]:

# get ref file
RefDs = dicom.read_file(lstFilesDCM[0])


# In[ ]:

print(RefDs)


# In[ ]:

# Load dimensions based on the number of rows, columns, and slices
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))


# In[ ]:

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))


# In[ ]:

x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])


# In[ ]:

# The array is sized based on 'ConstPixelDims'
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)


# In[ ]:

# loop all DICOM FILES
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    #store the raw image data
    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array


# In[ ]:

pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(x, y, np.flipud(ArrayDicom[:, :, 80]))


# In[ ]:

pyplot.show()


# In[ ]:

pyplot.pcolormesh(x, y, np.flipud(ArrayDicom[:, :, 100]))


# In[ ]:

pyplot.show()


# In[ ]:



