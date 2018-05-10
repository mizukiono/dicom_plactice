
# coding: utf-8

# In[ ]:


import os
import glob


# In[ ]:


def rename(path, filetype):
    files = glob.glob(path + '/*.%s'%filetype)
    
    for i, f in enumerate(files):
        os.rename(f, os.path.join(path, 'img_' + '{0:03d}'.format(i) + '.' + filetype))


# In[ ]:


rename('dicom_images_for_test/jpgs', filetype = 'jpg')

