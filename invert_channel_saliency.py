#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/tumble-weed/mahendran_inversion/blob/master/inversion_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:




# In[ ]:


import skimage.io
from PIL import Image


# In[3]:




# In[4]:



# In[5]:




# In[ ]:


from channel_api import get_model,prepare_for_inversion,invert


# In[7]:


modelname = 'resnet18'
model, model_imsize, preprocess,denormalize = get_model(modelname)


# In[ ]:


name_to_invert = 'avgpool'
layer_to_invert,hyperparams = prepare_for_inversion(modelname,model,name_to_invert)


# In[ ]:


im = skimage.io.imread('images-master/ILSVRC2012_val_00000013.JPEG')
im_pil = Image.fromarray(im)
ref = preprocess(im_pil).unsqueeze(0)
ref= ref.cuda()


# In[10]:


invert(ref,hyperparams,model,layer_to_invert)

