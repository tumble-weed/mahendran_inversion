#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/tumble-weed/mahendran_inversion/blob/master/inversion_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:




# In[ ]:


import skimage.io
from PIL import Image
import wandb



# In[3]:




# In[4]:



# In[5]:




# In[ ]:
#config = {
#    "modelname": "resnet18",
 #   "name_to_invert": "avgpool",
  #  "image": "images-master/ILSVRC2012_val_00000013.JPEG"
#}


#run = wandb.init(project="inversion", config=config)
#run.save()

from channel_inversion import get_model,prepare_for_inversion,invert


# In[7]:


modelname = 'resnet18'
model, model_imsize, preprocess,denormalize = get_model(modelname)


# In[ ]:


name_to_invert = 'avgpool'
layer_to_invert,hyperparams,modelname,name_to_invert = prepare_for_inversion(modelname,model,name_to_invert)


# In[ ]:

impath = "images-master/n01443537_16.JPEG"
#impath = 'images-master/ILSVRC2012_val_00000013.JPEG'
im = skimage.io.imread(impath)
im_pil = Image.fromarray(im)
ref = preprocess(im_pil).unsqueeze(0)
ref= ref.cuda()


# In[10]:

#num_channels = ref_feat.shape[1]
#num_channels = 25
num_channels = 512
#saliency_weights = torch.zeros(2048,device=device)#torch.rand(num_channels).cuda()
saliency_weights = np.zeros((512,)).astype(bool)#torch.rand(num_channels).cuda()
saliency_weights[:num_channels] = 1
#saliency_weights = saliency_weights.bool()
print(f'Initialised saliency weights with random values: {saliency_weights}')

#p46()
#hyperparams['tv_lambda'] =  hyperparams['tv_lambda'] * 1e2
hyperparams['alpha_lambda'] = 0
hyperparams['lr'] = 5e-2
#hyperparams['lr'] = 1e-1
hyperparams['clip'] = 1
hyperparams['pre_cnn_amplify'] = 1
#===============================================================
name = f'{modelname}_{name_to_invert}'
#run =  wandb.init(project='inversion',name=name)
wandb_project_name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
run = wandb.init(project=wandb_project_name,name=f'{modelname}-{name_to_invert}-tv{hyperparams["tv_lambda"]}-alpha{hyperparams["alpha_lambda"]}-lr{hyperparams["lr"]}-nchannels{num_channels}-clip{hyperparams["clip"]}-amp{hyperparams["pre_cnn_amplify"]}{dutils.get_n_wandb_runs(wandb_project_name)}')
#===============================================================
name = f'{modelname}_{name_to_invert}'

invert(ref,hyperparams,model,layer_to_invert,modelname,name_to_invert,saliency_weights = saliency_weights,
#lr = None,
#nepochs = None,
#x_mag = None,
#tv_lambda = None,
#alpha_lambda = None,
)
run.finish()
