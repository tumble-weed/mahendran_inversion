import torch,torchvision
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import skimage
import torchvision
from skimage import io
import os
from PIL import Image
import pdb
'''
For Reproducibility
'''
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

tensor_to_numpy = lambda t:t.detach().cpu().numpy()

def get_model(modelname):

    if modelname == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
        model.eval()
        model.cuda()
        model_imsize = 224,224

        

    elif modelname == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.eval()
        model.cuda()
        model_imsize = 224,224

        

    elif modelname == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        model.eval()
        model.cuda()
        model_imsize = 227,227

        

    vgg_mean = (0.485, 0.456, 0.406)
    vgg_std = (0.229, 0.224, 0.225)
    model_mean,model_std = vgg_mean,vgg_std 
    preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(model_imsize),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean = model_mean,std=model_std)
                                           ])
    denormalize = lambda t,vgg_mean=vgg_mean,vgg_std=vgg_std:(t * torch.tensor(vgg_std).view(1,3,1,1).to(t.device)) + torch.tensor(vgg_mean).view(1,3,1,1).to(t.device)
    return model, model_imsize, preprocess,denormalize


def prepare_for_inversion(modelname,model,name_to_invert):
    if modelname == 'alexnet':
        #features '0' to '12'
        #classifier '0' to '6'
#         name_to_invert = ('classifier','6')
        layer_to_invert  = model._modules[name_to_invert[0]]._modules[name_to_invert[1]]
        good_hyperparams = {('features','6'):{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-4,'alpha_lambda':0,'nepochs':1000},
                            ('features','8'):{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-3,'alpha_lambda':0,'nepochs':1000},
                           ('features','12'):{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-2,'alpha_lambda':0,'nepochs':1000},
                           ('classifier','0'):{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-2,'alpha_lambda':0,'nepochs':1000},
                           ('classifier','2'):{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-2,'alpha_lambda':0,'nepochs':1000},
                           ('classifier','6'):{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-2,'alpha_lambda':0,'nepochs':1000},} # the loss for classifier.6 oscillates a lot
        x_mag,lr,tv_lambda,alpha_lambda,nepochs = good_hyperparams[name_to_invert].values()
        x_mag = torch.tensor(x_mag).float().cuda()
        tv_lambda = torch.tensor(tv_lambda).float().cuda()
    elif modelname == 'resnet18':
#         name_to_invert = 'avgpool'
        if name_to_invert in ['conv1', 'bn1', 'relu', 'maxpool','avgpool', 'fc']:
            layer_to_invert = model._modules[name_to_invert]
        else:
            layer_to_invert = model._modules[name_to_invert[0]]._modules[name_to_invert[1]]._modules[name_to_invert[2]]
        good_hyperparams = {'maxpool':{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-6,'nepochs':1000},
                           ('layer1','1','bn2'):{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-6,'alpha_lambda':0,'nepochs':1000},
                           ('layer2','1','bn2'):{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-6,'alpha_lambda':0,'nepochs':1000},
                           ('layer3','1','bn2'):{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-6,'alpha_lambda':0,'nepochs':1000},
                            ('layer4','0','bn2'):{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-3,'alpha_lambda':1e-4,'nepochs':1000},
                           ('layer4','1','bn2'):{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-3,'alpha_lambda':1e-4,'nepochs':1000},
                           'avgpool':{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-2,'alpha_lambda':1e-3,'nepochs':1000},
                           'fc':{'x_mag':1e0,'lr':1e-1,'tv_lambda':1e-2,'alpha_lambda':1e-3,'nepochs':1000},} # Have checked these less precisely
        hyperparams = good_hyperparams[name_to_invert]


    def hook(self,input,output):
        self.our_feats = output
    hooked_layer = layer_to_invert.register_forward_hook(hook)
    
    return layer_to_invert,hyperparams


def tv(t,beta=2):
    #gets the smoothness or the edginess of an image
    tv_x = t[:,:,:,1:]-t[:,:,:,:-1]
    tv_y = t[:,:,1:,:]-t[:,:,:-1,:]
    
    tv_x = tv_x[:,:,:-1,:]
    tv_y = tv_y[:,:,:,:-1]
    if False: print(tv_x) 
    tv_2 = tv_x **2 + tv_y **2
    tv = tv_2.pow(beta/2.)
    total = tv.sum()
    return total

def alpha_norm(t,alpha=6):
    a = torch.sum(t.pow(alpha))
    return a

def invert(ref_image,hyperparams):
    lr = hyperparams['lr']
    nepochs = hyperparams['nepochs']
    x_mag = torch.tensor(hyperparams['x_mag']).float().cuda()
    tv_lambda = torch.tensor( hyperparams['tv_lambda']).float().cuda()
    alpha_lambda = torch.tensor(hyperparams['alpha_lambda']).float().cuda()
    
    ''' Reference Features '''
    ref_scores = model(ref)
    ref_feat = layer_to_invert.our_feats.detach()
    
    ''' to be Inverted tensor and its features'''
    x = x_mag*torch.randn(ref.shape).cuda()

    x = x.requires_grad_(True)

    x_np = tensor_to_numpy(x)
    x_im = np.transpose(x_np[0],(1,2,0))
    x_im = (x_im - x_im.min())/(x_im.max()-x_im.min())
    plt.imshow(x_im)

    x_scores = model(x) #should I pass it through preprocess?
    x_feat = layer_to_invert.our_feats

    opt = torch.optim.Adam([x],lr=lr)


    all_losses = {'mse':[],'tv':[],'total':[],'alpha':[]}

    nepochs = 1000
    for e in range(nepochs):
        print(e,end=',')
        x_scores = model(x) #should I pass it through preprocess?
        x_feat = layer_to_invert.our_feats
        mse_loss = torch.nn.functional.mse_loss(ref_feat,x_feat)
        tv_loss = tv(x)
        alpha_loss = alpha_norm(x)
        total_loss = mse_loss + tv_lambda*tv_loss + alpha_lambda * alpha_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        all_losses['mse'].append(tensor_to_numpy(mse_loss))
        all_losses['tv'].append(tensor_to_numpy(tv_loss))
        all_losses['alpha'].append(tensor_to_numpy(alpha_loss))
        all_losses['total'].append(tensor_to_numpy(total_loss))


    plt.figure()
    plt.plot(np.log10(all_losses['mse']),'r',label='mse')
    plt.legend()

    plt.figure()
    plt.plot(np.log10(all_losses['total']),'b',label='total loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(np.log10(all_losses['tv']),'g',label='tv')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(np.log10(all_losses['alpha']),'r',label='alpha')
    plt.legend()
    plt.show()

    plt.figure()
    x_np = tensor_to_numpy(x)
    x_im = np.transpose(x_np[0],(1,2,0))
    x_im = (x_im - x_im.min())/(x_im.max()-x_im.min())
    plt.imshow(x_im)
    pass
