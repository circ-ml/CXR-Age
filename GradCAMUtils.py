import torch
from fastai.vision import *

###Enables Grad-CAM to be used for an Inception model, registers the hook on branch 3 before ReLU
class InceptionCAM(nn.Module):
    def __init__(self,mdl,use_relu=True):
        super(InceptionCAM, self).__init__()
        # load up the model
        self.mdl = mdl
        
        ###Set to eval mode
        self.mdl.eval()
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.mdl[0][0][0:21]
        
        ###RESTORE THIS CODE
        self.branch0 = self.mdl[0][0][21].branch0

        #self.branch0_before = self.mdl[0][0][21].branch0.conv
        #self.branch0_after = nn.Sequential(self.mdl[0][0][21].branch0.bn,self.mdl[0][0][21].branch0.relu)
        
        self.branch1_0 = self.mdl[0][0][21].branch1_0
        self.branch1_1a = self.mdl[0][0][21].branch1_1a
        self.branch1_1b = self.mdl[0][0][21].branch1_1b

        self.branch2_0 = self.mdl[0][0][21].branch2_0
        self.branch2_1 = self.mdl[0][0][21].branch2_1
        self.branch2_2 = self.mdl[0][0][21].branch2_2
        self.branch2_3a = self.mdl[0][0][21].branch2_3a
        self.branch2_3b = self.mdl[0][0][21].branch2_3b
        
        
        
        
        self.branch3_before_hook = nn.Sequential(
            self.mdl[0][0][21].branch3[0],
            
            #BasicConv2d(1536, 256, kernel_size=1, stride=1)
            self.mdl[0][0][21].branch3[1].conv
            
        )
        self.branch3_after_hook = nn.Sequential(self.mdl[0][0][21].branch3[1].bn, self.mdl[0][0][21].branch3[1].relu)
        
        # get the max pool of the features stem
        #self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.mdl[1]
        
        # placeholder for the gradients
        self.gradients = None
    
        ###If we are plotting the activation map / activation index
        self.activation_map = False
        self.activation_index = -1
        
        self.use_relu=use_relu
    
    ###Which of the 512 activation maps is our heatmap?
    def set_activation_map(self,index):
        self.acivation_map = True
        self.activation_index = index
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        #RESTORE
        x0 = self.branch0(x)

        #x0_0 = self.branch0_before(x)
        #h = x0_0.register_hook(self.activations_hook)
        #x0 = self.branch0_after(x0_0)
        
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        
        
        ###RESTORE THIS CODE WHEN DONE WITH YOUR EXPERIMENT
        x3 = self.branch3_before_hook(x)
        h = x3.register_hook(self.activations_hook)

        x4 = self.branch3_after_hook(x3)

        

        out = torch.cat((x0, x1, x2, x4), 1)
        
        out = self.classifier(out)
        return out
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        
        x = self.features_conv(x)
        
        ###RESTORE
        
        x3 = self.branch3_before_hook(x)
        
        #x3 = self.branch0_before(x)
        
        
        return(x3)
    
    ###If we want the activation map instead of the class activation map (more useful for the positive class)
    def get_heatmap(self,img,activation_map=False,activation_index = None):        
        # pull the gradients out of the model
        gradients = self.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = gradients.cpu().numpy().sum((2,3)).reshape(-1)
        
        
        # get the activations of the last convolutional layer
        activations = self.get_activations(img).detach().cpu().numpy()

        # weight the channels by corresponding gradients
        #for i in range(activations.shape[1]):
        #    activations[:, i, :, :] *= pooled_gradients[i]
    
        # average the channels of the activations
       # heatmap = torch.mean(activations, dim=1).squeeze()
        
        heatmap = np.absolute(np.einsum('i,ijk',pooled_gradients,activations.reshape(256,5,5)))
        
        if(activation_map):
            heatmap = activations[:,activation_index,:,:]

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        
        #if(self.use_relu):
        #    heatmap = np.maximum(heatmap, 0)
        #else:
        #    heatmap = np.absolute(heatmap)

        # normalize the heatmap
        #heatmap = heatmap/np.max(heatmap)
        heatmap = (heatmap - np.min(heatmap)) /(np.max(heatmap)-np.min(heatmap))

        return(heatmap)
    
    def blendImage(self,heatmap,img,alpha=0.2,cmap='jet'):
        from PIL import Image
        from torchvision import transforms

        cm = plt.get_cmap(cmap)
        heatmap = np.uint8(heatmap*255)
        img_src = transforms.ToPILImage()(heatmap).convert('L')
        img_src = img_src.resize((img.shape[2],img.shape[1]),resample=PIL.Image.BILINEAR)

        im = np.array(img_src)
        im = cm(im)
        im = np.uint8(im*255)
        im = PIL.Image.fromarray(im).convert('RGB')

        xray = transforms.ToPILImage()(img)

        new_img = PIL.Image.blend(xray, im, alpha)
        #cv2.imwrite('./map.jpg', superimposed_img)

        return(new_img)

import torch
from fastai.vision import *

###Enables Grad-CAM to be used for a resnet model
class ResnetCAM(nn.Module):
    def __init__(self,mdl,use_relu=True):
        super(ResnetCAM, self).__init__()
        # load up the model
        self.mdl = mdl
        
        ###Set to eval mode
        self.mdl.eval()
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.mdl[0]
        
        # get the max pool of the features stem
        #self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.mdl[1]
        
        # placeholder for the gradients
        self.gradients = None
    
        ###If we are plotting the activation map / activation index
        self.activation_map = False
        self.activation_index = -1
        
        self.use_relu=use_relu
    
    ###Which of the 512 activation maps is our heatmap?
    def set_activation_map(self,index):
        self.acivation_map = True
        self.activation_index = index
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        #x = self.max_pool(x)
        #x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
    ###If we want the activation map instead of the class activation map (more useful for the positive class)
    def get_heatmap(self,img,activation_map=False,activation_index = None):        
        # pull the gradients out of the model
        gradients = self.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = gradients.cpu().numpy().sum((2,3)).reshape(-1)
        
        
        # get the activations of the last convolutional layer
        activations = self.get_activations(img).detach().cpu().numpy()

        # weight the channels by corresponding gradients
        #for i in range(activations.shape[1]):
        #    activations[:, i, :, :] *= pooled_gradients[i]
    
        # average the channels of the activations
       # heatmap = torch.mean(activations, dim=1).squeeze()
        
        heatmap = np.maximum(0,np.einsum('i,ijk',pooled_gradients,activations.reshape(activations.shape[1],activations.shape[2],activations.shape[3])))
        
        if(activation_map):
            heatmap = activations[:,activation_index,:,:]

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        
        #if(self.use_relu):
        #    heatmap = np.maximum(heatmap, 0)
        #else:
        #    heatmap = np.absolute(heatmap)

        # normalize the heatmap
        heatmap = (heatmap - np.min(heatmap)) /(np.max(heatmap)-np.min(heatmap))

        return(heatmap)
    
    
    def blendImage(self,heatmap,img,alpha=0.2,cmap='jet'):
        from PIL import Image
        from torchvision import transforms

        cm = plt.get_cmap(cmap)
        heatmap = np.uint8(heatmap*255)
        img_src = transforms.ToPILImage()(heatmap).convert('L')
        img_src = img_src.resize((img.shape[2],img.shape[1]),resample=PIL.Image.BILINEAR)

        im = np.array(img_src)
        im = cm(im)
        im = np.uint8(im*255)
        im = PIL.Image.fromarray(im).convert('RGB')

        xray = transforms.ToPILImage()(img)

        new_img = PIL.Image.blend(xray, im, alpha)
        #cv2.imwrite('./map.jpg', superimposed_img)

        return(new_img)
