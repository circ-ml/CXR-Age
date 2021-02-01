""" Simple convolutional neural network architectures, these all assume a 3 x 224 x 224 input image. The number of nodes in the output layer is an input parameter """

import os
import pandas as pd
import fastai
from fastai.vision import *
import pretrainedmodels
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import *
from fastai.callbacks import *
import math
class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x
            
class Unflatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], x.size()[1],1,1)
        return x


"""Returns the appropriate CNN sized based on the input string name"""

def get_simple_model(name,out_nodes,lin_layers = None):
    if(name=="LargeT"):
        return SimpleConv(out_nodes=out_nodes,final_max = False,lin_layers=lin_layers)
    elif(name=="LargeW"):
        return SimpleConv(out_nodes = out_nodes,min_layer_size = 64,lin_layers=lin_layers)
    elif(name=="Small"):
        return SimpleConv(out_nodes=out_nodes,min_layer_size=32,max_layer_size=256,lin_layers=lin_layers)
    elif(name=="Tiny"):
        return SimpleConv(out_nodes = out_nodes,min_layer_size=64,filter_size=5,lin_layers=lin_layers)
    else:
        return None

class SimpleConv(nn.Module):
    """ out-nodes = how many nodes in the final output layer
        min_layer_size = how many filters in the first convolutional layer
        max_layer_size = how many filters in teh final convolutional layer (before classification from linear layers
        filter_size = what should the kernel size of the convolutional layer be?
        final_max = should there be maxpooling before the final global avg pool and linear layers?
        max_pool_window = kernel size of max pooling
        max_pool_stride = stride length of max pooling
        lin_layers = Manually specified sizes of each of the final linear layers, otherwise we assume we move down to an intermediate size and then to the output layer size"""
    def __init__(self,out_nodes = 10,min_layer_size=32,max_layer_size=512,filter_size=7,final_max=True,max_pool_window=3,max_pool_stride=2,lin_layers = None,dropout_pct = 0.25,in_size = 224):
        super(SimpleConv,self).__init__()
        self.out_nodes = out_nodes
        curr_layer = min_layer_size
        
        self.mdl = nn.Sequential()
        ###Keep adding layers until we reach the max layer size
        
        count = 0
        while(curr_layer < max_layer_size):
            if(count==0):
                self.mdl.add_module("Conv" + str(count),nn.Conv2d(3,curr_layer,kernel_size=filter_size))
            else:
                self.mdl.add_module("Conv" + str(count),nn.Conv2d(curr_layer,curr_layer*2,kernel_size=filter_size))
                curr_layer = curr_layer * 2
            self.mdl.add_module("BN" + str(count),nn.BatchNorm2d(curr_layer))
            self.mdl.add_module("ReLU" + str(count), nn.ReLU())

            if(curr_layer < max_layer_size or final_max):
                self.mdl.add_module("Max" + str(count),nn.MaxPool2d(max_pool_window,max_pool_stride))
            count += 1

        self.mdl.eval()
        out_size = self.mdl(torch.randn(1,3,in_size,in_size)).shape
        
        if(out_size[2]>1):
            self.mdl.add_module("AvgPool",nn.AvgPool2d(out_size[2]))
        

        self.mdl.add_module("Flatten",Flatten())
        
        
        self.mdl.eval()
        
        out_size = self.mdl(torch.randn(1,3,in_size,in_size)).shape[1]
        self.mdl.train()
        
        ###Issue is how do we get the output size of the current network? 
        ###Issue is how do we get the output size of the current network? 
    
        ###Create simple classifier for linear layers
        if(lin_layers is not None and len(lin_layers)==0):
            self.mdl.add_module("Last_Linear",nn.Linear(out_size,out_nodes,bias=True))
        elif(lin_layers is not None):
            count = 0
            for i in lin_layers:
                self.mdl.add_module("Linear" + str(count),nn.Linear(out_size,lin_layers[i],bias=True))
                self.mdl.add_module("ReLU_Linear" + str(count),nn.ReLU())
                self.mdl.add_module("BN_Linear" + str(count),nn.BatchNorm1d(lin_layers[i]))
                self.mdl.add_module("Dropout" + str(count),nn.Dropout(dropout_pct))
                count += 1
                
            self.mdl.add_module("Last_Linear",nn.Linear(lin_layers[len(lin_layers)-1],out_nodes,bias=True))
        else:
            self.mdl.add_module("Linear1",nn.Linear(out_size,32,bias=True))
            self.mdl.add_module("ReLU_Linear1",nn.ReLU())
            self.mdl.add_module("BN_Linear1",nn.BatchNorm1d(32))
            self.mdl.add_module("Dropout",nn.Dropout(dropout_pct))
            self.mdl.add_module("Last_Linear",nn.Linear(32,out_nodes,bias=True))
    
    def forward(self,x):
        return self.mdl(x)