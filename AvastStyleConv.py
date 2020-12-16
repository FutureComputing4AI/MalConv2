from collections import deque
from collections import OrderedDict 

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from LowMemConv import LowMemConvBase


def getParams():
    #Format for this is to make it work easily with Optuna in an automated fashion.
    #variable name -> tuple(sampling function, dict(sampling_args) )
    params = {
        'channels'     : ("suggest_int", {'name':'channels', 'low':16, 'high':64}),
        'stride'   : ("suggest_int", {'name':'stride', 'low':2, 'high':4}),
        'window_size'  : ("suggest_int", {'name':'window_size', 'low':16, 'high':64}),
    }
    return OrderedDict(sorted(params.items(), key=lambda t: t[0]))

def initModel(**kwargs):
    new_args = {}
    for x in getParams():
        if x in kwargs:
            new_args[x] = kwargs[x]
            
    return AvastConv(**new_args)

def vec_bin_array(arr, m=8):
    """
    Arguments: 
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype(np.int8)

    return (ret*2-1).astype(np.float32)/16

class AvastConv(LowMemConvBase):
    
    def __init__(self, out_size=2, channels=48, window_size=32, stride=4):
        super(AvastConv, self).__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        for i in range(1, 257):
            self.embd.weight.data[i,:] = torch.tensor(vec_bin_array(np.asarray([i])))
        for param in self.embd.parameters():
             param.requires_grad = False
                
    
        self.conv_1 = nn.Conv1d(8, channels, window_size, stride=stride, bias=True)
        self.conv_2 = nn.Conv1d(channels, channels*2, window_size, stride=stride, bias=True)
        self.pool = nn.MaxPool1d(4)
        self.conv_3 = nn.Conv1d(channels*2, channels*3, window_size//2, stride=stride*2, bias=True)
        self.conv_4 = nn.Conv1d(channels*3, channels*4, window_size//2, stride=stride*2, bias=True)
        

        
        self.fc_1 = nn.Linear(channels*4, channels*4)
        self.fc_2 = nn.Linear(channels*4, channels*3)
        self.fc_3 = nn.Linear(channels*3, channels*2)
        self.fc_4 = nn.Linear(channels*2, out_size)
        
    
    def processRange(self, x):
        #Fixed embedding
#         cur_device = next(self.conv_1.parameters()).device
#         x = torch.tensor(vec_bin_array(x.cpu().data.numpy()))
#         print("chunk")
        with torch.no_grad():
            x = self.embd(x)
            x = torch.transpose(x,-1,-2)
         
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = self.pool(x)
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        
        return x
    
    def forward(self, x):
        post_conv = x = self.seq2fix(x)
        
        x = F.selu(self.fc_1(x))
        x = F.selu(self.fc_2(x))
        penult = x = F.selu(self.fc_3(x))
        x = self.fc_4(x)
        
        return x, penult, post_conv