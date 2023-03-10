#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:33:55 2023

@author: vog20246
"""

# https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
# and
# https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html 

import torch
import math
import gc 
# gc for deleting object from workspace

zeros = torch.zeros(2, 3)
print(zeros)

torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

x = torch.empty(2, 2, 3) # ACHTUNG, sind nicht 3 2*2-Matrizen, sondern 2 2*3-Matrizen
print(x.shape)
print(x)

ones_like_x = torch.ones_like(x) # suffix "_like" created tensors of shape etc like the ones given as an argument
print(ones_like_x.shape)
print(ones_like_x)

a = torch.ones((2, 3), dtype=torch.int16) # set data type at construction
print(a)

b = a.to(torch.bool) # ".to"-method for type conversion


ones = torch.zeros(2, 2) + 1 # arithmetic operations possible

# tensor broadcasting (similar in numpy)
rand = torch.rand(2, 4)
twos = torch.ones(1, 4) * 2
doubled = rand * twos

# most of the math functions have a version with an appended underscore (_) that will alter a tensor in place, but do not create a new one
# helpful for omitting intermediate tensors
a = torch.tensor(1.45)
torch.sin_(a)

# copying tensors
a = torch.ones(2,2)
b = a
a[0][1] = 2.3
print(a,b) # changing a also changes b!

# use the .clone - method instead
a = torch.ones(2, 2)
b = a.clone()

a[0][1] = 561      
print(b)  
# be careful when autograd is onm for a tensor

#%% GPU 
# either specify device at creation time
gpurand = torch.rand(2, 2, device='cuda')
# or move it to the GPU with the .to method; performs data type OR device conversions
y = torch.rand(2, 2)
y = y.to('cuda')