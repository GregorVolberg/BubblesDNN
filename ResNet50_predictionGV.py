#!/usr/bin/env python
# coding: utf-8

import torch
from PIL import Image
import torchvision.models as models

# additional imports
from pathlib import Path
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F

import torch.nn as nn

# ## Load stimuli
ddirmale   = Path('D:/BubblesDNN2/mfpics/male')
ddirfemale = Path('D:/BubblesDNN2/mfpics/female')
ddir       = Path('D:/BubblesDNN2/mfpics/')

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]),
}
    
num_classes = 2    
device = torch.device("cpu") # gpu gives error, https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least

## load weights when needed
model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, num_classes)).to(device)
model.load_state_dict(torch.load('./Documents/Git/BubblesLRP/scripts/models/pytorch/weights.h5'))

# further see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://stackoverflow.com/questions/73396203/how-to-use-trained-pytorch-model-for-prediction 

# switch model to eval mode (so that it does not learn)
model.eval()

validation_img_paths = [str(ddirmale) + '\image_17.png',
                        str(ddirmale) + '\image_18.png',
                        str(ddirfemale) + '\image_01.png',
                        str(ddirfemale) + '\image_16.png']
img_list = [Image.open(img_path) for img_path in validation_img_paths]

validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                for img in img_list])


pred_logits_tensor = model(validation_batch)
pred_probs         = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
# see answer 1 at https://stackoverflow.com/questions/60182984/how-to-get-the-predict-probability 
pred_probs


# plot
fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title("{:.0f}% Female, {:.0f}% Male".format(100*pred_probs[i,0],
                                                            100*pred_probs[i,1]))
    ax.imshow(img)
  
