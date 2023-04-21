#!/usr/bin/env python
# coding: utf-8

import torch
from PIL import Image
import torchvision.models as models

from pathlib import Path
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F

import torch.nn as nn
import pandas as pd
import numpy as np

from zennit.attribution import Gradient, SmoothGrad
#from zennit.core import Stabilizer
from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat
#from zennit.composites import SpecialFirstLayerMapComposite, NameMapComposite
#from zennit.image import imgify, imsave
#from zennit.rules import Epsilon, ZPlus, ZBox, Norm, Pass, Flat
#from zennit.types import Convolution, Activation, AvgPool, Linear as AnyLinear
#from zennit.types import BatchNorm, MaxPool
#from zennit.torchvision import VGGCanonizer, ResNetCanonizer

### --- select conditions and/or participants
condition_dict = {
	1: 'happyCorrect',
	2: 'happyIncorrect',
	3: 'happyNeutralCorrect',
	4: 'happyNeutralIncorrect',
	5: 'sadCorrect',
	6: 'sadIncorrect',
	7: 'sadNeutralCorrect',
	8: 'sadNeutralIncorrect'
	}


protocolFile      = './img/dsetComposite/BubblesProtocolComposite2.txt'
df = pd.read_csv(protocolFile, 
					 sep = '\t',
					 header = 0,
					 names = ['file', 'vp', 'const', 'condition', 'group', 'face', 'trialnr', 'nBubbles', 'rt'])

targetConditions  = [5, 7] # see condition dictionary
experimental      = df.loc[df['group'] == 'experimental', 'vp'].unique()
control           = df.loc[df['group'] == 'control', 'vp'].unique()

res50e = torch.load('./models/res50_sadCorrectVSsadNeutralCorrect_2G_experimental.h5')
res50c = torch.load('./models/res50_sadCorrectVSsadNeutralCorrect_2G_control.h5')

experimental_test = np.setdiff1d(experimental, np.array(res50e.participants))
control_test      = np.setdiff1d(control, np.array(res50c.participants))
participants      = np.concatenate((control_test, experimental_test), 0)     # leave empty for no selection

targetModel       = res50e

def get_fileList(csvfile, cond_dict, targetCons, vps):
	df = pd.read_csv(csvfile, 
					 sep = '\t',
					 header = 0,
					 names = ['file', 'vp', 'const', 'condition', 'group', 'face', 'trialnr', 'nBubbles', 'rt'])
	if (len(vps) == 0):
		df2 = df.loc[df['condition'].isin(targetCons)]
	else:
		df2 = df.loc[(df['condition'].isin(targetCons)) & (df['vp'].isin(vps))]
	return df2, list('./img/dsetComposite/' + df2['condition'].map(cond_dict) + '/' + df2['file'] + '.png'), df2['condition'].map(cond_dict)

### get list of image file names and updated data frame
newdf, allImages, allLabels = get_fileList(protocolFile, condition_dict, targetConditions, participants)


# only use validation transforms for prediction
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
		transforms.Grayscale(3),
		transforms.ToTensor()
	]),
}

num_classes = 2    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## load weights when needed
model = models.resnet50(pretrained = False).to(device)
in_features = model.fc.in_features

model.fc = nn.Sequential(
			   nn.Linear(in_features, num_classes)).to(device)


model.load_state_dict(targetModel) # see above
model.eval()

composite = EpsilonPlusFlat()
attributor = Gradient(model, composite)

tmp=np.zeros((224,224))
for oneImage in allImages:
    #img = Image.open(allImages[trl])
    img = Image.open(oneImage)
    tensor = data_transforms['validation'](img).to(device)
    data   = tensor[None]
    output = model.forward(tensor.unsqueeze(0))
    h_x    = nn.functional.softmax(output, 1).data.squeeze()
    output, relevance = attributor(data, h_x[None]) # first category
    im = relevance.squeeze().cpu().numpy().mean(0)
    tmp = tmp + im

ime = tmp/len(allImages)

colmin = -3
colmax = 3
#plt.imshow(im, cmap='hot', interpolation='nearest')
plt.imshow(ime, cmap='hot', interpolation='nearest', vmin = colmin, vmax = colmax)
plt.colorbar()
plt.title("LRP for resnet50e")
plt.savefig("./results/LRPres50e.png", dpi = 300, format = "png")

# now for controls model
targetModel       = res50c
model = models.resnet50(pretrained = False).to(device)
in_features = model.fc.in_features
model.fc = nn.Sequential(
			   nn.Linear(in_features, num_classes)).to(device)
model.load_state_dict(targetModel) # see above
model.eval()

attributor = Gradient(model, composite)

tmp=np.zeros((224,224))
for oneImage in allImages:
    #img = Image.open(allImages[trl])
    img = Image.open(oneImage)
    tensor = data_transforms['validation'](img).to(device)
    data   = tensor[None]
    output = model.forward(tensor.unsqueeze(0))
    h_x    = nn.functional.softmax(output, 1).data.squeeze()
    output, relevance = attributor(data, h_x[None]) # first category
    im = relevance.squeeze().cpu().numpy().mean(0)
    tmp = tmp + im
# Gespräch Robert 19.04.: jedes Bild (hier: "im") durch summe der Relevanzen pro pixel teilen, dann hat man die relative Relevanz, also ein p (Relevanzanteil). Das kann man in ein z überführen.
# für Skalenabhängigkeit braucht man die hooks. hier pro Kanal(Kernel) die gleiche Prozedur durchführen

imc = tmp/len(allImages)
plt.imshow(imc, cmap='hot', interpolation='nearest', vmin = colmin, vmax = colmax)
plt.colorbar()
plt.title("LRP resnet50c")
plt.savefig("./results/LRPres50c.png", dpi = 300, format = "png")



# in https://github.com/RobertBosek/Masterarbeit/blob/dev/scripts/analysis_clean.ipynb
# code  block 12
# [data, target, model] are defined in block 9. data is loaded and transformed image, model is pre-trained resnet, target is target category ODER prob für target
# geht auch ohne target
# alternative zu Zennit: Captum
# create a composite instance, create a gradient attributor, get relevance
# dat and target are here the image and the prob for cat 0
# then plot relevances

