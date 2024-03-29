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
targetConditions  = [1, 3] # see condition dictionary
participants      = []     # leave empty for no selection

condition_weights = './models/weights_HCvsNC.h5'    

def get_fileList(csvfile, cond_dict, targetCons, vps):
	df = pd.read_csv(csvfile, 
					 sep = '\t',
					 header = 0,
					 names = ['file', 'vp', 'const', 'condition', 'group', 'face', 'trialnr', 'nBubbles', 'rt'])
	if (len(participants) == 0):
		df2 = df.loc[df['condition'].isin(targetConditions)]
	else:
		df2 = df.loc[(df['condition'].isin(targetConditions)) & (df['vp'].isin(participants))]
	return df2, list('./img/dsetComposite/' + df2['condition'].map(condition_dict) + '/' + df2['file'] + '.png'), df2['condition'].map(condition_dict)

### get list of image file names and updated data frame
newdf, allImages, allLabels = get_fileList(protocolFile, condition_dict, targetConditions, participants)


# only unse validation transforms for prediction
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
model = models.resnet50(pretrained=False).to(device)
in_features = model.fc.in_features

model.fc = nn.Sequential(
#               nn.Linear(2048, 128),
# zwischenlayer(2048, 2)
#               nn.ReLU(inplace=True), ggfs drinlassen
			   nn.Linear(in_features, num_classes)).to(device)
model.load_state_dict(torch.load(condition_weights))


# further see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://stackoverflow.com/questions/73396203/how-to-use-trained-pytorch-model-for-prediction 
# altermativ
# https://androidkt.com/use-saved-pytorch-model-to-predict-single-and-multiple-images/

# switch model to eval mode (so that it does not learn)
model.eval()

img_list = [Image.open(img_path) for img_path in allimages]

# loop over images
p_happy = np.zeros(len(allimages))/0

for trl in range(0, len(allimages)):
	img = Image.open(validation_img_paths[trl])
	tensor = data_transforms['validation'](img).unsqueeze(0).to(device)
	output = model.forward(tensor)
	probs = torch.nn.functional.softmax(output, dim=1).cpu().data.numpy()
	p_happy[trl] = probs[0,0]

####
newdf.loc[:,'prob_happy'] = p_happy
newdf.to_csv(str('probs_for_' + condition_weights + '.csv'))



# # plot
# fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
# for i, img in enumerate(img_list):
# 	ax = axs[i]
# 	ax.axis('off')
# 	ax.set_title("{:.0f}% Female, {:.0f}% Male".format(100*pred_probs[i,0],
# 															100*pred_probs[i,1]))
# 	ax.imshow(img)
  
