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
import itertools

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
resultsFileName   = './results/probs_for_G2_crossvalidation.csv'

df = pd.read_csv(protocolFile, 
					 sep = '\t',
					 header = 0,
					 names = ['file', 'vp', 'const', 'condition', 'group', 'face', 'trialnr', 'nBubbles', 'rt'])

experimental      = df.loc[df['group'] == 'experimental', 'vp'].unique()
control           = df.loc[df['group'] == 'control', 'vp'].unique()

targetConditions = ['happy_e', 'sad_e', 'happy_c', 'sad_c',]
conditionCodes   = {'happy_e': [1, 3], 'sad_e': [5, 7], 'happy_c': [1, 3], 'sad_c': [5, 7]}
res50  = {'happy_e': torch.load('./models/res50_happyCorrectVShappyNeutralCorrect_2G_experimental.h5'),
		  'sad_e':  torch.load('./models/res50_sadCorrectVSsadNeutralCorrect_2G_experimental.h5'),
		  'happy_c': torch.load('./models/res50_happyCorrectVShappyNeutralCorrect_2G_control.h5'),
		  'sad_c': torch.load('./models/res50_sadCorrectVSsadNeutralCorrect_2G_control.h5')}

experimental_test = np.setdiff1d(experimental, np.array(res50['happy_e'].participants)) # same for happy and sad
control_test      = np.setdiff1d(control, np.array(res50['happy_c'].participants))      # same for happy and sad
participants      = np.concatenate((control_test, experimental_test), 0) 

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
			   nn.Linear(in_features, num_classes)).to(device)

def get_fileList(csvfile, cond_dict, targetCons, vps):
	df = pd.read_csv(csvfile, 
					 sep = '\t',
					 header = 0,
					 names = ['file', 'vp', 'const', 'condition', 'group', 'face', 'trialnr', 'nBubbles', 'rt'])
	if (len(participants) == 0):
		df2 = df.loc[df['condition'].isin(targetCons)]
	else:
		df2 = df.loc[(df['condition'].isin(targetCons)) & (df['vp'].isin(participants))]
	return df2, list('./img/dsetComposite/' + df2['condition'].map(condition_dict) + '/' + df2['file'] + '.png'), df2['condition'].map(condition_dict)


dummy = get_fileList(protocolFile, condition_dict, conditionCodes['happy_e'], participants)[0]
alldf = dummy[0:0]
alldf['prob'] = 0
alldf['model'] = 0

for tcon in targetConditions:
	model.load_state_dict(res50[tcon]) # see above
	model.eval() # switch model to eval mode (so that it does not learn)
	newdf, allImages, allLabels = get_fileList(protocolFile, condition_dict, conditionCodes[tcon], participants)

	# loop over images
	p = np.zeros(len(allImages))/0
	for trl in range(0, len(allImages)):
		img = Image.open(allImages[trl])
		tensor = data_transforms['validation'](img).unsqueeze(0).to(device)
		output = model.forward(tensor)
		probs = torch.nn.functional.softmax(output, dim=1).cpu().data.numpy()
		p[trl] = probs[0,0]

	####
	newdf.loc[:,'prob'] = p
	newdf.loc[:,'model'] = tcon
	alldf = pd.concat([alldf, newdf], axis = 0)

alldf.to_csv(resultsFileName)

# further see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://stackoverflow.com/questions/73396203/how-to-use-trained-pytorch-model-for-prediction 
# alternativ https://androidkt.com/use-saved-pytorch-model-to-predict-single-and-multiple-images/

