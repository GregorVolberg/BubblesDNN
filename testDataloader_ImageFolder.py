#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision.models as models

# additional imports
from pathlib import Path
import os
import torchvision.transforms as transforms
from torchvision.io import read_image

import torch.nn as nn
from torchvision import datasets
from torchvision.models import ResNet50_Weights
from torchvision.datasets.vision import VisionDataset

from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import pandas as pd

from torch.utils.data import Dataset

# declare new class 'CostumDataset' and overwrite the 'find_classes' - method
# https://www.geeksforgeeks.org/method-overriding-in-python/
# https://discuss.pytorch.org/t/using-only-some-classes-of-a-data-set/103348/3
# or superclass "DatasetFolder"



# make_dataset muss auch überschrieben werden

condition_weights = './models/weights_HCvsNC.h5' 

#### anderer Versuch
# see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataset

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

protocolFile      = './img/testannotations.txt'
targetConditions  = [1, 3] # see condition dictionary
participants      = []     # leave empty for no selection
imageDir          = './img/allimages/'

# write csv file to image dir
# read that file from image dir with find_classes

def write_datasetcsv(csvfile, cond_dict, targetCons, vps, imDir):
	df = pd.read_csv(csvfile, 
					 sep = '\t',
					 header = 0,
					 names = ['file', 'vp', 'const', 'condition', 'group', 'face', 'trialnr', 'nBubbles', 'rt'])
	if (len(participants) == 0):
		df2 = df.loc[df['condition'].isin(targetConditions)]
		vpsel = 'all'
	else:
		df2 = df.loc[(df['condition'].isin(targetConditions)) & (df['vp'].isin(participants))]
		vlsel = vps
	df2.loc[:,'file'] = df2['file'] + '.png'
	df2['condition'] = df2['condition'].map(condition_dict)
	fname1 = imDir + 'imageList.txt'
	fname2 = './results/targets' + str(targetConditions).replace(' ', '').replace(',','.') + vpsel + '.txt'
	df2.to_csv(fname1)
	df2.to_csv(fname2)
	return fname1

write_datasetcsv('./img/testannotations.txt', condition_dict, targetConditions, participants, imageDir)

class CostumImageFolder (datasets.ImageFolder):
	def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
		df = pd.read_csv(directory + 'imageList.txt')
		classes = sorted(df['condition'].unique())
		class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
		return classes, class_to_idx
	def make_dataset(self, directory: str, class_to_idx: Dict[str, int]):
		df = pd.read_csv(directory + 'imageList.txt')
		fnames = directory + df['file']
		indices = df['condition'].map(class_to_idx)
		samples = list(zip(fnames, indices))
		return samples





class CustomImageDataset(Dataset):
	def __init__(self, annotations_file, cond_dict, targetCons, vps, img_dir, transform=None, target_transform=None):
		self.img_labels = get_annotations(annotations_file, cond_dict, targetCons, vps) #
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform
	def __len__(self):
		return len(self.img_labels)
	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
		image = read_image(img_path)
		label = self.img_labels.iloc[idx, 1]
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label

#### set up model (resnet50)
# https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html


data_transforms = transforms.Compose([transforms.Resize((224,224)),
		transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
	])
	
#image_datasets =  datasets.ImageFolder(ddir, data_transforms)
#img3_datasets  = CostumImageFolder(ddir, data_transforms)
img4 = CustomImageDataset(protocolFile, condition_dict, targetConditions, participants, imageDir, data_transforms)

#indizz = (0,1)
#img2 = torch.utils.data.Subset(image_datasets, indizz)

#image_datasetsF =  datasets.ImageFolder(ddirfemale, data_transforms)


#train, test = torch.utils.data.random_split(image_datasets['train'], [0.8, 0.2])
	
# dataloaders = {
# 	'train':
# 	torch.utils.data.DataLoader(image_datasets['train'],
# 								batch_size=32,
# 								shuffle=True),
# 	'validation':
# 	torch.utils.data.DataLoader(image_datasets['validation'],
# 								batch_size=32,
# 								shuffle=False)
# }
	
# random split
#train_size = int(0.8 * len(full_dataset))
#test_size = len(full_dataset) - train_size
#train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])    

num_classes = 2    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") # gpu gives error, https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
model  = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=True).to(device)

## freeze weights optionally
# for param in model.parameters():
#    param.requires_grad = False # requires_grad = F freezes weights

# replace fc layer
#model.fc = nn.Linear(512, num_classes)
in_features = model.fc.in_features # vorher hart auf 2048 eingestellt

#model.fc = nn.Sequential(
#               nn.Linear(in_features, 128),
#               nn.ReLU(inplace=True),
#               nn.Linear(128, num_classes)).to(device)

model.fc = nn.Sequential(
#               nn.Linear(in_features, 128),
#               nn.ReLU(inplace=True),
			   nn.Linear(in_features, num_classes)).to(device)


criterion = nn.CrossEntropyLoss()
# but for binary classification BCELoss might be better, see step 3 in https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/ 
#criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.fc.parameters())    

# again, from https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch

def train_model(model, criterion, optimizer, num_epochs):
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch+1, num_epochs))
		print('-' * 10)
		for phase in ['train', 'test']:
			if phase == 'train':
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			running_corrects = 0
			train_db, test_db = torch.utils.data.random_split(img4, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
			train_db.dataset.transform = data_transforms['train']
			test_db.dataset.transform  = data_transforms['test']
			dataloaders = {
				'train':
				torch.utils.data.DataLoader(train_db,
											batch_size=64, #previously 10
											shuffle=True,
											pin_memory=True,
											drop_last=True),
				'test':
				torch.utils.data.DataLoader(test_db,
											batch_size=64, #previously 10
											shuffle=False,
											pin_memory=True,
											drop_last=True)
			}
			dbsizes = {
				 'train': len(train_db),
				 'test':  len(test_db)
			 }
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				if phase == 'train':
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
				_, preds = torch.max(outputs, 1)
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			epoch_loss = running_loss / dbsizes[phase]
			epoch_acc = running_corrects.double() / dbsizes[phase]
			print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
														epoch_loss,
														epoch_acc))
	return model


model_trained = train_model(model, criterion, optimizer, num_epochs = 100)
