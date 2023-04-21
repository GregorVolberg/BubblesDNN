#!/usr/bin/env python
# coding: utf-8

# pytorch
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import ResNet50_Weights

# additional imports
from pathlib import Path
import pandas as pd
from datetime import datetime

# condition dictionary
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

#### condition, participants, image selection
# files and dirs
# for file output see https://howtodoinjava.com/python-examples/python-print-to-file/ (epochs, loss, accuracy etc)
protocolFile      = './img/dsetComposite/BubblesProtocolComposite2.txt'
imageDir          = './img/allimages/'
dummyDir          = './img/dummyDir/' # for a dummy ImageFolder object
resultsPath       = './models/'
nameSuffix        = '' # for results file, e.g. '_onlyControls'

df = pd.read_csv(protocolFile, 
	sep = '\t',
	header = 0,
	names = ['file', 'vp', 'const', 'condition', 'group', 'face', 'trialnr', 'nBubbles', 'rt'])

# participants and conditions
targetConditions  = [5, 7] # see condition dictionary
trialNr_min       = 1
trialNr_max       = float('inf')
experimental      = df.loc[df['group'] == 'experimental', 'vp'].unique()
control           = df.loc[df['group'] == 'control', 'vp'].unique()
all_participants  = df['vp'].unique()

#participants = all_participants
participantsList = [experimental, control, all_participants]
suffixList       = ['_2L_experimental', '_2L_control', '_2L_allParticipants']
#participantsList = [all_participants]
#suffixList       = ['_allParticipants']
# function for selecting images (conditions, participants)
def update_ImageFolder(ImageFolder, cond_dict, df, targetCons, vps, imageDir, t1, t2):
	### use df to select images
	### and set classes, class_to_idx etc in ImageFolder object
	df2 = df.loc[(df['condition'].isin(targetConditions)) & (df['vp'].isin(participants)) & (df['trialnr'].between(t1, t2))]
	df2.loc[:,'file'] = df2.loc[:,'file'] + '.png'
	df2.loc[:,'condition'] = df2.loc[:,'condition'].map(cond_dict)
	classes = sorted(df2['condition'].unique())
	class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
	fnames = imageDir + df2['file']
	indices = df2['condition'].map(class_to_idx)
	samples = list(zip(fnames, indices))
	ImageFolder.class_to_idx = class_to_idx
	ImageFolder.classes = classes
	ImageFolder.samples = samples
	ImageFolder.imgs = samples
	ImageFolder.targets = indices
	return ImageFolder

for participants, nameSuffix in zip(participantsList, suffixList):
	# imageFolders (datasets)
	dummy    = datasets.ImageFolder(dummyDir)
	all_imgs = update_ImageFolder(dummy, condition_dict, df, targetConditions, participants, imageDir, trialNr_min, trialNr_max)

	# random split 
	data_transforms = {
		'train':
		transforms.Compose([
			transforms.Resize((224,224)),
			#transforms.RandomAffine(0, shear=10, scale=(0.9,1.1)), 
			transforms.RandomAffine(0, translate=(0.13, 0.13), scale=(0.9,1.1)),
			transforms.RandomHorizontalFlip(0.5),
			transforms.ToTensor()
		]),
		'test':
		transforms.Compose([
			transforms.Resize((224,224)),
			transforms.ToTensor()
		]),
	}
		
	# prepare model in gpu
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model  = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=True).to(device)

	# replace model fc layer
	in_features = model.fc.in_features
	num_classes = len(all_imgs.classes)  
	num_hidden  = 46 # 46 action units in Ekman & Friesen FACS
	model.fc    = nn.Sequential(
	               nn.Linear(in_features, num_hidden),
	               nn.ReLU(),
				nn.Linear(num_hidden, num_classes)).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.fc.parameters())    
	
	# from https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch
	def train_model(model, criterion, optimizer, num_epochs, protFileName):
		for epoch in range(num_epochs):
			print('Epoch {}/{}'.format(epoch+1, num_epochs), file = protFileName)
			print('-' * 10, file = protFileName)
			for phase in ['train', 'test']:
				if phase == 'train':
					model.train()
				else:
					model.eval()
				running_loss = 0.0
				running_corrects = 0
				train_db, test_db = torch.utils.data.random_split(all_imgs, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
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
															epoch_acc), file = protFileName)
		return model

	# results file names
	modelname = resultsPath + 'res50_' + 'VS'.join(all_imgs.classes) + nameSuffix + '.h5'
	resultsFileName = resultsPath + 'res50_' + 'VS'.join(all_imgs.classes) + nameSuffix + '.txt' 

	prtFile = open(resultsFileName, 'w')
	model_trained = train_model(model, criterion, optimizer, num_epochs = 100, protFileName = prtFile)
	prtFile.close()

	res50 = model_trained.state_dict()
	setattr(res50, 'participants', participants)
	setattr(res50, 'class_to_index', all_imgs.class_to_idx)
	setattr(res50, 'trialRange', [trialNr_min, trialNr_max])
	setattr(res50, 'dateAndTime', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

	# save weights
	torch.save(res50, modelname)


