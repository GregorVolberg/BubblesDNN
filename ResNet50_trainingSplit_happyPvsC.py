#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision.models as models

# additional imports
from pathlib import Path
#import os
import torchvision.transforms as transforms

import torch.nn as nn
from torchvision import datasets
from torchvision.models import ResNet50_Weights

# ## Load stimuli
#dirhc    = Path('./dsetComposite/happyCorrect/')
# dirhnc   = Path('./dsetComposite/happyNeutralCorrect/')
#ddirfemale = Path('D:/BubblesDNN2/mfpics/female')
#ddir       = Path('D:/BubblesDNN2/mfpics/')
ddir       = Path('./happyPvsC/')

all_db = datasets.ImageFolder(ddir)

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
       

#num_classes = 2  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") # gpu gives error, https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
model  = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=True).to(device)

## freeze weights optionally
# for param in model.parameters():
#    param.requires_grad = False # requires_grad = F freezes weights

# replace fc layer
#model.fc = nn.Linear(512, num_classes)
in_features = model.fc.in_features
num_classes = len(all_db.classes)  

model.fc = nn.Sequential(
#               nn.Linear(2048, 128),
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

            train_db, test_db = torch.utils.data.random_split(all_db, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
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

# save weights
#os.mkdir('C:/Users/LocalAdmin/Documents/Git/BubblesLRP/scripts/models')
#os.mkdir('C:/Users/LocalAdmin/Documents/Git/BubblesLRP/scripts/models/pytorch')
torch.save(model_trained.state_dict(), './weights_happyPvsC.h5')

