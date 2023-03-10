#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision.models as models

# additional imports
from pathlib import Path
import os
import torchvision.transforms as transforms

import torch.nn as nn
from torchvision import datasets
from torchvision.models import ResNet50_Weights

# ## Load stimuli
#ddirmale   = Path('D:/BubblesDNN2/mfpics/male')
#ddirfemale = Path('D:/BubblesDNN2/mfpics/female')
ddir       = Path('./mfpics/')

# =============================================================================
# def listdir_fullpath(d):
#     return [os.path.join(d, f) for f in os.listdir(d)]
# 
# allfiles = listdir_fullpath(ddirmale) + listdir_fullpath(ddirfemale)
# 
# 
# img = Image.open(allfiles[0])
# plt.imshow(img)
# #del img
# 
# =============================================================================

#### set up model (resnet50)

# https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

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
    
image_datasets = {
    'train': 
    datasets.ImageFolder(ddir, data_transforms['train']),
    'validation': 
    datasets.ImageFolder(ddir, data_transforms['validation'])
}

#train, test = torch.utils.data.random_split(image_datasets['train'], [0.8, 0.2])
    
dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,
                                shuffle=True),
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=32,
                                shuffle=False)
}
    
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
def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

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

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
    return model


model_trained = train_model(model, criterion, optimizer, num_epochs=100)

# save weights
#os.mkdir('C:/Users/LocalAdmin/Documents/Git/BubblesLRP/scripts/models')
#os.mkdir('C:/Users/LocalAdmin/Documents/Git/BubblesLRP/scripts/models/pytorch')
#Nettorch.save(model_trained.state_dict(), 'C:/Users/LocalAdmin/Documents/Git/BubblesLRP/scripts/models/pytorch/weights.h5')

