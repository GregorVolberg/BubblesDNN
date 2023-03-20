#!/usr/bin/env python
# coding: utf-8

from itertools import islice

import torch
from torch.nn import Linear
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms import ToTensor, Normalize
import torchvision.models as models

# from zennit.attribution import Gradient, SmoothGrad
# from zennit.core import Stabilizer
# from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat
# from zennit.composites import SpecialFirstLayerMapComposite, NameMapComposite
# from zennit.image import imgify, imsave
# from zennit.rules import Epsilon, ZPlus, ZBox, Norm, Pass, Flat
# from zennit.types import Convolution, Activation, AvgPool, Linear as AnyLinear
# from zennit.types import BatchNorm, MaxPool
# from zennit.torchvision import VGGCanonizer, ResNetCanonizer

# additional imports
# import torchvision.models as models
from pathlib import Path
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from random import sample
import torch.nn as nn

# https://discuss.pytorch.org/t/how-to-load-image-using-dataloader-with-a-list-of-image-file-names/4171
# use Resnet50TrainingGV.py as a test
 

 # Zelle 12 in roberts jupyter-book, 

 # output, relevance = attributor(data, target)
 # data ist bild (224x224), target ist vektor mit Klassen aus des Modells, zB [1 0], gibt featires die relevant sind für Klassifikation in Klasse 1