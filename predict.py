import argparse 

import numpy as np
import pandas as pd
import os
import json

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from PIL import Image
from matplotlib import pyplot as plt

parser= argparse.ArgumentParser()
# defining argument
parser.add_argument('--image', action='store', default='image.jpg'  , help='image file must be provided to classifier')
parser.add_argument('--checkpoint', action='store', default='checkpoint.pth', help='trained network')
parser.add_argument('--top_k', action='store', default= 5, type= int, help='define de top k classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='category name')
parser.add_argument('--gpu', action='store', help='gpu')

argument= parser.parse_args()


# process an image
def process_image(image):
    image_PIL= Image.open(image)
    
    process= transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    
    image_tensor= process(image_PIL)
    
    return image_tensor



# check argument

if argument.top_k is None:
    top_k= 5
    
with opern(argument.category_names, 'r') as f:
    cat_to_name= json.load(f)

    
# load the model data from checkpoint
model_info= torch.load(argument.checkpoint, map_location=lambda storage, loc:storage)

# load a model from torchvision
if model_info['model'] == 'vgg16_bn':
    model = models.vgg16_bn(pretrained =True)
    
model.classifier = model_info['classifier']  
model.load_state_dict(model_info["state_dict"])
model.class_to_idx = model_info["idx_to_class"]
model.eval()    
    
# pre-process the image, return tensor
image = process_image(arguments.image)

# predit image
def predict(image_path, model, topk=5):
    with torch.no_grad():
        # numpy to pytorch
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.float()
        # remove runtime error
        image_tensor.unsqueeze_(0) 
        
        out = model(image_tensor)
        proba, classes = torch.exp(out).topk(topk)      
        return proba[0].tolist(), classes[0].add(1).tolist()
    top_proba = proba[0].tolist()
    top_classes = classes[0].add(1).tolist()
    return top_proba, top_classes
top_proba, top_classes = predict(image, model, topk=5)

print("Top Probabilities: " , top_proba)
print("Top Classes: ", top_classes)
    

