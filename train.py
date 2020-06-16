import argparse

import os
import numpy as np
import pandas as pd
import json

from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


# define data source folders
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# creat a parser
parser= argparse.ArgumentParser()

# defining argument
parser.add_argument('--data_dir', action='store', type= str, default='flowers', help='data directory')
parser.add_argument('--arch', action='store', type=str, default='vgg16_bn', help='define the model')
parser.add_argument('--gpu', action='store', help='gpu for training')
parser.add_argument('--epochs', action='store', type= int, default= 20, help='define epochs for training')
parser.add_argument('--hidden_layers', action='store', type=int, default=4096, help='define hidden layers on the classifier')
parser.add_argument('--learning_rate', action='store', type= float, default= 0.003, help='learning rate')
parser.add_argument('--save_dir', action='store', dest='save_dir', type=str)

argument= parser.parse_arg()

# check argument
if argument.arch is None:
    arch_type= 'vgg16_bn
    
if argument.arch != 'vgg16_bn':
    print("Error, network must be vgg14_bn")
    argument.arch= 'vgg16_bn'
    
if argument.learning_rate <= 0:
    print("Error, learning rate must be larger than zero")
    argument.learning_rate= 0.003
    

if argument.hidden_layers <= 0:
    print("Error, hidden layers must be larger than zero")
    

if argument.epochs <= 0:
    print("Error, epochs must be larger than zero")



# show the inputs for training
print("reading data from: " + argument.data_dir)
print("classifier based on model: " + argument.arch)
print("number of epochs: " + str(argument.epochs))
print("learning rate: " + str(argument.learning_rate))

# Define the transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ColorJitter(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

valid_transforms= transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

test_transforms= transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])                  
                          
# load datasets 
train_data= datasets.ImageFolder(train_dir, transform= train_transforms)
valid_data= datasets.ImageFolder(valid_dir, transform= valid_transforms)
test_data= datasets.ImageFolder(test_dir, transform= test_transforms)
# Using the image datasets and the trainforms, define the dataloaders
trainloader= torch.utils.data.DataLoader(train_data, batch_size= 64, shuffle=True)
validloader= torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader= torch.utils.data.DataLoader(test_data, batch_size=64)

# load a model
if argument.arch == 'vgg16_bn':
    model= models.vgg16_bn(pretrained=True)
    input_layer= 25088
    output= 102
    
# freeze the parameters
for parameter in model.parameters():
    parameter.requires_grad= False

# define classfier
model.classifier= nn.Sequential(nn.Linear(25088,2048),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(2048,512),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(512,102),
                               nn.LogSoftmax(dim=1))

# define the loss and the optimizer
criterion= nn.NLLLoss()
optimizer= optim.Adam(model.classifier.parameters(), lr= argument.learning_rate)
    

    
# train the model
print_every= 1

results= [] 
steps= 0 
    
for epoch in range(argument.epochs):
    train_loss= 0
    valid_loss= 0
        
    train_accuracy= 0
    valid_accuracy= 0
        
    # set training
    model.train()
        
    # training
    for images, labels in trainloader:
        steps += 1
        # gpu
        if torch.cuda.is_available():
            images, labels= images.cuda(), labels.cuda()
            
        optimizer.zero_grad()
        # get log probabilities from model
        logps= model(images)
        # get the loss
        loss= criterion(logps, labels)
        loss.backward()
        optimizer.step()
        # increment train loss 
        train_loss += loss.item()
            
        # calculate  train accuracy
        ps= torch.exp(logps)
        top_p, top_class= ps.topk(1, dim=1)
        equals= top_class == labels.view(*top_class.shape)
        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    # after training loop end, start validation
    if steps % print_every == 0:
        with torch.no_grad():
            # set to evaluation mode
            model.eval() 
            
            for images, labels in validloader:
                # gpu
                if torch.cuda.is_available():
                    images, labels= images.cuda(), labels.cuda()
                    
                logps= model(images)
                loss= criterion(logps, labels)
                # increment valid loss
                valid_loss += loss.item()
                    
                # calculate validation accuracy
                ps= torch.exp(logps)
                top_p, top_class= ps.topk(1, dim=1)
                equals= top_class == labels.view(*top_class.shape)
                valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            # average loss
            train_loss= train_loss/len(trainloader)
            valid_loss= valid_loss/len(validloader)
            
            # average accuracy
            train_accuracy= train_accuracy/len(trainloader)
            valid_accuracy= valid_accuracy/len(validloader)
            
            results.append([train_loss, valid_loss, train_accuracy, valid_accuracy])
            
            # print training and validation results
            if (epoch + 1) % print_every == 0:
                print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                print(f'\t\tTraining Accuracy: {100 * train_accuracy:.2f}%\t Validation Accuracy: {100 * valid_accuracy:.2f}%')    
                
                            
# format results
results= pd.DataFrame(results, columns=['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy'])


# Do validation on the test set

def predict_test(testloader):
    total= 0
    correct= 0
    with torch.no_grad():
        model.eval()
        for target in testloader:
            images, labels= target
            # gpu
            if torch.cuda.is_available():
                images, labels= images.cuda(), labels.cuda()
            
            
            out= model(images)
            _, pred= torch.max(out.data, 1)
            
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
    print("Accuracy on test set: %d %%" % (100 * correct / total))
            
predict_test(testloader)


# save the checkpoint
model.idx_to_class= {i:j for i,j in trainloader.dataset.class_to_idx.items()}

checkpoint= {"model": "vgg16_bn",
            "pretrained": True,
            "features": model.features,
            "classifier": model.classifier,
            "optimizer": optimizer.state_dict(),
            "state_dict": model.state_dict(),
             "idx_to_class": model.idx_to_class,
             "classifier_input_size": 25088,
             "epochs": 22,
             "classifier_output_size": 102}

torch.save(checkpoint, "checkpoint.pth")