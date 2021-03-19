# -*- coding: utf-8 -*-
""" 
    training module 
"""

import os 
import copy
import torch
import numpy as np
from os.path import join as pjoin
from evaluation import EvaluationReport


# +
def train(model, dataloaders, dataset_sizes, model_path, 
                             criterion, optimizer, epochs):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    print('Training started')
    print('-' * 10)
    
    for epoch in range(epochs):
        running_corrects = 0
        running_loss = 0.0
        
        for inputs, labels in dataloaders['train']:
            # send inputs and labels to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # reset the gradient
            optimizer.zero_grad()
            
            # compute loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim = 1)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss/dataset_sizes['train']
        print('Epoch {}/{} - train loss = {}'.format(epoch, epochs - 1, epoch_loss))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'loss': epoch_loss,  
        }
        # save best model until now
        torch.save(checkpoint, pjoin(model_path, 'epoch-{}.pt'.format(epoch)))

    return model


def train_early_stopping(model, dataloaders, dataset_sizes, model_path, 
                             criterion, optimizer, n_steps=2, patience=2):
    """ Train a model applying early stopping 
        
        Parameters:
            model (torch.nn.Module) - Model to be trained
            dataloaders (array-like shape) - Dataloaders (train and val)
            dataset_sizes (array-like shape) - Datasets' sizes (train and val)
            model_path (string) - Path to save the best model
            criterion () - Loss function 
            optimizer () - Optimizer 
            n_steps (int) - Number of steps between evaluations
            patience (int) - Number of times to observe worsening 
            validation set error before giving up
    """
    epoch = 0
    fails = 0
    best_model = {
        'epoch': 0,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'loss': 1e8,
        'val_loss': 1e8,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Training started')
    print('-' * 10)
    
    while fails < patience:
        
        # train during n epochs
        model.train()
        for i in range(n_steps):
            running_loss = 0.0
            
            for inputs, ground_truths in dataloaders['train']:
                inputs = inputs.to(device)
                ground_truths  = ground_truths.to(device)
                
                optimizer.zero_grad()
                
                # forward 
                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                loss = criterion(outputs, ground_truths)
                
                # backward
                loss.backward()
                optimizer.step()
                    
                running_loss += loss.item() * inputs.size(0)
              
            epoch_loss = running_loss / dataset_sizes['train']
            print('Epoch: {} - [Train] Loss: {:.4f}'.format(
                    epoch + i, epoch_loss))
        epoch += n_steps
        
        # evaluate validation error
        model.eval()
        running_loss = 0.0
        for inputs, ground_truths in dataloaders['val']:
            with torch.no_grad():
                inputs = inputs.to(device)
                ground_truths = ground_truths.to(device)

                # forward 
                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                loss = criterion(outputs, ground_truths)

                running_loss += loss.item() * inputs.size(0)
                
        if running_loss < best_model['val_loss']:
            fails = 0
            best_model = {
                'epoch': epoch-1,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'loss': epoch_loss,
                'val_loss': running_loss,       
            }
            # save best model until now
            torch.save(best_model, pjoin(model_path, 'best_model.pt'))
        else:
            fails += 1
            
        print('Epoch: {} - [Val] Loss: {:.4f}, fails: {}'.format(
                epoch-1, running_loss, fails))
 
    # load best model weights
    model.load_state_dict(best_model['model_state_dict'])
    
    return model
