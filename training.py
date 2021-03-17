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


def train_early_stopping(model, dataloaders, dataset_sizes, labels, 
            model_path, criterion, optimizer, n_steps=2, patience=2):
    """ Train a model applying early stopping 
        
        Parameters:
            model (torch.nn.Module) - Model to be trained
            dataloaders (array-like shape) - Dataloaders (train and val)
            dataset_sizes (array-like shape) - Datasets' sizes (train and val)
            labels (array-like shape) - Target labels
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
        'loss': 0.0,
        'val_f1': 0,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Training started')
    print('-' * 10)
    
    while fails < patience:
        
        # train during n epochs
        for i in range(n_steps):
            model.train()
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
        eval_report = evaluation.EvaluationReport.from_model(dataloaders['val'], model, labels)
        val_f1 = eval_report.f1_score(average="macro")
        
        if val_f1 > best_model['val_f1']:
            fails = 0
            best_model = {
                'epoch': epoch-1,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'loss': epoch_loss,
                'val_f1': val_f1,
                
            }
            # save best model until now
            torch.save(best_model, pjoin(model_path, 'best_model.pt'))
        else:
            fails += 1
            
        print('Epoch: {} - [Val.] F1-score: {:.4f}, fails: {}'.format(
                epoch-1, val_f1, fails))
 
    # load best model weights
    model.load_state_dict(best_model['model_state_dict'])
    
    return model
