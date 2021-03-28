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
    """ Train a model for a fixed number of epochs, saving its
        weights at the end of every epoch.
        
    Args:
        model (torch.nn.Module) - Model to be trained
        dataloaders (array-like shape) - Dataloaders (train)
        dataset_sizes (array-like shape) - Datasets' sizes (train)
        model_path (string) - Path to save the model after each epoch
        criterion (torch.nn.functional) - Loss function 
        optimizer (torch.optim) - Optimizer 
        epochs (int) - Number of epochs to train
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print('Training started')
    print('-' * 20)
    model.train()
    
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
            #running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / dataset_sizes['train']
        #epoch_acc = running_corrects.double() / dataset_sizes['train']
        
        print('Epoch {}/{} - train loss = {}'.format(epoch, epochs - 1, epoch_loss))
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'loss': epoch_loss,  
        }
        # save the model 
        torch.save(checkpoint, pjoin(model_path, 'epoch-{}.pt'.format(epoch)))

    return model

def train_val(model, dataloaders, dataset_sizes, model_path, 
                                 criterion, optimizer, epochs):
    """ Train a model evaluating it over the validation set at the end of 
        each epoch and return the model with best loss on the validation set.
        
    Args:
        model (torch.nn.Module) - Model to be trained
        dataloaders (array-like shape) - Dataloaders (train and val)
        dataset_sizes (array-like shape) - Datasets' sizes (train and val)
        model_path (string) - Path to save the best model
        criterion (torch.nn.functional) - Loss function 
        optimizer (torch.optim) - Optimizer 
        epochs (int) - Number of epochs to train
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_model = {
        'epoch': 0,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'val_loss': 1e8,
    } 
        
    print('Training started')
    print('-' * 20)
    model.train()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)
            

            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('[{}] Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss > best_model['val_loss']:
                best_acc = epoch_acc
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    'val_loss': epoch_loss,
                } 
                # save best model until now
                torch.save(best_model, pjoin(model_path, 'best_model.pt'))

        print()

    print('Best val Loss: {:4f}'.format(best_model['val_loss']))

    # load best model weights
    model.load_state_dict(best_model['model_state_dict'])
    return model

def train_early_stopping(model, dataloaders, dataset_sizes, model_path, 
                             criterion, optimizer, n_steps=2, patience=2):
    """ Train a model applying early stopping 
        
    Args:
        model (torch.nn.Module) - Model to be trained
        dataloaders (array-like shape) - Dataloaders (train and val)
        dataset_sizes (array-like shape) - Datasets' sizes (train and val)
        model_path (string) - Path to save the best model
        criterion (torch.nn.functional) - Loss function 
        optimizer (torch.optim) - Optimizer 
        n_steps (int) - Number of steps between evaluations
        patience (int) - Number of times to observe worsening validation set error 
        before giving up
    """
    epoch = 0
    fails = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    
    print('Training started')
    print('-' * 20)
    
    best_model = {
        'epoch': 0,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'loss': 1e8,
        'val_loss': 1e8,
    }
    
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
        
        epoch_loss = running_loss / dataset_sizes['val']
        
        if epoch_loss < best_model['val_loss']:
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
                epoch-1, epoch_loss, fails))
 

    # load best model weights
    model.load_state_dict(best_model['model_state_dict'])
    
    return model


# -
class EarlyStopping:
    """ Early stops the training if validation loss doesn't improve after a given patience.
    
    Ref: 
        https://github.com/Bjarten/early-stopping-pytorch
        
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """ Saves model when validation loss decrease. """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
