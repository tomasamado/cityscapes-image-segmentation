#!/usr/bin/env python
# coding: utf-8

""" 
    utils module 
"""

import torch 
import numpy as np 
import matplotlib.pyplot as plt


def median_frequency_balancing(training_data, n_classes):
    """ The weight of class c  is computed as weight(c) = median freq/freq(c) where 
        freq(c) is the number of pixels of class c divided by the total number of 
        pixels in images where c is present, and median freq is the median of these 
        frequencies. Based on the paper `Predicting Depth, Surface Normals and Semantic 
        Labels with a Common Multi-Scale Convolutional Architecture <http://https://arxiv.org/abs/1411.4734>`_

        Parameters:
            training_data (array) - Training dataset of pairs (image, labels)
            n_classes (int) - Number of classes

        Returns: 
            (array) - Weight of each class
    """
    freqs = np.zeros(n_classes) # Frequencies of each class
    pixel_counts = np.zeros(n_classes) # Number of pixels of class c
    total_counts = np.zeros(n_classes) # Total number of pixels in images where c is present

    for p in range(len(training_data)):
        pixels = training_data[p][1].numpy()
        classes, counts = np.unique(pixels, return_counts=True)
        total_pixels = (pixels.shape[0] * pixels.shape[1]) 

        for i, c in enumerate(classes): 
            pixel_counts[c] += counts[i]
            total_counts[c] += total_pixels

    for i in range(freqs.shape[0]):
        freqs[i] = pixel_counts[i] / total_counts[i]

    median = np.median(freqs)  
    weights = median / freqs 

    return weights

def plot_seg_results(images, ground_truths, predictions):
    """ Plot a grid of several images, their ground-truth segmentations
        and their predicted segmentations.
      
      Args:
        images (numpy.array) - Images
        ground_truths (numpy.array) - Ground-truth segmentations
        predictions () - Predicted segmentations
    """
    f, axarr = plt.subplots(len(images), 3)
    f.set_size_inches(10,3*len(images))
    
    for i in range(len(images)):
        axarr[i,0].imshow(images[i])
        axarr[i,1].imshow(ground_truths[i])
        axarr[i,2].imshow(predictions[i].squeeze())
    
    # Remove axis
    for i in range(len(images)):
        for j in range(3):
            axarr[i,j].xaxis.set_visible(False)
            axarr[i,j].yaxis.set_visible(False)
        
    # Set columns titles
    axarr[0,0].set_title('IMAGE')
    axarr[0,1].set_title('GROUND TRUTH')
    axarr[0,2].set_title('PREDICTION')
    
    plt.show()

def plot_seg_result(image, ground_truth, prediction):
    """ Show a grid of several images, their ground-truth segmentations
        and their predicted segmentations.
      
      Parameters:
        image (torch.Tensor or numpy.array) - Image
        ground_truth (torch.Tensor or numpy.array) - Ground-truth segmentation
        prediction (torch.Tensor or numpy.array) - Predicted segmentation
    """
    f, axarr = plt.subplots(1,3)
    f.set_size_inches(5, 10)
    
    axarr[0].imshow(image)
    axarr[1].imshow(ground_truth)
    axarr[2].imshow(prediction)
    
    # Remove axis
    for j in range(3):
        axarr[i,j].xaxis.set_visible(False)
        axarr[i,j].yaxis.set_visible(False)
    
    # Set columns titles
    axarr[0].set_title('IMAGE')
    axarr[1].set_title('GROUND TRUTH')
    axarr[2].set_title('PREDICTION')
    
    plt.show()

def plot_metric(metric_history, label, color='b'): 
    """ Plot a metric vs. the epochs 
      
      Args: 
        metric_history (numpy.array): history of the metric's values order
        from older to newer.
        label (string): y-axis label
        title (string): title for the plot
    """
    epochs = range(len(metric_history))
    plt.plot(epochs, metric_history, color, label=label)
    plt.title(label + " vs. Epochs")
    plt.xticks(np.arange(0, len(epochs), 1.0))
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.show()


class EarlyStopping:
    """ Early stops the training if validation loss doesn't improve after a given patience.
    
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
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
