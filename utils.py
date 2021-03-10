""" 
    utils module 
"""

import torch 
import numpy as np 
import matplotlib.pyplot as plt

# +
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
        images (torch.Tensor or numpy.array) - Images
        ground_truths (torch.Tensor or numpy.array) - Ground-truth segmentations
        predictions () - Predicted segmentations
    """
    f, axarr = plt.subplots(len(images), 3)
    f.set_size_inches(10,3*len(images))
    
    for i in range(len(images)):
        axarr[i,0].imshow(images[i].permute(1, 2, 0))
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
    
def plot_metric(metric_history, label): 
    """ Plot a metric vs. the epochs 
      
      Args: 
        metric_history (numpy.array): history of the metric's values order
        from older to newer.
        label (string): y-axis label
        title (string): title for the plot
    """
    epochs = range(len(metric_history))
    plt.plot(epochs, metric_history, 'b', label=label)
    plt.title(label + " vs. Epochs")
    plt.xticks(np.arange(0, len(epochs), 1.0))
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.show()
# -


