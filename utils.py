""" 
    utils module 
"""

import torch 
import numpy as np 

def median_frequency_balancing(training_data, n_classes):
  """ The weight of class c  is computed as weight(c) = median freq/freq(c) where 
    freq(c) is the number of pixels of class c divided by the total number of 
    pixels in images where c is present, and median freq is the median of these 
    frequencies. Based on the paper `Predicting Depth, Surface Normals and Semantic 
    Labels with a Common Multi-Scale Convolutional Architecture <http://https://arxiv.org/abs/1411.4734>`_
    
  Args:
    training_data (array): training dataset of pairs (image, labels)
    n_classes (int): number of classes
  
  Returns: 
    (torch.Tensor): weight of each class
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

  return torch.from_numpy(weights).float()
