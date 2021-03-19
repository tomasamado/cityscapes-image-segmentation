#!/usr/bin/python

""" 
    Cityscapes Pixel-Level Semantic Labeling dataset loader

    Author: pabvald

    Code based on the 'pascalVocDataset class' of `Dikshant Gupta <https://github.com/dikshant2210>`_.

"""

import os
import collections
import glob
import torch
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import cityscapesscripts.helpers.csHelpers as cs

# +
from PIL import Image
from torch.utils import data
from os.path import join as pjoin

from torchvision import transforms
import torchvision.transforms.functional as TF
from cityscapesscripts.preparation import createTrainIdLabelImgs
from cityscapesscripts.helpers.labels import labels as cityscapes_labels


# -

class cityscapesDataset(data.Dataset):
    """ Data loader for the Cityscapes Pixel-Level Semantic Labeling Task  """
    
    def __init__(self, root, split="train", encoding="trainId", is_transform=True, 
                                            img_size=(512, 256), augmentations=None):
        
        assert split in ["train", "val", "test"], "invalid split `{}`".format(split)
        assert encoding in ["id", "trainId"], "unkown encoding `{}`".format(encoding)
        
        self.root = root 
        self.split = split
        self.encoding = encoding
        self.is_transform = is_transform 
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)      
        self.augmentations = augmentations
        
        # Obtain the file names
        expected = {'train': 2975, 'val': 500, 'test': 1525}
        for split in ["train", "val", "test"]:
            path = pjoin(self.root , "leftImg8bit", split, "*", "*.png")
            file_list = glob.glob(path, recursive=True)
            file_list.sort()
            self.files[split] = file_list         
            assert len(file_list) == expected[split], "unexpedted data size" 
        
        # Generate the trainId labels if necessary
        if encoding == "trainId":
            os.environ["CITYSCAPES_DATASET"] = root
            searchFine   = os.path.join(root , "gtFine"   , "*" , "*" , "*_labelTrainIds.png")
            filesFine = glob.glob(searchFine)
            # Create the labels using the trainId encoding
            if len(filesFine) != 5000:
                createTrainIdLabelImgs.main()
            else:
                print("Annotations files processed")
            
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.2869, 0.3251, 0.2839], [0.1743, 0.1793, 0.1761]),
            ]
        )
        
    def __len__(self):
        return len(self.files[self.split])
    
    def __getitem__(self, index): 
        im_path = self.files[self.split][index]
        im_name = cs.getCoreImageFileName(im_path)
        im_dir = cs.getDirectory(im_path)
        lbl_ending = "TrainIds" if self.encoding == "trainId" else "Ids"
        lbl_path = pjoin(self.root, 'gtFine', self.split, im_dir, 
                                    im_name + "_gtFine_label" + lbl_ending + ".png")
        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
        
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl
    
    def label_ids(self):
        """ Return the classes' ids of the corresponding encoding """
        if self.encoding == "trainId":
            labels = [label.trainId for label in cityscapes_labels]
        else: 
            labels = [label.id for label in cityscapes_labels]
        return sorted(np.unique(labels))
    
    def label_colours(self):
        """ Return the classes' color of the corresponding encoding"""
        if self.encoding == "trainId":
            label_colours  = { label.trainId : label.color for label in cityscapes_labels}
            label_colours[255] = (0,0,0)
        else:
            label_colours  = { label.id : label.color for label in cityscapes_labels}
        return label_colours
    
    def label_names(self):
        """ Return the classes' name of the corresponding encoding"""
        if self.encoding == "trainId":
            label_names  = { label.trainId : label.name for label in cityscapes_labels}
            label_names[255] = 'unlabeled'
        else:
            label_names  = { label.id : label.name for label in cityscapes_labels}
        return label_names
        
    def transform(self, img, lbl):
        """ Apply the specified transformations to the image and resize both image 
            and label accordingly.
        
        Args:
            img (PIL.Image) - Image 
            lbl (PIL.Image) - Label mask

        Returns:
            Transformed image and label mask.
            
        """
        
        img = img.resize((self.img_size[0], self.img_size[1]), Image.NEAREST)  # uint8 with RGB mode
        lbl = lbl.resize((self.img_size[0], self.img_size[1]), Image.NEAREST)
        if self.split == 'train':
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(256, 512))
            img = TF.crop(img, i, j, h, w)
            lbl = TF.crop(lbl, i, j, h, w)
        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()   
        
        return img, lbl
    
    def encode_segmap(self, mask):
        """Encode segmentation label images as cityscapes classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Cityscapes classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for label in self.label_ids():
            colour = self.label_colours()[label]
            label_mask[np.where(np.all(mask == colour, axis=-1))[:2]] = label
        label_mask = label_mask.astype(int)
        # print(np.unique(label_mask))
        return label_mask
    
    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.label_colours()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in self.label_ids():
            r[label_mask == ll] = label_colours[ll][0]
            g[label_mask == ll] = label_colours[ll][1]
            b[label_mask == ll] = label_colours[ll][2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        if plot:
            plt.imshow(rgb)
            plt.axis('off')
            plt.show()
        else:
            return rgb


