""" 
    Pascal VOC semantic segmentation dataset

    Author: dikshant2210
"""
# +
import collections
import glob
import torch
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import cityscapesscripts.helpers.csHelpers as cs

from PIL import Image
from torch.utils import data
from os.path import join as pjoin
from torchvision import transforms
from cityscapesscripts.helpers.labels import labels as cityscapes_labels
# -

cityscapesPath = '../Cityscapes'


class cityscapesDataset(data.Dataset):
    """ Data loader for the Cityscapes Pixel-Level Semantic Labeling Task 
        
        Code based on the 'pascalVocDataset class' of `Dikshant Gupta <https://github.com/dikshant2210>`_.
    """
    
    def __init__(self, root, split="train", encoding="trainId", is_transform=True, 
                                            img_size=(512, 256), augmentations=None):
        
        assert split in ["train", "val", "test"], "Invalid split `{}`".format(split)
        assert encoding in ["id", "trainId"], "Unkown endoing `{}`".format(encoding)
        
        self.root = root 
        self.split = split
        self.encoding = encoding
        self.is_transform = is_transform 
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)      
        self.augmentations = augmentations
        
        # Obtain the file names
        for split in ["train", "val", "test"]:
            path = pjoin(self.root , "leftImg8bit", split, "*", "*.png")
            file_list = glob.glob(path, recursive=True)
            file_list.sort()
            self.files[split] = file_list 
    
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                # Add Normalization
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
    
    def class_ids(self):
        """ Return the classes' ids of the corresponding encoding """
        if self.encoding == "trainId":
            labels = [label.trainId for label in cityscapes_labels]
        else: # self.encoding = "id"
            labels = [label.id for label in cityscapes_labels]
        return sorted(np.unique(labels))
    
    def transform(self, img, lbl):
        """ Apply the specified transformations to the image and resize both image 
            and label accordingly.
        
        Args:
            img (PIL.Image) - Image 
            lbl (PIL.Image) - Label mask

        Returns:
            Transformed image and label mask.
        """
        img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        #lbl[lbl == 255] = 0
        return img, lbl
        
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
        if self.encoding == "trainId":
            label_colours  = { label.trainId : label.color for label in cityscapes_labels}
        else:
            label_colours  = { label.id : label.color for label in cityscapes_labels}
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in self.class_ids():
            r[label_mask == ll] = label_colours[ll][0]
            g[label_mask == ll] = label_colours[ll][1]
            b[label_mask == ll] = label_colours[ll][2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

