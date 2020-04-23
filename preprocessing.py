#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
from scipy.io import loadmat
import keras.utils
from collections import defaultdict

import anchor as ac

from utils import convert_coords, convert_type


def draw_box(bbs, image):
    """bbs is a list of boxes, each has form of topleft"""
#     image = cv2.imread(image_path)
    fig,ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    
    for i, bb in enumerate(bbs):
        # changed color and width to make it visible
        rect = patches.Rectangle((bb[0], bb[1]),bb[2],bb[3],linewidth=2, edgecolor='r',facecolor='none')
        
    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

def draw_resized_box(bbs, image_path, target_size = 224):
    # load an image from path
    image = cv2.imread(image_path)
    image_shape = image.shape[:2]
    
    # resize and convert bounding box to top_left form for drawing purpose
    bbs = [resize_bb(bb, image_shape, target_size) for bb in bbs]
    bbs = [convert_coords(bb, 'centroids_to_topleft') for bb in bbs]
    
    #resize image
    image = cv2.resize(image, (target_size, target_size))
    
    #draw
    draw_box(bbs, image )


# In[103]:


def resize_bb(bb, image_shape, target_size = 224):
    """ resize bbs to corresponding resized_image 
        bbs is a list of bb which has topleft form
        return bb in new size, of centroids form"""
    h, w = image_shape
    w_scale = target_size/w
    h_scale = target_size/h
    
    #convert bb to corners form
    bb = convert_coords(bb, kind = 'topleft_to_corners') 
    
    #rescale bbs
    bb[0] *= w_scale
    bb[1] *= h_scale
    bb[2] *= w_scale
    bb[3] *= h_scale
    # convert in to centroids forms
    return convert_coords(bb, kind = 'corners_to_centroids') 






def get_raw_labels(path, key):
    """ return dict labs = {ids: [bbs]} fromlabel path
    key is 'label_train' for training or 'label_test' for testing data'"""
    
    #load label
    labels = loadmat(os.path.join(path))
    
    labels = labels[key][0]
    
    # grabs all bounding boxes in to a dict. 
    # Im intend to implement an face detection so no classes label included
    labs = defaultdict()
    for lab in labels:
        bbs = [bb[:4] for bb in lab[-1]]
        labs[lab[-2][0]] = bbs

    return labs





def spatial_shape(path):
    return cv2.imread(path).shape[:2]


# In[4]:


from collections import defaultdict

def get_labels(imgs_path, labels_path,key ,  target_size = 224):
    """create dict of labels {id: pad invalid bbs, resize bb}
       key is 'label_train' for training or 'label_test' for testing data'
       """
    # create a dict labs = {ids: [bbs]} each bb: top_left, wh [x1, y1, w, h] dtype = int16
    _labs = get_raw_labels(labels_path, key)
    
    #list of paths for images
    img_paths = os.listdir(imgs_path)
    
    #create an array to contain labels for each image
    
    labs = defaultdict(list)
    for i, ID in enumerate(img_paths):
        
        #get spatial shape of an image
        shape = spatial_shape(imgs_path + '/' + ID)
        
        # loop though each box of an image, 
        # resize, normalize, can convert each bb into centroids form
        for lab in _labs[ID]:

            bbox = resize_bb(bb = lab, image_shape = shape, target_size = target_size)
            bbox = bbox/target_size
            labs[ID].append(bbox)          
    return labs



def show_image(path):
    plt.imshow(preprocess_image(path))
    plt.show()


# In[32]:


def preprocess_image(path):
    "load + resize + normalize image"
#     print(path)
    img = cv2.imread(path)[:, :, ::-1]
    img = cv2.resize(img, (224, 224))
    img = img/255
    return img



class DataGenerator(keras.utils.Sequence):
    
    def __init__(self,images_path,list_ids, labels, label_anchor, spatial_dim = (224, 224), 
                 batch_size = 128, n_channels = 3, shuffle = True):
        """create an data_generator instance
            anchor_box is an instance of class LabelAnchor
            m: maximum number of bounding boxes in an image (can be our choice?)
            labels: dict {ID: [true_bbs]}
        ex: train_data = DataGenerator(list_ids = train_ids, labels= train_labels)"""
        
        self.label_anchor = label_anchor
        self.list_ids = list_ids
        self.images_path = images_path
        self.labels = labels

        self.spatial_dim = spatial_dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def on_epoch_end(self):
        """shuffle indicies of ids after each epoch to shuffle data"""
        self.indices = np.arange(len(self.list_ids))
        if self.shuffle == True : np.random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def _data_generator(self, batch_list_ids):
        """load and preprocess data
        batch_list_ids: list of ids for this mini_batch
        -output: minibatch of
            -train_images: nd.array (batch_size, 224, 244, 3)
            -labels: nd.array (batch_size, m , 4 )

        Because images can have various number of bb, 
        for imgs that have less than m boxes, we pad it with invalid boxes which bb = [0, 0, 0, 0]
        """
        # get number of all anchor boxes in an image
        total_boxes = len(self.label_anchor.ab)
        
        X = np.empty((self.batch_size, *self.spatial_dim, self.n_channels))
        Y = np.empty((self.batch_size, total_boxes , 6))

        for i, ID in enumerate(batch_list_ids):
            X[i] = preprocess_image(path = self.images_path +'/' + ID )
            true_bbs = np.array(self.labels[ID])
            anchor_boxes_labels = self.label_anchor.labeling_anchors(true_bbs)
            Y[i] = anchor_boxes_labels

        return X, Y
    
    def __getitem__(self, idx):
        """get 1 minibatch for data
        idx: index of the batch , ex: batch_0, batch_1
        ex: train_data = DataGenerator(list_ids = train_ids, labels= train_labels)
            train_data[0] will return (np.array) images and labels of a batch"""

        # grab indices from self.indices
        indices = self.indices[idx*self.batch_size : (idx + 1)*self.batch_size]

        # get ids of data from their coresponding indices
        batch_list_ids = [self.list_ids[ID] for ID in indices]

        #grab images and labels from batch_list_ids
        return self._data_generator(batch_list_ids )

