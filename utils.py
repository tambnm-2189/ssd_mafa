#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
from scipy.io import loadmat
import keras.utils
import tensorflow as tf

def prediction(y_pred, score_threshold, iou_threshold, top_k_boxes):
    """y_pred (b, total_boxes, 2 + 4 + 8)
    
    ## labels of y_pred
    0:bg, 1:face, 
    2: offset_x, 3: offset_y, 4: offset_w, 5: offset_h,
    6: anchor_x, 7: anchor_y, 8:anchor_w, 9:anchor_h, 
    -4: variance_x, -3, variance_y, -2, variance_w, -1:variance_h
    
    2: cls, 4: offset prediction, 8 : 4 anchor boxes coords in centroids, 4 variance
    return (b, top_k, 4)
    """
    # convert y_pred boxes offset to centroids
    cx_pred = y_pred[..., 2]*y_pred[...,-4]*y_pred[...,8] + y_pred[..., 6]
    cy_pred = y_pred[..., 3]*y_pred[...,-3]*y_pred[...,9] + y_pred[..., 7]
    w_pred = tf.math.exp(y_pred[..., 4]*y_pred[...,-2])*y_pred[...,8] 
    h_pred = tf.math.exp(y_pred[..., 5]*y_pred[...,-1])*y_pred[...,9]
    
    # convert centroids to (ymin, xmin, ymax, xmax)
    xmin = tf.expand_dims(cx_pred - w_pred/2, axis= -1)
    ymin = tf.expand_dims(cy_pred - h_pred/2, axis= -1)
    xmax = tf.expand_dims(cx_pred + w_pred/2, axis= -1)
    ymax = tf.expand_dims(cy_pred + h_pred/2, axis= -1)
    
    all_boxes = tf.concat([ymin, xmin, ymax, xmax], axis= -1)
    
    #nms for each image
    def nms(boxes):
        pad_indices = tf.image.non_max_suppression(boxes= boxes, 
                                    scores= y_pred[...,1], 
                                    max_output_size = top_k_boxes, 
                                    iou_threshold = iou_threshold, 
                                    score_threshold = score_threshold) 
        
        selected_boxes = tf.gather(boxes, selected_indices) #(n_boxes, 4)
        #padding 
        paddings = tf.constant([[0, top_k_boxes - len(pad_indices)],[0, 0]])
        return tf.pad(selected_boxes, paddings, "CONSTANT")
    
    
    return tf.map_fn(fn=lambda i: nms(i),
            elems=all_boxes,
            dtype=None,
            parallel_iterations=128,
            back_prop=False,
            swap_memory=False,
            infer_shape=True)



class LossSSD():
    def __init__(self, alpha = 1):
        self.alpha = 1
        
    def smooth_l1_loss(self, y_true, y_pred ):
        """(b, total_boxes, 4)"""
        abs_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5*(tf.square(y_true - y_pred))
        difference = tf.where(abs_loss < 1, square_loss, abs_loss - 0.5)
        return tf.reduce_sum(difference, axis= -1)
        
    def cross_entropy_loss(self, y_true, y_pred):
        y_pred = tf.maximum(1e-7, y_pred)
        return -tf.reduce_sum(y_true*tf.math.log(y_pred),axis= -1)


    def compute_loss(self, y_true, y_pred):
        """y_true, y_pred (b, total_boxes, 2 + 4)
            alpha: coeffienct weight between red_loss and cls_loss"""

        # calculate reg_loss, cls_loss (b, total_boxes)
        cls_loss = self.cross_entropy_loss(y_true[...,:2], y_pred[...,:2])
        reg_loss = self.smooth_l1_loss(y_true[...,2:], y_pred[...,2:])

        #recompute reg_loss that only account for pos_true_boxes
        pos_mask = y_true[..., 1] #(b, total_boxes)
        reg_loss *= pos_mask
        n_pos_box = tf.maximum(tf.reduce_sum(pos_mask, axis= -1), 1e-7)
        reg_loss = tf.reduce_sum(reg_loss, axis= -1)/n_pos_box
        reg_loss = tf.reduce_mean(reg_loss)

        #recompute cls_loss that only accounts for neg and pos true label
        non_neutral_mask = tf.reduce_sum(y_true[..., :2], axis= -1)
        cls_loss *= non_neutral_mask
        n_non_neutral_cls = tf.maximum(tf.reduce_sum(non_neutral_mask, axis= -1), 1e-7)
        cls_loss = tf.reduce_sum(cls_loss, axis= -1)/n_non_neutral_cls
        cls_loss = tf.reduce_mean(cls_loss)
        
        
        return cls_loss + self.alpha*reg_loss



def IOU(boxesA, boxesB):
    """A, B is a numpy array of shape (batch, m, 4), (batch, n, 4) in corners form respectively
    return a numpy array of length m*n """
    m, n = len(boxesA), len(boxesB)
    
    boxesA = np.tile(np.expand_dims(boxesA, axis= 1), reps= (1, n, 1))
    side_len_A = boxesA[..., 2:] - boxesA[..., : 2]
    boxesA_area = side_len_A[..., 0]*side_len_A[..., 1]
    
    boxesB = np.tile(np.expand_dims(boxesB, axis= 0), reps= (m, 1, 1))
    side_len_B = boxesB[..., 2:] - boxesB[..., : 2]
    boxesB_area = side_len_B[..., 0]*side_len_B[..., 1]
    
    
#     boxesB = np.tile(boxesB, (m, 1))
#     side_len_B = boxesB[:, 2:]- boxesB[:, :2]
#     boxesB_area = side_len_B[:, 0]*side_len_B[:, 1]
    
    
#     boxesA = np.repeat(boxesA, n, axis = 0)
#     side_len_A = boxesA[:, 2:]- boxesA[:, :2]
#     boxesA_area = side_len_A[:, 0]*side_len_A[:, 1]
    
    #calculate intersection of area (m, n)
    xmin = np.maximum(boxesA[..., 0], boxesB[..., 0])
    ymin = np.maximum(boxesA[..., 1], boxesB[..., 1])
    xmax = np.minimum(boxesA[..., 2], boxesB[..., 2])
    ymax = np.minimum(boxesA[..., 3], boxesB[..., 3])
    
    
    w = np.maximum(0, (xmax - xmin))
    h = np.maximum(0, (ymax - ymin))
    intersect_areas = w*h
    denom = (boxesA_area + boxesB_area - intersect_areas)
    if np.any(denom == 0):
        position = np.where(denom == 0)
        print('boxesA_area: {}'.format(boxesA_area[position]))
        print('boxesB_area: {}'.format(boxesB_area[position]))
        print('intersect_areas: {}'.format(intersect_areas[position])) 
        print('position: {}'.format(position))
    
    iou = intersect_areas /(boxesA_area + boxesB_area - intersect_areas)
    return iou





def convert_coords(bbs, kind):
    """kind : between centroids, corners and topleft
       bb1: in 2D numpy array"""
    bb1 = np.zeros_like(bbs)
    
    if kind == 'centroids_to_corners':
        x, y, w, h = bbs[..., 0], bbs[..., 1], bbs[..., 2], bbs[..., 3]
        bb1[..., 0] = x - w/2
        bb1[..., 1] = y - h/2
        bb1[..., 2] = x + w/2
        bb1[..., 3] = y + h/2
    
    elif kind == 'corners_to_centroids':
        xmin, ymin, xmax, ymax = bbs[..., 0], bbs[..., 1], bbs[..., 2], bbs[..., 3]
        bb1[..., 0] = (xmin + xmax)/2
        bb1[..., 1] = (ymin + ymax)/2
        bb1[..., 2] = xmax - xmin
        bb1[..., 3] = ymax - ymin
    
    elif kind == 'centroids_to_topleft':
        x, y, w, h = bbs[..., 0], bbs[..., 1], bbs[..., 2], bbs[..., 3]
        bb1[..., 0] = x - w/2
        bb1[..., 1] = y - h/2
        bb1[..., 2] = w
        bb1[..., 3] = h
    
    elif kind == 'topleft_to_centroids':
        xmin, ymin, w, h = bbs[..., 0], bbs[..., 1], bbs[..., 2], bbs[..., 3]
        bb1[..., 0] = xmin + w/2
        bb1[..., 1] = ymin + h/2
        bb1[..., 2] = w
        bb1[..., 3] = h
        
    elif kind == 'topleft_to_corners':
        xmin, ymin, w, h = bbs[..., 0], bbs[..., 1], bbs[..., 2], bbs[..., 3]
        bb1[..., 0] = xmin 
        bb1[..., 1] = ymin 
        bb1[..., 2] = xmin + w
        bb1[..., 3] = ymin + h
    
    elif kind == 'corners_to_topleft':
        xmin, ymin, xmax, ymax = bbs[..., 0], bbs[..., 1], bbs[..., 2], bbs[..., 3]
        bb1[..., 0] = xmin 
        bb1[..., 1] = ymin 
        bb1[..., 2] = xmax - xmin
        bb1[..., 3] = ymax - ymin
    else: raise ValueError('kind is not supported')
    
#     assert not np.min(bb1) < 0, 'value of an bb can be smaller than 0' 
    return bb1




### only used for drawing
def convert_type(bb, kind, image_shape):
    """bb is an array of any form centroids, corners, topleft
       kind: between absolute pixel values or relative in range [0, 1]"""
    w, h = image_shape
    box_scale = np.array([w,h, w, h])
    
    if kind == 'abs2rel':
        return np.array(bb/ box_scale)
    elif kind == 'rel2abs':
        return np.array(bb*box_scale)
    else: raise ValueError('your kind is not supported')






