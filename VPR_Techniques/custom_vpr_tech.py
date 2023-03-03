#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:49:42 2020

@author: mubariz
"""
import cv2
import numpy as np
import time

grid_x = 20
grid_y = 20
btchsize = 100
innrstep = 30
epoch_num = 20
tau_str = '1e2'
tau = float(tau_str[:-1] + '-' + tau_str[-1])

def compute_map_features(ref_map, vpr_ntwrk):  #ref_map is a 1D list of images in this case.
    
    ref_desc_list=[]
    
    for ref_image in ref_map:
        
        if ref_image is not None:   
            ref_image = cv2.resize(ref_image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            gs_img = np.dot(ref_image, [0.299, 0.587, 0.114]) #RGB to gray-scale
            gs_img = gs_img.reshape((1,-1))
            gs_img = gs_img[:,None,:]
            inpt_btch = np.tile(gs_img, (btchsize,innrstep,1))
            out_btch = vpr_ntwrk.predict(inpt_btch, verbose=0)
            cstm_vpr_desc = out_btch[vpr_ntwrk.model.probes[1]][0,-1,:]
            cstm_vpr_desc.resize(len(cstm_vpr_desc),1)
            
        ref_desc_list.append(cstm_vpr_desc)
        #print(cstm_vpr_desc)
    
    #sim.close()

    return ref_desc_list

def compute_query_desc(query, vpr_ntwrk):
        
    que_image = cv2.resize(query, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    gs_img = np.dot(que_image, [0.299, 0.587, 0.114]) #RGB to gray-scale
    gs_img = gs_img.reshape((1,-1))
    gs_img = gs_img[:,None,:]
    inpt_btch = np.tile(gs_img, (btchsize,innrstep,1))
    out_btch = vpr_ntwrk.predict(inpt_btch, verbose=0)
    cstm_vpr_que_desc = out_btch[vpr_ntwrk.model.probes[1]][0,-1,:]
    cstm_vpr_que_desc.resize(len(cstm_vpr_que_desc),1)
    
    #print(query_desc)
    #print(query_desc.shape)
    
    return cstm_vpr_que_desc

def perform_VPR(query_desc,ref_map_features): #ref_map_features is a 1D list of feature descriptors of reference images in this case.

    confusion_vector=np.zeros(len(ref_map_features))
    itr=0
    for ref_desc in ref_map_features:
        #t1=time.time()
        query_desc=query_desc.astype('float64')
        ref_desc=ref_desc.astype('float64')
        score=np.dot(query_desc.T,ref_desc)/(np.linalg.norm(query_desc)*np.linalg.norm(ref_desc))
        #t2=time.time()
        #print('HOG tm:',t2-t1)
        confusion_vector[itr]=score
        itr=itr+1
        
    return np.amax(confusion_vector), np.argmax(confusion_vector), confusion_vector
