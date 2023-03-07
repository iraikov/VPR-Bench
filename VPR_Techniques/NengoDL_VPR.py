import cv2
import numpy as np
import time

def compute_map_features(ref_map, vpr_model):  #ref_map is a 1D list of images in this case.
    
    ref_desc_list=[]
    
    for ref_image in ref_map:
        
        if ref_image is not None:   
            ref_image = cv2.resize(ref_image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            gs_img = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
            gs_img = gs_img[:,None,:]
            input_img = np.tile(gs_img, (1, innrstep, 1))
            out = vpr_model.predict(input_img, verbose=0)
            vpr_desc = out[vpr_model.model.probes[1]][0,-1,:]
            vpr_desc.resize(len(vpr_desc),1)
            
        ref_desc_list.append(vpr_desc)

    return ref_desc_list

def compute_query_desc(query, vpr_model):
        
    query_image = cv2.resize(query, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    gs_img = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    gs_img = gs_img.reshape((1,-1))
    gs_img = gs_img[:,None,:]
    input_image = np.tile(gs_img, (1,innrstep,1))
    out = vpr_model.predict(input_image, verbose=0)
    vpr_query_desc = out[vpr_model.model.probes[1]][0,-1,:]
    vpr_query_desc.resize(len(vpr_query_desc), 1)
    
    return vpr_query_desc

def perform_VPR(query_desc, ref_map_features): #ref_map_features is a 1D list of feature descriptors of reference images in this case.

    confusion_vector=np.zeros(len(ref_map_features))
    for itr, ref_desc in enumerate(ref_map_features):
        #t1=time.time()
        query_desc=query_desc.astype('float64')
        ref_desc=ref_desc.astype('float64')
        score=np.dot(query_desc.T,ref_desc)/(np.linalg.norm(query_desc)*np.linalg.norm(ref_desc))
        #t2=time.time()
        #print('HOG tm:',t2-t1)
        confusion_vector[itr]=score
        
    return np.amax(confusion_vector), np.argmax(confusion_vector), confusion_vector
