#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Originally created on Tue Mar 24 12:49:47 2020

@author: mubariz
"""
from vpr_system import compute_image_descriptors
from vpr_system import place_match
import cv2
import os
import glob
import numpy as np

def evaluate_vpr_techniques(dataset_dir,precomputed_directory,techniques, save_descriptors, cstm_ntwrk, scale_percent=100):
    
    everything_precomputed=1 
    for vpr_tech in techniques:
        if (vpr_tech.find('Precomputed')==-1):
            everything_precomputed=0
    
    if (everything_precomputed==0):
        query_dir=dataset_dir+'query/' #Creating path of query directory as per the template proposed in our work.
        ref_dir=dataset_dir+'ref/' #Creating path of ref directory as per the template proposed in our work.
    
        ref_images_list=[]    
        ref_images_names=[os.path.basename(x) for x in glob.glob(ref_dir+'*.jpg')]  
        query_images_names=[os.path.basename(x) for x in glob.glob(query_dir+'*.jpg')]
    
        for image_name in sorted(ref_images_names,key=lambda x:int(x.split(".")[0])):  #Reading all the reference images into a list
            print(('Reading Image: ' + ref_dir+image_name))
            ref_image=cv2.imread(ref_dir+image_name)
            if (ref_image is not None):
                ################### Optional Resize Provision ###################
#                scale_percent = 100 # percent of original size
                width = int(ref_image.shape[1] * scale_percent / 100)
                height = int(ref_image.shape[0] * scale_percent / 100)
                dim = (width, height)
                # resize image
                ref_image = cv2.resize(ref_image, dim, interpolation = cv2.INTER_AREA)
                #####################################################
                ref_images_list.append(ref_image)
                print((ref_image.shape[1],ref_image.shape[0]))
                ref_image=None
            else:
                print((ref_dir+image_name+' not a valid image!'))
            
    query_indices_dict={}
    matching_indices_dict={}
    matching_scores_dict={}
    encoding_time_dict={}
    matching_time_dict={}
    all_retrievedindices_scores_allqueries_dict={}
    descriptor_shape_dict={}
    
    for vpr_tech in techniques:
        matching_indices_list=[]
        matching_scores_list=[]
        query_indices_list=[]
        encoding_time=0
        matching_time=0        
        all_retrievedindices_scores_allqueries=[]
        print(vpr_tech)
        
        if (vpr_tech.find('Precomputed')==-1):
            ref_images_desc= compute_image_descriptors(ref_images_list, cstm_ntwrk, vpr_tech) #Compute descriptors of all reference images for the VPR technique.
    
            itr=0
            for image_name in sorted(query_images_names,key=lambda x:int(x.split(".")[0])):     #Iterating over each query image instead of loading them all at once, to save RAM space
                query_image=cv2.imread(query_dir+image_name)
                if (query_image is not None):
                    
                    ################### Optional Resize Provision ###################
#                    scale_percent = 100 # percent (0-100) of original size
                    width = int(query_image.shape[1] * scale_percent / 100)
                    height = int(query_image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    # resize image
                    query_image = cv2.resize(query_image, dim, interpolation = cv2.INTER_AREA)
                    #####################################################
    
                    matched, matching_index, score, t_e, t_m, all_retrievedindices_scores_perquery  = place_match(query_image,ref_images_desc, cstm_ntwrk, vpr_tech)  #Matches a given query image with all reference images.
                    
                    query_indices_list.append(itr)
                    matching_indices_list.append(matching_index)
                    matching_scores_list.append(score)
                    all_retrievedindices_scores_allqueries.append(all_retrievedindices_scores_perquery)
                    itr=itr+1
                    encoding_time=encoding_time+t_e  #Feature Encoding time per query image
                    matching_time=matching_time+t_m  #Descriptor Matching time for 2 image descriptors    
                    descriptor_shape=str(ref_images_desc[0].shape)+' '+str(ref_images_desc[0].dtype)
                    
                else:
                    print((query_dir+image_name+' not a valid image!'))
            

            query_indices_dict[vpr_tech]=query_indices_list        
            matching_indices_dict[vpr_tech]=matching_indices_list
            matching_scores_dict[vpr_tech]=matching_scores_list
            all_retrievedindices_scores_allqueries_dict[vpr_tech]=all_retrievedindices_scores_allqueries
            encoding_time_dict[vpr_tech]=encoding_time/itr  #Average Feature Encoding Time 
            matching_time_dict[vpr_tech]=matching_time/itr  #Average Descriptor Matching Time
            descriptor_shape_dict[vpr_tech]=descriptor_shape

            query_indices_array = np.asarray(query_indices_list)
            matching_indices_array = np.asarray(matching_indices_list)
            matching_scores_array = np.asarray(matching_scores_list)
            all_retrievedindices_scores_allqueries_array = np.asarray(all_retrievedindices_scores_allqueries)

            if (save_descriptors==1):
                cwd=os.getcwd()
                
                if not os.path.exists(cwd+'/'+precomputed_directory+vpr_tech):
                    os.makedirs(cwd+'/'+precomputed_directory+vpr_tech)
                np.savez(cwd+'/'+precomputed_directory+vpr_tech+'/'+'precomputed_data_corrected.npz',
                query_indices=query_indices_array,
                matching_indices=matching_indices_array,                            
                matching_scores=matching_scores_array,
                all_retrievedindices_scores_allqueries=all_retrievedindices_scores_allqueries_array,
                encoding_time=np.asarray([encoding_time//itr]),
                matching_time=np.asarray([matching_time//itr]))

        else:
            cwd=os.getcwd()
            precomputed_data=np.loadz(cwd+'/'+precomputed_directory+vpr_tech.replace("_Precomputed","")+'/'+'precomputed_data_corrected.npz' )

            query_indices_dict[vpr_tech]=precomputed_data['query_indices']
            matching_indices_dict[vpr_tech]=precomputed_data['matching_indices']
            matching_scores_dict[vpr_tech]=precomputed_data['matching_scores']
            all_retrievedindices_scores_allqueries_dict[vpr_tech]=precomputed_data['all_retrievedindices_scores_allqueries']
            encoding_time_dict[vpr_tech]=precomputed_data['encoding_time']  #NOTE: If the descriptors were not computed on the same computational platform as the one running this code, this value of encoding time is compromised and accordingly all the metrics that utilise this (like RMF etc) are not applicable. 
            matching_time_dict[vpr_tech]=precomputed_data['matching_time']  #NOTE: If the descriptors were not computed on the same computational platform as the one running this code, this value of matching time is compromised and accordingly all the metrics that utilise this (like RMF etc) are not applicable.

    return query_indices_dict, matching_indices_dict, matching_scores_dict, encoding_time_dict, matching_time_dict, all_retrievedindices_scores_allqueries_dict, descriptor_shape_dict
