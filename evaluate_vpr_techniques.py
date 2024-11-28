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


def evaluate_vpr_techniques(
    dataset_dir,
    precomputed_directory,
    techniques,
    save_descriptors,
    scale_percent=100,
    query_images_file=None,
    query_labels_file=None,
    ref_images_file=None,
    ref_labels_file=None,
    model_config=None,
):
    ref_images_names = None
    query_images_names = None

    everything_precomputed = 1
    for vpr_tech in techniques:
        if vpr_tech.find("Precomputed") == -1:
            everything_precomputed = 0

    query_dir = (
        dataset_dir + "/query/"
    )  # Creating path of query directory as per the template proposed in our work.
    ref_dir = (
        dataset_dir + "/ref/"
    )  # Creating path of ref directory as per the template proposed in our work.
    
    if everything_precomputed == 0:

        ref_images_list = []

        if ref_images_file is None:
            ref_images_names = [
                os.path.basename(x) for x in glob.glob(ref_dir + "*.jpg")
            ]

            for image_name in sorted(
                ref_images_names, key=lambda x: int(x.split(".")[0])
            ):  # Reading all the reference images into a list
                print(("Reading Image: " + ref_dir + image_name))
                ref_image = cv2.imread(ref_dir + image_name)
                if ref_image is not None:
                    ################### Optional Resize Provision ###################
                    #                scale_percent = 100 # percent of original size
                    width = int(ref_image.shape[1] * scale_percent / 100)
                    height = int(ref_image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    # resize image
                    ref_image = cv2.resize(ref_image, dim, interpolation=cv2.INTER_AREA)
                    #####################################################
                    ref_images_list.append(ref_image)
                    ref_image = None
                else:
                    print((ref_dir + image_name + " not a valid image!"))
        else:
            ref_images_list_0 = np.load(os.path.join(ref_dir, ref_images_file))
            ref_images_list = None
            if scale_percent < 100:
                ref_images_list = []
                for ref_image in ref_images_list_0:
                    width = int(ref_image.shape[1] * scale_percent / 100)
                    height = int(ref_image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    ref_image = np.asarray(ref_image*255, dtype=np.float32)
                    # resize image
                    ref_image = cv2.resize(ref_image, dim, interpolation=cv2.INTER_AREA)
                    ref_images_list.append(ref_image)
            else:
                ref_images_list = [np.asarray(ref_image*255, dtype=np.float32) 
                                   for ref_image in ref_images_list_0]

    query_indices_dict = {}
    matching_indices_dict = {}
    matching_scores_dict = {}
    encoding_time_dict = {}
    matching_time_dict = {}
    all_retrievedindices_scores_allqueries_dict = {}
    descriptor_shape_dict = {}

    query_images = []
    if query_images_file is None:
        query_images_names = [
            os.path.basename(x) for x in glob.glob(query_dir + "*.jpg")
        ]
        for image_name in query_images_names:
            query_image = cv2.imread(query_dir + image_name)
            if query_image is not None:
                ################### Optional Resize Provision ###################
                #                    scale_percent = 100 # percent (0-100) of original size
                width = int(query_image.shape[1] * scale_percent / 100)
                height = int(query_image.shape[0] * scale_percent / 100)
                dim = (width, height)
                # resize image
                query_image = cv2.resize(query_image, dim, interpolation=cv2.INTER_AREA)
                #####################################################
                query_images_list.append(query_image)
            else:
                print((query_dir + image_name + " not a valid image!"))
    else:
        query_images_list_0 = np.load(os.path.join(query_dir, query_images_file))
        query_images_list = None
        if scale_percent < 100:
            query_images_list = []
            for query_image in query_images_list_0:
                width = int(query_image.shape[1] * scale_percent / 100)
                height = int(query_image.shape[0] * scale_percent / 100)
                dim = (width, height)
                query_image = np.asarray(query_image*255, dtype=np.float32)
                # resize image
                query_image = cv2.resize(query_image, dim, interpolation=cv2.INTER_AREA)
                query_images_list.append(query_image)
        else:
            query_images_list = [np.asarray(query_image*255, dtype=np.float32)
                                 for query_image in query_images_list_0]

    for vpr_tech in techniques:
        matching_indices_list = []
        matching_scores_list = []
        query_indices_list = []
        encoding_time = 0
        matching_time = 0
        all_retrievedindices_scores_allqueries = []
        print(vpr_tech)

        if vpr_tech.find("Precomputed") == -1:
            ref_images_desc = compute_image_descriptors(
                ref_images_list, vpr_tech, model_config
            )  # Compute descriptors of all reference images for the VPR technique.

            itr = 0
            for query_image in query_images_list:
                if query_image is not None:
                    (
                        matched,
                        matching_index,
                        score,
                        t_e,
                        t_m,
                        all_retrievedindices_scores_perquery,
                    ) = place_match(
                        query_image,
                        ref_images_desc,
                        vpr_tech,
                        model_config=model_config,
                    )  # Matches a given query image with all reference images.

                    query_indices_list.append(itr)
                    matching_indices_list.append(matching_index)
                    matching_scores_list.append(score)
                    all_retrievedindices_scores_allqueries.append(
                        all_retrievedindices_scores_perquery
                    )
                    itr = itr + 1
                    encoding_time = (
                        encoding_time + t_e
                    )  # Feature Encoding time per query image
                    matching_time = (
                        matching_time + t_m
                    )  # Descriptor Matching time for 2 image descriptors
                    descriptor_shape = (
                        str(ref_images_desc[0].shape)
                        + " "
                        + str(ref_images_desc[0].dtype)
                    )

                else:
                    raise RuntimeError(" not a valid image!")

            assert itr > 0
            query_indices_dict[vpr_tech] = query_indices_list
            matching_indices_dict[vpr_tech] = matching_indices_list
            matching_scores_dict[vpr_tech] = matching_scores_list
            all_retrievedindices_scores_allqueries_dict[
                vpr_tech
            ] = all_retrievedindices_scores_allqueries
            encoding_time_dict[vpr_tech] = (
                encoding_time / itr
            )  # Average Feature Encoding Time
            matching_time_dict[vpr_tech] = (
                matching_time / itr
            )  # Average Descriptor Matching Time
            descriptor_shape_dict[vpr_tech] = descriptor_shape

            query_indices_array = np.asarray(query_indices_list)
            matching_indices_array = np.asarray(matching_indices_list)
            matching_scores_array = np.asarray(matching_scores_list)
            all_retrievedindices_scores_allqueries_array = np.asarray(
                all_retrievedindices_scores_allqueries
            )

            if save_descriptors == 1:
                cwd = os.getcwd()

                if not os.path.exists(cwd + "/" + precomputed_directory + vpr_tech):
                    os.makedirs(cwd + "/" + precomputed_directory + vpr_tech)
                np.savez(
                    cwd
                    + "/"
                    + precomputed_directory
                    + vpr_tech
                    + "/"
                    + "precomputed_data_corrected.npz",
                    query_indices=query_indices_array,
                    matching_indices=matching_indices_array,
                    matching_scores=matching_scores_array,
                    all_retrievedindices_scores_allqueries=all_retrievedindices_scores_allqueries_array,
                    encoding_time=np.asarray([encoding_time // itr]),
                    matching_time=np.asarray([matching_time // itr]),
                )

        else:
            cwd = os.getcwd()
            try:
                precomputed_data = np.load(
                    cwd
                    + "/"
                    + precomputed_directory
                    + vpr_tech.replace("_Precomputed", "")
                    + "/"
                    + "precomputed_data_corrected.npy",
                    allow_pickle=True,
                    encoding="latin1",
                )

                query_indices_dict[vpr_tech] = precomputed_data[0]
                matching_indices_dict[vpr_tech] = precomputed_data[1]
                matching_scores_dict[vpr_tech] = precomputed_data[2]
                all_retrievedindices_scores_allqueries_dict[
                    vpr_tech
                ] = precomputed_data[3]
                encoding_time_dict[vpr_tech] = precomputed_data[
                    4
                ]  # NOTE: If the descriptors were not computed on the same computational platform as the one running this code, this value of encoding time is compromised and accordingly all the metrics that utilise this (like RMF etc) are not applicable.
                matching_time_dict[vpr_tech] = precomputed_data[
                    5
                ]  # NOTE: If the descriptors were not computed on the same computational platform as the one running this code, this value of matching time is compromised and accordingly all the metrics that utilise this (like RMF etc) are not applicable.
            except FileNotFoundError:
                precomputed_data = np.load(os.path.join(precomputed_directory,
                                                        vpr_tech.replace("_Precomputed", "") + "/",
                                                        "precomputed_data_corrected.npz"),
                                           allow_pickle=True,
                                           encoding="latin1")

                query_indices_dict[vpr_tech] = precomputed_data["query_indices"]
                matching_indices_dict[vpr_tech] = precomputed_data["matching_indices"]
                matching_scores_dict[vpr_tech] = precomputed_data["matching_scores"]
                all_retrievedindices_scores_allqueries_dict[
                    vpr_tech
                ] = precomputed_data["all_retrievedindices_scores_allqueries"]
                encoding_time_dict[vpr_tech] = precomputed_data[
                    "encoding_time"
                ]  # NOTE: If the descriptors were not computed on the same computational platform as the one running this code, this value of encoding time is compromised and accordingly all the metrics that utilise this (like RMF etc) are not applicable.
                matching_time_dict[vpr_tech] = precomputed_data[
                    "matching_time"
                ]  # NOTE: If the descriptors were not computed on the same computational platform as the one running this code, this value of matching time is compromised and accordingly all the metrics that utilise this (like RMF etc) are not applicable.

    return (
        query_indices_dict,
        matching_indices_dict,
        matching_scores_dict,
        encoding_time_dict,
        matching_time_dict,
        all_retrievedindices_scores_allqueries_dict,
        descriptor_shape_dict,
    )
