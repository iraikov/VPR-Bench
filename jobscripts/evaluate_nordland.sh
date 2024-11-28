
export ref_images_file="Partitioned_Nordland_Dataset_lowres_train_images.npy"
export ref_labels_file="Partitioned_Nordland_Dataset_lowres_train_labels.npy"
export query_images_file="Partitioned_Nordland_Dataset_lowres_test_images.npy"
export query_labels_file="Partitioned_Nordland_Dataset_lowres_test_labels.npy"
                                                                                                          
                                                                                                          
python3 main.py -em 0 -sm 1 \
 -dn nordland -ddir datasets/nordland \
 -mdir ./VPR/precomputed_matches/nordland \
 -qimf $query_images_file -qlabf $query_labels_file \
 -rimf $ref_images_file -rlabf $ref_labels_file \
 -techs CoHOG #NetVLAD_Precomputed
