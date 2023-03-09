import sys
import time
import importlib
import cv2
import numpy as np
import nengo_dl
import tensorflow as tf
import tqdm

def sim_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    global sim
    if sim is not None:
        sim.close()


sys_excepthook = sys.excepthook
sys.excepthook = sim_excepthook


classification_accuracy = lambda y_true, y_pred: tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])

sim = None
net = None

def initialize_sim(model_config):
    global sim, net
    if sim is None:

        model_source_config = model_config["model source"]
        model_classname = model_source_config["class"]
        model_module = model_source_config.get("module", "__main__")
        model_kwargs = model_source_config.get("kwargs", {})
        model_params_path = model_config["model params"]
        
        input_shape=model_config.get("input shape", (28, 28))
        nsteps=model_config.get("nsteps", 20)

        if model_module not in sys.modules:
            importlib.import_module(model_module)
        model_kons = eval(model_classname, sys.modules[model_module].__dict__)
        net = model_kons(**model_kwargs)
        
        sim = nengo_dl.Simulator(net)
        sim.load_params(model_params_path)
        sim.compile(loss={net.out_p: classification_accuracy})
        
    return sim, net

def compute_map_features(ref_map, model_config, **kwargs):  #ref_map is a 1D list of images in this case.

    input_shape=model_config.get("input shape", (28, 28))
    nsteps=model_config.get("nsteps", 20)
    batch_size=100

    sim, net = initialize_sim(model_config)
    
    ref_desc_list=[]
    input_image_batch = []
    
    for i, ref_image in enumerate(tqdm.tqdm(ref_map)):
        
        ref_image = cv2.resize(ref_image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
        gs_img = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY).reshape((1,-1))
        input_img = np.tile(gs_img[:,None,:], (1, nsteps, 1))
        input_image_batch.append(input_img)
        if (i+1) % batch_size == 0:
            input_image_batch_array = np.vstack(input_image_batch)
            out = sim.predict(input_image_batch_array, verbose=0)
            res = out[net.out_p_filt]
            vpr_desc = res[:,-1,:].reshape((batch_size, -1))
            ref_desc_list.append(vpr_desc)
            input_image_batch = []
    if len(input_image_batch) > 0:
        input_image_batch_array = np.vstack(input_image_batch)
        N = input_image_batch_array.shape[0]
        out = sim.predict(input_image_batch_array, verbose=0)
        res = out[net.out_p_filt]
        vpr_desc = res[:,-1,:].reshape((N, -1))
        ref_desc_list.append(vpr_desc)
        
    return np.concatenate(ref_desc_list)


def compute_query_desc(query, model_config, **kwargs):

    input_shape=model_config.get("input shape", (28, 28))
    nsteps=model_config.get("nsteps", 20)
    batch_size=1
    
    sim, net = initialize_sim(model_config)
        
    query_image = cv2.resize(query, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
    gs_img = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY).reshape((1,-1))
    input_image = np.tile(gs_img[:,None,:], (batch_size,nsteps,1))
    out = sim.predict(input_image, verbose=0)
    vpr_query_desc = out[net.out_p_filt][0,-1,:].reshape((-1,1))
    
    return vpr_query_desc

def perform_VPR(query_desc, ref_map_features, **kwargs): #ref_map_features is a 1D list of feature descriptors of reference images in this case.

    confusion_vector=np.zeros(len(ref_map_features))
    for itr, ref_desc in enumerate(ref_map_features):
        #t1=time.time()
        query_desc=query_desc.astype('float64')
        ref_desc=ref_desc.astype('float64')
        score=np.dot(query_desc.T,ref_desc)/(np.linalg.norm(query_desc)*np.linalg.norm(ref_desc))
        #t2=time.time()
        confusion_vector[itr]=score
        
    return np.amax(confusion_vector), np.argmax(confusion_vector), confusion_vector
