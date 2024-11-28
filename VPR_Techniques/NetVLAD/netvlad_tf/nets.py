import numpy as np
import os
from os.path import dirname
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from . import layers as layers

def defaultCheckpoint():
    return os.path.join(dirname(dirname(__file__)), 
                        'checkpoints', 
                        'vd16_pitts30k_conv5_3_vlad_preL2_intra_white')

def vgg16NetvladPca(image_batch):
    ''' Assumes rank 4 input, first 3 dims fixed or dynamic, last dim 1 or 3. 
    '''
    assert len(image_batch.shape) == 4
    
    with tf.variable_scope('vgg16_netvlad_pca'):
        # Gray to color if necessary.
        if image_batch.shape[3] == 1:
            x = keras.ops.conv(
                image_batch,
                np.ones((1, 1, 1, 3)),  # kernel
                strides=(1, 1, 1, 1),   # equivalent to np.ones(4).tolist()
                padding='valid'         # 'VALID' in TF becomes lowercase 'valid'
            )
        else :
            assert image_batch.shape[3] == 3
            x = image_batch
        
        # Subtract trained average image.
        average_rgb = tf.get_variable(
                'average_rgb', 3, dtype=image_batch.dtype)
        x = x - average_rgb
        
        # VGG16
        def vggConv(inputs, numbers, out_dim, with_relu):
            if with_relu:
                activation = tf.nn.relu
            else:
                activation = None
            return keras.layers.Conv2D(filters=out_dim,          # number of output channels
                                       kernel_size=(3, 3),       # [3, 3] becomes tuple (3, 3)
                                       strides=1,                # can be int instead of tuple when same in all dimensions
                                       padding='same',           # stays the same
                                       activation=activation,     # stays the same
                                       name='conv%s' % numbers   # stays the same
                                       )(inputs)
        def vggPool(inputs):
            return keras.layers.MaxPooling2D(pool_size=2,    # can be int instead of tuple when same in both dimensions
                                             strides=2       # when not specified, strides defaults to same as pool_size
                                             )(inputs)
        
        x = vggConv(x, '1_1', 64, True)
        x = vggConv(x, '1_2', 64, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '2_1', 128, True)
        x = vggConv(x, '2_2', 128, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '3_1', 256, True)
        x = vggConv(x, '3_2', 256, True)
        x = vggConv(x, '3_3', 256, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '4_1', 512, True)
        x = vggConv(x, '4_2', 512, True)
        x = vggConv(x, '4_3', 512, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '5_1', 512, True)
        x = vggConv(x, '5_2', 512, True)
        x = vggConv(x, '5_3', 512, False)
        
        # NetVLAD
        x = tf.nn.l2_normalize(x, dim=-1)
        x = layers.netVLAD(x, 64)
        
        # PCA
        x = keras.layers.Conv2D(filters=4096, 
                                kernel_size=1, 
                                strides=1, 
                                name='WPCA'
                                )(keras.ops.expand_dims(keras.ops.expand_dims(x, 1), 1))
        flattened = keras.ops.reshape(x, (-1, np.prod(x.shape[1:])))
        x = tf.nn.l2_normalize(flattened, dim=-1)
        
    return x
