import numpy as np
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def netVLAD(inputs, num_clusters, assign_weight_initializer=None, 
            cluster_initializer=None, skip_postnorm=False):
    ''' skip_postnorm: Only there for compatibility with mat files. '''
    K = num_clusters
    # D: number of (descriptor) dimensions.
    D = inputs.get_shape()[-1]

    # soft-assignment.
    s = keras.layers.Conv2D(filters=K,                             # output channels
                            kernel_size=1,                         # size 1 becomes int 1
                            use_bias=False,                        # stays the same
                            kernel_initializer=assign_weight_initializer,  # stays the same
                            name='assignment'                      # stays the same
                            )(inputs)
    
    a = tf.nn.softmax(s)

    # Dims used hereafter: batch, H, W, desc_coeff, cluster
    # Move cluster assignment to corresponding dimension.
    a = tf.expand_dims(a, -2)

    # VLAD core.
    C = tf.get_variable('cluster_centers', [1, 1, 1, D, K],
                        initializer=cluster_initializer,
                        dtype=inputs.dtype)

    v = tf.expand_dims(inputs, -1) + C
    v = a * v
    v = tf.reduce_sum(v, axis=[1, 2])
    v = tf.transpose(v, perm=[0, 2, 1])

    if not skip_postnorm:
        # Result seems to be very sensitive to the normalization method
        # details, so sticking to matconvnet-style normalization here.
        v = matconvnetNormalize(v, 1e-12)
        v = tf.transpose(v, perm=[0, 2, 1])
        flattened = keras.ops.reshape(v, (-1, np.prod(v.shape[1:])))
        v = matconvnetNormalize(flattened, 1e-12)

    return v

def matconvnetNormalize(inputs, epsilon):
    return inputs / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=-1, keep_dims=True)
                            + epsilon)
