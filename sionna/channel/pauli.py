#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layer for simulating an AWGN channel"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from sionna.utils import expand_to_rank 

class Pauli(Layer):
    r"""Pauli(dtype=tf.complex64, **kwargs)

    Add iid Pauli noise to the inputs.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    This layer adds Pauli noise with `px` `py` `pz` to the input.
    It can be either a scalar or a tensor which can be broadcast to the shape
    of the input.

    Example
    --------

    Setting-up:

    >>> pauli_channel = Pauli()

    Running:

    >>> # x is the channel input
    >>> # px, py, pz are the probabilities of an X, Y=iXZ, Z noise happening 
    >>> y_x, y_z, noise_x, noise_z = pauli_channel((x, px, py, pz))

    Parameters
    ----------
        dtype : Complex tf.DType
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `tf.complex64`.

    Input
    -----

        (x, px, py, pz) :
            Tuple:

        x :  Tensor, tf.uint8
            Channel input (binary)

        px : Scalar or Tensor, tf.float
            Scalar or tensor whose shape can be broadcast to the shape of ``x``.
            If ``px`` is a scalar, noise of the same variance will be added to the input.
            If ``px`` is a tensor, it must have a shape that can be broadcast to
            the shape of ``x``. This allows, e.g., adding noise of different
            flip probability to each example in a batch. If ``px`` has a lower rank than
            ``x``, then ``px`` will be broadcast to the shape of ``x`` by adding
            dummy dimensions after the last axis.

    Output
    -------
        y_x : Tensor with the same shape as ``x``, tf.uint8
            Channel output
        y_z : Tensor with the same shape as ``x``, tf.uint8
            Channel output
        noise_x : Tensor with the same shape as ``x``, tf.uint8
            Added channel X-type (bit-flip) noise 
        noise_z : Tensor with the same shape as ``x``, tf.uint8
            Added channel Z-type (phase-flip) noise
    """

    def __init__(self, dtype=tf.uint8, wt=False, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._wt = wt
        self._real_dtype = tf.dtypes.as_dtype(self._dtype).real_dtype

    def call(self, inputs):

        if self._wt:
            cx, cz, wt = inputs
            bs, n = tf.shape(cx)[0], tf.shape(cx)[1]
            wt = tf.cast(wt, tf.int32)
            # pos = tf.map_fn(lambda i: tf.cast(tf.numpy_function(np.random.choice, [n, wt, False], tf.int64), tf.int32), tf.range(bs))
            pos = tf.map_fn(lambda i: tf.random.shuffle(tf.range(0, n, dtype=tf.int32))[:wt], tf.range(bs))
            # row_idx = tf.map_fn(lambda i: tf.cast(tf.numpy_function(np.full, [wt, i], tf.int32), tf.int32), tf.range(bs))
            row_idx = tf.tile(tf.expand_dims(tf.range(0, bs, dtype=tf.int32), -1), (1, wt))
            pos = tf.reshape(tf.stack((row_idx, pos), axis=-1), (-1,2))

            noise = tf.random.uniform([tf.shape(pos)[0]], 0, 1, tf.float32)
            noise_x_ind = tf.squeeze(tf.gather(pos, tf.where(noise < 2/3), axis=0), axis=1)
            noise_x = tf.zeros_like(cx, dtype=tf.bool)
            noise_x = tf.tensor_scatter_nd_update(noise_x, noise_x_ind, tf.ones(tf.shape(noise_x_ind)[0], dtype=tf.bool))
            noise_z_ind = tf.squeeze(tf.gather(pos, tf.where(noise > 1/3), axis=0), axis=1)
            noise_z = tf.zeros_like(cx, dtype=tf.bool)
            noise_z = tf.tensor_scatter_nd_update(noise_z, noise_z_ind, tf.ones(tf.shape(noise_z_ind)[0], dtype=tf.bool))

        else:
            cx, cz, px, py, pz= inputs
            noise = tf.random.uniform(tf.shape(cx), 0, 1, tf.float32)
            px_tile = tf.tile([[px]], tf.shape(cx))
            py_tile = tf.tile([[py]], tf.shape(cx))
            pz_tile = tf.tile([[pz]], tf.shape(cx))

            noise_x = noise < px_tile
            mask1 = noise >= (px_tile - py_tile)
            mask2 = noise < ((px_tile + pz_tile) - py_tile)
            noise_z = tf.math.logical_and(mask1, mask2)

        if cx is not None and cz is not None:
            cx_bool = tf.cast(cx, dtype=tf.bool)
            cz_bool = tf.cast(cz, dtype=tf.bool)
            y_x = tf.math.logical_xor(cx_bool, noise_x)
            y_z = tf.math.logical_xor(cz_bool, noise_z) 
            return y_x, y_z, noise_x, noise_z
        else:
            return noise_x, noise_z
