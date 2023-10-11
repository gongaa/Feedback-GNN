#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for channel decoding and utility functions."""

import tensorflow as tf
import numpy as np
import scipy as sp # for sparse H matrix computations
from tensorflow.keras.layers import Layer
from sionna.fec.utils import int_mod_2
import matplotlib.pyplot as plt

class QLDPCBPDecoder(Layer):
    def __init__(self,
                 code,
                 trainable=False,
                 cn_type='boxplus',
                 hard_out=True,
                 track_exit=False,
                 num_iter=32,
                 normalization_factor=0.625,
                 output_dtype=tf.float32,
                 loss_type='boxplus-phi', # or sine
                 stage_one=False,
                 stage_two=False,
                 **kwargs):

        super().__init__(dtype=output_dtype, **kwargs)

        # init decoder parameters
        self._trainable = trainable
        self._pcm_x, self._pcm_x_perp = code.hx, code.hx_perp 
        self._pcm_z, self._pcm_z_perp = code.hz, code.hz_perp 
        if stage_one or stage_two:
            self._pcm_x_perp = code.hz
            self._pcm_z_perp = code.hx
        # self._pcm_x, self._pcm_x_perp = code.hx, np.vstack([code.hz, code.lz])
        # self._pcm_z, self._pcm_z_perp = code.hz, np.vstack([code.hx, code.lx])
        self._cn_type = cn_type
        self._hard_out = hard_out
        self._track_exit = track_exit
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)
        self._normalization_factor = normalization_factor
        # self._normalization_factor = tf.Variable(tf.constant(normalization_factor), trainable=True, dtype=tf.float32)
        self._output_dtype = output_dtype

        # clipping value for the atanh function is applied (tf.float32 is used)
        self._atanh_clip_value = 1 - 1e-7
        # internal value for llr clipping
        self._llr_max = tf.constant(20., tf.float32)

        # init code parameters
        self._num_cns_x = self._pcm_x.shape[0] # total number of check nodes
        self._num_cns_z = self._pcm_z.shape[0] # total number of check nodes
        self._num_vns = self._pcm_x.shape[1] # total number of variable nodes

        # make pcm sparse first if ndarray is provided
        if isinstance(self._pcm_x, np.ndarray):
            self._pcm_x = sp.sparse.csr_matrix(self._pcm_x)
            self._pcm_z = sp.sparse.csr_matrix(self._pcm_z)
        # find all edges from variable and check node perspective
        self._cn_con_x, self._vn_con_x, _ = sp.sparse.find(self._pcm_x) # I,J,V (all-one). J is non-decreasing (scipy 1.10).
        self._cn_con_z, self._vn_con_z, _ = sp.sparse.find(self._pcm_z) 
        # fix for scripy>=1.11
        idx_x = np.argsort(self._vn_con_x)
        self._cn_con_x = self._cn_con_x[idx_x]
        self._vn_con_x = self._vn_con_x[idx_x]
        idx_z = np.argsort(self._vn_con_z)
        self._cn_con_z = self._cn_con_z[idx_z]
        self._vn_con_z = self._vn_con_z[idx_z]

        # number of edges equals number of non-zero elements in the
        # parity-check matrix
        self._num_edges_x = len(self._vn_con_x)
        self._num_edges_z = len(self._vn_con_z)

        # permutation index to rearrange messages into check node perspective
        self._ind_cn_x = np.argsort(self._cn_con_x)
        self._ind_cn_z = np.argsort(self._cn_con_z)

        # inverse permutation index to rearrange messages back into variable
        # node perspective
        self._ind_cn_inv_x = np.argsort(self._ind_cn_x)
        self._ind_cn_inv_z = np.argsort(self._ind_cn_z)

        # generate row masks (array of integers defining the row split pos.)
        self._vn_row_splits_x = self._gen_node_mask_row(self._vn_con_x, self._num_edges_x)
        self._vn_row_splits_z = self._gen_node_mask_row(self._vn_con_z, self._num_edges_z)
        # self._cn_row_splits = self._gen_node_mask_row(self._cn_con[self._ind_cn])

        # if self._trainable:
        _, _, _, self._cn_gather_mask_x_perp = self._gen_gather_mask(self._pcm_x_perp)
        _, _, _, self._cn_gather_mask_z_perp = self._gen_gather_mask(self._pcm_z_perp)

        # pre-load the CN function for performance reasons
        if self._cn_type=='boxplus':
            # check node update using the tanh function
            self._cn_update = self._cn_update_tanh
        elif self._cn_type=='boxplus-phi':
            # check node update using the "_phi" function
            self._cn_update = self._cn_update_phi
        elif self._cn_type=='minsum':
            # check node update using the min-sum approximation
            self._cn_update = self._cn_update_minsum
        else:
            raise ValueError('Unknown node type.')

        self._loss_type = loss_type
        self._stage_one = stage_one
        self._stage_two = stage_two

        # for Neural BP
        # init trainable weights if needed
        # self._edge_weights_v2c_x = [tf.Variable(tf.ones(self._num_edges_x),
        #                                  trainable=self._trainable,
        #                                  dtype=tf.float32) for i in range(self._num_iter)]
        # self._edge_weights_v2c_z = [tf.Variable(tf.ones(self._num_edges_z),
        #                                  trainable=self._trainable,
        #                                  dtype=tf.float32) for i in range(self._num_iter)]
        # self._edge_weights_c2v_x = [tf.Variable(tf.constant(normalization_factor, shape=self._num_edges_x),
        #                                     trainable=self._trainable,
        #                                     dtype=tf.float32) for i in range(self._num_iter)] # +1 not used for now
        # self._edge_weights_c2v_z = [tf.Variable(tf.constant(normalization_factor, shape=self._num_edges_z),
        #                                     trainable=self._trainable,
        #                                     dtype=tf.float32) for i in range(self._num_iter)] # +1 not used for now
        # self._llr_weights_x = [tf.Variable(tf.ones(self._num_vns),
        #                                     trainable=self._trainable,
        #                                     dtype=tf.float32) for i in range(self._num_iter+1)]
        # self._llr_weights_y = [tf.Variable(tf.ones(self._num_vns),
        #                                     trainable=self._trainable,
        #                                     dtype=tf.float32) for i in range(self._num_iter+1)]
        # self._llr_weights_z = [tf.Variable(tf.ones(self._num_vns),
        #                                     trainable=self._trainable,
        #                                     dtype=tf.float32) for i in range(self._num_iter+1)]

    def show_weights(self, size=7, type="v2c", iter=0):
        """Show histogram of trainable weights.

        Input
        -----
            size: float
                Figure size of the matplotlib figure.

        """
        # only plot if weights exist
        if type == "v2c":
            weights = np.concatenate([self._edge_weights_v2c_x[iter].numpy(), self._edge_weights_v2c_z[iter].numpy()])
        elif type == "c2v":
            weights = np.concatenate([self._edge_weights_c2v_x[iter].numpy(), self._edge_weights_c2v_z[iter].numpy()])
        elif type == "llr":
            weights = np.concatenate([self._llr_weights_x[iter].numpy(), self._llr_weights_y[iter].numpy(), self._llr_weights_z[iter].numpy()])
        elif type == "normalization":
            weights = self._normalization_factor.numpy()

        plt.figure(figsize=(size,size))
        plt.hist(weights, density=True, bins=20, align='mid')
        plt.xlabel('weight value')
        plt.ylabel('density')
        plt.grid(True, which='both', axis='both')
        plt.title('Weight Distribution')

    #########################
    # Utility methods
    #########################
    def _gen_gather_mask(self, pcm):
        edges = np.stack(np.where(pcm), axis=1)
        # Create 2D ragged tensor of shape [num_cn,...]
        # cn_edges[i] contains the edge ids for CN i
        num_cn, num_vn = pcm.shape
        cn_edges = []
        checked_vns = []
        for i in range(num_cn):
            v = np.where(edges[:,0]==i)
            cn_edges.append(v[0])
            checked_vns.append(edges[v][:,1])
        cn_edges = tf.ragged.constant(cn_edges)
        checked_vns = tf.ragged.constant(checked_vns)

        # Create 2D ragged tensor of shape [num_vn,...]
        # vn_edges[i] contains the edge ids for VN i
        vn_edges = []
        for i in range(num_vn):
            vn_edges.append(np.where(edges[:,1]==i)[0])
        vn_edges = tf.ragged.constant(vn_edges)

        return edges, cn_edges, vn_edges, checked_vns
        
    def _gen_node_mask(self, con, num_edges):
        """ Generates internal node masks indicating which msg index belongs
        to which node index.
        """
        ind = np.argsort(con)
        con = con[ind]

        node_mask = []

        cur_node = 0
        cur_mask = []
        for i in range(num_edges):
            if con[i] == cur_node:
                cur_mask.append(ind[i])
            else:
                node_mask.append(cur_mask)
                cur_mask = [ind[i]]
                cur_node += 1
        node_mask.append(cur_mask)
        return node_mask

    def _gen_node_mask_row(self, con, num_edges):
        """ Defining the row split positions of a 1D vector consisting of all
        edges messages.

        Used to build a ragged Tensor of incoming node messages.
        """
        node_mask = [0] # the first element indicates the first node index (=0)

        cur_node = 0
        for i in range(num_edges):
            if con[i] != cur_node:
                node_mask.append(i)
                cur_node += 1
        node_mask.append(num_edges) # last element must be the number of
        # elements (delimiter)
        return node_mask

    def _vn_update(self, msg_x, msg_z, llr_ch, it, sum_only=False):
        """ Variable node update function.

        This function implements the (extrinsic) variable node update
        function. It takes the sum over all incoming messages ``msg`` excluding
        the intrinsic (= outgoing) message itself.

        Additionally, the channel LLR ``llr_ch`` is added to each message.
        """
        # aggregate all incoming messages per node
        llrx = llr_ch[0]
        llry = llr_ch[1]
        llrz = llr_ch[2]
        # llrx = self._mult_weights(llr_ch[0], self._llr_weights_x, it)
        # llry = self._mult_weights(llr_ch[1], self._llr_weights_y, it)
        # llrz = self._mult_weights(llr_ch[2], self._llr_weights_z, it)

        llrx_hx = tf.reduce_sum(msg_z, axis=1, keepdims=False)
        llrz_hz = tf.reduce_sum(msg_x, axis=1, keepdims=False)
        llry_all = tf.add(llrx_hx, llrz_hz) + llry
        llrx_hx = llrx_hx + llrx
        llrz_hz = llrz_hz + llrz

        if sum_only:
            return llrx_hx, llry_all, llrz_hz

        # For XLA
        llrz_hx = tf.ragged.map_flat_values(lambda x, y, row_ind: x+tf.gather(y, row_ind), -1.*msg_x, llrz_hz, msg_x.value_rowids())
        llry_hx = tf.ragged.map_flat_values(lambda x, y, row_ind: x+tf.gather(y, row_ind), -1.*msg_x, llry_all, msg_x.value_rowids())
        llrx_hz = tf.ragged.map_flat_values(lambda x, y, row_ind: x+tf.gather(y, row_ind), -1.*msg_z, llrx_hx, msg_z.value_rowids())
        llry_hz = tf.ragged.map_flat_values(lambda x, y, row_ind: x+tf.gather(y, row_ind), -1.*msg_z, llry_all, msg_z.value_rowids())
        # in Eager and Graph mode, the above four lines can be written as
        # llrz_hx = llrz_hz - msg_x
        # llry_hx = llry_all - msg_x
        # llrx_hz = llrx_hx - msg_z
        # llry_hz = llry_all - msg_z

        # replace with a numerical stable version
        num_hx = tf.math.softplus(-1.*llrx_hx)
        denom_hx = tf.ragged.map_flat_values(lambda x, y: tf.math.reduce_logsumexp(-1.*tf.stack([x, y], axis=-1), axis=-1), llrz_hx, llry_hx)
        # new_msg_x = num_hx - denom_hx
        new_msg_x = tf.ragged.map_flat_values(lambda x, y, row_ind: tf.gather(y, row_ind)-x, denom_hx, num_hx, denom_hx.value_rowids())

        num_hz = tf.math.softplus(-1.*llrz_hz)
        denom_hz = tf.ragged.map_flat_values(lambda x, y: tf.math.reduce_logsumexp(-1.*tf.stack([x, y], axis=-1), axis=-1), llrx_hz, llry_hz)
        # new_msg_z = num_hz - denom_hz
        new_msg_z = tf.ragged.map_flat_values(lambda x, y, row_ind: tf.gather(y, row_ind)-x, denom_hz, num_hz, denom_hz.value_rowids())
        
        return new_msg_x, new_msg_z, llrx_hx, llry_all, llrz_hz

    def _extrinsic_min(self, msg):
        """ Provides the extrinsic min operation for the minsum approximation
        of the CN function.

        This function implements the extrinsic min operation, i.e.,
        the min is taken over all values excluding the value at the current
        index.

        Note that the input is expected to be a Tensor and NOT a ragged Tensor.
        """
        num_val = tf.shape(msg)[0]
        msg = tf.transpose(msg, (1,0))
        msg = tf.expand_dims(msg, axis=1)
        id_mat = tf.eye(num_val)

        msg = (tf.tile(msg, (1, num_val, 1)) # create outgoing tensor per value
               + 1000. * id_mat) # "ignore" intrinsic msg by adding large const.


        msg = tf.math.reduce_min(msg, axis=2)
        msg = tf.transpose(msg, (1,0))
        return msg

    def _where_ragged(self, msg):
        """Helper to replace 0 elements from ragged tensor (called with
        map_flat_values)."""
        return tf.where(tf.equal(msg, 0), tf.ones_like(msg) * 1e-12, msg)

    def _where_ragged_inv(self, msg):
        """Helper to replace small elements from ragged tensor (called with
        map_flat_values) with exact `0`."""
        msg_mod =  tf.where(tf.less(tf.abs(msg), 1e-7),
                            tf.zeros_like(msg),
                            msg)
        return msg_mod

    def _cn_update_tanh(self, msg, syndrome):
        """Check node update function implementing the exact boxplus operation.

        This function implements the (extrinsic) check node update
        function. It calculates the boxplus function over all incoming messages
        "msg" excluding the intrinsic (=outgoing) message itself.
        The exact boxplus function is implemented by using the tanh function.

        The input is expected to be a ragged Tensor of shape
        `[num_cns, None, batch_size]`.

        Note that for numerical stability clipping is applied.
        """

        msg = msg / 2
        # tanh is not overloaded for ragged tensors
        msg = tf.ragged.map_flat_values(tf.tanh, msg) # tanh is not overloaded

        # for ragged tensors; map to flat tensor first
        msg = tf.ragged.map_flat_values(self._where_ragged, msg)

        msg_prod = tf.reduce_prod(msg, axis=1) # [_num_cns, bs]
        msg_prod = tf.math.multiply(msg_prod, syndrome)

        # TF2.9 does not support XLA for the multiplication of ragged tensors
        # the following code provides a workaround that supports XLA

        # ^-1 to avoid division
        # Note this is (potentially) numerically unstable
        # msg = msg**-1 * tf.expand_dims(msg_prod, axis=1) # remove own edge

        msg = tf.ragged.map_flat_values(lambda x, y, row_ind :
                                        x * tf.gather(y, row_ind),
                                        msg**-1,
                                        msg_prod,
                                        msg.value_rowids())

        # Overwrite small (numerical zeros) message values with exact zero
        # these are introduced by the previous "_where_ragged" operation
        # this is required to keep the product stable (cf. _phi_update for log
        # sum implementation)
        msg = tf.ragged.map_flat_values(self._where_ragged_inv, msg)

        msg = tf.clip_by_value(msg,
                               clip_value_min=-self._atanh_clip_value,
                               clip_value_max=self._atanh_clip_value)

        # atanh is not overloaded for ragged tensors
        msg = 2 * tf.ragged.map_flat_values(tf.atanh, msg)
        # tf.print("in cn update tanh, msg= ", msg)
        return msg

    def _phi(self, x):
        """Helper function for the check node update.

        This function implements the (element-wise) `"_phi"` function as defined
        in [Ryan]_.
        """
        # the clipping values are optimized for tf.float32
        x = tf.clip_by_value(x, clip_value_min=8.5e-8, clip_value_max=16.635532)
        return tf.math.softplus(x) - tf.math.log(tf.math.exp(x)-1)
        return tf.math.log(tf.math.exp(x)+1) - tf.math.log(tf.math.exp(x)-1)

    def _cn_update_phi(self, msg, syndrome):
        """Check node update function implementing the exact boxplus operation.

        This function implements the (extrinsic) check node update function
        based on the numerically more stable `"_phi"` function (cf. [Ryan]_).
        It calculates the boxplus function over all incoming messages ``msg``
        excluding the intrinsic (=outgoing) message itself.
        The exact boxplus function is implemented by using the `"_phi"` function
        as in [Ryan]_.

        The input is expected to be a ragged Tensor of shape
        `[num_cns, None, batch_size]`.

        Note that for numerical stability clipping is applied.
        """

        sign_val = tf.sign(msg)

        sign_val = tf.where(tf.equal(sign_val, 0),
                            tf.ones_like(sign_val),
                            sign_val)

        sign_node = tf.reduce_prod(sign_val, axis=1)
        sign_node = tf.math.multiply(sign_node, syndrome)

        # TF2.9 does not support XLA for the multiplication of ragged tensors
        # the following code provides a workaround that supports XLA

        # sign_val = sign_val * tf.expand_dims(sign_node, axis=1)
        sign_val = tf.ragged.map_flat_values(lambda x, y, row_ind :
                                             x * tf.gather(y, row_ind),
                                             sign_val,
                                             sign_node,
                                             sign_val.value_rowids())

        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

        # apply _phi element-wise (does not support ragged Tensors)
        msg = tf.ragged.map_flat_values(self._phi, msg)
        msg_sum = tf.reduce_sum(msg, axis=1)

        # TF2.9 does not support XLA for the addition of ragged tensors
        # the following code provides a workaround that supports XLA

        # msg = tf.add( -msg, tf.expand_dims(msg_sum, axis=1)) # remove own edge
        msg = tf.ragged.map_flat_values(lambda x, y, row_ind :
                                        x + tf.gather(y, row_ind),
                                        -1.*msg,
                                        msg_sum,
                                        msg.value_rowids())

        # apply _phi element-wise (does not support ragged Tensors)
        msg = self._stop_ragged_gradient(sign_val) * tf.ragged.map_flat_values(
                                                            self._phi, msg)
        # tf.print("cn update phi, msg", msg)
        return msg

    @tf.function(autograph=False)
    def _cn_update_phi_loss(self, msg):
        sign_val = tf.sign(msg)

        sign_val = tf.where(tf.equal(sign_val, 0),
                            tf.ones_like(sign_val),
                            sign_val)

        sign_node = tf.reduce_prod(sign_val, axis=1)

        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

        # apply _phi element-wise (does not support ragged Tensors)
        msg = tf.ragged.map_flat_values(self._phi, msg)
        msg = tf.reduce_sum(msg, axis=1)

        # apply _phi element-wise (does not support ragged Tensors)
        # msg = self._stop_ragged_gradient(sign_node) * tf.ragged.map_flat_values(self._phi, msg)
        msg = sign_node * self._phi(msg)
        # tf.print("cn update phi loss, msg", msg)
        return msg

    @tf.function(autograph=False)
    def cal_logit(self, llrx, llry, llrz):
        # for bce loss later
        num_no_z = tf.math.softplus(-1.*llrx) 
        denom_is_z = tf.math.reduce_logsumexp(-1.*tf.stack([llrz, llry], axis=-1), axis=-1)
        llr_z = num_no_z - denom_is_z # log((p_I+p_X)/(p_Z+p_Y)) = log((1-p'_Z)/p'_Z)

        num_no_x = tf.math.softplus(-1.*llrz)
        denom_is_x = tf.math.reduce_logsumexp(-1.*tf.stack([llrx, llry], axis=-1), axis=-1)
        llr_x = num_no_x - denom_is_x # log((p_I+p_Z)/(p_X+o_Y)) = log((1-p'_X)/p'_X)

        msg_vn_x_perp = tf.gather(llr_x, self._cn_gather_mask_x_perp, axis=0)
        msg_vn_z_perp = tf.gather(llr_z, self._cn_gather_mask_z_perp, axis=0)
        x_perp_logit = self._cn_update_phi_loss(msg_vn_x_perp)
        z_perp_logit = self._cn_update_phi_loss(msg_vn_z_perp)

        return x_perp_logit, z_perp_logit
        
    def _cal_prob(self, llrx, llry, llrz):
        # for sine loss later
        num_no_z = tf.math.softplus(-1.*llrx)
        denom_is_z = tf.math.reduce_logsumexp(-1.*tf.stack([llrz, llry], axis=-1), axis=-1)
        llr_z = num_no_z - denom_is_z # log((p_I+p_X)/(p_Z+p_Y)) = log((1-p'_Z)/p'_Z)
        p_z = tf.math.sigmoid(-1.*llr_z) # p'_Z

        num_no_x = tf.math.softplus(-1.*llrz)
        denom_is_x = tf.math.reduce_logsumexp(-1.*tf.stack([llrx, llry], axis=-1), axis=-1)
        llr_x = num_no_x - denom_is_x # log((p_I+p_Z)/(p_X+o_Y)) = log((1-p'_X)/p'_X)
        p_x = tf.math.sigmoid(-1.*llr_x) # p'_X

        return p_x, p_z 

    def _stop_ragged_gradient(self, rt):
        """Helper function as TF 2.5 does not support ragged gradient
        stopping"""
        return rt.with_flat_values(tf.stop_gradient(rt.flat_values))

    def _sign_val_minsum(self, msg):
        """Helper to replace find sign-value during min-sum decoding.
        Must be called with `map_flat_values`."""

        sign_val = tf.sign(msg)
        sign_val = tf.where(tf.equal(sign_val, 0),
                            tf.ones_like(sign_val),
                            sign_val)
        return sign_val

    def _cn_update_minsum_mapfn(self, msg):
        """ Check node update function implementing the min-sum approximation.

        This function approximates the (extrinsic) check node update
        function based on the min-sum approximation (cf. [Ryan]_).
        It calculates the "extrinsic" min function over all incoming messages
        ``msg`` excluding the intrinsic (=outgoing) message itself.

        The input is expected to be a ragged Tensor of shape
        `[num_vns, None, batch_size]`.

        This function uses tf.map_fn() to call the CN updates.
        It is currently not used, but can be used as template to implement
        modified CN functions (e.g., offset-corrected minsum).
        Please note that tf.map_fn lowers the throughput significantly.
        """

        sign_val = tf.ragged.map_flat_values(self._sign_val_minsum, msg)

        sign_node = tf.reduce_prod(sign_val, axis=1)
        sign_val = self._stop_ragged_gradient(sign_val) * tf.expand_dims(
                                                             sign_node, axis=1)

        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

        # calculate extrinsic messages and include the sign
        msg_e = tf.map_fn(self._extrinsic_min, msg, infer_shape=False)

        # ensure shape after map_fn
        msg_fv = msg_e.flat_values
        msg_fv = tf.ensure_shape(msg_fv, msg.flat_values.shape)
        msg_e = msg.with_flat_values(msg_fv)

        msg = sign_val * msg_e

        return msg

    def _cn_update_minsum(self, msg, syndrome):
        """ Check node update function implementing the min-sum approximation.

        This function approximates the (extrinsic) check node update
        function based on the min-sum approximation (cf. [Ryan]_).
        It calculates the "extrinsic" min function over all incoming messages
        ``msg`` excluding the intrinsic (=outgoing) message itself.

        The input is expected to be a ragged Tensor of shape
        `[num_vns, None, batch_size]`.
        """
        # a constant used overwrite the first min
        LARGE_VAL = 10000. # pylint: disable=invalid-name

        # clip values for numerical stability
        msg = tf.clip_by_value(msg,
                               clip_value_min=-self._llr_max,
                               clip_value_max=self._llr_max)

        # calculate sign of outgoing msg
        sign_val = tf.ragged.map_flat_values(self._sign_val_minsum, msg)

        sign_node = tf.reduce_prod(sign_val, axis=1)
        sign_node = tf.math.multiply(sign_node, syndrome)

        # TF2.9 does not support XLA for the multiplication of ragged tensors
        # the following code provides a workaround that supports XLA

        # sign_val = self._stop_ragged_gradient(sign_val) \
        #             * tf.expand_dims(sign_node, axis=1)
        sign_val = tf.ragged.map_flat_values(
                                        lambda x, y, row_ind:
                                        tf.multiply(x, tf.gather(y, row_ind)),
                                        self._stop_ragged_gradient(sign_val),
                                        sign_node,
                                        sign_val.value_rowids())

        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

        # Calculate the extrinsic minimum per CN, i.e., for each message of
        # index i, find the smallest and the second smallest value.
        # However, in some cases the second smallest value may equal the
        # smallest value (multiplicity of mins).
        # Please note that this needs to be applied to raggedTensors, e.g.,
        # tf.top_k() is currently not supported and the ops must support graph
        # # mode.

        # find min_value per node
        min_val = tf.reduce_min(msg, axis=1, keepdims=True)

        # TF2.9 does not support XLA for the subtraction of ragged tensors
        # the following code provides a workaround that supports XLA

        # and subtract min; the new array contains zero at the min positions
        # benefits from broadcasting; all other values are positive
        # msg_min1 = msg - min_val
        msg_min1 = tf.ragged.map_flat_values(lambda x, y, row_ind:
                                             x- tf.gather(y, row_ind),
                                             msg,
                                             tf.squeeze(min_val, axis=1),
                                             msg.value_rowids())

        # replace 0 (=min positions) with large value to ignore it for further
        # min calculations
        msg = tf.ragged.map_flat_values(lambda x:
                                        tf.where(tf.equal(x, 0), LARGE_VAL, x),
                                        msg_min1)

        # find the second smallest element (we add min_val as this has been
        # subtracted before)
        min_val2 = tf.reduce_min(msg, axis=1, keepdims=True) + min_val

        # Detect duplicated minima (i.e., min_val occurs at two incoming
        # messages). As the LLRs per node are <LLR_MAX and we have
        # replace at least 1 position (position with message "min_val") by
        # LARGE_VAL, it holds for the sum < LARGE_VAL + node_degree*LLR_MAX.
        # if the sum > 2*LARGE_VAL, the multiplicity of the min is at least 2.
        node_sum = tf.reduce_sum(msg, axis=1, keepdims=True) - (2*LARGE_VAL-1.)
        # indicator that duplicated min was detected (per node)
        double_min = 0.5*(1-tf.sign(node_sum))

        # if a duplicate min occurred, both edges must have min_val, otherwise
        # the second smallest value is taken
        min_val_e = (1-double_min) * min_val + (double_min) * min_val2

        # replace all values with min_val except the position where the min
        # occurred (=extrinsic min).
        msg_e = tf.where(msg==LARGE_VAL, min_val_e, min_val)

        # it seems like tf.where does not set the shape of tf.ragged properly
        # we need to ensure the shape manually
        msg_e = tf.ragged.map_flat_values(
                                    lambda x:
                                    tf.ensure_shape(x, msg.flat_values.shape),
                                    msg_e)

        # TF2.9 does not support XLA for the multiplication of ragged tensors
        # the following code provides a workaround that supports XLA

        # and apply sign
        #msg = sign_val * msg_e
        msg = tf.ragged.map_flat_values(tf.multiply,
                                        sign_val,
                                        msg_e)
        # tf.print("cn update minsum, msg:", msg)
        return msg

    def _mult_weights(self, x, wt, it):
        """Multiply messages with trainable weights for weighted BP."""
        # transpose for simpler broadcasting of training variables
        x = tf.transpose(x, (1, 0))
        x = tf.math.multiply(x, tf.gather(wt, it))
        x = tf.transpose(x, (1, 0))
        return x

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        pass

    def call(self, inputs):
        """Iterative BP decoding function.

        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated codeword.

        Args:
        llr_ch or (llr_ch, msg_vn):

            llr_ch (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

            msg_vn (tf.float32) : Ragged tensor containing the VN
                messages, or None. Required if ``stateful`` is set to True.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]` containing
            bit-wise soft-estimates (or hard-decided bit-values) of all
            codeword bits.

        Raises:
            ValueError: If ``inputs`` is not of shape `[batch_size, n]`.

            InvalidArgumentError: When rank(``inputs``)<2.
        """

        # Extract inputs
        llr_ch, syndrome_x, syndrome_z = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Invalid input dtype.')

        # internal calculations still in tf.float32
        llr_ch = tf.cast(llr_ch, tf.float32)

        # clip llrs for numerical stability
        # llr_ch = tf.clip_by_value(llr_ch,
        #                           clip_value_min=-self._llr_max,
        #                           clip_value_max=self._llr_max)

        # last dim must be of length n
        tf.debugging.assert_equal(tf.shape(llr_ch)[-1],
                                  self._num_vns,
                                  'Last dimension must be of length n.')

        # must be done during call, as XLA fails otherwise due to ragged
        # indices placed on the CPU device.
        # create permutation index from cn perspective
        self._cn_mask_tf_x = tf.ragged.constant(self._gen_node_mask(self._cn_con_x, self._num_edges_x),
                                              row_splits_dtype=tf.int32)
        self._cn_mask_tf_z = tf.ragged.constant(self._gen_node_mask(self._cn_con_z, self._num_edges_z),
                                              row_splits_dtype=tf.int32)


        # shape of llr_ch: [bs, 3, _num_vns]
        bs = tf.shape(llr_ch)[0]
        llr_ch = tf.transpose(llr_ch, (1,2,0))
        # expect the shape of llr_ch to be [3, _num_vns, bs]

        # map 0->+1, 1->-1
        syndrome_x = tf.cast(1 - 2 * syndrome_x, tf.float32)
        syndrome_z = tf.cast(1 - 2 * syndrome_z, tf.float32)
        # syndrome_x, syndrome_z has shape [_num_cn_x, bs] and [_num_cn_z, bs], contains +1 or -1

        msg_shape_x = tf.stack([tf.constant(self._num_edges_x), bs], axis=0)
        msg_shape_z = tf.stack([tf.constant(self._num_edges_z), bs], axis=0)
        msg_vn_x = tf.zeros(msg_shape_x, dtype=tf.float32)
        msg_vn_z = tf.zeros(msg_shape_z, dtype=tf.float32)


        llr_hat = tf.TensorArray(tf.float32, size=2*self._num_iter+2)
        x_logit, z_logit = tf.zeros([self._pcm_x_perp.shape[0], bs]), tf.zeros([self._pcm_z_perp.shape[0], bs])
        for it in range(self._num_iter):
            # tf.print("it", it)
            msg_vn_x = tf.RaggedTensor.from_row_splits(
                        values=msg_vn_x,
                        row_splits=tf.constant(self._vn_row_splits_x, tf.int32))
            msg_vn_z = tf.RaggedTensor.from_row_splits(
                        values=msg_vn_z,
                        row_splits=tf.constant(self._vn_row_splits_z, tf.int32))

            msg_vn_x, msg_vn_z, llrx, llry, llrz = self._vn_update(msg_vn_x, msg_vn_z, llr_ch, it)

            if self._trainable or self._stage_two: # cannot do it > 0 for XLA compatibility
                x_logit, z_logit = self.cal_logit(llrx, llry, llrz)
                llr_hat = llr_hat.write(2*it, x_logit)
                llr_hat = llr_hat.write(2*it+1, z_logit)

            # msg_vn_x = tf.ragged.map_flat_values(self._mult_weights, msg_vn_x, self._edge_weights_v2c_x, it)
            # msg_vn_z = tf.ragged.map_flat_values(self._mult_weights, msg_vn_z, self._edge_weights_v2c_z, it)

            # permute edges into CN perspective
            msg_cn_x = tf.gather(msg_vn_x.flat_values, self._cn_mask_tf_x, axis=None)
            msg_cn_z = tf.gather(msg_vn_z.flat_values, self._cn_mask_tf_z, axis=None)

            # check node update using the pre-defined function
            msg_cn_x = self._cn_update(msg_cn_x, syndrome_x)
            msg_cn_z = self._cn_update(msg_cn_z, syndrome_z)

            msg_cn_x = msg_cn_x * self._normalization_factor
            msg_cn_z = msg_cn_z * self._normalization_factor

            # msg_cn_x = tf.ragged.map_flat_values(self._mult_weights, msg_cn_x, self._edge_weights_c2v_x, it)
            # msg_cn_z = tf.ragged.map_flat_values(self._mult_weights, msg_cn_z, self._edge_weights_c2v_z, it)

            # re-permute edges to variable node perspective
            msg_vn_x = tf.gather(msg_cn_x.flat_values, self._ind_cn_inv_x, axis=None)
            msg_vn_z = tf.gather(msg_cn_z.flat_values, self._ind_cn_inv_z, axis=None)


        # make hard decision
        msg_vn_x = tf.RaggedTensor.from_row_splits(
                    values=msg_vn_x,
                    row_splits=tf.constant(self._vn_row_splits_x, tf.int32))
        msg_vn_z = tf.RaggedTensor.from_row_splits(
                    values=msg_vn_z,
                    row_splits=tf.constant(self._vn_row_splits_z, tf.int32))
        llrx, llry, llrz = self._vn_update(msg_vn_x, msg_vn_z, llr_ch, self._num_iter, sum_only=True)
        x_logit, z_logit = self.cal_logit(llrx, llry, llrz)
        if self._trainable or self._stage_two:
            llr_hat = llr_hat.write(2*self._num_iter  , x_logit)
            llr_hat = llr_hat.write(2*self._num_iter+1, z_logit)
        # restore batch dimension to first dimension
        llrx = tf.transpose(llrx, (1,0))
        llrz = tf.transpose(llrz, (1,0))
        llry = tf.transpose(llry, (1,0))
        i_hat = tf.zeros_like(llrx)
        all_hat = tf.stack([i_hat, llrx, llrz, llry], axis=0)
        decision = tf.math.argmin(all_hat, axis=0)
        x_hat = int_mod_2(decision)  # faster implementation of tf.math.mod(decision, 2)
        z_hat = tf.subtract(decision, x_hat) / 2

        if self._stage_one: # work as stage-one decoder and not trainable
            return llrx, llry, llrz, x_hat, z_hat, x_logit, z_logit
        elif self._trainable or self._stage_two:
            return llr_hat.stack(), x_hat, z_hat 
        else: 
            return x_hat, z_hat
