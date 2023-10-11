# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


##### Utility functions for graph neural networks #####

import os
import numpy as np
import scipy as sp # for sparse H matrix computations
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.losses import BinaryCrossentropy 
import pickle
from sionna.utils.metrics import compute_bler
from time import time
import warnings # ignore some internal TensorFlow warnings

# for e2e model
from sionna.utils import BinarySource 
from sionna.channel import Pauli
from sionna.fec.utils import int_mod_2

class MLP(Layer):
    """Simple MLP layer.

    Parameters
    ----------
    units : List of int
        Each element of the list describes the number of units of the
        corresponding layer.

    activations : List of activations
        Each element of the list contains the activation to be used
        by the corresponding layer.

    use_bias : List of booleans
        Each element of the list indicates if the corresponding layer
        should use a bias or not.
    """
    def __init__(self, units, activations, use_bias):
        super().__init__()
        self._num_units = units
        self._activations = activations
        self._use_bias = use_bias

    def build(self, input_shape):
        self._layers = []
        for i, units in enumerate(self._num_units):
            self._layers.append(Dense(units,
                                      self._activations[i],
                                      use_bias=self._use_bias[i],
                                      bias_initializer='ones'))
            
    # @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None]),))
    @tf.function(autograph=False)
    def call(self, inputs):
        # print(f"Tracing MLP for num_units={self._num_units}, input shape={inputs.shape}, input type={type(inputs)}.")
        outputs = inputs
        for layer in self._layers:
            outputs = layer(outputs)
        return outputs

class GNN_BP4(Layer):
    """GNN-based message passing decoder.

    Parameters
    ---------
    code: css_code
        A CSS quantum LDPC code.

    num_embed_dims: int
        Number of dimensions of the vertex embeddings.

    num_msg_dims: int
        Number of dimensions of a message.

    num_hidden_units: int
        Number of hidden units of the MLPs used to compute
        messages and to update the vertex embeddings.

    num_mlp_layers: int
        Number of layers of the MLPs used to compute
        messages and to update the vertex embeddings.

    num_iter: int
        Number of iterations.

    reduce_op: str
        A string defining the vertex aggregation function.
        Currently, "mean", "max", "min" and "sum" is supported.

    activation: str
        A string defining the activation function of the hidden MLP layers to
        be used. Defaults to "relu".

    clip_llr_to: float or None
        If set, the absolute value of the input LLRs will be clipped to this value.

    use_attributes: Boolean
        Defaults to False. If True, trainable node and edge attributes will be
        applied per node/edge, respectively.

    node_attribute_dims: int
        Number of dimensions of each node attribute.

    msg_attribute_dims: int
        Number of dimensions of each message attribute.

    use_bias: Boolean
        Defaults to False. Indicates if the MLPs should use a bias or not.

    Input
    -----
    llr : [batch_size, num_vn], tf.float32
        Tensor containing the LLRs of all bits.

    Output
    ------
    llr_hat: : [batch_size, num_vn], tf.float32
        Tensor containing the LLRs at the decoder output.
        If `output_all_iter`==True, a list of such tensors will be returned.
    """
    def __init__(self,
                 code,
                 num_embed_dims,
                 num_msg_dims,
                 num_hidden_units,
                 num_mlp_layers,
                 num_iter,
                 reduce_op="mean",
                 activation="tanh",
                 clip_llr_to=None,
                 use_attributes=False,
                 node_attribute_dims=0,
                 msg_attribute_dims=0,
                 use_bias=False,
                 input_embed=False,
                 loss_type="boxplus-phi"):

        super().__init__()

        self._pcm_x, self._pcm_lx = code.hx, code.lx # Parity check matrix
        self._pcm_z, self._pcm_lz = code.hz, code.lz
        # pcm = np.vstack((self._pcm_x, self._pcm_z))
        self._num_cn_x, self._num_vn = self._pcm_x.shape # number of X-type check nodes
        self._num_cn_z = self._pcm_z.shape[0]
        self._num_edges_x = int(np.sum(self._pcm_x))
        self._num_edges_z = int(np.sum(self._pcm_z))

        self._edges_x, self._cn_edges_x, self._vn_edges_x, self._cn_gather_mask_x = self.create_edges(self._pcm_x)
        self._edges_z, self._cn_edges_z, self._vn_edges_z, self._cn_gather_mask_z = self.create_edges(self._pcm_z)

        _, _, _, self._cn_gather_mask_lx = self.create_edges(self._pcm_lx)
        _, _, _, self._cn_gather_mask_lz = self.create_edges(self._pcm_lz)

        # lx = hz_perp \ hx
        self._cn_gather_mask_z_perp = tf.concat([self._cn_gather_mask_x, self._cn_gather_mask_lx], axis=0)
        self._cn_gather_mask_x_perp = tf.concat([self._cn_gather_mask_z, self._cn_gather_mask_lz], axis=0)
        
        # Number of dimensions for vertex embeddings
        self._num_embed_dims = num_embed_dims

        # Number of dimensions for messages
        self._num_msg_dims = num_msg_dims

        # Number of hidden units for MLPs computing messages and embeddings
        self._num_hidden_units = num_hidden_units

        # Number of layers for MLPs computing messages and embeddings
        self._num_mlp_layers = num_mlp_layers

        # Number of BP iterations, can be modified
        self._num_iter = num_iter

        # Reduce operation for message aggregation
        self._reduce_op = reduce_op

        # Activation function of the hidden MLP layers
        self._activation = activation

        # Defines the (internal) LLR clipping value
        self._clip_llr_to = clip_llr_to

        # Actives (trainable) attributes
        self._use_attributes = use_attributes

        # Node /Edge attribute dimensions
        self._node_attribute_dims = node_attribute_dims
        self._msg_attribute_dims = msg_attribute_dims

        # Activate bias of MLP layers
        self._use_bias = use_bias

        # Internal state for initialization
        self._is_built = False

        self._input_embed = input_embed

        self._loss_type = loss_type

    @property
    def num_iter(self):
        return self._num_iter

    @num_iter.setter # no retracing of graph (=no effect in graph mode)
    def num_iter(self, value):
        self._num_iter = value

    def create_edges(self, pcm):
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

    def build(self, input_shape):
        if not self._is_built: # only build once
            self._is_built=True

            self._syndrome_embed_x = Dense(self._num_embed_dims, use_bias=self._use_bias)
            self._syndrome_embed_z = Dense(self._num_embed_dims, use_bias=self._use_bias)

            # NN to transform VN embedding to llr_x, llr_y, llr_z
            self._llr_inv_embed = Dense(3, use_bias=self._use_bias, bias_initializer='ones',
                                        kernel_initializer='zeros')

            # CN embedding update function
            self.update_h_cn = UpdateCNEmbeddings(self._num_msg_dims,
                                                self._num_hidden_units,
                                                self._num_mlp_layers,
                                                # Flip columns: "from VN to CN"
                                                (np.flip(self._edges_x, 1), np.flip(self._edges_z, 1)),
                                                (self._cn_edges_x, self._cn_edges_z),
                                                self._reduce_op,
                                                self._activation,
                                                self._use_attributes,
                                                self._node_attribute_dims,
                                                self._msg_attribute_dims,
                                                self._use_bias)

            # VN embedding update function
            # TODO: reduce op for messages coming from Hx and Hz respectively
            self.update_h_vn = UpdateVNEmbeddings(self._num_msg_dims,
                                                self._num_hidden_units,
                                                self._num_mlp_layers,
                                                # "from CN to VN"
                                                (self._edges_x, self._edges_z), 
                                                (self._vn_edges_x, self._vn_edges_z),
                                                self._reduce_op,
                                                self._activation,
                                                self._use_attributes,
                                                self._node_attribute_dims,
                                                self._msg_attribute_dims,                                                
                                                self._use_bias)

    @tf.function(autograph=False)
    def embed_to_llr(self, h_vn):
        """Transform VN embeddings to LLR for X,Y,Z flips."""
        print(f"tracing embed to llr for h_vn shape={h_vn.shape}")
        embed = self._llr_inv_embed(h_vn)   # shape [bs, n, 3]
        embed = tf.transpose(embed, (2,1,0)) # shape [3, n, bs]
        llrx, llry, llrz = embed[0], embed[1], embed[2]
        return llrx, llry, llrz

    @tf.function(autograph=False)
    def cal_logit(self, h_vn):
        print(f"tracing cal_logit for h_vn shape={h_vn.shape}")
        llrx, llry, llrz = self.embed_to_llr(h_vn) # each has shape [n, bs]
        # sum_llr = tf.reduce_mean(llrx) + 2*tf.reduce_mean(llry) + tf.reduce_mean(llrz)
        # sum_llr = tf.reduce_mean(llrx) + tf.reduce_mean(llrz)
        # batch dimension is the last
        num_no_z = tf.math.softplus(-1.*llrx) 
        denom_is_z = tf.math.reduce_logsumexp(-1.*tf.stack([llrz, llry], axis=-1), axis=-1)
        llr_z = num_no_z - denom_is_z # log((p_I+p_X)/(p_Z+p_Y)) = log((1-p'_Z)/p'_Z)

        num_no_x = tf.math.softplus(-1.*llrz)
        denom_is_x = tf.math.reduce_logsumexp(-1.*tf.stack([llrx, llry], axis=-1), axis=-1)
        llr_x = num_no_x - denom_is_x # log((p_I+p_Z)/(p_X+o_Y)) = log((1-p'_X)/p'_X)

        msg_vn_hz = tf.gather(llr_x, self._cn_gather_mask_z, axis=0)
        msg_vn_lz = tf.gather(llr_x, self._cn_gather_mask_lz, axis=0)
        msg_vn_hx = tf.gather(llr_z, self._cn_gather_mask_x, axis=0)
        msg_vn_lx = tf.gather(llr_z, self._cn_gather_mask_lx, axis=0)
        hz_logit = self._cn_update_phi_loss(msg_vn_hz)
        lz_logit = self._cn_update_phi_loss(msg_vn_lz)
        hx_logit = self._cn_update_phi_loss(msg_vn_hx)
        lx_logit = self._cn_update_phi_loss(msg_vn_lx)
        x_perp_logit = tf.concat([hz_logit, lz_logit], axis=0)
        z_perp_logit = tf.concat([hx_logit, lx_logit], axis=0)
        return tf.transpose(hx_logit), tf.transpose(hz_logit), x_perp_logit, z_perp_logit 

    # @tf.function(autograph=False)
    def cal_prob(self, h_vn):
        # for sine loss 
        print("tracing cal_prob")
        llrx, llry, llrz = self.embed_to_llr(h_vn) # each has shape [n, bs]

        num_no_z = tf.math.softplus(-1.*llrx)
        denom_is_z = tf.math.reduce_logsumexp(-1.*tf.stack([llrz, llry], axis=-1), axis=-1)
        llr_z = num_no_z - denom_is_z # log((p_I+p_X)/(p_Z+p_Y)) = log((1-p'_Z)/p'_Z)
        p_z = tf.math.sigmoid(-1.*llr_z) # p'_Z

        num_no_x = tf.math.softplus(-1.*llrz)
        denom_is_x = tf.math.reduce_logsumexp(-1.*tf.stack([llrx, llry], axis=-1), axis=-1)
        llr_x = num_no_x - denom_is_x # log((p_I+p_Z)/(p_X+o_Y)) = log((1-p'_X)/p'_X)
        p_x = tf.math.sigmoid(-1.*llr_x) # p'_X

        return p_x, p_z
    
    @tf.function(autograph=False)
    def _phi(self, x):
        print("tracing _phi")
        x = tf.clip_by_value(x, clip_value_min=8.5e-8, clip_value_max=16.635532)
        return tf.math.log(tf.math.exp(x)+1) - tf.math.log(tf.math.exp(x)-1)

    @tf.function(autograph=False)
    def _cn_update_phi_loss(self, msg):
        print(f"tracing cn update phi loss for msg shape={msg.shape}")
        sign_val = tf.sign(msg)

        sign_val = tf.where(tf.equal(sign_val, 0),
                            tf.ones_like(sign_val),
                            sign_val)

        sign_node = tf.reduce_prod(sign_val, axis=1)

        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

        msg = tf.ragged.map_flat_values(self._phi, msg)
        msg = tf.reduce_sum(msg, axis=1)
        msg = sign_node * self._phi(msg)
        return msg
    
    def make_hard_decision(self, h_vn):
        print("tracing make hard decision")
        llrx, llry, llrz = self.embed_to_llr(h_vn)
        llri = tf.zeros_like(llrx)
        all = tf.stack([llri, llrx, llrz, llry], axis=0)
        decision = tf.math.argmin(all, axis=0)
        x_out = int_mod_2(decision)
        z_out = tf.subtract(decision, x_out) / 2
        return x_out, z_out

    def _gen_gather_mask(self, I, J):
        gather_mask = []
        cur_node = 0
        temp = []
        for i in range(len(I)):
            if I[i] != cur_node:
                gather_mask.append(temp)
                temp = [J[i]]
                cur_node += 1
            else: 
                temp.append(J[i])
        gather_mask.append(temp)
        row_lengths = [len(t) for t in gather_mask]
        return gather_mask, row_lengths

    def call(self, inputs):
        """Run the decoder."""
        syndrome_x, syndrome_z = inputs
        batch_size = tf.shape(syndrome_x)[0]
        print(f"Tracing GNN_BP4 for values batch_size={batch_size}.")
        
        # map 0->+1, 1->-1
        syndrome_x = 1 - 2 * syndrome_x
        syndrome_z = 1 - 2 * syndrome_z
        h_cn_x = tf.zeros([batch_size, self._num_cn_x, self._num_embed_dims])
        h_cn_z = tf.zeros([batch_size, self._num_cn_z, self._num_embed_dims])

        # Initialize vertex embeddings
        h_vn = tf.ones([batch_size, self._num_vn, self._num_embed_dims])

        # BP iterations
        llr_hat = []
        h_cn_x, h_cn_z = self.update_h_cn(h_vn, h_cn_x, h_cn_z, 
                tf.zeros_like(syndrome_x, dtype=tf.float32), tf.zeros_like(syndrome_z, dtype=tf.float32))
        for i in range(self._num_iter):

            # Update VNs
            h_vn = self.update_h_vn(h_cn_x, h_cn_z, h_vn, syndrome_x, syndrome_z)

            if self._loss_type == "boxplus-phi":
                hx_logit, hz_logit, x_logit, z_logit, sum_llr = self.cal_logit(h_vn)
                llr_hat.append((x_logit, z_logit, sum_llr))
            elif self._loss_type == "sine":
                x_logit, z_logit = self.cal_prob(h_vn)
                llr_hat.append((x_logit, z_logit))

            if i == self._num_iter - 1:
                break
            # Update CNs
            hx_logit = tf.math.multiply(hx_logit, tf.cast(syndrome_x, dtype=hx_logit.dtype)) # shape [bs, cn]
            hz_logit = tf.math.multiply(hz_logit, tf.cast(syndrome_z, dtype=hz_logit.dtype))
            h_cn_x, h_cn_z = self.update_h_cn(h_vn, h_cn_x, h_cn_z, hx_logit, hz_logit)

        x_hat, z_hat = self.make_hard_decision(h_vn)
        
        return llr_hat, x_hat, z_hat


class UpdateCNEmbeddings(Layer):
    """
    Update vertex embeddings of the GNN BP decoder.

    This layer computes first the messages that are sent across the edges
    of the graph, then sums the incoming messages at each vertex, finally and
    updates their embeddings.

    Parameters
    ----------
    num_msg_dims: int
        Number of dimensions of a message.

    num_hidden_units: int
        Number of hidden units of MLPs used to compute
        messages and to update the vertex embeddings.

    num_mlp_layers: int
        Number of layers of the MLPs used to compute
        messages and to update the vertex embeddings.

    from_to_ind: [num_egdes, 2], np.array
        Two dimensional array containing in each row the indices of the
        originating and receiving vertex for an edge.

    gather_ind: [`num_vn` or `num_cn`, None], tf.ragged.constant
        Ragged tensor that contains for each receiving vertex the list of
        edge indices from which to aggregate the incoming messages. As each
        vertex can have a different degree, a ragged tensor is used.

    reduce_op: str
        A string defining the vertex aggregation function.
        Currently, "mean", "max", "min" and "sum" is supported.

    activation: str
        A string defining the activation function of the hidden MLP layers to
        be used. Defaults to "relu".

    use_attributes: Boolean
        Defaults to False. If True, trainable node and edge attributes will be
        applied per node/edge, respectively.

    node_attribute_dims: int
        Number of dimensions of each node attribute.

    msg_attribute_dims: int
        Number of dimensions of each message attribute.

    use_bias: Boolean
        Defaults to False. Indicates if the MLP should use a bias or not.
        
    Input
    -----
    h_from : [batch_size, num_cn or num_vn, num_embed_dims], tf.float32
        Tensor containing the embeddings of the "transmitting" vertices.

    h_to : [batch_size, num_vn or num_cn, num_embed_dims], tf.float32
        Tensor containing the embeddings of the "receiving" vertices.

    Output
    ------
    h_to_new : Same shape and type as `h_to`
        Tensor containing the updated embeddings of the "receiving" vertices.
    """
    def __init__(self,
                 num_msg_dims,
                 num_hidden_units,
                 num_mlp_layers,
                 from_to_ind,
                 gather_ind,
                 reduce_op="sum",
                 activation="relu",
                 use_attributes=False,
                 node_attribute_dims=0,
                 msg_attribute_dims=0,                 
                 use_bias=False):

        super().__init__()
        self._num_msg_dims = num_msg_dims
        self._num_hidden_units = num_hidden_units
        self._num_mlp_layers = num_mlp_layers
        from_to_ind_x, from_to_ind_z = from_to_ind
        self._from_ind_x = from_to_ind_x[:,0]
        self._to_ind_x = from_to_ind_x[:,1]
        self._from_ind_z = from_to_ind_z[:,0]
        self._to_ind_z = from_to_ind_z[:,1]
        self._gather_ind_x, self._gather_ind_z = gather_ind
        self._reduce_op = reduce_op
        self._activation = activation
        self._use_attributes = use_attributes
        self._node_attribute_dims = node_attribute_dims
        self._msg_attribute_dims = msg_attribute_dims
        self._use_bias = use_bias

        # add node attributes
        if self._use_attributes:
            num_nodes_x = self._gather_ind_x.shape[0]
            num_edges_x = self._from_ind_x.shape[0]
            num_nodes_z = self._gather_ind_z.shape[0]
            num_edges_z = self._from_ind_z.shape[0]
            # node attributes
            self._g_node_x = tf.Variable(tf.zeros((num_nodes_x, self._node_attribute_dims),tf.float32), trainable=True)
            self._g_node_z = tf.Variable(tf.zeros((num_nodes_z, self._node_attribute_dims),tf.float32), trainable=True)
            # edge attributes
            self._g_msg_x = tf.Variable(tf.zeros((num_edges_x, self._msg_attribute_dims), tf.float32), trainable=True)
            self._g_msg_z = tf.Variable(tf.zeros((num_edges_z, self._msg_attribute_dims), tf.float32), trainable=True)

    def build(self, input_shape):

        num_embed_dims = input_shape[-1]

        # MLP to compute messages
        units = [self._num_hidden_units]*(self._num_mlp_layers-1) + [self._num_msg_dims]
        activations = [self._activation]*(self._num_mlp_layers-1) + [None]
        use_bias = [self._use_bias]*self._num_mlp_layers

        # X and Z type of CN uses different MLPs
        self._msg_mlp_x = MLP(units, activations, use_bias)
        self._msg_mlp_z = MLP(units, activations, use_bias)


        # MLP to update embeddings from accumulated messages
        units[-1] = num_embed_dims
        # different embedding for messages to Hx and Hz
        self._embed_mlp_x = MLP(units, activations, use_bias)
        self._embed_mlp_z = MLP(units, activations, use_bias)

    def reduce_msg(self, messages, gather_ind):
        print(f"Tracing reduce msg in update CN embedding for msg shape={messages.shape}")
        # Reduce messages at each receiving (to) vertex
        # note: bring batch dim to last dim for improved performance with ragged tensors
        messages = tf.transpose(messages, (1,2,0)) # [num_edges, embed_dim, bs]
        m_ragged = tf.gather(messages, gather_ind, axis=0)
        # m_ragged shape [23, None, 20, None]
        if self._reduce_op=="sum":
            m = tf.reduce_sum(m_ragged, axis=1)
        elif self._reduce_op=="mean":
            m = tf.reduce_mean(m_ragged, axis=1)
        elif self._reduce_op=="max":
            m = tf.reduce_max(m_ragged, axis=1)
        elif self._reduce_op=="min":
            m = tf.reduce_min(m_ragged, axis=1)
        else:
            raise ValueError("unknown reduce operation")
        # after reduction, m shape [23, 20, 256]
        return tf.transpose(m, (2,0,1)) # batch-dim back to first dim
    
    @tf.function(autograph=False)
    def call(self, h_from, h_to_x, h_to_z, hx_logit, hz_logit):
        print(f"Tracing update CN embedding.")

        # Concatenate embeddings of the transmitting (from) and receiving (to) vertex for each edge
        features_x = tf.concat([tf.gather(h_from, self._from_ind_x, axis=1),
                              tf.gather(h_to_x, self._to_ind_x, axis=1)], axis=-1)

        features_z = tf.concat([tf.gather(h_from, self._from_ind_z, axis=1),
                              tf.gather(h_to_z, self._to_ind_z, axis=1)], axis=-1)

        if self._use_attributes:
            # attributes shared between samples
            attr_x = tf.tile(tf.expand_dims(self._g_msg_x, axis=0), [tf.shape(features_x)[0], 1, 1])
            attr_z = tf.tile(tf.expand_dims(self._g_msg_z, axis=0), [tf.shape(features_z)[0], 1, 1])
            features_x = tf.concat([features_x, attr_x], axis=-1)
            features_z = tf.concat([features_z, attr_z], axis=-1)

        messages_x = self._msg_mlp_x(features_x)
        messages_z = self._msg_mlp_z(features_z)

        m_x = self.reduce_msg(messages_x, self._gather_ind_x)
        m_z = self.reduce_msg(messages_z, self._gather_ind_z)

        if self._use_attributes:
            attr_x = tf.tile(tf.expand_dims(self._g_node_x, axis=0), [tf.shape(m_x)[0], 1, 1])
            attr_z = tf.tile(tf.expand_dims(self._g_node_z, axis=0), [tf.shape(m_z)[0], 1, 1])
            m_x = tf.concat([m_x, attr_x], axis=-1)
            m_z = tf.concat([m_z, attr_z], axis=-1)

        hx_logit = tf.expand_dims(hx_logit, -1)
        hz_logit = tf.expand_dims(hz_logit, -1)

        # Compute new embeddings
        h_to_x_new = self._embed_mlp_x(tf.concat([m_x, h_to_x, hx_logit], axis=-1))
        h_to_z_new = self._embed_mlp_z(tf.concat([m_z, h_to_z, hz_logit], axis=-1))

        return h_to_x_new, h_to_z_new

class UpdateVNEmbeddings(Layer):
    """
    from_to_ind: [num_egdes, 2], np.array
        Two dimensional array containing in each row the indices of the
        originating and receiving vertex for an edge.

    gather_ind: [`num_vn` or `num_cn`, None], tf.ragged.constant
        Ragged tensor that contains for each receiving vertex the list of
        edge indices from which to aggregate the incoming messages. As each
        vertex can have a different degree, a ragged tensor is used.

    Input
    -----
    h_from : [batch_size, num_cn or num_vn, num_embed_dims], tf.float32
        Tensor containing the embeddings of the "transmitting" vertices.

    h_to : [batch_size, num_vn or num_cn, num_embed_dims], tf.float32
        Tensor containing the embeddings of the "receiving" vertices.

    Output
    ------
    h_to_new : Same shape and type as `h_to`
        Tensor containing the updated embeddings of the "receiving" vertices.
    """
    def __init__(self,
                 num_msg_dims,
                 num_hidden_units,
                 num_mlp_layers,
                 from_to_ind,
                 gather_ind,
                 reduce_op="sum",
                 activation="relu",
                 use_attributes=False,
                 node_attribute_dims=0,
                 msg_attribute_dims=0,
                 use_bias=False):

        super().__init__()
        self._num_msg_dims = num_msg_dims
        self._num_hidden_units = num_hidden_units
        self._num_mlp_layers = num_mlp_layers
        from_to_ind_x, from_to_ind_z = from_to_ind
        self._from_ind_x = from_to_ind_x[:,0]
        self._to_ind_x = from_to_ind_x[:,1]
        self._from_ind_z = from_to_ind_z[:,0]
        self._to_ind_z = from_to_ind_z[:,1]
        self._gather_ind_x, self._gather_ind_z = gather_ind
        self._reduce_op = reduce_op
        self._activation = activation
        self._use_attributes = use_attributes
        self._node_attribute_dims = node_attribute_dims
        self._msg_attribute_dims = msg_attribute_dims
        self._use_bias = use_bias

        # add node attributes
        if self._use_attributes:
            num_nodes = self._gather_ind_x.shape[0]
            num_edges_x = self._from_ind_x.shape[0]
            num_edges_z = self._from_ind_z.shape[0]
            # node attributes
            self._g_node = tf.Variable(tf.zeros((num_nodes, self._node_attribute_dims),tf.float32), trainable=True)
            # edge attributes
            self._g_msg_x = tf.Variable(tf.zeros((num_edges_x, self._msg_attribute_dims), tf.float32), trainable=True)
            self._g_msg_z = tf.Variable(tf.zeros((num_edges_z, self._msg_attribute_dims), tf.float32), trainable=True)

    def build(self, input_shape):

        num_embed_dims = input_shape[-1]

        # MLP to compute messages
        units = [self._num_hidden_units]*(self._num_mlp_layers-1) + [self._num_msg_dims]
        activations = [self._activation]*(self._num_mlp_layers-1) + [None]
        use_bias = [self._use_bias]*self._num_mlp_layers
        # TODO: use two MLPs for X and Z
        self._msg_mlp_x = MLP(units, activations, use_bias)
        self._msg_mlp_z = MLP(units, activations, use_bias)

        # MLP to update embeddings from accumulated messages
        units[-1] = num_embed_dims
        self._embed_mlp = MLP(units, activations, use_bias)
    
    def reduce_msg(self, messages, gather_ind):
        print(f"Tracing reduce msg in update VN embedding for msg shape={messages.shape}")
        # Reduce messages at each receiving (to) vertex
        # note: bring batch dim to last dim for improved performance with ragged tensors
        messages = tf.transpose(messages, (1,2,0)) # [num_edges, embed_dim, bs]
        m_ragged = tf.gather(messages, gather_ind, axis=0)
        # m_ragged shape [46, None, 20, None]
        if self._reduce_op=="sum":
            m = tf.reduce_sum(m_ragged, axis=1)
        elif self._reduce_op=="mean":
            m = tf.reduce_mean(m_ragged, axis=1)
        elif self._reduce_op=="max":
            m = tf.reduce_max(m_ragged, axis=1)
        elif self._reduce_op=="min":
            m = tf.reduce_min(m_ragged, axis=1)
        else:
            raise ValueError("unknown reduce operation")
        # after reduction, m shape [46, 20, 256]

        return tf.transpose(m, (2,0,1)) # batch-dim back to first dim
    
    @tf.function(autograph=False)
    def call(self, h_from_x, h_from_z, h_to, syndrome_x, syndrome_z):
        print(f"Tracing update VN embedding.")

        # Concatenate embeddings of the transmitting (from) and receiving (to) vertex for each edge
        features_x = tf.concat([tf.gather(h_from_x, self._from_ind_x, axis=1),
                                tf.gather(h_to, self._to_ind_x, axis=1)], axis=-1)
        features_z = tf.concat([tf.gather(h_from_z, self._from_ind_z, axis=1),
                                tf.gather(h_to, self._to_ind_z, axis=1)], axis=-1)

        if self._use_attributes:
            attr_x = tf.tile(tf.expand_dims(self._g_msg_x, axis=0), [tf.shape(features_x)[0], 1, 1])
            attr_z = tf.tile(tf.expand_dims(self._g_msg_z, axis=0), [tf.shape(features_z)[0], 1, 1])
            features_x = tf.concat([features_x, attr_x], axis=-1)
            features_z = tf.concat([features_z, attr_z], axis=-1)


        # Compute messsages for all edges
        messages_x = self._msg_mlp_x(features_x) # syndrome shape [bs, cn]
        syndrome_x_mask = tf.cast(tf.gather(syndrome_x, self._from_ind_x, axis=1), messages_x.dtype)
        messages_x = tf.math.multiply(messages_x, tf.expand_dims(syndrome_x_mask, -1))
        messages_z = self._msg_mlp_z(features_z)
        syndrome_z_mask = tf.cast(tf.gather(syndrome_z, self._from_ind_z, axis=1), messages_z.dtype)
        messages_z = tf.math.multiply(messages_z, tf.expand_dims(syndrome_z_mask, -1))

        # Reduce messages at each receiving (to) vertex
        # note: bring batch dim to last dim for improved performance
        # with ragged tensors
        m_x = self.reduce_msg(messages_x, self._gather_ind_x)
        m_z = self.reduce_msg(messages_z, self._gather_ind_z)

        if self._use_attributes:
            attr = tf.tile(tf.expand_dims(self._g_node, axis=0), [tf.shape(m_x)[0], 1, 1])
            m_z = tf.concat([m_z, attr], axis=-1)
        # Compute new embeddings
        h_to_new = self._embed_mlp(tf.concat([m_x, m_z, h_to], axis=-1))

        return h_to_new

######### Utility functions #########

def save_weights(system, model_path):
    """Save model weights.

    This function saves the weights of a Keras model ``system`` to the
    path as provided by ``model_path``.

    Parameters
    ----------
        system: Keras model
            A model containing the weights to be stored.

        model_path: str
            Defining the path where the weights are stored.

    """
    weights = system.get_weights()
    with open(model_path, 'wb') as f:
        pickle.dump(weights, f)

def load_weights(system, model_path):
    """Load model weights.

    This function loads the weights of a Keras model ``system`` from a file
    provided by ``model_path``.

    Parameters
    ----------
        system: Keras model
            The target model into which the weights are loaded.

        model_path: str
            Defining the path where the weights are stored.

    """
    with open(model_path, 'rb') as f:
        weights = pickle.load(f)
    system.set_weights(weights)

