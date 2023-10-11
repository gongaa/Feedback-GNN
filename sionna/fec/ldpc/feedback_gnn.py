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
from sionna.channel import Pauli, BinarySymmetricChannel
from sionna.fec.utils import int_mod_2
from .gnn import MLP


class Feedback_GNN(Layer):
    def __init__(self,
                 code,
                 num_msg_dims,
                 num_hidden_units,
                 num_mlp_layers,
                 reduce_op="mean",
                 activation="tanh",
                 use_bias=False):

        super().__init__()

        self._pcm_x, self._pcm_lx = code.hx, code.lx # Parity check matrix
        self._pcm_z, self._pcm_lz = code.hz, code.lz
        self._num_cn_x, self._num_vn = self._pcm_x.shape # number of X-type check nodes
        self._num_cn_z = self._pcm_z.shape[0]
        self._num_edges_x = int(np.sum(self._pcm_x))
        self._num_edges_z = int(np.sum(self._pcm_z))

        self._edges_x, self._cn_gather_ind_x, self._vn_gather_ind_x, self._cn_gather_mask_x = self.create_edges(self._pcm_x)
        self._edges_z, self._cn_gather_ind_z, self._vn_gather_ind_z, self._cn_gather_mask_z = self.create_edges(self._pcm_z)

        self._cn_from_ind_x = np.flip(self._edges_x, 1)[:,0]
        self._cn_to_ind_x   = np.flip(self._edges_x, 1)[:,1]
        self._cn_from_ind_z = np.flip(self._edges_z, 1)[:,0]
        self._cn_to_ind_z   = np.flip(self._edges_z, 1)[:,1]
        self._vn_from_ind_x = self._edges_x[:,0]
        self._vn_to_ind_x   = self._edges_x[:,1]
        self._vn_from_ind_z = self._edges_z[:,0]
        self._vn_to_ind_z   = self._edges_z[:,1]

        _, _, _, self._cn_gather_mask_lx = self.create_edges(self._pcm_lx)
        _, _, _, self._cn_gather_mask_lz = self.create_edges(self._pcm_lz)

        # lx = hz_perp \ hx
        self._cn_gather_mask_z_perp = tf.concat([self._cn_gather_mask_x, self._cn_gather_mask_lx], axis=0)
        self._cn_gather_mask_x_perp = tf.concat([self._cn_gather_mask_z, self._cn_gather_mask_lz], axis=0)
        
        # Number of dimensions for messages
        self._num_msg_dims = num_msg_dims

        # Number of hidden units for MLPs computing messages and embeddings
        self._num_hidden_units = num_hidden_units

        # Number of layers for MLPs computing messages and embeddings
        self._num_mlp_layers = num_mlp_layers

        # Reduce operation for message aggregation
        self._reduce_op = reduce_op

        # Activation function of the hidden MLP layers
        self._activation = activation

        # Activate bias of MLP layers
        self._use_bias = use_bias

        # Internal state for initialization
        self._is_built = False

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

            # NN to transform VN embedding to llr_x, llr_y, llr_z
            self._llr_inv_embed = Dense(3, use_bias=self._use_bias, bias_initializer='ones',
                                        kernel_initializer='zeros')


            units = [self._num_hidden_units]*(self._num_mlp_layers-1) + [self._num_msg_dims]
            activations = [self._activation]*(self._num_mlp_layers-1) + [None]
            use_bias = [self._use_bias]*self._num_mlp_layers
            self.vn_msg_mlp_x = MLP(units, activations, use_bias)
            self.vn_msg_mlp_z = MLP(units, activations, use_bias)

            units = [self._num_hidden_units]*(self._num_mlp_layers-1)
            activations = [self._activation]*(self._num_mlp_layers-1)
            use_bias = [self._use_bias]*(self._num_mlp_layers-1)
            self.vn_embed_mlp   = MLP(units, activations, use_bias)

    @tf.function(autograph=False)
    def reduce_msg(self, messages, gather_ind):
        # print(f"Tracing reduce msg in update VN embedding for msg shape={messages.shape}")
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
    def embed_to_llr(self, h_vn):
        """Transform VN embeddings to LLR for X,Y,Z flips."""
        # print(f"tracing embed to llr for h_vn shape={h_vn.shape}")
        embed = self._llr_inv_embed(h_vn)   # shape [bs, n, 3]
        embed = tf.transpose(embed, (2,1,0)) # shape [3, n, bs]
        llrx, llry, llrz = embed[0], embed[1], embed[2]
        return llrx, llry, llrz

    def call(self, inputs):
        """Run the decoder."""
        h_vn, logit_hx, logit_hz, syndrome_x, syndrome_z = inputs
        # batch_size = tf.shape(syndrome_x)[0]
        # h_vn shape [bs, n, 3]

        # map 0->+1, 1->-1
        syndrome_x = tf.cast(1 - 2 * syndrome_x, tf.float32)
        syndrome_z = tf.cast(1 - 2 * syndrome_z, tf.float32)

        h_cn_x = tf.expand_dims(tf.transpose(tf.math.multiply(logit_hx, syndrome_x), (1,0)), -1) # [bs, cn, 1]
        h_cn_z = tf.expand_dims(tf.transpose(tf.math.multiply(logit_hz, syndrome_z), (1,0)), -1)

        # VN update
        edge_c2v_features_x = tf.concat([tf.gather(h_cn_x, self._vn_from_ind_x, axis=1),
                                tf.gather(h_vn, self._vn_to_ind_x, axis=1)], axis=-1)
        edge_c2v_features_z = tf.concat([tf.gather(h_cn_z, self._vn_from_ind_z, axis=1),
                                tf.gather(h_vn, self._vn_to_ind_z, axis=1)], axis=-1)

        messages_c2v_x = self.vn_msg_mlp_x(edge_c2v_features_x)
        messages_c2v_z = self.vn_msg_mlp_z(edge_c2v_features_z)
        
        m_x = self.reduce_msg(messages_c2v_x, self._vn_gather_ind_x)
        m_z = self.reduce_msg(messages_c2v_z, self._vn_gather_ind_z)
        
        h_vn = self._llr_inv_embed(self.vn_embed_mlp(tf.concat([m_x, m_z, h_vn], axis=-1)))

        return h_vn
    
class BP_BSC_Model(tf.keras.Model):
    def __init__(self, pcm, decoder, logical_pcm=None, p0=None):

        super().__init__()
        
        # store values internally
        self.pcm = pcm
        self.logical_pcm = logical_pcm
        _, self.n = pcm.shape
        self.source = BinarySource()
       
        self.channel = BinarySymmetricChannel()

        # FEC encoder / decoder
        self.decoder = decoder
        self.p0 = p0

    @tf.function(jit_compile=True, reduce_retracing=True) # XLA mode during evaluation
    def call(self, batch_size, ebno_db):

        p0 = ebno_db if self.p0 is None else self.p0
        llr_const = -tf.math.log((1.-p0)/p0)
           
        c = tf.zeros([batch_size, self.n])
        noise = self.channel((c, ebno_db))  # [bs, self.n]
        llr = tf.fill(tf.shape(noise), llr_const)  
        noise_int = tf.transpose(tf.cast(noise, self.pcm.dtype), (1,0)) # [self.n, bs]
        syndrome = int_mod_2(tf.matmul(self.pcm, noise_int))
        noise_hat = self.decoder((llr, syndrome)) 
        if self.logical_pcm is None:
            return noise, tf.cast(noise_hat, noise.dtype) 
        else:
            noise_hat = tf.transpose(tf.cast(noise_hat, self.pcm.dtype), (1,0)) # [self.n, bs]
            noise_diff = tf.math.logical_xor(tf.cast(noise_int, tf.bool), tf.cast(noise_hat, tf.bool)) 
            noise_diff = tf.cast(noise_diff, self.logical_pcm.dtype)
            s_hat = int_mod_2(tf.matmul(self.pcm, noise_diff))
            ls_hat = int_mod_2(tf.matmul(self.logical_pcm, noise_diff))
            s_hat = tf.transpose(s_hat, (1,0)) # bs should be the first dimension!
            ls_hat = tf.transpose(ls_hat, (1,0)) 
            return s_hat, ls_hat


class Sandwich_BP_GNN_Evaluation_Model(tf.keras.Model):
    """System model for channel coding BER simulations.
    
    This model allows to simulate BLERs over an iid quantum Pauli channel.
    
    Parameters
    ----------
        code: css_code
            A css code with PCM hx and hz.
            
        decoder: Keras layer
            A Keras layer that decodes llr tensors.


    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.
        
        p: float or tf.float
            A float defining the flip probability, for depolarizing noise model.
            
    Output
    ------
        (u, u_hat):
            Tuple:
        
        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.           

        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.           
    """
    def __init__(self, code, decoders, feedbacks, num_layers=4, wt=False, p0=0.05):

        super().__init__()
        
        # store values internally
        self.k = code.K      
        self.n = code.N
        self.hx = code.hx # [cn, n]
        self.hz = code.hz
        self.lx = code.lx
        self.lz = code.lz

        self.hx_perp = code.hx_perp
        self.hz_perp = code.hz_perp
        self.code_name = code.name
        self.num_checks = code.hx.shape[0] + code.hz.shape[0]
  
        self.source = BinarySource()
        self.channel = Pauli(wt=wt)
        
        self.decoders  = decoders
        self.feedbacks = feedbacks
        self.num_layers = num_layers

        self.wt = wt
        self.p0 = p0
        
      
    @tf.function(jit_compile=True, reduce_retracing=True) # XLA mode during evaluation
    def call(self, batch_size, ebno_db):

        p = ebno_db
        # depolarizing noise
        px, py, pz = 2*p/3, p/3, 2*p/3
        c_dummy = tf.zeros([batch_size, self.n])
        if self.wt: # train using errors or weight p
            noise_x, noise_z = self.channel([c_dummy, None, p])
        else:
            noise_x, noise_z = self.channel([c_dummy, None, px, py, pz])  # [bs, self.n]
            
        noise_x_T, noise_z_T = tf.transpose(noise_x, (1,0)), tf.transpose(noise_z, (1,0))
        noise_x_int = tf.cast(noise_x_T, self.hz.dtype)
        noise_z_int = tf.cast(noise_z_T, self.hx.dtype)
        syndrome_x = int_mod_2(tf.matmul(self.hx, noise_z_int))
        syndrome_z = int_mod_2(tf.matmul(self.hz, noise_x_int))

        p0 = p if self.p0 is None else self.p0
        llr_ch_x = tf.fill(tf.shape(noise_x), tf.math.log(3.*(1.-p0)/p0))
        llr = tf.tile(tf.expand_dims(llr_ch_x, axis=1), multiples=tf.constant([1, 3, 1], tf.int32))
        # shape of llr: [bs, 3, self.n]
        gt_x = int_mod_2(tf.matmul(self.hz, noise_x_int)) # [cn, bs]
        gt_z = int_mod_2(tf.matmul(self.hx, noise_z_int))
        
        gt = tf.concat([gt_x, gt_z], axis=0) # [cn_x+cn_z, bs]
        gt = tf.transpose(gt, (1,0))         # [bs, cn_x+cn_z]        
       
        llrx, llry, llrz, x_hat, z_hat, logit_hx_perp, logit_hz_perp = self.decoders[0]((llr, syndrome_x, syndrome_z))
        errors = tf.ones([batch_size], dtype=tf.bool)
        for i in range(1, self.num_layers):
            sx = int_mod_2(tf.matmul(self.hz, tf.transpose(tf.cast(x_hat, self.hz.dtype), (1,0)))) # [cn, bs]
            sz = int_mod_2(tf.matmul(self.hx, tf.transpose(tf.cast(z_hat, self.hx.dtype), (1,0))))        
            # find where flagged errors happened during first stage
            s_hat = tf.transpose(tf.concat([sx, sz], axis=0), (1,0)) # [bs, cn_x+cn_z]
            new_errors = tf.reduce_any(tf.not_equal(gt, s_hat), axis=-1)

            errors = tf.math.logical_and(errors, new_errors)
#             tf.print("num of errors after i=", i, "is", tf.math.reduce_sum(tf.cast(errors, tf.int32)))

            h_vn = tf.stack([llrx, llry, llrz], axis=-1) # [bs, n, 3] GNN variable node values  
            # first feedback
            new_llr = self.feedbacks[i-1]((h_vn, logit_hz_perp, logit_hx_perp, syndrome_x, syndrome_z))
            llrx, llry, llrz, x_hat_update, z_hat_update, logit_hx_perp, logit_hz_perp = self.decoders[i]((tf.transpose(new_llr, (0,2,1)), syndrome_x, syndrome_z))

            # update the second-stage results for where flagged errors happened
            x_hat = tf.tensor_scatter_nd_update(x_hat, tf.where(errors), x_hat_update[errors])
            z_hat = tf.tensor_scatter_nd_update(z_hat, tf.where(errors), z_hat_update[errors])   

        # final decision
        x_hat = tf.transpose(tf.cast(x_hat, tf.bool), (1,0)) # [self.n, bs]
        z_hat = tf.transpose(tf.cast(z_hat, tf.bool), (1,0))

        x_diff = tf.cast(tf.math.logical_xor(noise_x_T, x_hat), self.hx_perp.dtype)
        z_diff = tf.cast(tf.math.logical_xor(noise_z_T, z_hat), self.hz_perp.dtype)

        sx = int_mod_2(tf.matmul(self.hz, x_diff))
        sz = int_mod_2(tf.matmul(self.hx, z_diff))
        
        lsx = int_mod_2(tf.matmul(self.hx_perp, x_diff))
        lsz = int_mod_2(tf.matmul(self.hz_perp, z_diff))
            
        s_hat = tf.concat([sx, sz], axis=0)        # for flagged error counts
        ls_hat = tf.concat([lsx, lsz], axis=0)      # for total logical error counts

        s_hat = tf.transpose(s_hat, (1,0))
        ls_hat = tf.transpose(ls_hat, (1,0))         # bs should be the first dimension!!!

        return s_hat, ls_hat       

        
class First_Stage_BP_Model(tf.keras.Model):
# first block of BP runs during training, nothing trainable here, jit enabled for speed
    def __init__(self, code, decoder, p0=0.05):

        super().__init__()

        self.hx = code.hx
        self.hz = code.hz
        
        self.decoder  = decoder
        self.p0 = p0
      
    @tf.function(jit_compile=True, reduce_retracing=True)  
    def call(self, noise_x, noise_z):

        noise_x_T, noise_z_T = tf.transpose(noise_x, (1,0)), tf.transpose(noise_z, (1,0))
        noise_x_int = tf.cast(noise_x_T, self.hz.dtype)
        noise_z_int = tf.cast(noise_z_T, self.hx.dtype)
        syndrome_x = int_mod_2(tf.matmul(self.hx, noise_z_int))
        syndrome_z = int_mod_2(tf.matmul(self.hz, noise_x_int))

        p0 = self.p0 # do not know the real p
        llr_ch_x = tf.fill(tf.shape(noise_x), tf.math.log(3.*(1.-p0)/p0))
        llr = tf.tile(tf.expand_dims(llr_ch_x, axis=1), multiples=tf.constant([1, 3, 1], tf.int32))
        # shape of llr: [bs, 3, self.n]

        llrx, llry, llrz, x_hat, z_hat, logit_hx_perp, logit_hz_perp = self.decoder((llr, syndrome_x, syndrome_z))
        h_vn = tf.stack([llrx, llry, llrz], axis=-1) # [bs, n, 3]
        return h_vn, logit_hx_perp, logit_hz_perp
      
    
class Second_Stage_GNN_BP_Model(tf.keras.Model):

    def __init__(self, code, feedback, decoder, num_iter=16, trainable=True, loss_from=8):

        super().__init__()
        
        # store values internally
        self.k = code.K      
        self.n = code.N
        self.hx = code.hx
        self.hz = code.hz
        self.lx = code.lx
        self.lz = code.lz
        self.hx_perp = code.hx_perp
        self.hz_perp = code.hz_perp
        self.code_name = code.name
        self.num_checks = code.hx.shape[0] + code.hz.shape[0]
        
        self.feedback = feedback # the only thing to train
        self.decoder  = decoder
        self.num_iter = num_iter
        self.loss_from = loss_from

        self.trainable = trainable
        self.bce = BinaryCrossentropy(from_logits=True)
        
      
    @tf.function()  # graph mode during training, because computing gradient for TensorArray in XLA mode is not supported
    def call(self, noise_x, noise_z, h_vn, logit_hx_perp, logit_hz_perp):

        noise_x_T, noise_z_T = tf.transpose(noise_x, (1,0)), tf.transpose(noise_z, (1,0))
        noise_x_int = tf.cast(noise_x_T, self.hz.dtype)
        noise_z_int = tf.cast(noise_z_T, self.hx.dtype)
        syndrome_x = int_mod_2(tf.matmul(self.hx, noise_z_int))
        syndrome_z = int_mod_2(tf.matmul(self.hz, noise_x_int))

        gt_x = tf.transpose(1 - syndrome_z, (1,0)) # Don't forget to flip the label for bce!
        gt_z = tf.transpose(1 - syndrome_x, (1,0))
        
        loss = 0.0 

        new_llr = self.feedback((h_vn, logit_hz_perp, logit_hx_perp, syndrome_x, syndrome_z))
        llr_hat, x_hat, z_hat = self.decoder((tf.transpose(new_llr, (0,2,1)), syndrome_x, syndrome_z))
        
        for i in range(self.loss_from, self.num_iter): # skip the first two + first eight
            x_logit, z_logit = llr_hat[2*i+2], llr_hat[2*i+3]
            loss += self.bce(gt_x, tf.transpose(x_logit, (1,0)))
            loss += self.bce(gt_z, tf.transpose(z_logit, (1,0)))   
        
        x_hat = tf.transpose(tf.cast(x_hat, tf.bool), (1,0)) # [self.n, bs]
        z_hat = tf.transpose(tf.cast(z_hat, tf.bool), (1,0))

        x_diff = tf.cast(tf.math.logical_xor(noise_x_T, x_hat), self.hx_perp.dtype)
        z_diff = tf.cast(tf.math.logical_xor(noise_z_T, z_hat), self.hz_perp.dtype)

        sx = int_mod_2(tf.matmul(self.hz, x_diff))
        sz = int_mod_2(tf.matmul(self.hx, z_diff))
        
        lsx = int_mod_2(tf.matmul(self.hx_perp, x_diff))
        lsz = int_mod_2(tf.matmul(self.hz_perp, z_diff))
            
        s_hat = tf.concat([sx, sz], axis=0)
        ls_hat = tf.concat([lsx, lsz], axis=0)

        s_hat = tf.transpose(s_hat, (1,0))   # all-zero if no flag errors
        ls_hat = tf.transpose(ls_hat, (1,0)) # all-zero if no logical errors
        # bs should be the first dimension!!!

        return s_hat, ls_hat, loss        
