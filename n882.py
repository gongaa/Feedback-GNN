import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

import argparse
argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--p", help="Physical error rate p to simulate.")
argParser.add_argument("-id", "--gpu_id", help="GPU id")

args = argParser.parse_args()

nG = 5
p = float(args.p)
gpu_num = int(args.gpu_id)

gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

from sionna.fec.ldpc import QLDPCBPDecoder, Feedback_GNN, load_weights 
from sionna.fec.ldpc import Sandwich_BP_GNN_Evaluation_Model 
from sionna.fec.ldpc import *
from sionna.utils.plotting import PlotBER

print(f"Running for {nG} rounds of GNN feedback at p={p} on GPU {gpu_num}.")

GHP_n882_k24 = create_QC_GHP_codes(63, create_cyclic_permuting_matrix(7, [27,54,0]), [0,1,6]) # 18 <= d <= 24
code = GHP_n882_k24

ber_plot = PlotBER()

bs = tf.constant(5000) 
max_iter = tf.constant(100000)
n = tf.constant(code.N)
cn_z = tf.constant(code.hz.shape[0])
cn_x = tf.constant(code.hx.shape[0])

G = Feedback_GNN(code=code, 
                 num_msg_dims=tf.constant(20),
                 num_hidden_units=tf.constant(40),
                 num_mlp_layers=2,
                 reduce_op="mean",     
                 activation="tanh",
                 use_bias=True)
G((tf.zeros((bs, n, 3)), tf.zeros((cn_x, bs)), tf.zeros((cn_z, bs)), 
                  tf.zeros((cn_x, bs)), tf.zeros((cn_z, bs))))
load_weights(G, f"./sionna/fec/ldpc/weights/feedback_GNN_n882_k24_wt_4_60_iter_64_16_mixed.npy") 

num_iter1 = tf.constant(64)
num_iter2 = tf.constant(16)
factor1 = tf.constant(1.0)
factor2 = tf.constant(1.0)

decoder1 = QLDPCBPDecoder(code=code, num_iter=num_iter1, normalization_factor=factor1, cn_type="boxplus-phi", trainable=False, stage_one=True)
decoder2 = QLDPCBPDecoder(code=code, num_iter=num_iter2, normalization_factor=factor2, cn_type="boxplus-phi", trainable=False, stage_one=True)


model_eval = Sandwich_BP_GNN_Evaluation_Model(code, [decoder1]+[decoder2]*nG, [G]*nG, num_layers=(nG+1))
ber_plot.simulate(model_eval,
              ebno_dbs=[p],
              batch_size=bs,
              num_target_block_errors=100,
              legend=f"feedback GNN {factor1.numpy():.2f} {nG} rounds",
              soft_estimates=True,
              max_mc_iter=max_iter, 
              early_stop=True, 
              add_bler=True, 
              show_fig=False, 
              qldpc=True,
              forward_keyboard_interrupt=False)

print(f"at {ber_plot._snrs[1]}, BLER is {ber_plot._bers[1]}")