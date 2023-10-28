<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Feedback GNN

This repo contains the source code of the paper [Graph Neural Networks for Enhanced Decoding of Quantum LDPC Codes](https://arxiv.org/pdf/2304.04743.pdf).

First, I extend the classical error correction simulation framework [Sionna](https://nvlabs.github.io/sionna/) to do quantum error correction. The entire framework is built on top of the open-source software library [TensorFlow](https://www.tensorflow.org) for machine learning.

Second, I trained an intermediate GNN layer (called feedback GNN) between consecutive quaternary BP runs. This feedback GNN leverages the knowledge from the previous BP run, in order to find a suitable initialization for the next BP run.

Since this extension has not yet been merged into Sionna, you do not need to install Sionna. If you already have Sionna installed, it is recommended to create a new environment where Sionna is not installed and run the notebooks there. 

[Python](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/) are required.
In order to run the tutorial notebooks on your machine, you also need [Jupyter](https://jupyter.org/).
It is recommended to run the demonstration notebooks on GPUs.

We refer to the [TensorFlow GPU support tutorial](https://www.tensorflow.org/install/gpu) for GPU support and the required driver setup.

Once everything is set up, you can run the [examples/QLDPC.ipynb](examples/QLDPC.ipynb) for the demonstration of code construction and binary/quaternary BP decoder.

For the demonstration of the feedback GNN evaluation on the $[[1270,28]]$ code and the $[[882,24]]$ code, please refer to the [examples/n1270.ipynb](examples/n1270.ipynb) and [examples/n882.ipynb](examples/n882.ipynb) notebooks. The two scripts [n1270.py](n1270.py) and [n882.py](n882.py) are for evaluations running on GPUs for days (at small physical error rates).

For training, either you can download the [datasets](https://drive.google.com/drive/folders/1BnjUUDRleT4B3IZQEk-fEYu2wBPQ35Hf?usp=sharing) and put the files under `sionna/fec/ldpc/datasets` and directly follow [examples/Feedback_GNN.ipynb](examples/Feedback_GNN.ipynb) to train on them. Or you can follow [examples/Generate_dataset.ipynb](examples/Generate_dataset.ipynb) to generate your own dataset.


## Directory Layout
    .
    ├── examples                        # contains all the jupyter notebooks
    └── sionna                   
        ├── channel
        │   ├── pauli.py                # independent Pauli noise
        │   └── discrete_channel.py     # Sionna's differentiable BSC
        └── fec                         # Forward Error Correction
            └── ldpc                    
                ├── codes_q.py          # CSS code construction
                ├── decoding.py         # Sionna's binary BP, add syndrome decoding
                ├── decoding_q.py       # quaternary BP
                ├── feedback_gnn.py     # feedback GNN models for training and evaluation
                ├── weights             # trained feedback GNN model weights
                ├── datasets            # empty
                └── gnn.py              # full GNN decoder for QLDPC codes, results not shown in the paper



