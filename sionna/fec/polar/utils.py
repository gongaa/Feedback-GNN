#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions and layers for the Polar code package."""

import numpy as np
import numbers
from numpy.core.numerictypes import issubdtype
import matplotlib.pyplot as plt
from scipy.special import comb
from importlib_resources import files, as_file
from . import codes # pylint: disable=relative-beyond-top-level

def generate_5g_ranking(k, n, sort=True):
    """Returns information and frozen bit positions of the 5G Polar code
    as defined in Tab. 5.3.1.2-1 in [3GPPTS38212]_ for given values of ``k``
    and ``n``.

    Input
    -----
        k: int
            The number of information bit per codeword.

        n: int
            The desired codeword length. Must be a power of two.

        sort: bool
            Defaults to True. Indicates if the returned indices are
            sorted.

    Output
    ------
        [frozen_pos, info_pos]:
            List:

        frozen_pos: ndarray
            An array of ints of shape `[n-k]` containing the frozen
            position indices.

        info_pos: ndarray
            An array of ints of shape `[k]` containing the information
            position indices.

    Raises
    ------
        AssertionError
            If ``k`` or ``n`` are not positve ints.

        AssertionError
            If ``sort`` is not bool.

        AssertionError
            If ``k`` or ``n`` are larger than 1024

        AssertionError
            If ``n`` is less than 32.

        AssertionError
            If the resulting coderate is invalid (`>1.0`).

        AssertionError
            If ``n`` is not a power of 2.
    """
    #assert error if r>1 or k,n are negativ
    assert isinstance(k, int), "k must be integer."
    assert isinstance(n, int), "n must be integer."
    assert isinstance(sort, bool), "sort must be bool."
    assert k>-1, "k cannot be negative."
    assert k<1025, "k cannot be larger than 1024."
    assert n<1025, "n cannot be larger than 1024."
    assert n>31, "n must be >=32."
    assert n>=k, "Invalid coderate (>1)."
    assert np.log2(n)==int(np.log2(n)), "n must be a power of 2."

    # load the channel ranking from csv format in folder "codes"
    source = files(codes).joinpath("polar_5G.csv")
    with as_file(source) as codes.csv:
        ch_order = np.genfromtxt(codes.csv, delimiter=";")
    ch_order = ch_order.astype(int)

    # find n smallest values of channel order (2nd row)
    ind = np.argsort(ch_order[:,1])
    ch_order_sort = ch_order[ind,:]
    # only consider the first n channels
    ch_order_sort_n = ch_order_sort[0:n,:]
    # and sort again according to reliability
    ind_n = np.argsort(ch_order_sort_n[:,0])
    ch_order_n = ch_order_sort_n[ind_n,:]

    # and calculate frozen/information positions for given n, k
    # assume that pre_frozen_pos are already frozen (rate-matching)
    frozen_pos = np.zeros(n-k)
    info_pos = np.zeros(k)
    #the n-k smallest positions of ch_order denote frozen pos.
    for i in range(n-k):
        frozen_pos[i] = ch_order_n[i,1] # 2. row yields index to freeze
    for i in range(n-k, n):
        info_pos[i-(n-k)] = ch_order_n[i,1] # 2. row yields index to freeze

    # sort to have channels in ascending order
    if sort:
        info_pos = np.sort(info_pos)
        frozen_pos = np.sort(frozen_pos)

    return [frozen_pos.astype(int), info_pos.astype(int)]

def generate_polar_transform_mat(n_lift):
    """Generate the polar transformation matrix (Kronecker product).

    Input
    -----
        n_lift: int
            Defining the Kronecker power, i.e., how often is the kernel lifted.

    Output
    ------
       : ndarray
            Array of `0s` and `1s` of shape `[2^n_lift , 2^n_lift]` containing
            the Polar transformation matrix.
    """

    assert int(n_lift)==n_lift, "n_lift must be integer"
    assert n_lift>=0, "n_lift must be positive"

    assert n_lift<12, "Warning: the resulting code length is large (=2^n_lift)."

    gm = np.array([[1, 0],[ 1, 1]])

    gm_l = np.copy(gm)
    for _ in range(n_lift-1):
        gm_l_new = np.zeros([2*np.shape(gm_l)[0],2*np.shape(gm_l)[1]])
        for j in range(np.shape(gm_l)[0]):
            for k in range(np.shape(gm_l)[1]):
                gm_l_new[2*j:2*j+2, 2*k:2*k+2] = gm_l[j,k]*gm
        gm_l = gm_l_new
    return gm_l

def generate_rm_code(r, m):
    """Generate frozen positions of the (r, m) Reed Muller (RM) code.

    Input
    -----
        r: int
            The order of the RM code.

        m: int
            `log2` of the desired codeword length.

    Output
    ------
        [frozen_pos, info_pos, n, k, d_min]:
            List:

        frozen_pos: ndarray
            An array of ints of shape `[n-k]` containing the frozen
            position indices.

        info_pos: ndarray
            An array of ints of shape `[k]` containing the information
            position indices.

        n: int
            Resulting codeword length

        k: int
            Number of information bits

        d_min: int
            Minimum distance of the code.

    Raises
    ------
        AssertionError
            If ``r`` is larger than ``m``.

        AssertionError
            If ``r`` or ``m`` are not positive ints.

    """
    assert isinstance(r, int), "r must be int."
    assert isinstance(m, int), "m must be int."
    assert r<=m, "order r cannot be larger than m."
    assert r>=0, "r must be positive."
    assert m>=0, "m must be positive."

    n = 2**m
    d_min = 2**(m-r)

    # calc k to verify results
    k = 0
    for i in range(r+1):
        k += int(comb(m,i))

    # select positions to freeze
    # freeze all rows that have weight < m-r
    w = np.zeros(n)
    for i in range(n):
        x_bin = np.binary_repr(i)
        for x_i in x_bin:
            w[i] += int(x_i)
    frozen_vec = w < m-r
    info_vec = np.invert(frozen_vec)
    k_res = np.sum(info_vec)
    frozen_pos = np.arange(n)[frozen_vec]
    info_pos = np.arange(n)[info_vec]

    # verify results
    assert k_res==k, "Error: resulting k is inconsistent."

    return frozen_pos, info_pos, n, k, d_min


def generate_dense_polar(frozen_pos, n, verbose=True):
    """Generate *naive* (dense) Polar parity-check and generator matrix.

    This function follows Lemma 1 in [Goala_LP]_ and returns a parity-check
    matrix for Polar codes.

    Note
    ----
    The resulting matrix can be used for decoding with the
    :class:`~sionna.fec.ldpc.LDPCBPDecoder` class. However, the resulting
    parity-check matrix is (usually) not sparse and, thus, not suitable for
    belief propagation decoding as the graph has many short cycles.
    Please consider :class:`~sionna.fec.polar.PolarBPDecoder` for iterative
    decoding over the encoding graph.

    Input
    -----
    frozen_pos: ndarray
        Array of `int` defining the ``n-k`` indices of the frozen positions.

    n: int
        The codeword length.

    verbose: bool
        Defaults to True. If True, the code properties are printed.

    Output
    ------
    pcm: ndarray of `zeros` and `ones` of shape [n-k, n]
        The parity-check matrix.

    gm: ndarray of `zeros` and `ones` of shape [k, n]
        The generator matrix.

    """

    assert isinstance(n, numbers.Number), "n must be a number."
    n = int(n) # n can be float (e.g. as result of n=k*r)
    assert issubdtype(frozen_pos.dtype, int), "frozen_pos must \
                                                consist of ints."
    assert len(frozen_pos)<=n, "Number of elements in frozen_pos cannot \
                                be greater than n."

    assert np.log2(n)==int(np.log2(n)), "n must be a power of 2."

    k = n - len(frozen_pos)

    # generate info positions
    info_pos = np.setdiff1d(np.arange(n), frozen_pos)
    assert k==len(info_pos), "Internal error: invalid " \
                                            "info_pos generated."

    gm_mat = generate_polar_transform_mat(int(np.log2(n)))

    gm_true = gm_mat[info_pos,:]
    pcm = np.transpose(gm_mat[:,frozen_pos])

    if verbose:
        print("Shape of the generator matrix: ", gm_true.shape)
        print("Shape of the parity-check matrix: ", pcm.shape)
        plt.spy(pcm)

    # Verify result, i.e., check that H*G has an all-zero syndrome.
    # Note: we have no proof that Lemma 1 holds for all possible
    # frozen_positions. Thus, it seems to be better to verify the generated
    # results individually.
    s = np.mod(np.matmul(pcm, np.transpose(gm_true)),2)
    assert np.sum(s)==0, "Non-zero syndrom for H*G'."

    return pcm, gm_true

def generate_quantum_polar_code_pw(kx, kz, n, beta=2**0.25):
    """Generate frozen positions in X and Z basis, 
       and the kx+kz-n quantum infomation (logical) positions of the (n, kx, kz) quantum polar code.
       Based on the Polarization Weight (PW) method.

    Input
    -----
        kx: int
            The X-basis quantum polar code number of information bits.

        kz: int
            The Z-basis quantum polar code number of information bits.

        n: int
            The codeword length, must be a power of two (for now).

        beta: float
            The beta used in the polarization weight beta-expansion calculation.


    Output
    ------
        [frozen_pos, info_pos, n, k, d_min]:
            List:

        frozen_pos_X: ndarray
            An array of ints of shape `[n-kx]` containing the frozen
            position indices.

        info_pos_X: ndarray
            An array of ints of shape `[kx]` containing the information
            position indices.

        frozen_pos_Z: ndarray
            An array of ints of shape `[n-kz]` containing the frozen
            position indices.

        info_pos_Z: ndarray
            An array of ints of shape `[kz]` containing the information
            position indices.

        q_info_pos: ndarray
            An array of ints of shape ``[kx+kz-n]`` containing the quantum
            information position indices of the info vector u after profiling
            (just before the polar encoding matrix)

        u_info_pos: ndarray
            An array of ints of shape ``[kx+kz-n]`` containing the quantum 
            information position indices in the info vector u. 

        d_min: int
            Minimum distance of the code.

    Raises
    ------
        AssertionError
            If ``kx`` or ``kz`` is larger than ``n``.

        AssertionError
            If ``kx``, ``kz`` or ``n`` are not positive ints.

        AssertionError
            If ``n`` is not a power of 2.

        AssertionError
            If sum of ``kx`` and ``kz`` is not larger than ``n``.

    """
    assert isinstance(kx, int), "kx must be int."
    assert isinstance(kz, int), "kz must be int."
    assert isinstance(n, int),  "n must be int."
    assert kx<=n, "kx cannot be larger than n."
    assert kz<=n, "kz cannot be larger than n."
    assert kx>=0, "kx must be positive."
    assert kz>=0, "kz must be positive."
    assert n>=0,  "n must be positive."
    assert np.log2(n)==int(np.log2(n)), "n must be a power of 2."

    m = int(np.log2(n))
    k = kx + kz - n
    assert k>0, "kz+kz must be larger than n"
    # select positions to freeze
    # freeze all rows that have the lowest polarization weight 
    power = [beta**i for i in range(m)]
    pw = np.zeros(n)
    for i in range(n):
        x_bin = np.binary_repr(i)
        j = 0
        for c in reversed(x_bin):
            if c == '1':
                pw[i] += power[j]
            j += 1

    ch_order_sort = np.argsort(pw) # ch_order_sort is the same as np.arange(n)[ch_order_sort]
    q_info_pos = np.sort(ch_order_sort[n-kz:kx])
    u_info_pos = np.zeros_like(q_info_pos)
    d_min = 2 ** min([bin(int(c)).count("1") for c in q_info_pos]) # TODO: add explanation

    frozen_pos_X = np.sort(ch_order_sort[:n-kx])
    frozen_pos_Z = np.sort(ch_order_sort[:n-kz])

    info_pos_X = np.sort(ch_order_sort[n-kx:n])
    info_pos_Z = np.sort(ch_order_sort[n-kz:n])

    info_mask_Z = np.zeros(n)
    info_mask_Z[info_pos_Z] = 1
    for idx, i in enumerate(q_info_pos):
        u_info_pos[idx] = np.sum(info_mask_Z[:i])

    return frozen_pos_X, info_pos_X, frozen_pos_Z, info_pos_Z, q_info_pos, u_info_pos, d_min