from sionna.utils import BinarySource 
from sionna.channel import Pauli, BinarySymmetricChannel
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.fec.utils import int_mod_2
    

class OSD0_Decoder(Layer):
# For decoding error strings that the BP decoder failed to decode
    def __init__(self, n):
        super().__init__()
        self.n = n
        
    @tf.function(jit_compile=True, reduce_retracing=True)
    def find_mrb(self, pcm, bs):
        # the parity check matrix must be full-rank
        # find most reliable basis for the parity check matrix of size [bs, rank, n]
        # column-wise gaussian elimination
        # find pivot for each row

        _, rank, _ = pcm.shape
        s = pcm.shape

        idx_pivot = tf.TensorArray(tf.int32, rank, dynamic_size=False)
        
        for row in tf.range(rank):
            pcm = tf.ensure_shape(pcm, s)
            
            # find pivot and store it in TensorArray
            idx_p = tf.argmax(pcm[:, row, :], axis=-1, output_type=tf.int32) # size [bs]
            idx_pivot = idx_pivot.write(row, idx_p)
            
            # gather the idx_pivot'th column from pcm
            c = tf.gather(pcm, idx_p, batch_dims=1, axis=-1) # [bs, rank]
            
            # do not eliminate current row, i.e., c[:,row]=0
            all_zero = tf.zeros((bs, 1), dtype=tf.int32)
            c = tf.concat([c[:,:row], all_zero, c[:,row+1:]], axis=1) # or use tensor_scatter_nd_update?
            c = tf.tile(tf.expand_dims(c, axis=-1), (1, 1, self.n+1)) # [bs, rank, n+1]
            
            # use current row to eliminate all other rows
            current_row = tf.expand_dims(pcm[:,row,:], axis=1) # [bs, 1 ,n]                        
            pcm = int_mod_2(pcm + c * current_row)
            
        idx_pivot = tf.transpose(idx_pivot.stack()) # [bs, rank]
        sol = tf.cast(pcm[:,:,-1], tf.bool) # last column, [bs, rank]
        return idx_pivot, sol

    @tf.function(jit_compile=True, reduce_retracing=True)    
    def call(self, llr, pcm, s, bs):
        # use llrz together with code.hx and syndrome_x, return noise_z_hat

        sort_order = tf.argsort(llr) # [bs, n]
        pcm = tf.cast(pcm, tf.int32)

        permuted_pcm = tf.gather(pcm, sort_order, batch_dims=1, axis=-1)

        inv_sort = tf.argsort(sort_order) # is it correct ???
        
        syndrome = tf.cast(tf.expand_dims(tf.transpose(s, (1,0)), -1), permuted_pcm.dtype) # [bs, rank, 1]
        pcm_syndrome = tf.concat((permuted_pcm, syndrome), axis=-1) # [bs, rank, n+1]

        idx_pivot, sol = self.find_mrb(pcm_syndrome, bs)

        # [bs, rank, rank] * [bs, rank, 1] = [bs, rank, 1]
        _, rank = sol.shape
        ii, _ = tf.meshgrid(tf.range(bs), tf.range(rank), indexing='ij')
        # for sample i in the batch, update using idx_pivot[i]
        
        ii = tf.cast(ii, tf.int32)
        idx_pivot = tf.cast(idx_pivot, tf.int32)
        idx_updates = tf.stack([ii, idx_pivot], axis=-1) # [bs, rank, 2]

        e_hat = tf.tensor_scatter_nd_update(tf.zeros_like(llr, dtype=tf.bool), idx_updates, sol)
        e_hat = tf.gather(e_hat, inv_sort, batch_dims=1, axis=-1)

        return e_hat        

    
class BP4_OSD_Model(tf.keras.Model):
# For storing error strings that the BP decoder failed to decode
    def __init__(self, code, bp4_decoder, osd_decoder):

        super().__init__()
        
        self.k = code.K      
        self.n = code.N
        self.hx = code.hx
        self.hz = code.hz
        self.rank_hx, self.pivot_hx, self.hx_basis = code.rank_hx, code.pivot_hx, code.hx_basis
        self.rank_hz, self.pivot_hz, self.hz_basis = code.rank_hz, code.pivot_hz, code.hz_basis
        self.lx = code.lx
        self.lz = code.lz
        self.hx_perp = code.hx_perp # contains Im(hz.T)
        self.hz_perp = code.hz_perp # contains Im(hx.T)
        self.code_name = code.name
        self.num_checks = code.hx.shape[0] + code.hz.shape[0]
        
        self.source = BinarySource()
        self.channel = Pauli(wt=False)
        self.bp4_decoder = bp4_decoder
        self.osd_decoder = osd_decoder
        
    @tf.function(jit_compile=True, reduce_retracing=True) # XLA mode
    def generate_noise_and_bp4(self, batch_size, p):
        # depolarizing noise
        px, py, pz = 2*p/3, p/3, 2*p/3
        c_dummy = tf.zeros([batch_size, self.n])
        noise_x, noise_z = self.channel([c_dummy, None, px, py, pz])  # [bs, self.n]
        noise_x_T, noise_z_T = tf.transpose(noise_x, (1,0)), tf.transpose(noise_z, (1,0))
        noise_x_int = tf.cast(noise_x_T, self.hz.dtype)
        noise_z_int = tf.cast(noise_z_T, self.hx.dtype)
        syndrome_x = int_mod_2(tf.matmul(self.hx, noise_z_int))
        syndrome_z = int_mod_2(tf.matmul(self.hz, noise_x_int))

        llr_ch_x = tf.fill(tf.shape(noise_x), tf.math.log(3.*(1.-p)/p))
        llr = tf.tile(tf.expand_dims(llr_ch_x, axis=1), multiples=tf.constant([1, 3, 1], tf.int32))
        # shape of llr: [bs, 3, self.n]
        
        llrx, llry, llrz, x_hat, z_hat, _, _ = self.bp4_decoder((llr, syndrome_x, syndrome_z))
       
        x_hat = tf.transpose(tf.cast(x_hat, tf.bool), (1,0)) # [self.n, bs]
        z_hat = tf.transpose(tf.cast(z_hat, tf.bool), (1,0))

        x_diff = tf.cast(tf.math.logical_xor(noise_x_T, x_hat), self.hx_perp.dtype)
        z_diff = tf.cast(tf.math.logical_xor(noise_z_T, z_hat), self.hz_perp.dtype)

        # identify where flagged errors occurred
        sx = int_mod_2(tf.matmul(self.hz, x_diff))  
        sz = int_mod_2(tf.matmul(self.hx, z_diff))
        s_hat = tf.concat([sx, sz], axis=0)
        s_hat = tf.transpose(s_hat, (1,0))
        err = tf.reduce_any(tf.not_equal(tf.zeros_like(s_hat), s_hat), axis=-1)
        
        err_llrx, err_llry, err_llrz = llrx[err], llry[err], llrz[err]
        num_hx = tf.math.softplus(-1.*err_llrx) # I or X, both commute with X-type check
        denom_hx = tf.ragged.map_flat_values(lambda x, y: tf.math.reduce_logsumexp(-1.*tf.stack([x, y], axis=-1), axis=-1), err_llrz, err_llry)
        osd_llrz = num_hx - denom_hx

        num_hz = tf.math.softplus(-1.*err_llrz) # I or Z, both commute with Z-type check
        denom_hz = tf.ragged.map_flat_values(lambda x, y: tf.math.reduce_logsumexp(-1.*tf.stack([x, y], axis=-1), axis=-1), err_llrx, err_llry)
        osd_llrx = num_hz - denom_hz
        
        return noise_x, noise_z, x_hat, z_hat, err, osd_llrx, osd_llrz


    @tf.function(jit_compile=True, reduce_retracing=True) # XLA mode
    def call_osd(self, nx, nz, err, llrx, llrz, bs):
        # re-calculate syndrome to avoid XLA bug of masking
        syndrome_x = int_mod_2(tf.matmul(self.hx, tf.cast(tf.transpose(nz, (1,0)), self.hz.dtype)))
        syndrome_z = int_mod_2(tf.matmul(self.hz, tf.cast(tf.transpose(nx, (1,0)), self.hx.dtype)))

        reduced_sx = tf.gather(syndrome_x, self.pivot_hx, axis=0) # [rank, new_bs]
        reduced_sz = tf.gather(syndrome_z, self.pivot_hz, axis=0) # [rank, new_bs]

        full_rank_hx = tf.broadcast_to(tf.expand_dims(self.hx_basis, axis=0),
                             (bs, self.rank_hx, self.n))
        full_rank_hz = tf.broadcast_to(tf.expand_dims(self.hz_basis, axis=0),
                             (bs, self.rank_hz, self.n))

        z_hat_osd = self.osd_decoder(llrz, full_rank_hx, reduced_sx, bs)
        x_hat_osd = self.osd_decoder(llrx, full_rank_hz, reduced_sz, bs)
        
        return x_hat_osd, z_hat_osd

    def call(self, batch_size, ebno_db):

        noise_x, noise_z, x_hat, z_hat, err, osd_llrx, osd_llrz = self.generate_noise_and_bp4(batch_size, ebno_db)

        nx, nz = noise_x[err], noise_z[err]
        new_bs = tf.reduce_sum(tf.cast(err, tf.int32))
        x_hat_osd, z_hat_osd = self.call_osd(nx, nz, err, osd_llrx, osd_llrz, new_bs)

        new_x_hat = tf.tensor_scatter_nd_update(tf.transpose(x_hat, (1,0)), tf.where(err), x_hat_osd)
        new_z_hat = tf.tensor_scatter_nd_update(tf.transpose(z_hat, (1,0)), tf.where(err), z_hat_osd)

        new_x_hat_T = tf.transpose(new_x_hat, (1,0))
        new_z_hat_T = tf.transpose(new_z_hat, (1,0))
        x_diff = tf.cast(tf.math.logical_xor(tf.transpose(noise_x, (1,0)), new_x_hat_T), self.hx_perp.dtype)
        z_diff = tf.cast(tf.math.logical_xor(tf.transpose(noise_z, (1,0)), new_z_hat_T), self.hz_perp.dtype)

        # lsx = int_mod_2(tf.matmul(self.hx_perp, x_diff))
        # lsz = int_mod_2(tf.matmul(self.hz_perp, z_diff))
        lsx = int_mod_2(tf.matmul(self.lz, x_diff))
        lsz = int_mod_2(tf.matmul(self.lx, z_diff))
            
        ls_hat = tf.concat([lsx, lsz], axis=0)      # for total logical error counts

        ls_hat = tf.transpose(ls_hat, (1,0))         # bs should be the first dimension!!!

        return tf.zeros_like(ls_hat), ls_hat  

        
class BP2_OSD_Model(tf.keras.Model):
    def __init__(self, pcm, pcm_basis, pivot_pcm, logical_pcm, bp2_decoder, osd_decoder):

        super().__init__()
        
        # store values internally
        self.pcm = pcm
        self.pcm_basis = pcm_basis
        self.pivot_pcm = pivot_pcm
        self.logical_pcm = logical_pcm
        self.rank, self.n = pcm_basis.shape
        self.source = BinarySource()
       
        self.channel = BinarySymmetricChannel()

        # FEC encoder / decoder
        self.bp2_decoder = bp2_decoder
        self.osd_decoder = osd_decoder

    @tf.function(jit_compile=True, reduce_retracing=True) # XLA mode
    def generate_noise_and_bp2(self, batch_size, p):        
        
        llr_const = -tf.math.log((1.-p)/p)
    
        c = tf.zeros([batch_size, self.n])
        noise = tf.cast(self.channel((c, p)), tf.bool)  # [bs, self.n]
        llr = tf.fill(tf.shape(noise), llr_const)  
        noise_T = tf.transpose(noise, (1,0))
        noise_int_T = tf.cast(noise_T, self.pcm.dtype) # [self.n, bs]
        syndrome = int_mod_2(tf.matmul(self.pcm, noise_int_T))
        llr_hat = -1.0 * self.bp2_decoder((llr, syndrome)) 
        noise_hat = tf.less(llr_hat, 0.0)

        noise_hat_T = tf.transpose(noise_hat, (1,0)) # [self.n, bs]
        noise_diff = tf.math.logical_xor(noise_T, noise_hat_T) 
        noise_diff = tf.cast(noise_diff, self.logical_pcm.dtype)
        s_hat = int_mod_2(tf.matmul(self.pcm, noise_diff))
        s_hat = tf.transpose(s_hat, (1,0)) # bs should be the first dimension!

        # identify where flagged errors occurred
        err = tf.reduce_any(tf.not_equal(tf.zeros_like(s_hat), s_hat), axis=-1)
        
        osd_llr = llr_hat[err]
   
        return noise, noise_hat, err, osd_llr


    @tf.function(jit_compile=True, reduce_retracing=True) # XLA mode
    def call_osd(self, n, err, llr, bs):
        # re-calculate syndrome to avoid XLA bug of masking
        syndrome = int_mod_2(tf.matmul(self.pcm, tf.cast(tf.transpose(n, (1,0)), self.pcm.dtype)))

        reduced_s = tf.gather(syndrome, self.pivot_pcm, axis=0) # [rank, new_bs]

        full_rank_pcm = tf.broadcast_to(tf.expand_dims(self.pcm_basis, axis=0),
                             (bs, self.rank, self.n))

        noise_hat_osd = self.osd_decoder(llr, full_rank_pcm, reduced_s, bs)
        
        return noise_hat_osd
    
    
    def call(self, batch_size, ebno_db):
        noise, noise_hat, err, osd_llr = self.generate_noise_and_bp2(batch_size, ebno_db)

        n = noise[err]
        new_bs = tf.reduce_sum(tf.cast(err, tf.int32))
        noise_hat_osd = self.call_osd(n, err, osd_llr, new_bs)

        new_noise_hat = tf.tensor_scatter_nd_update(noise_hat, tf.where(err), noise_hat_osd)

        new_noise_hat_T = tf.transpose(new_noise_hat, (1,0))
        noise_diff = tf.cast(tf.math.logical_xor(tf.transpose(noise, (1,0)), new_noise_hat_T), self.logical_pcm.dtype)

        ls_hat = int_mod_2(tf.matmul(self.logical_pcm, noise_diff))
        ls_hat = tf.transpose(ls_hat, (1,0)) 
        return tf.zeros_like(ls_hat), ls_hat  

            
            
        
