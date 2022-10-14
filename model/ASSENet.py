import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
from torch.distributions import Bernoulli, Categorical
import os
import sys
sys.path.append(os.path.dirname(__file__))
from deformable.deform_conv import th_batch_map_offsets, th_generate_grid
from DCCRN.ConvSTFT import ConvSTFT, ConviSTFT

class ConvOffset2D(nn.Conv2d):
    """ConvOffset2D
    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """
    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters*2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)
        #print(offsets.size())
        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)
        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)
        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self,x))
        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)
        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x

class GLayerNorm2d(nn.Module):
    
    def __init__(self, in_channel, eps=1e-12):
        super(GLayerNorm2d, self).__init__()
        self.eps = eps 
        self.beta = nn.Parameter(torch.ones([1, in_channel,1,1]))
        self.gamma = nn.Parameter(torch.zeros([1, in_channel,1,1]))
    
    def forward(self,inputs):
        mean = torch.mean(inputs,[1,2,3], keepdim=True)
        var = torch.var(inputs,[1,2,3], keepdim=True)
        outputs = (inputs - mean)/ torch.sqrt(var+self.eps)*self.beta+self.gamma
        return outputs

class NonLocalAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, kernel_size=7, stride=1, height_dim=True, inference=False):
        super(NonLocalAttention, self).__init__()
        self.depth = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.height_dim = height_dim
        self.dh = self.depth // self.num_heads
        
        assert self.depth % self.num_heads == 0, "depth should be divided by num_heads. (example: depth: 32, num_heads: 8)"

        self.kqv_conv = nn.Conv1d(in_channels, self.depth * 2, kernel_size=1, bias=False)
        self.kqv_bn = nn.BatchNorm1d(self.depth * 2)
        self.logits_bn = nn.BatchNorm2d(num_heads * 3)
        # Positional encodings
        self.rel_encoding = nn.Parameter(torch.randn(self.dh * 2, kernel_size * 2 - 1), requires_grad=True)
        key_index = torch.arange(kernel_size)
        query_index = torch.arange(kernel_size)
        # Shift the distance_matrix so that it is >= 0. Each entry of the
        # distance_matrix distance will index a relative positional embedding.
        distance_matrix = (key_index[None, :] - query_index[:, None]) + kernel_size - 1
        self.register_buffer('distance_matrix', distance_matrix.reshape(kernel_size*kernel_size))

        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x):
        if self.height_dim:
            x = x.permute(0, 3, 1, 2)  # batch_size, width, depth, height
        else:
            x = x.permute(0, 2, 1, 3)  # batch_size, height, depth, width
            
        batch_size, width, depth, height = x.size()
        x = x.reshape(batch_size * width, depth, height)

        # Compute q, k, v
        kqv = self.kqv_conv(x)

        kqv = self.kqv_bn(kqv) # apply batch normalization on k, q, v
        k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height), [self.dh // 2, self.dh // 2, self.dh], dim=2)

        # Positional encodings
        rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * 2, self.kernel_size, self.kernel_size)
        q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh // 2, self.dh // 2, self.dh], dim=0)

        # qk + qr + kr
        qk = torch.matmul(q.transpose(2, 3), k)
        qr = torch.einsum('bhdx,dxy->bhxy', q, q_encoding)
        kr = torch.einsum('bhdx,dxy->bhxy', k, k_encoding).transpose(2, 3)

        logits = torch.cat([qk, qr, kr], dim=1)
        logits = self.logits_bn(logits) # apply batch normalization on qk, qr, kr
        logits = logits.reshape(batch_size * width, 3, self.num_heads, height, height).sum(dim=1)
        
        weights = F.softmax(logits, dim=3)

        if self.inference:
            self.weights = nn.Parameter(weights)
            
        attn = torch.matmul(weights, v.transpose(2,3)).transpose(2,3)
        attn_encoding = torch.einsum('bhxy,dxy->bhdx', weights, v_encoding)
        attn_out = torch.cat([attn, attn_encoding], dim=-1).reshape(batch_size * width, self.depth * 2, height)
        output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)

        if self.height_dim:
            output = output.permute(0, 2, 3, 1)
        else:
            output = output.permute(0, 2, 1, 3)
        
        return output

class LocalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = ConvOffset2D(256)
		
        self.q = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
			
        self.k = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))	
			
        self.v = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))			

        self.softmax = nn.Softmax(dim=1) 		

    def forward(self, x):

        q = self.offset(x)
        q = self.q(q)
		
        k = self.k(x)
        v = self.v(x)

        output = self.softmax(q * k) * v
		
        return output

class NonCausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class NonCausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x
 
class Model(nn.Module):

    def __init__(self, channel_amp = 1, channel_phase=2):
        super(Model, self).__init__()
        self.stft = ConvSTFT(512, 256, 512, 'hanning', 'complex', True)
        self.istft = ConviSTFT(512, 256, 512, 'hanning', 'complex', True)

        self.amp_conv1 = nn.Sequential(
                                nn.Conv2d(2, channel_amp, 
                                        kernel_size=[7,1],
                                        padding=(3,0)
                                    ),
                                nn.BatchNorm2d(channel_amp),
                                nn.ReLU(),
                                nn.Conv2d(channel_amp, channel_amp, 
                                        kernel_size=[1,7],
                                        padding=(0,3)
                                    ),
                                nn.BatchNorm2d(channel_amp),
                                nn.ReLU(),
                        )
        self.phase_conv1 = nn.Sequential(
                                nn.Conv2d(2, channel_phase, 
                                        kernel_size=[3,5],
                                        padding=(1,2)
                                    ),
                                nn.Conv2d(channel_phase, channel_phase, 
                                        kernel_size=[3,25],
                                        padding=(1, 12)
                                    ),
                        )
        self.amp_conv2 = nn.Sequential(
                        nn.Conv2d(channel_amp, 1, kernel_size=[1, 1]),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                    )
        self.phase_conv2 = nn.Sequential(
                        nn.Conv1d(3,8,kernel_size=[1,1])
                    )

        self.phase_conv5 = nn.Sequential(
                        nn.Conv1d(8, 2, kernel_size=(1,1))
                    )
        self.phase_conv3 = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=(5,5), padding=(2,2)),
                        GLayerNorm2d(8),
                    )
        self.phase_conv4 = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=(1,25), padding=(0,12)),
                        GLayerNorm2d(8),
                    )

        self.rnn = nn.GRU(
                        257,
                        300,
                        bidirectional=True
                    )
        self.fcs = nn.Sequential(
                    nn.Linear(300*2,600),
                    nn.ReLU(),
                    nn.Linear(600,600),
                    nn.ReLU(),
                    nn.Linear(600,514//2),
                    nn.Sigmoid()
                )	
        # Encoder
        self.conv_block_1 = NonCausalConvBlock(1, 16)
        self.conv_block_2 = NonCausalConvBlock(16, 32)
        self.conv_block_3 = NonCausalConvBlock(32, 64)
        self.conv_block_4 = NonCausalConvBlock(64, 128)
        self.conv_block_5 = NonCausalConvBlock(128, 256)
        # Decoder
        self.tran_conv_block_1 = NonCausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = NonCausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = NonCausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = NonCausalTransConvBlock(32 + 32, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = NonCausalTransConvBlock(16 + 16, 1, is_last=True)
        # Local Attention
        self.local = LocalAttention()
		
        # Non-local Attention
        self.nonlocal_time = NonLocalAttention(256, kernel_size=7, height_dim=True)
        self.nonlocal_frequency = NonLocalAttention(256, kernel_size=251, height_dim=False)
		
        # ASN
        self.select_block_1 = NonCausalConvBlock(64, 128)
        self.select_block_2 = NonCausalConvBlock(128, 256)
        
        self.lstm_layer = nn.LSTM(input_size=251, hidden_size=251, num_layers=2, batch_first=True)
        self.logit = nn.Linear(251, 4)
        self.sigmoid = nn.Sigmoid()
    def full_forward(self, x):
	
        cmp_spec = self.stft(x)
        cmp_spec = torch.unsqueeze(cmp_spec, 1)

        # to [B, 2, D, T]
        cmp_spec = torch.cat([
                                cmp_spec[:,:,:257,:],
                                cmp_spec[:,:,257:,:],
                                ],
                                1)
        # mean = torch.mean(cmp_spec, [1, 2, 3], keepdim = True)
        # std = torch.std(cmp_spec, [1, 2, 3], keepdim = True)
        # cmp_spec = (cmp_spec - mean) / (std + 1e-8)    
        amp_spec = torch.sqrt(
                            torch.abs(cmp_spec[:,0])**2+
                            torch.abs(cmp_spec[:,1])**2,
                        )
        amp_spec = torch.unsqueeze(amp_spec, 1)
        spec = self.amp_conv1(cmp_spec)
	
	

        e1 = self.conv_block_1(spec)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)		
	##################################################################################
	# Dynamic block 1
	    # selector
        s = self.select_block_2(self.select_block_1(e3))
        batch_size, n_channels, n_f_bins, n_frame_size = s.shape
        lstm_in = s.reshape(batch_size, n_channels * n_f_bins, n_frame_size)
        lstm_out, _ = self.lstm_layer(lstm_in)
        lstm_out = lstm_out.mean(dim=1)
        probs = self.sigmoid(self.logit(lstm_out))
        probs = torch.randint(0, 2, [1, 4], dtype=torch.float).to(s.device)
		
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = e5
        if probs[:, 0] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs[:, 1] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5

        e5 = l + n
            
        if probs[:, 2] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs[:, 3] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5	
			
        e5 = l + n		
		
        d1 = self.tran_conv_block_1(torch.cat([e5, e6], dim = 1))		
        d2 = self.tran_conv_block_2(torch.cat([e4, d1], dim = 1))		

        e3_2 = e3 + d2
    # Dynamic block 2
      # selector

        s = self.select_block_2(self.select_block_1(e3))
        batch_size, n_channels, n_f_bins, n_frame_size = s.shape
        lstm_in = s.reshape(batch_size, n_channels * n_f_bins, n_frame_size)
        lstm_out, _ = self.lstm_layer(lstm_in)
        lstm_out = lstm_out.mean(dim=1)
        probs1 = self.sigmoid(self.logit(lstm_out))
        probs1 = torch.randint(0, 2, [1, 4], dtype=torch.float).to(s.device)

        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = e5
        if probs1[:, 0] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs1[:, 1] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5

        e5 = l + n
            
        if probs1[:, 2] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs1[:, 3] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5	
			
        e5 = l + n		
		
        d1 = self.tran_conv_block_1(torch.cat([e5, e6], dim = 1))		
        d2 = self.tran_conv_block_2(torch.cat([e4, d1], dim = 1))

        e3_3 = e3_2 + d2
    
	# Dynamic block 3
	    # selector

        s = self.select_block_2(self.select_block_1(e3))
        batch_size, n_channels, n_f_bins, n_frame_size = s.shape
        lstm_in = s.reshape(batch_size, n_channels * n_f_bins, n_frame_size)
        lstm_out, _ = self.lstm_layer(lstm_in)
        lstm_out = lstm_out.mean(dim=1)
        probs2 = self.sigmoid(self.logit(lstm_out))

        probs2 = torch.randint(0, 2, [1, 4], dtype=torch.float).to(s.device)

        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = e5
        if probs2[:, 0] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs2[:, 1] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5

        e5 = l + n
            
        if probs2[:, 2] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs2[:, 3] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5	
			
        e5 = l + n		
		
        d1 = self.tran_conv_block_1(torch.cat([e5, e6], dim = 1))		
        d2 = self.tran_conv_block_2(torch.cat([e4, d1], dim = 1))

        e3_4 = e3_3 + d2
	####################################################################
		
        d3 = self.tran_conv_block_3(torch.cat([e3_4, d2], dim = 1))		
        d4 = self.tran_conv_block_4(torch.cat([e2, d3], dim = 1))		
        d = self.tran_conv_block_5(torch.cat([e1, d4], dim = 1))


        spec = torch.transpose(d, 1,3)
        #print(spec.size())
        B, T, D, C = spec.size()
        spec = torch.reshape(spec, [B, T, D*C])
        spec = self.rnn(spec)[0]
        spec = self.fcs(spec)
        
        spec = torch.reshape(spec, [B,T,D,1]) 
        spec = torch.transpose(spec, 1,3)

	
        phase_pro = self.phase_conv1(cmp_spec)			
        phase_input = torch.cat([phase_pro, self.amp_conv2(d)], dim = 1)
      
        phase_input = self.phase_conv2(phase_input)	
        p1 = self.phase_conv3(phase_input)
        p1 = self.phase_conv4(p1)
		
        p2 = self.phase_conv3(p1 + phase_input)
        p2 = self.phase_conv4(p2)
		
        p3 = self.phase_conv3(p2 + p1)
        p3 = self.phase_conv4(p3)

        p5 = self.phase_conv5(p3)
        p5 = phase_pro + p5
        p5 = p5/(torch.sqrt(
                            torch.abs(p5[:,0])**2+
                            torch.abs(p5[:,1])**2)
                        +1e-8).unsqueeze(1)
        est_spec = amp_spec * d * p5
        est_spec = torch.cat([est_spec[:,0], est_spec[:,1]], 1)
        est_wav = self.istft(est_spec, None)
        est_wav = torch.squeeze(est_wav, 1)
        return est_spec, est_wav

    def forward(self, x):
	
        cmp_spec = self.stft(x)
        cmp_spec = torch.unsqueeze(cmp_spec, 1)

        # to [B, 2, D, T]
        cmp_spec = torch.cat([
                                cmp_spec[:,:,:257,:],
                                cmp_spec[:,:,257:,:],
                                ],
                                1)
        # mean = torch.mean(cmp_spec, [1, 2, 3], keepdim = True)
        # std = torch.std(cmp_spec, [1, 2, 3], keepdim = True)
        # cmp_spec = (cmp_spec - mean) / (std + 1e-8)    
        amp_spec = torch.sqrt(
                            torch.abs(cmp_spec[:,0])**2+
                            torch.abs(cmp_spec[:,1])**2,
                        )
        amp_spec = torch.unsqueeze(amp_spec, 1)
        spec = self.amp_conv1(cmp_spec)
	
	

        e1 = self.conv_block_1(spec)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)		
	##################################################################################
	# Dynamic block 1
	    # selector
        s = self.select_block_2(self.select_block_1(e3))
        batch_size, n_channels, n_f_bins, n_frame_size = s.shape
        lstm_in = s.reshape(batch_size, n_channels * n_f_bins, n_frame_size)
        lstm_out, _ = self.lstm_layer(lstm_in)
        lstm_out = lstm_out.mean(dim=1)
        probs = self.sigmoid(self.logit(lstm_out)).mean(0)
        probs = 0.8 * probs + (1 - probs) * 0.2
        distr_1 = Bernoulli(probs)
        probs = distr_1.sample()     
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = e5
        if probs[0] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs[1] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5

        e5 = l + n
            
        if probs[2] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs[3] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5	
			
        e5 = l + n		
		
        d1 = self.tran_conv_block_1(torch.cat([e5, e6], dim = 1))		
        d2 = self.tran_conv_block_2(torch.cat([e4, d1], dim = 1))		

        e3_2 = e3 + d2
    # Dynamic block 2
      # selector

        s = self.select_block_2(self.select_block_1(e3))
        batch_size, n_channels, n_f_bins, n_frame_size = s.shape
        lstm_in = s.reshape(batch_size, n_channels * n_f_bins, n_frame_size)
        lstm_out, _ = self.lstm_layer(lstm_in)
        lstm_out = lstm_out.mean(dim=1)
        probs1 = self.sigmoid(self.logit(lstm_out)).mean(0)
        probs1 = 0.8 * probs1 + (1 - probs1) * 0.2
        distr_2 = Bernoulli(probs1)
        probs1 = distr_2.sample()

        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = e5
        if probs1[0] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs1[1] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5

        e5 = l + n
            
        if probs1[2] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs1[3] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5	
			
        e5 = l + n		
		
        d1 = self.tran_conv_block_1(torch.cat([e5, e6], dim = 1))		
        d2 = self.tran_conv_block_2(torch.cat([e4, d1], dim = 1))

        e3_3 = e3_2 + d2
    
	# Dynamic block 3
	    # selector

        s = self.select_block_2(self.select_block_1(e3))
        batch_size, n_channels, n_f_bins, n_frame_size = s.shape
        lstm_in = s.reshape(batch_size, n_channels * n_f_bins, n_frame_size)
        lstm_out, _ = self.lstm_layer(lstm_in)
        lstm_out = lstm_out.mean(dim=1)
        probs2 = self.sigmoid(self.logit(lstm_out)).mean(0)
        probs2 = 0.8 * probs2 + (1 - probs2) * 0.2
        distr_3 = Bernoulli(probs2)
        probs2 = distr_3.sample()

        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = e5
        if probs2[0] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs2[1] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5

        e5 = l + n
            
        if probs2[2] >= 0.5:
            l = self.local(e5)
        else:
            l = e5
			
        if probs2[3] >= 0.5:
            n_time = self.nonlocal_time(e5)
            n_frequency = self.nonlocal_frequency(e5)
            n = n_time + n_frequency
        else:
            n = e5	
			
        e5 = l + n		
		
        d1 = self.tran_conv_block_1(torch.cat([e5, e6], dim = 1))		
        d2 = self.tran_conv_block_2(torch.cat([e4, d1], dim = 1))

        e3_4 = e3_3 + d2
	####################################################################
		
        d3 = self.tran_conv_block_3(torch.cat([e3_4, d2], dim = 1))		
        d4 = self.tran_conv_block_4(torch.cat([e2, d3], dim = 1))		
        d = self.tran_conv_block_5(torch.cat([e1, d4], dim = 1))


        spec = torch.transpose(d, 1,3)
        #print(spec.size())
        B, T, D, C = spec.size()
        spec = torch.reshape(spec, [B, T, D*C])
        spec = self.rnn(spec)[0]
        spec = self.fcs(spec)
        
        spec = torch.reshape(spec, [B,T,D,1]) 
        spec = torch.transpose(spec, 1,3)

	
        phase_pro = self.phase_conv1(cmp_spec)			
        phase_input = torch.cat([phase_pro, self.amp_conv2(d)], dim = 1)
      
        phase_input = self.phase_conv2(phase_input)	
        p1 = self.phase_conv3(phase_input)
        p1 = self.phase_conv4(p1)
		
        p2 = self.phase_conv3(p1 + phase_input)
        p2 = self.phase_conv4(p2)
		
        p3 = self.phase_conv3(p2 + p1)
        p3 = self.phase_conv4(p3)

        p5 = self.phase_conv5(p3)
        p5 = phase_pro + p5
        p5 = p5/(torch.sqrt(
                            torch.abs(p5[:,0])**2+
                            torch.abs(p5[:,1])**2)
                        +1e-8).unsqueeze(1)
        est_spec = amp_spec * d * p5
        est_spec = torch.cat([est_spec[:,0], est_spec[:,1]], 1)
        est_wav = self.istft(est_spec, None)
        est_wav = torch.squeeze(est_wav, 1)
        return est_spec, est_wav, probs, probs1, probs2, distr_1, distr_2, distr_3

    def loss(self, est, labels, mode='Mix'):
        '''
        mode == 'Mix'
            est: [B, F*2, T]
            labels: [B, F*2,T]
        mode == 'SiSNR'
            est: [B, T]
            labels: [B, T]
        '''
        if mode == 'SiSNR':
            if labels.dim() == 3:
                labels = torch.squeeze(labels,1)
            if est.dim() == 3:
                est = torch.squeeze(est,1)
            return -si_snr(est, labels)         
        elif mode == 'Mix':
            b, d, t = est.size()
            gth_cspec = self.stft(labels)
            est_cspec = est  
            gth_mag_spec = torch.sqrt(
                                    gth_cspec[:, :self.feat_dim, :]**2
                                    +gth_cspec[:, self.feat_dim:, :]**2 + 1e-8
                               )
            est_mag_spec = torch.sqrt(
                                    est_cspec[:, :self.feat_dim, :]**2
                                    +est_cspec[:, self.feat_dim:, :]**2 + 1e-8
                                )
            
            # power compress 
            gth_cprs_mag_spec = gth_mag_spec**0.3
            est_cprs_mag_spec = est_mag_spec**0.3
            amp_loss = F.mse_loss(
                                gth_cprs_mag_spec, est_cprs_mag_spec
                            )*d
            compress_coff = (gth_cprs_mag_spec/(1e-8+gth_mag_spec)).repeat(1,2,1)
            phase_loss = F.mse_loss(
                                gth_cspec*compress_coff,
                                est_cspec*compress_coff
                            )*d
            
            all_loss = amp_loss*0.5 + phase_loss*0.5
            return all_loss, amp_loss, phase_loss

def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True) 
    data = data - mean
    return data
def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)
    
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)


		
if __name__ == '__main__':
    net = Model()
    x = torch.randn(1, 64000)
    y, y1 = net(x)
    print(y.size())		
		
		
		
		
		
		
		
		
		
		
		
		
		
		