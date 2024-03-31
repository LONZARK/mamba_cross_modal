import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import copy
import math


# from nets.ViS4mer_mamba import S4

import sys
sys.path.append('/home/jxl220096/code/ai_assignment')
from mamba.mamba_ssm.modules.mamba_simple import Mamba_mutilModal


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)

        return output_a, output_v


class HAN_Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, mamba_flag = 'None', crossmodal = 'None', norm=None):
        super(HAN_Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

        self.crossmodal = crossmodal
        self.mamba_flag = mamba_flag

        # print('HAN_Encoder - self.crossmodal', self.crossmodal)

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v,  src_mask=mask, 
                                    src_key_padding_mask=src_key_padding_mask)  #  mamba_flag, crossmodal,
            output_v = self.layers[i](src_v, src_a,  src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)

        return output_a, output_v

class Encoder_Mamba(nn.Module):

    def __init__(self, encoder_layer, num_layers, mamba_flag = 'None', crossmodal = 'None',  norm=None):
        super(Encoder_Mamba, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a, output_v, output_av, output_va = self.layers[i](src_a, src_v, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)
            output_av = self.norm2(output_av)
            output_va = self.norm2(output_va)

        return output_a, output_v, output_av, output_va


class HANLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, mamba_flag = 'None', crossmodal = 'None'):
        super(HANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.crossmodal = crossmodal
        self.mamba_flag = mamba_flag


        # Add mamba layer
        if self.mamba_flag != 'None':
            mamba_params = {'d_model_a': 512, 'd_model_v': 512, 'd_state': 16, 'd_conv': 4, 'expand': 2, 'layer_idx': 0}
            self.mamba = HAN_Encoder(Mamba_mutilModal(**mamba_params, crossmodal=self.crossmodal), \
                mamba_flag=self.mamba_flag, crossmodal=self.crossmodal, num_layers=5)

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)


        if self.mamba_flag == 'None':
            # ----------- original --------------
            src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]

        if self.mamba_flag == 'han cmatt to mamba': 
            # ----------- change cm_attn to mamba --------------
            src1, _ = self.mamba(src_q, src_v, self.crossmodal)[0]      
            src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]

        if self.mamba_flag == 'han selfatt to mamba': 
            # ----------- change self_attn to mamba --------------
            src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]  
            src2, _ = self.mamba(src_q, src_q, self.crossmodal)[0]

        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)



class MMIL_Net(nn.Module):

    def __init__(self, mamba_flag='None', crossmodal='None'):
        super(MMIL_Net, self).__init__()


        self.crossmodal = crossmodal
        self.mamba_flag = mamba_flag


        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_av_att = nn.Linear(512, 25)
        self.fc_a =  nn.Linear(128, 512)
        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)
        self.audio_encoder = nn.TransformerEncoder \
            (nn.TransformerEncoderLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)
        self.visual_encoder = nn.TransformerEncoder \
            (nn.TransformerEncoderLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)
        self.cmt_encoder = Encoder(CMTLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)

        self.han_layer = HANLayer(d_model=512, nhead=1, dim_feedforward=512, mamba_flag=self.mamba_flag, crossmodal=self.crossmodal)
        self.hat_encoder = HAN_Encoder(self.han_layer, mamba_flag=self.mamba_flag, crossmodal=self.crossmodal, num_layers=1)

        # Jia: add mamba model
        # self.Vmamba_encoder = Encoder(ViS4mer(d_input=512, l_max=512, d_output=512, d_model=512, n_layers=1, dropout=0.1, prenorm=True,), num_layers=1)
        
        # self.mamba_encoder = Encoder_Mamba(Mamba(d_model_a=512, d_model_v=512, d_state=16, d_conv=4, expand=2, layer_idx = 0), num_layers=10)
        
    def forward(self, audio, visual, visual_st, mamba_flag, crossmodal):

        x1 = self.fc_a(audio)

        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        vid_st = self.fc_st(visual_st)
        x2 = torch.cat((vid_s, vid_st), dim =-1)
        x2 = self.fc_fusion(x2)

        # HAN
        x1, x2 = self.hat_encoder(x1, x2)

        # # Jia : change encoder to Mamba / Mar-12, 2024
        # out_a, out_v, out_av, out_va = self.mamba_encoder(x1, x2)  # the shape of input: batch=16/1, seqlen=10, dim=512
        # # x1, x2 = out_a, out_v
        # x1, x2 = out_va, out_av

        # prediction
        x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2)
        frame_prob = torch.sigmoid(self.fc_prob(x))

        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
        av_att = torch.softmax(self.fc_av_att(x), dim=2)
        temporal_prob = (frame_att * frame_prob)
        global_prob = (temporal_prob*av_att).sum(dim=2).sum(dim=1)


        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
        v_prob =temporal_prob[:, :, 1, :].sum(dim=1)

        return global_prob, a_prob, v_prob, frame_prob

class CMTLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(CMTLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src_q, src_v, src_v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout1(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q


# self.mamba_encoder = Encoder(Mamba(d_input=512, l_max=512, d_output=512, d_model=512, n_layers=1, dropout=0.1, prenorm=True,), num_layers=1)


# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model_a=dim_a, # Model dimension d_model
#     d_model_v=dim_v,
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")

# print(model)




class ViS4mer(nn.Module):

    def __init__(
            self,
            d_input,
            l_max,
            d_output,
            d_model,
            n_layers,
            dropout=0.2,
            prenorm=True,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.d_model = d_model
        self.d_input = d_input

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.gelus = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(H=d_model, l_max=l_max, dropout=dropout, transposed=True)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))
            self.pools.append(nn.AvgPool1d(2))
            self.linears.append(nn.Linear(d_model, int(d_model/2)))
            self.gelus.append(nn.GELU())
            d_model = int(d_model/2)
            l_max = int(l_max/2)

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x1, x2, src_mask=None, src_key_padding_mask=None):
        """
        Input x is shape (B, L, d_input)
        """
        x1 = x1.to(torch.float32)

        if self.d_model != self.d_input:
            x1 = self.encoder(x1)  # (B, L, d_input) -> (B, L, d_model)

        x1 = x1.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout, pool,linear, gelu in \
                zip(self.s4_layers, self.norms, self.dropouts, self.pools, self.linears, self.gelus):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z1 = x1

            if self.prenorm:
                # Prenorm
                z1 = norm(z1.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            # print('z1.shape', z1.shape)
            z1, _ = layer(z1)

            # Dropout on the output of the S4 block
            z1 = dropout(z1)

            # Residual connection
            x1 = z1 + x1

            if not self.prenorm:
                # Postnorm
                x1 = norm(x1.transpose(-1, -2)).transpose(-1, -2)
            #pooling layer
            x1 = pool(x1)
            # MLP
            x1 = x1.transpose(-1, -2)
            x1 = linear(x1)
            x1 = gelu(x1)
            x1 = x1.transpose(-1, -2)
        x1 = x1.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x1 = x1.mean(dim=1)

        #x = x.max(dim=1)
        # Decode the outputs
        x1 = self.decoder(x1)  # (B, d_model) -> (B, d_output)

        return x1
    