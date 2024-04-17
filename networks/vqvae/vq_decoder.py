import os, sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.vqvae.layers import *
from networks.vqvae.resnet import Resnet1D
from networks.utils.positional_encoding import PositionalEncoding
from funcs.hml3d.conversion import *
import importlib

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class VQDecoder(nn.Module):
    """Conv + Transformer decoder. 
    1. The Convs decode the codebooks back to sub-local motion sequences. 
       Then we convert the sub-local motion sequence to sub-global motion sequence.
       We define the sub-local motion sequence as: we set the global translation of 
       the root joints as the offsets relative to previous frame. 
    2. Transformers then maps the sub-local motion sequence to global motion sequence.
    """
    def __init__(self, input_size, channels, n_resblk, n_up, hidden_dims, num_layers, num_heads, dropout, activation="gelu", **kwargs):
        super(VQDecoder, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        assert len(channels) == n_up + 1
        
        # Build convs (codebook to sub-local)
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]
        
        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1), 
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.convs = nn.Sequential(*layers)
        self.convs.apply(init_weight)
        
        # Build transformers (sub-local to global)
        output_size = channels[-1]
        self.s2g_linear = nn.Linear(output_size, input_size, bias=False)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, 
                                                               nhead=num_heads, 
                                                               dim_feedforward=hidden_dims, 
                                                               dropout=dropout, 
                                                               activation=activation)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, 
                                                 num_layers=num_layers)
        self.position_encoding = PositionalEncoding(d_model=input_size, dropout=dropout, max_len=5000)
        self.final = nn.Conv1d(input_size, output_size, kernel_size=3, stride=1, padding=1)
        self.s2g_linear.apply(init_weight)
        self.final.apply(init_weight)
        
    def forward(self, inputs, return_pose=False):
        """
        :param inputs: [batch_size, n_tokens, dim]
        """
        # Decode codebook to sub-local motion sequence
        self.sublocal_outputs = self.convs(inputs.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, seq_len, out_dim]
        # Convert sub-local to sub-global
        subglobal_inputs = self.convert_sublocal_to_subglobal(self.sublocal_outputs)
        # Convert sub-global to global
        global_outputs = self.convert_subglobal_to_global(subglobal_inputs)
        return global_outputs
        
    def convert_sublocal_to_subglobal(self, inputs):
        """
        :param input: [batch_size, seq_len, dim]
        """
        sl_trans = inputs[..., :3]
        pose = inputs[..., 3:]
        
        batch_size = sl_trans.size(0)
        seq_len = sl_trans.size(1)
        
        sg_trans = [torch.zeros(batch_size, 3).float().to(inputs.device)]
        for i in range(1, seq_len):
            sg_trans.append(sg_trans[-1] + sl_trans[:, i])
        sg_trans = torch.stack(sg_trans, dim=1)
        return torch.cat([sg_trans, pose], dim=-1)
        
    def convert_subglobal_to_global(self, inputs):
        x = self.s2g_linear(inputs)
        x = x.permute(1, 0, 2)
        x = self.position_encoding(x)
        x = self.transformer(x).permute(1, 0, 2)
        y = self.final(x.permute(0, 2, 1))
        return y.permute(0, 2, 1)
        
    def get_sublocal_outputs(self):
        return self.sublocal_outputs
    
class VQDecoderV2(nn.Module):
    """Conv1D decoder.
    It decodes the latent features and reconstructs motion sequence in one-stage manner.
    """
    def __init__(self, input_size, channels, n_resblk, n_up, **kwargs):
        super(VQDecoderV2, self).__init__()
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs, return_pose=False):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs  # [bs, nframes, 263]
    
class VQDecoderV3(VQDecoderV2):
    """Conv1D decoder.
    It decodes the latent features and reconstructs motion sequence in two-stage manner.
    """
    def __init__(self, input_size, channels, n_resblk, n_up, mean_dir=None, std_dir=None, **kwargs):
        super(VQDecoderV3, self).__init__(input_size, channels, n_resblk, n_up, **kwargs)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.mean = torch.from_numpy(np.load(mean_dir)).float().to(self.device)
        # self.std = torch.from_numpy(np.load(std_dir)).float().to(self.device)
        self.build_stage_two_model(d_input=channels[-1], d_aux=input_size, **kwargs)
    
    def recover_from_ric(self, data, joints_num):
        data = data * self.std + self.mean  # Apply inverse transform
        r_rot_quat, r_pos = recover_root_rot_pos(data)
        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        return positions
    
    def build_stage_two_model(self, d_input, d_aux, channels_aux, **kwargs):
        """Build the second-stage decoder.
        It takes as input the reconstruction results of stage-one decoder, and the latent vectors.
        """
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                nn.Conv1d(d_input+d_aux, channels_aux[0], kernel_size=3, stride=1, padding=1), 
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        for i in range(1, len(channels_aux), 1):
            layer = nn.Sequential(
                nn.Conv1d(channels_aux[i-1], channels_aux[i], kernel_size=4, stride=2, padding=1), 
                ResBlock(channels_aux[i])
            )
            self.layers.append(layer)
        
        channels_aux_rev = channels_aux[::-1]
        for i in range(1, len(channels_aux_rev), 1):
            layer = nn.Sequential(
                ResBlock(channels_aux_rev[i-1]),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels_aux_rev[i-1], channels_aux_rev[i], 3, 1, 1)
            )
            self.layers.append(layer)
        self.layers.append(
            nn.Sequential(
                nn.Conv1d(channels_aux_rev[-1], channels_aux_rev[-1], kernel_size=3, stride=1, padding=1), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Conv1d(channels_aux_rev[-1], d_input, kernel_size=3, stride=1, padding=1)
            )
        )
    
    def decode_stage_one(self, inputs):
        return super().forward(inputs=inputs)
        
    def decode_stage_two(self, recons, latents):
        """
        :param recons: [batch_size, seq_len, num_dim_1]
        :param latents: [batch_size, seq_len//4, num_dim_2]
        :param lengths: [batch_size]
        """
        B, T = recons.shape[:2]
        # # 1. Convert motion parameters to joints
        # joints = [self.recover_from_ric(recon, joints_num=22) for recon in recons]
        # joints = torch.stack(joints, dim=0)
        # joints = joints.contiguous().view(B, T, -1)
        
        # 2. Interpolate the latents
        latents = latents.permute(0, 2, 1)
        latents = nn.Upsample(scale_factor=4, mode="linear")(latents)
        latents = latents.permute(0, 2, 1)
        
        inputs = torch.cat([recons, latents], dim=-1)   # [B, T, C]
        
        x = inputs.permute(0, 2, 1)                     # [B, C, T]
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 2, 1)
        
    def forward(self, inputs, return_pose=False, return_dict=False):
        """
        :param inputs: [batch_size, seq_len, num_dim]
        """
        recons = {}
        # Decode the latents using stage-one decoder
        recons["stage_one"] = self.decode_stage_one(inputs=inputs)
        # y = self.decode_stage_one(inputs=inputs)
        # Decode the latents using stage-two decoder
        recons["stage_two"] = self.decode_stage_two(recons=recons["stage_one"], latents=inputs)
        
        if return_dict:
            return recons
        else:
            return recons["stage_two"]
    
class VQDecoderHML(nn.Module):
    def __init__(
        self,
        input_emb_width = 263,
        output_emb_width = 512,
        down_t = 2,
        stride_t = 2,
        width = 512,
        depth = 3,
        dilation_growth_rate = 3, 
        activation='relu',
        norm=None, 
        **kwargs
    ):
        super(VQDecoderHML, self).__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs, return_pose=False):
        x = inputs.permute(0, 2, 1)
        y = self.model(x)
        return y.permute(0, 2, 1)

if __name__ == "__main__":
    
    Conf = {
        "input_size": 512,
        "channels": [1024, 1024, 263], 
        "n_resblk": 3, 
        "n_up": 2, 
        "channels_aux": [128, 512, 1024], 
        "dilation_growth_rate": 3, 
        "activation": "relu", 
        "mean_dir": "tools/mean.npy", 
        "std_dir": "tools/std.npy"
    }
    
    model = VQDecoderV3(**Conf)
    model = model.to(model.device)
    latents = torch.randn(2, 10, 512).to(model.device)
    model(latents)