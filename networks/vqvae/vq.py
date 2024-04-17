import os, sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.vqvae.resnet import Resnet1D
from networks.vqvae.vq_encoder import *
from networks.vqvae.vq_decoder import *
from networks.vqvae.quantizer import *

if __name__ == "__main__":
    
    # ckpt = torch.load("networks/vqvae/pretrained/net_last.pth")
    
    # encoder_ckpt = {}
    # decoder_ckpt = {}
    # quantizer_ckpt = {}
    # for name, val in ckpt["net"].items():
    #     if 'vqvae.encoder.' in name:
    #         encoder_ckpt[name.replace('vqvae.encoder.', '')] = val
    #     if 'vqvae.decoder.' in name:
    #         decoder_ckpt[name.replace('vqvae.decoder.', '')] = val
    #     if 'vqvae.quantizer.' in name:
    #         quantizer_ckpt[name.replace('vqvae.quantizer.', '')] = val
    # out_ckpt = {
    #     'body_vqencoder': encoder_ckpt, 
    #     'body_vqdecoder': decoder_ckpt, 
    #     'body_quantizer': quantizer_ckpt
    # }
    # torch.save(out_ckpt, "networks/vqvae/pretrained/net_reorganized.pth")
    
    
    encoder = VQEncoderHML(
        input_emb_width=263, 
        output_emb_width=512, 
        down_t=2, 
        stride_t=2, 
        width=512, 
        depth=3, 
        dilation_growth_rate=3, 
        activation='relu', 
        norm=None).cuda()
    decoder = VQDecoderHML(
        input_emb_width=263, 
        output_emb_width=512, 
        down_t=2, 
        stride_t=2, 
        width=512, 
        depth=3, 
        dilation_growth_rate=3, 
        activation='relu', 
        norm=None).cuda()
    # quantizer = QuantizeEMAReset(
    #     nb_code=512, 
    #     code_dim=512, 
    #     mu=0.99).cuda()
    quantizer = Quantizer(n_e=1024, e_dim=512, beta=1.0).cuda()
    
    for name, _ in encoder.named_parameters(): print(name)
    for name, _ in decoder.named_parameters(): print(name)
    for name, _ in quantizer.named_parameters(): print(name)
    
    motion = torch.randn(1, 100, 263).cuda()
    
    z = encoder(motion)
    print(z.shape)
    loss, z_q, *_ = quantizer(z)
    print(z_q.shape)
    rec = decoder(z_q)
    # z = z.contiguous().view(-1, z.size(-1))
    tokens = quantizer.map2index(z)
    print(tokens.shape)
    z_q = quantizer.get_codebook_entry(tokens.unsqueeze(dim=0))
    print(z_q.shape)
    
    