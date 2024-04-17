import os, sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.vqvae.layers import *
from networks.utils.positional_encoding import PositionalEncoding
import importlib

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, **kwargs):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding, there are two terms in the loss: 
        # 1) z_q - z.detach(): pull codebook embeddings to close to encoded motion embedding, and 
        # 2) z_q.detach() - z: pull encoded motion embeddings to close to codebook embeddings.
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return loss, z_q, min_encoding_indices, perplexity

    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def get_z_to_code_distance(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        return d.reshape(z.shape[0], z.shape[1], -1)

    def get_codebook_entry(self, indices, is_onhot=False):
        """
        Get the token embeddings by given token indices (onehot distribution)
        :param indices: [B, seq_len] or [B, seq_len, n_dim]
        :return z_q(B, seq_len, e_dim):
        """
        if not is_onhot:
            index_flattened = indices.view(-1)
            z_q = self.embedding(index_flattened)
            z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        else:
            z_q = torch.matmul(indices, self.embedding.weight).contiguous()  # [B, seq_len, n_dim]
        return z_q

    def multinomial(self, indices, K=10, temperature=2):
        """Sample tokens according to the embedding distance.
        :param indices: [batch_size, seq_len]
        :param K: top-k
        :param temperature: temperature parameter controls the smoothness of pdist distribution
        """
        # Get query embedding
        B, T = indices.shape
        z_emb = self.get_codebook_entry(indices)    # [B, T, C]
        # Calculate the pairwise distance between query embedding and all embedding
        q_emb = self.embedding.weight.clone()
        pdist = torch.cdist(z_emb.contiguous().view(B*T, -1), q_emb)                 # [B*T, N]
        pdist = torch.exp(-pdist / temperature)
        # Select top-k
        _, topk_idx = torch.topk(pdist, k=K, dim=-1)
        topk_data = [pdist[i, topk_idx[i]] for i in range(topk_idx.size(0))]
        topk_data = torch.stack(topk_data, dim=0)
        # Sample 1 data from the selected top-K data (the returns are index)
        sampled_idx_ = torch.multinomial(topk_data, num_samples=1, replacement=False)
        # We get the corresponding indices in pre-topk data space
        sampled_idx = [topk_idx[i, sampled_idx_[i]] for i in range(sampled_idx_.size(0))]
        # We get the corresponding sampled data
        sampled_data = [topk_data[i, sampled_idx_[i]] for i in range(sampled_idx_.size(0))]
        sampled_idx = torch.stack(sampled_idx, dim=0)
        sampled_data = torch.stack(sampled_data, dim=0)
        
        sampled_idx = sampled_idx.reshape(B, T)
        sampled_data = sampled_data.reshape(B, T, -1)
        topk_data = topk_data.reshape(B, T, -1)
        
        return sampled_idx, sampled_data, topk_data

    def multinomial_prob(self, indices, K=10, temperature=2):
        """Convert the onehot probability to multinomial probability.
        :param indices: [batch_size, seq_len]
        :param K: top-k
        :param temperature: temperature parameter controls the smoothness of pdist distribution
        """
        device = indices.device
        # Get query embedding
        B, T = indices.shape
        z_emb = self.get_codebook_entry(indices)    # [B, T, C]
        # Calculate the pairwise distance between query embedding and all embedding
        q_emb = self.embedding.weight.clone()
        pdist = torch.cdist(z_emb.contiguous().view(B*T, -1), q_emb)                 # [B*T, N]
        prob = torch.exp(-pdist)
        # Select top-K (actually the smallest top-k)
        _, topk_idx = torch.topk(pdist, k=self.n_e-K, dim=-1)
        topk_mask = [torch.zeros(self.n_e).to(device).scatter_(dim=0, index=idx, src=torch.ones(self.n_e-K).to(device)) for idx in topk_idx]
        topk_mask = torch.stack(topk_mask, dim=0)
        topk_mask = topk_mask.to(device)
        prob = prob.masked_fill(topk_mask == 1, float('-inf'))
        # Convert to multinomial probability
        prob = F.softmax(prob.contiguous().view(B, T, -1) / temperature, dim=-1)
        return prob
    
class QuantizeEMAReset(nn.Module):
    def __init__(
        self, 
        nb_code = 512, 
        code_dim = 512, 
        mu = 0.99, 
        beta=1.0, 
        **kwargs
    ):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu   # 0.99
        self.beta = beta
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        if torch.cuda.is_available():
            self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())
        else:
            self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        N, T, width = x.shape

        # Preprocess
        # x = self.preprocess(x)

        # # Init codebook if not inited
        # if self.training and not self.init:
        #     self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        # if self.training:
        #     perplexity = self.update_codebook(x, code_idx)
        # else : 
        #     perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).contiguous()   #(N, T, DIM)
        
        # compute loss for embedding, there are two terms in the loss: 
        # 1) z_q - z.detach(): pull codebook embeddings to close to encoded motion embedding, and 
        # 2) z_q.detach() - z: pull encoded motion embeddings to close to codebook embeddings.
        loss = torch.mean((x_d - x.detach())**2) + self.beta * \
               torch.mean((x_d.detach() - x)**2)
        
        return loss, x_d, code_idx, 0.0
    
    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        index = self.quantize(x=z)
        return index.squeeze(dim=0)

    def get_codebook_entry(self, indices, is_onhot=False):
        """
        Get the token embeddings by given token indices (onehot distribution)
        :param indices: [B, seq_len] or [B, seq_len, n_dim]
        :return z_q(B, seq_len, e_dim):
        """
        if not is_onhot:
            index_flattened = indices.view(-1)
            z_q = F.embedding(index_flattened, self.codebook)
            z_q = z_q.view(indices.shape + (self.code_dim, )).contiguous()
        else:
            pass
        return z_q
    
    
    
    