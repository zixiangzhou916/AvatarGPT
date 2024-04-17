import os, sys
sys.path.append(os.getcwd())
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (
    LoraConfig, 
    PeftConfig, 
    get_peft_model, 
    prepare_model_for_int8_training, 
    TaskType, 
    set_peft_model_state_dict
)
from peft.tuners.lora import LoraLayer
from transformers import (
    GenerationConfig, 
    LlamaTokenizer, 
    # LlamaForCausalLM, 
    LlamaConfig, 
    BitsAndBytesConfig
)
import math
from typing import List, Optional, Tuple, Union

from networks.utils.model_utils import *
from networks.encodec import EncodecModel
from networks.encodec.utils import convert_audio
from networks.llama.llama_heads import build_llama_heads
from networks.llama.modeling_llama import LlamaForCausalLM
# from networks.llama.quantization_config import BitsAndBytesConfig

def load_partial_parameters(model, checkpoint, logger=None):
    loaded_params = dict()
    for name, val in checkpoint.items():
        name_new = name.replace('module.', '') if 'module.' in name else name
        loaded_params[name_new] = val
                
    model_params = dict()
    num_condition_encoder = 0
    for name, val in model.state_dict().items():
        name_new = name.replace('module.', '') if 'module.' in name else name
        model_params[name_new] = val

    valid_params = dict()
    valid_num_condition_encoder = 0
    for src_name, src_val in loaded_params.items():
        if src_name not in model_params.keys():
            continue
        src_val_shape = ', '.join(map(str, src_val.size()))
        dst_val = model_params[src_name]
        dst_val_shape = ', '.join(map(str, dst_val.size()))
        if src_val_shape != dst_val_shape:
            print("shape of {:s} does not match: {:s} <-> {:s}".format(src_name, src_val_shape, dst_val_shape))
            continue
        suffix = 'module.' if hasattr(model, "module") else ''
        valid_params[suffix + src_name] = src_val
            
    return valid_params

def top_k_logits(logits, k):
    """
    :param logits: [num_seq, num_dim]
    """
    dim = logits.dim()
    if dim == 3:
        logits = logits.squeeze(dim=1)
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    if dim == 3:
        out = out.unsqueeze(dim=1)
    return out

def print_log(logger, log_str):
    if logger is None:
        print(log_str)
    else:
        logger.info(log_str)

class AvatarGPT(nn.Module):
    def __init__(
        self, conf, logger=None, m_quantizer=None, a_quantizer=None
    ):
        super(AvatarGPT, self).__init__()
        self.model_dtype = conf.get("model_dtype", "float32")

        self.conf = conf
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Build LLaMA tokenizer
        print_log(logger=logger, log_str="Building LLaMA tokenizer")
        self.tokenizer = LlamaTokenizer.from_pretrained(conf["tokenizer"], add_eos_token=conf.get("add_eos_token", False))
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        self.tokenizer.padding_side = "right"
        
        print_log(logger=logger, log_str='Special token <SOS>: {:d}'.format(self.tokenizer.bos_token_id))
        print_log(logger=logger, log_str='Special token <EOS>: {:d}'.format(self.tokenizer.eos_token_id))
        print_log(logger=logger, log_str='Special token <PAD>: {:d}'.format(self.tokenizer.pad_token_id))
            
        # Build LLaMA model
        print_log(logger=logger, log_str="Building LLaMA model")

        # torch.set_default_tensor_type(torch.HalfTensor)
        if self.conf.get("quantization", None) is None:
            llama_model = LlamaForCausalLM.from_pretrained(conf["model"])
        else:
            quant_conf = self.conf.get("quantization")
            llama_model = LlamaForCausalLM.from_pretrained(
                conf["model"], 
                load_in_4bit=quant_conf.get("load_in_4bit", True), 
                load_in_8bit=quant_conf.get("load_in_8bit", False), 
            )
        
        # llama_model = LlamaForCausalLM.from_pretrained(conf["model"], torch_dtype=torch.float16)
        # torch.set_default_tensor_type(torch.FloatTensor)
        num_token_embeddings = len(self.tokenizer)
        llama_model.resize_token_embeddings(num_token_embeddings)
        # Prepare model for int8 training.
        llama_model = prepare_model_for_int8_training(llama_model, output_embedding_layer_name="not-used")
        
        # Define LoRA configuration file
        self.lora_config = LoraConfig(
            **conf["lora_config"]
        )
        # Add LoRA adaptor
        # self.llama_model = llama_model.to(self.device)
        if self.model_dtype == "float32":
            self.llama_model = get_peft_model(llama_model, self.lora_config).to(self.device)
        elif self.model_dtype == "float16":
            self.llama_model = get_peft_model(llama_model, self.lora_config).bfloat16().to(self.device)
        self.llama_model.print_trainable_parameters()
        
        for name, module in self.llama_model.named_modules():
            if isinstance(module, LoraLayer):
                if self.model_dtype == "float16":
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if self.model_dtype == "float16" and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
                    elif self.model_dtype == "float32":
                        module = module.to(torch.float32)
                
        # Learnable tokens
        if self.model_dtype == "float32":
            self.token_embed = nn.Embedding(num_embeddings=3, embedding_dim=conf["d_embed"])
        elif self.model_dtype == "float16":
            self.token_embed = nn.Embedding(num_embeddings=3, embedding_dim=conf["d_embed"]).bfloat16()
            
        # LLM linears
        self.lm_linears = nn.ModuleDict()
        for name, opt in self.conf["linear"].items():
            if self.model_dtype == "float32":
                self.lm_linears[name] = build_llama_heads(**opt)
            elif self.model_dtype == "float16":
                self.lm_linears[name] = build_llama_heads(**opt).bfloat16()
            
        # LLM headers
        self.lm_heads = nn.ModuleDict()
        for name, opt in self.conf["head"].items():
            if self.model_dtype == "float32":
                self.lm_heads[name] = build_llama_heads(**opt)
            elif self.model_dtype == "float16":
                self.lm_heads[name] = build_llama_heads(**opt).bfloat16()
    
    def set_quantizer(self, quantizer, type="audio"):
        assert type in ["motion_t", "motion_a", "audio"]
        
        use_learnable_embedding = self.conf.get("use_learnabled_embedding", {})
        
        if type == "motion_t" and not use_learnable_embedding.get("motion_t", True):
            if self.model_dtype == "float32":
                self.motion_quantizer_t = copy.deepcopy(quantizer)
            elif self.model_dtype == "float16":
                self.motion_quantizer_t = copy.deepcopy(quantizer)
                for name, module in self.motion_quantizer_t.named_modules():
                    if self.model_dtype == "float16":
                        module = module.to(torch.bfloat16)
            for p in self.motion_quantizer_t.parameters():
                p.requires_grad = False
        elif type == "motion_a" and not use_learnable_embedding.get("motion_a", True):
            if self.model_dtype == "float32":
                self.motion_quantizer_a = copy.deepcopy(quantizer)
            elif self.model_dtype == "float16":
                self.motion_quantizer_a = copy.deepcopy(quantizer)
                for name, module in self.motion_quantizer_a.named_modules():
                    if self.model_dtype == "float16":
                        module = module.to(torch.bfloat16)
            for p in self.motion_quantizer_a.parameters():
                p.requires_grad = False
        elif type == "audio" and not use_learnable_embedding.get("audio", True):
            if self.model_dtype == "float32":
                self.audio_quantizer = copy.deepcopy(quantizer)
            elif self.model_dtype == "float16":
                self.audio_quantizer = copy.deepcopy(quantizer)
                for name, module in self.audio_quantizer.named_modules():
                    if self.model_dtype == "float16":
                        module = module.to(torch.bfloat16)
            for p in self.audio_quantizer.parameters():
                p.requires_grad = False
    
    def set_trainables(self):
        for name, param in self.llama_model.named_parameters():
            for t_name in self.conf["trainable_params"]:
                if t_name in name: param.requires_grad = True
    
    def train(self):
        self.llama_model.train()
        self.token_embed.train()
        self.lm_linears.train()
        self.lm_heads.train()

    def eval(self):
        self.llama_model.eval()
        self.token_embed.eval()
        self.lm_linears.eval()
        self.lm_heads.eval()
        
    def get_trainable_parameters(self):
        state_dict = {}
        for key, val in super().state_dict().items():
            if "llama_model" not in key and "motion_quantizer_a" not in key and "motion_quantizer_a" not in key and "audio_quantizer" not in key:
                state_dict[key] = val
        return state_dict
    
    def save_model(self, output_path):
        # Save LoRA model
        self.llama_model.save_pretrained(output_path)
        # Save other trainable parameters
        trainable_parameters = self.get_trainable_parameters()
        torch.save(trainable_parameters, os.path.join(output_path, "trainable.pth"))
        
    def load_model(self, input_path, logger=None, strict=True):
        # Load adapter model parameters
        adapter_model_ckpt = torch.load(os.path.join(input_path, "adapter_model.bin"), map_location=self.device)
        try:
            set_peft_model_state_dict(self.llama_model, adapter_model_ckpt)
            print_log(logger=logger, log_str='LoRA Adapter model parameters loaded from {:s} successfully.'.format(os.path.join(input_path, "adapter_model.bin")))
        except:
            print_log(logger=logger, log_str='LoRA Adapter model parameters initialied!')
        # Load other learnable parameters
        learnable_param_ckpt = torch.load(os.path.join(input_path, "trainable.pth"), map_location=self.device)
        valid_learnable_param = load_partial_parameters(self, learnable_param_ckpt)
        super().load_state_dict(valid_learnable_param, strict=False)
        print_log(logger=logger, log_str='Trainable parameters loaded from {:s} successfully.'.format(os.path.join(input_path, "trainable.pth")))
            
    def forward(self, **kwargs):
        pass
    
    def get_special_token_id(self, token, is_learnable=True):
        assert token in ["sos", "eos", "pad"]
        if token == "sos":
            if is_learnable:
                return 0
            else:
                return self.tokenizer.bos_token_id
        elif token == "eos":
            if is_learnable:
                return 1
            else:
                return self.tokenizer.eos_token_id
        elif token == "pad":
            if is_learnable:
                return 2
            else:
                return self.tokenizer.pad_token_id
    
    @staticmethod
    def generate_prompts(task, num_prompts=1):
        assert task in ["t2m", "tm2m", "m2t", "m2m", "a2m", "am2m", "m2a"]
        if task == "t2m":       # Text-to-Motion
            prompts = ["Generate a sequence of motion tokens matching the following natural language description: "] * num_prompts
            return prompts
        elif task == "tm2m":    # Text-Motion-to-Motion
            prompts_1 = ["Generate a sequence of motion tokens following the given motion tokens: "] * num_prompts
            prompts_2 = [", and matching the following natural language description: "] * num_prompts
            return prompts_1, prompts_2
        elif task == "m2t":     # Motion-to-Text
            prompts = ["Describe the motion demonstrated by the following motion token sequence: "] * num_prompts
            return prompts
        elif task == "a2m":     # Audio-to-Motion
            prompts = ["Generate a sequence of motion tokens matching the following audio sequence: "] * num_prompts
            return prompts
        elif task == "am2m":    # Audio-Motion-to-Motion
            prompts_1 = ["Generate a sequence of motion tokens matching the following audio sequence: "] * num_prompts
            prompts_2 = [", and following the given motion tokens: "] * num_prompts
            return prompts_1, prompts_2
        elif task == "m2a":     # Motion-to-Audio
            prompts = ["Compose an audio sequence following the same choreography demonstrated by following motion tokens: "] * num_prompts
            return prompts
        elif task == "t2t":
            prompts = ["Predict the next action status given scene information and current action status: "] * num_prompts
            return prompts
        else:
            raise ValueError

    @staticmethod
    def calc_prediction_accuracy(pred, target, ignore_cls, model_dtype):
        if model_dtype == "float32":
            acc_mask = pred.eq(target).float()
            valid_mask = target.ne(ignore_cls).float()
        elif model_dtype == "float16":
            acc_mask = pred.eq(target).bfloat16()
            valid_mask = target.ne(ignore_cls).bfloat16()
        accuracy = acc_mask.sum() / valid_mask.sum()
        return accuracy
    
    @staticmethod
    def update_predicted_tokens(inp_tokens, special_token, shifting_token=None):
        """
        :param inp_tokens: [T, 1] or [T, N, 1]
        :param special_token: integer
        :param shifting_token: integer
        """
        if inp_tokens.dim() == 2:
            mask = inp_tokens.gt(special_token).squeeze(dim=-1) # [T]
            out_tokens = inp_tokens[mask]   # [T, 1]
            if shifting_token is not None:
                out_tokens -= shifting_token
        elif inp_tokens.dim() == 3:
            out_tokens = []
            for i in range(inp_tokens.size(1)):
                mask = inp_tokens[:, i].gt(special_token).squeeze(dim=-1)   # [T]
                out_token = inp_tokens[mask, i]
                if shifting_token is not None:
                    out_token -= shifting_token
                out_tokens.append(out_token)
            out_tokens = torch.stack(out_tokens, dim=1)
        return out_tokens   # [T, 1] or [T, N, 1]
    
    @staticmethod
    def prepare_generation_tokens(
        inp_tokens, sos, eos, pad, targ_length, 
        preparation_type="bidirectional", 
        shifting_token=None
    ):
        """
        :param inp_tokens: [T, 1] or [T, N, 1]
        """
        assert preparation_type in ["pre", "post", "bidirectional"]
        device = inp_tokens.device
        sos_token = torch.tensor(sos).long().to(device).view(1, 1)
        eos_token = torch.tensor(eos).long().to(device).view(1, 1)
        pad_token = torch.tensor(pad).long().to(device).view(1, 1)
        
        out_tokens = inp_tokens.clone()
        if shifting_token is not None:
            out_tokens += shifting_token
        
        if preparation_type == "pre":
            if out_tokens.dim() == 2:
                out_tokens = torch.cat([sos_token, out_tokens], dim=0)
            elif out_tokens.dim() == 3:
                out_tokens = torch.cat([sos_token.view(1, 1, 1).repeat(1, out_tokens.size(1), 1), out_tokens], dim=0)
        elif preparation_type == "post":
            if out_tokens.dim() == 2:
                out_tokens = torch.cat([out_tokens, eos_token], dim=0)
            elif out_tokens.dim() == 3:
                out_tokens = torch.cat([out_tokens, eos_token.view(1, 1, 1).repeat(1, out_tokens.size(1), 1)], dim=0)
        elif preparation_type == "bidirectional":
            if out_tokens.dim() == 2:
                out_tokens = torch.cat([sos_token, out_tokens, eos_token], dim=0)
            elif out_tokens.dim() == 3:
                out_tokens = torch.cat([sos_token.view(1, 1, 1).repeat(1, out_tokens.size(1), 1), 
                                        out_tokens, 
                                        eos_token.view(1, 1, 1).repeat(1, out_tokens.size(1), 1)], dim=0)
        
        mot_len = out_tokens.size(0)
        pad_len = targ_length - mot_len
        if pad_len > 0:
            if out_tokens.dim() == 2:
                pad_token = pad_token.repeat(pad_len, 1)
            elif out_tokens.dim() == 3:
                pad_token = pad_token.view(1, 1, 1).repeat(pad_len, out_tokens.size(1), 1)
            out_tokens = torch.cat([out_tokens, pad_token], dim=0)
        return out_tokens
    
    def generate_m2t_labels(self, inp_labels, mask):
        ignore_cls = self.get_special_token_id("pad", is_learnable=False)
        mask = inp_labels.eq(ignore_cls)
        out_labels = inp_labels.masked_fill(mask, -100)
        return out_labels
    
    def generate_t2m_labels(self, inp_labels, mask):
        ignore_cls = self.get_special_token_id("pad", is_learnable=True)
        mask = inp_labels.eq(ignore_cls)
        out_labels = inp_labels.masked_fill(mask, -100)
        return out_labels
    
    def generate_a2m_labels(self, inp_labels, mask):
        """Behaves exactly as t2m."""
        return self.generate_t2m_labels(inp_labels, mask)
    
    def generate_m2a_labels(self, inp_labels, mask):
        ignore_cls = self.get_special_token_id("pad", is_learnable=True)
        eos_cls = self.get_special_token_id("eos", is_learnable=True)
        valid_mask = inp_labels.gt(eos_cls)
        ignore_mask = inp_labels.eq(ignore_cls)
        out_labels = inp_labels.clone()
        out_labels[valid_mask] -= self.conf["tokens"]["motion"]
        out_labels = out_labels.masked_fill(ignore_mask, -100)
        return out_labels
    
    def generate_embeddings_from_motion_tokens(
        self, inp_tokens, sos, eos, pad, 
        padding_type="bidirectional", 
        targ_len=30, 
        shifting_token=None, 
        tokenizer="motion_t"
    ):
        """
        :param inp_tokens: [seq_len]
        """
        assert tokenizer in ["motion_t", "motion_a"]
        
        mask = inp_tokens.gt(pad)
        tokens = inp_tokens[mask]
        
        if shifting_token is not None and tokens.size(0) != 0:
            tokens += shifting_token
        
        if tokens.size(0) != 0:
            if tokenizer == "motion_t":
                if hasattr(self, "motion_quantizer_t"):
                    embeddings = self.motion_quantizer_t.get_codebook_entry(tokens)   # [T, C]
                    # embeddings = self.lm_linears["motion"](embeddings)
                    embeddings = self.lm_linears[tokenizer](embeddings)
                else:
                    embeddings = self.lm_linears[tokenizer](tokens)
            elif tokenizer == "motion_a":
                if hasattr(self, "motion_quantizer_a"):
                    embeddings = self.motion_quantizer_a.get_codebook_entry(tokens)   # [T, C]
                    # embeddings = self.lm_linears["motion"](embeddings)
                    embeddings = self.lm_linears[tokenizer](embeddings)
                else:
                    embeddings = self.lm_linears[tokenizer](tokens)
        else:
            if self.model_dtype == "float32":
                embeddings = torch.empty(0, self.conf["d_embed"]).float().to(self.device)
            elif self.model_dtype == "float16":
                embeddings = torch.empty(0, self.conf["d_embed"]).bfloat16().to(self.device)
            
        if padding_type == "pre":
            sos_embedding = self.token_embed(torch.tensor(sos).long().to(self.device)).contiguous().view(1, -1)
            embeddings = torch.cat([sos_embedding, embeddings], dim=0)
        elif padding_type == "post":
            eos_embedding = self.token_embed(torch.tensor(eos).long().to(self.device)).contiguous().view(1, -1)
            embeddings = torch.cat([embeddings, eos_embedding], dim=0)
        elif padding_type == "bidirectional":
            sos_embedding = self.token_embed(torch.tensor(sos).long().to(self.device)).contiguous().view(1, -1)
            eos_embedding = self.token_embed(torch.tensor(eos).long().to(self.device)).contiguous().view(1, -1)
            embeddings = torch.cat([sos_embedding, embeddings, eos_embedding], dim=0)
        
        emb_len = embeddings.size(0)
        if emb_len < targ_len:
            pad_embedding = self.token_embed(torch.tensor(pad).long().to(self.device)).contiguous().view(1, -1)
            embeddings = torch.cat([embeddings, pad_embedding.repeat(targ_len-emb_len, 1)], dim=0)
        
        attn_mask = torch.zeros(embeddings.size(0)).long().to(self.device)
        attn_mask[:emb_len] = 1
        
        return embeddings, attn_mask
    
    def generate_embeddings_from_audio_tokens(
        self, inp_tokens, sos, eos, pad, 
        padding_type="bidirectional", 
        targ_len=30, 
        shifting_token=None
    ):
        """
        :param inp_tokens: [seq_len] or [n_channels, seq_len]
        """
        
        def controllable_padding(embeddings, sos, eos, pad, padding_type, targ_len):
            if padding_type == "pre":
                sos_embedding = self.token_embed(torch.tensor(sos).long().to(self.device)).contiguous().view(1, -1)
                embeddings = torch.cat([sos_embedding, embeddings], dim=0)
            elif padding_type == "post":
                eos_embedding = self.token_embed(torch.tensor(eos).long().to(self.device)).contiguous().view(1, -1)
                embeddings = torch.cat([embeddings, eos_embedding], dim=0)
            elif padding_type == "bidirectional":
                sos_embedding = self.token_embed(torch.tensor(sos).long().to(self.device)).contiguous().view(1, -1)
                eos_embedding = self.token_embed(torch.tensor(eos).long().to(self.device)).contiguous().view(1, -1)
                embeddings = torch.cat([sos_embedding, embeddings, eos_embedding], dim=0)

            emb_len = embeddings.size(0)
            if emb_len < targ_len:
                pad_embedding = self.token_embed(torch.tensor(pad).long().to(self.device)).contiguous().view(1, -1)
                embeddings = torch.cat([embeddings, pad_embedding.repeat(targ_len-emb_len, 1)], dim=0)
            
            attn_mask = torch.zeros(embeddings.size(0)).long().to(self.device)
            attn_mask[:emb_len] = 1    
            
            return embeddings, attn_mask
        
        if inp_tokens.dim() == 1:
            mask = inp_tokens.gt(pad)
            tokens = inp_tokens[mask]
            
            if shifting_token is not None:
                tokens += shifting_token
            
            # Replace 'audio_quantizer' with 'audio_quantizer'
            if hasattr(self, "audio_quantizer"):
                embeddings = self.audio_quantizer.decode(tokens.contiguous().view(1, 1, -1))   # [1, 128, T]
                embeddings = self.lm_linears["audio"](embeddings.view(128, -1).t())             # [T, C]
            else:
                embeddings = self.lm_linears["audio"](tokens)
            embeddings, attn_masks = controllable_padding(embeddings, sos, eos, pad, padding_type, targ_len)
        elif inp_tokens.dim() == 2:
            
            tokens = []
            for inp_tok in inp_tokens:
                mask = inp_tok.gt(pad)
                tokens.append(inp_tok[mask])
            tokens = torch.stack(tokens, dim=0)
            
            if shifting_token is not None:
                tokens += shifting_token
            
            N, T = tokens.shape
            if hasattr(self, "audio_quantizer"):
                embeddings = self.audio_quantizer.decode(tokens.contiguous().view(N, 1, T)) # [1, 128, T]
                embeddings = self.lm_linears["audio"](embeddings.view(128, -1).t())               # [T, C]
            else:
                embeddings = self.lm_linears["audio"](tokens)
            embeddings, attn_masks = controllable_padding(embeddings, sos, eos, pad, padding_type, targ_len)
                        
        else:
            raise ValueError
        
        return embeddings, attn_masks
    
    def prepare_training_embeddings(
        self, inp_tokens, sos, eos, pad, 
        token_type="motion_t", 
        padding_type="bidirectional", 
        targ_len=30, 
        shifting_token=None
    ):
        """
        :param inp_tokens: [batch_size, seq_len] or [batch_size, n_channels, seq_len]
        """
        assert token_type in ["motion_t", "motion_a", "audio"]
        if "motion" in token_type:
            embeddings, attn_masks = [], []
            for x in  inp_tokens:
                embed, attn_mask = self.generate_embeddings_from_motion_tokens(
                    x, sos, eos, pad, 
                    padding_type=padding_type, 
                    targ_len=targ_len, 
                    tokenizer=token_type, 
                    shifting_token=shifting_token)
                embeddings.append(embed)
                attn_masks.append(attn_mask)
            embeddings = torch.stack(embeddings, dim=0) # [B, T, C]
            attn_masks = torch.stack(attn_masks, dim=0) # [B, T]
        elif "audio" in token_type:
            embeddings, attn_masks = [], []
            for x in inp_tokens:
                embed, attn_mask = self.generate_embeddings_from_audio_tokens(
                    x, sos, eos, pad, 
                    padding_type=padding_type, 
                    targ_len=targ_len, 
                    shifting_token=shifting_token)
                embeddings.append(embed)
                attn_masks.append(attn_mask)
            embeddings = torch.stack(embeddings, dim=0) # [B, T, C]
            attn_masks = torch.stack(attn_masks, dim=0) # [B, T]
        else:
            raise ValueError
        
        return embeddings, attn_masks
    
    def generate_audio_to_motion_once(
        self, a_tokens, m_tokens, topk=1, max_num_tokens=1
    ):
        """
        :param a_tokens: [batch_size, seq_len(audio)] or [batch_size, n_channels, seq_len(audio)], with <SOS>, <EOS>, <PAD> appended
        :param m_tokens: [batch_size, seq_len(motion)], with <SOS> appended
        """
        # Generate prompts
        prompts = self.generate_prompts(task="a2m", num_prompts=len(a_tokens))
        
        # Tokenize the prompts
        prompt_tokens = []
        prompt_mask = []
        for pro in prompts:
            tokenizer_outputs = self.tokenizer(pro, return_tensors="pt")
            tokens = tokenizer_outputs.input_ids.to(self.device)
            mask = tokenizer_outputs.attention_mask.long().to(self.device)
            prompt_tokens.append(tokens)
            prompt_mask.append(mask)
        prompt_tokens = torch.cat(prompt_tokens, dim=0)
        prompt_mask = torch.cat(prompt_mask, dim=0)
        
        # Get embeddings from prompt tokens
        input_embeddings = self.llama_model.get_input_embeddings()
        prompt_emb = input_embeddings(prompt_tokens)
        
        # Get embeddings from audio tokens
        audio_emb, audio_mask = self.prepare_training_embeddings(
            inp_tokens=a_tokens, 
            sos=self.get_special_token_id("sos", is_learnable=True), 
            eos=self.get_special_token_id("eos", is_learnable=True), 
            pad=self.get_special_token_id("pad", is_learnable=True), 
            token_type="audio", padding_type="bidirectional", 
            targ_len=a_tokens.size(-1), 
            shifting_token=-(self.conf["tokens"]["motion"]+3)
        )
        
        # Get embeddings from motion tokens
        if m_tokens is None:
            m_tokens = torch.tensor(self.get_special_token_id("sos", is_learnable=True)).view(1, 1).long().to(self.device)
        motion_emb, motion_mask = self.prepare_training_embeddings(
            inp_tokens=m_tokens, token_type="motion_a", 
            sos=self.get_special_token_id("sos", is_learnable=True), 
            eos=self.get_special_token_id("eos", is_learnable=True), 
            pad=self.get_special_token_id("pad", is_learnable=True), 
            padding_type="pre", 
            targ_len=m_tokens.size(-1), 
            shifting_token=-3   # We need to shift the tokens by -3 because we have shifted it by +3 previously.
        )

        # Get input embeddings and attention mask
        embeddings = torch.cat([prompt_emb, audio_emb, motion_emb], dim=1)
        attn_masks = torch.cat([prompt_mask, audio_mask, motion_mask], dim=1)
        cond_len = prompt_emb.size(1) + audio_emb.size(1)
        
        pred_tokens = []
        for _ in range(max_num_tokens):
            outputs = self.llama_model(
                inputs_embeds=embeddings, 
                attention_mask=attn_masks, 
                output_hidden_states=True, 
                output_attentions=True)
            last_hidden_state = outputs.hidden_states[-1]
            pred_logits = self.lm_heads["motion_a"](last_hidden_state[:, -1:])    # [1, 1, D]
            if topk == 1:
                pred_logits = F.softmax(pred_logits, dim=-1)
                pred_token = pred_logits.argmax(dim=-1)
            else:
                pred_logits = top_k_logits(pred_logits, k=topk)
                pred_logits = F.softmax(pred_logits, dim=-1)
                # pred_token = pred_logits.argmax(dim=-1)
                pred_token = torch.multinomial(pred_logits[:, 0], num_samples=1)
            
            # Make sure <EOS> is not sampled between max_num_tokens is met!
            while pred_token == self.get_special_token_id("eos", is_learnable=True) or \
                pred_token == self.get_special_token_id("pad", is_learnable=True) or \
                    pred_token == self.get_special_token_id("sos", is_learnable=True):
                
                pred_logits = top_k_logits(pred_logits, k=10)
                pred_logits = F.softmax(pred_logits, dim=-1)
                pred_token = torch.multinomial(pred_logits[:, 0], num_samples=1)
            
            pred_tokens.append(pred_token)
            
            # Update input embeddings and attention masks
            pred_emb, pred_mask = self.prepare_training_embeddings(
                pred_token, token_type="motion_a", 
                sos=self.get_special_token_id("sos", is_learnable=True), 
                eos=self.get_special_token_id("eos", is_learnable=True), 
                pad=self.get_special_token_id("pad", is_learnable=True), 
                padding_type="none", 
                targ_len=1, 
                shifting_token=-3)
            embeddings = torch.cat([embeddings, pred_emb], dim=1)
            attn_masks = torch.cat([attn_masks, pred_mask], dim=1)
        
        pred_tokens = torch.cat(pred_tokens, dim=1) # [1, T]
        return pred_tokens.t()                      # [T, 1]
    
    def generate_motion_to_audio_once(
        self, m_tokens, a_tokens, topk=1, max_num_tokens=1
    ):
        """
        :param m_tokens: [batch_size, seq_len(motion)], with <SOS>, <EOS>, <PAD> appended
        :param a_tokens: (optional) [batch_size, seq_len(audio)] or [batch_size, n_channels, seq_len(audio)], with <SOS>, <EOS>, <PAD> appended
        """
        n_channels = self.conf.get("n_encodec_channels_for_m2a", 1) # Number of channels used for motion-to-audio
        
        # Generate prompts
        prompts = self.generate_prompts(task="m2a", num_prompts=len(a_tokens))
        
        # Tokenize the prompts
        prompt_tokens = []
        prompt_mask = []
        for pro in prompts:
            tokenizer_outputs = self.tokenizer(pro, return_tensors="pt")
            tokens = tokenizer_outputs.input_ids.to(self.device)
            mask = tokenizer_outputs.attention_mask.long().to(self.device)
            prompt_tokens.append(tokens)
            prompt_mask.append(mask)
        prompt_tokens = torch.cat(prompt_tokens, dim=0)
        prompt_mask = torch.cat(prompt_mask, dim=0)
        
        # Get embeddings from prompt tokens
        input_embeddings = self.llama_model.get_input_embeddings()
        prompt_emb = input_embeddings(prompt_tokens)
        
        # Get embeddings from motion tokens
        motion_emb, motion_mask = self.prepare_training_embeddings(
            inp_tokens=m_tokens, token_type="motion_a", 
            sos=self.get_special_token_id("sos", is_learnable=True), 
            eos=self.get_special_token_id("eos", is_learnable=True), 
            pad=self.get_special_token_id("pad", is_learnable=True), 
            padding_type="bidirectional", 
            targ_len=m_tokens.size(-1), 
            shifting_token=-3   # We need to shift the tokens by -3 because we have shifted it by +3 previously.
        )
        
        # Get embeddings from audio tokens
        audio_emb, audio_mask = self.prepare_training_embeddings(
            inp_tokens=a_tokens if a_tokens.dim() == 2 else a_tokens[:, :n_channels], 
            sos=self.get_special_token_id("sos", is_learnable=True), 
            eos=self.get_special_token_id("eos", is_learnable=True), 
            pad=self.get_special_token_id("pad", is_learnable=True), 
            token_type="audio", padding_type="bidirectional", 
            targ_len=a_tokens.size(-1), 
            shifting_token=-(self.conf["tokens"]["motion"]+3)
        )
        
        # Get input embeddings and attention mask
        embeddings = torch.cat([prompt_emb, motion_emb, audio_emb], dim=1)
        attn_masks = torch.cat([prompt_mask, motion_mask, audio_mask], dim=1)
        cond_len = prompt_emb.size(1) + motion_emb.size(1)
        
        pred_tokens = []
        for _ in range(max_num_tokens):
            outputs = self.llama_model(inputs_embeds=embeddings, 
                                        attention_mask=attn_masks, 
                                        output_hidden_states=True, 
                                        output_attentions=True)
            last_hidden_state = outputs.hidden_states[-1]
            pred_logits = self.lm_heads["audio"](last_hidden_state[:, -1:])
            if topk == 1:
                pred_logits = F.softmax(pred_logits, dim=-1)
                pred_token = pred_logits.argmax(dim=-1)
            else:
                if pred_logits.dim() == 3:
                    pred_logits = top_k_logits(pred_logits, k=topk)
                else:
                    pred_logits = [top_k_logits(pred_logits[:, i], k=topk) for i in range(pred_logits.size(1))]
                    pred_logits = torch.stack(pred_logits, dim=1)
                pred_logits = F.softmax(pred_logits, dim=-1)
                if pred_logits.dim() == 3:
                    pred_token = torch.multinomial(pred_logits[:, 0], num_samples=1)    # [1, T] or [1, N, T]
                else:
                    pred_token = [torch.multinomial(pred_logits[:, i, 0], num_samples=1) for i in range(pred_logits.size(1))]
                    pred_token = torch.stack(pred_token, dim=1)
            
            # Make sure <EOS> is not sampled between max_num_tokens is met!
            while torch.any(pred_token.eq(self.get_special_token_id("sos", is_learnable=True))) or \
                torch.any(pred_token.eq(self.get_special_token_id("eos", is_learnable=True))) or \
                    torch.any(pred_token.eq(self.get_special_token_id("pad", is_learnable=True))):
                
                pred_logits[..., :3] *= 0.0
                if pred_token.dim() == 2:
                    pred_token = torch.multinomial(pred_logits[:, 0], num_samples=1)
                elif pred_token.dim() == 3:
                    for j in range(pred_token.size(1)):
                        pred_token[:, j] = torch.multinomial(pred_logits[:, j, 0], num_samples=1)
            pred_tokens.append(pred_token)
            
            # Update input embeddings and attention masks
            pred_emb, pred_mask = self.prepare_training_embeddings(
                pred_token, token_type="audio", 
                sos=self.get_special_token_id("sos", is_learnable=True), 
                eos=self.get_special_token_id("eos", is_learnable=True), 
                pad=self.get_special_token_id("pad", is_learnable=True), 
                padding_type="none", 
                targ_len=1, 
                shifting_token=-3)
            embeddings = torch.cat([embeddings, pred_emb], dim=1)
            attn_masks = torch.cat([attn_masks, pred_mask], dim=1)
            
        pred_tokens = torch.cat(pred_tokens, dim=-1) # [1, T] or [1, N, T]
        if pred_tokens.dim() == 2:
            return pred_tokens.t()                      # [T, 1]
        else:
            return pred_tokens.permute(2, 1, 0)         # [T, N, 1]
    
    def text_to_motion(self, texts, m_tokens, loss_type=["pred"]):
        """
        :param texts: list of strings.
        :param m_tokens: [batch_size, seq_len] with <SOS>, <EOS>, and <PAD> appended.
        """
        # Generate prompts
        prompts = self.generate_prompts(task="t2m", num_prompts=len(texts))
        texts_w_propmts = [t1 + t2 for (t1, t2) in zip(prompts, texts)]

        # Tokenize the input text prompts.
        text_tokenizer_outputs = self.tokenizer.batch_encode_plus(texts_w_propmts, **self.conf["tokenizer_config"])
        text_tokens = text_tokenizer_outputs["input_ids"]
        text_mask = text_tokenizer_outputs["attention_mask"].squeeze().long().to(self.device)
        
        # Get embeddings from text tokens
        input_embeddings = self.llama_model.get_input_embeddings()
        text_emb = input_embeddings(text_tokens.long().to(self.device))
    
        # Get embeddings from motion tokens
        motion_emb, motion_mask = self.prepare_training_embeddings(
            inp_tokens=m_tokens, token_type="motion_t", 
            sos=self.get_special_token_id("sos", is_learnable=True), 
            eos=self.get_special_token_id("eos", is_learnable=True), 
            pad=self.get_special_token_id("pad", is_learnable=True), 
            padding_type="bidirectional", 
            targ_len=m_tokens.size(-1), 
            shifting_token=-3   # We need to shift the tokens by -3 because we have shifted it by +3 previously.
        )
        
        # Get input embeddings and attention mask
        embeddings = torch.cat([text_emb, motion_emb], dim=1)
        attn_masks = torch.cat([text_mask, motion_mask], dim=1)
        
        outputs = self.llama_model(
            inputs_embeds=embeddings, 
            attention_mask=attn_masks, 
            output_hidden_states=True, 
            output_attentions=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # logits
        cond_logits = outputs.logits[:, :text_emb.size(1), :]
        if "motion" in self.lm_heads.keys():
            pred_logits = self.lm_heads["motion"](last_hidden_state[:, text_emb.size(1):])
        else:
            pred_logits = self.lm_heads["motion_t"](last_hidden_state[:, text_emb.size(1):])

        # Calc losses.
        loss_fn = nn.CrossEntropyLoss()
        
        losses = {}
        accuracy = {}
        # Condition loss
        if "cond" in loss_type:
            pass
        
        # Prediction loss
        if "pred" in loss_type:
            pred_shift_logits = pred_logits[..., :-1, :]
            pred_shift_pred = pred_shift_logits.argmax(dim=-1)
            pred_shift_labels = self.generate_t2m_labels(m_tokens[..., 1:], mask=None)
            losses["pred"] = loss_fn(
                pred_shift_logits.contiguous().view(-1, self.conf["tokens"]["motion"]+3), 
                pred_shift_labels.contiguous().view(-1))
            accuracy["pred"] = self.calc_prediction_accuracy(
                pred_shift_pred, pred_shift_labels, 
                self.get_special_token_id("pad", is_learnable=True), 
                model_dtype=self.model_dtype)
        
        return {"losses": losses, "accuracy": accuracy, "pred_tokens": pred_shift_pred, "target_tokens": pred_shift_labels}
        
    def motion_to_text(self, texts, m_tokens, loss_type=["pred"]):
        
        # Generate prompts
        prompts = self.generate_prompts(task="m2t", num_prompts=len(texts))
        
        # Tokenize the input text and prompts
        # 1. Tokenize input text
        text_tokenizer_outputs = self.tokenizer.batch_encode_plus(texts, **self.conf["tokenizer_config"])
        text_tokens = text_tokenizer_outputs["input_ids"].long().to(self.device)
        text_mask = text_tokenizer_outputs["attention_mask"].squeeze().long().to(self.device)
        # 2. Tokenize prompts
        prompt_tokens = []
        prompt_mask = []
        for prompt in prompts:
            tokenizer_outputs = self.tokenizer(prompt, return_tensors="pt")
            prompt_tokens.append(tokenizer_outputs["input_ids"])
            prompt_mask.append(tokenizer_outputs["attention_mask"])
        prompt_tokens = torch.cat(prompt_tokens, dim=0).long().to(self.device)
        prompt_mask = torch.cat(prompt_mask, dim=0).long().to(self.device)
        # 3. Get text and prompts embedding
        input_embeddings = self.llama_model.get_input_embeddings()
        text_emb = input_embeddings(text_tokens)
        prompt_emb = input_embeddings(prompt_tokens)
        
        # Get embeddings from motion tokens
        motion_emb, motion_mask = self.prepare_training_embeddings(
            inp_tokens=m_tokens, token_type="motion_t", 
            sos=self.get_special_token_id("sos", is_learnable=True), 
            eos=self.get_special_token_id("eos", is_learnable=True), 
            pad=self.get_special_token_id("pad", is_learnable=True), 
            padding_type="bidirectional", 
            targ_len=m_tokens.size(-1), 
            shifting_token=-3   # We need to shift the tokens by -3 because we have shifted it by +3 previously.
        )
        
        # Get input embeddings and attention mask
        embeddings = torch.cat([prompt_emb, motion_emb, text_emb], dim=1)
        attn_masks = torch.cat([prompt_mask, motion_mask, text_mask], dim=1)
        outputs = self.llama_model(
            inputs_embeds=embeddings, 
            attention_mask=attn_masks, 
            output_hidden_states=True, 
            output_attentions=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # logits
        cond_len = prompt_emb.size(1) + motion_emb.size(1)
        cond_logits = outputs.logits[:, :cond_len, :]
        pred_logits = outputs.logits[:, cond_len:, :]
        
        # Calc losses.
        loss_fn = nn.CrossEntropyLoss()
        
        losses = {}
        accuracy = {}
        # Condition loss
        if "cond" in loss_type:
            pass

        # Prediction loss
        if "pred" in loss_type:
            pred_shift_logits = pred_logits[..., :-1, :]
            pred_shift_pred = pred_shift_logits.argmax(dim=-1)
            pred_shift_labels = self.generate_m2t_labels(text_tokens[..., 1:], mask=None)
            losses["pred"] = loss_fn(
                pred_shift_logits.contiguous().view(-1, pred_logits.size(-1)), 
                pred_shift_labels.contiguous().view(-1))
            accuracy["pred"] = self.calc_prediction_accuracy(
                pred_shift_pred, pred_shift_labels, 
                self.get_special_token_id("pad", is_learnable=False), 
                model_dtype=self.model_dtype)
        
        return {"losses": losses, "accuracy": accuracy, "pred_tokens": pred_shift_pred, "target_tokens": pred_shift_labels}
    
    def audio_to_motion(self, a_tokens, m_tokens, cond_m_tokens=None, loss_type=["pred"]):
        """
        :param a_tokens: [batch_size, seq_len(audio)] or [batch_size, n_channels, seq_len(audio)], with <SOS>, <EOS>, <PAD> appended
        :param m_tokens: [batch_size, seq_len(motion)], with <SOS>, <EOS>, <PAD> appended
        :param cond_m_tokens: (optional)[batch_size, seq_len(cond_motion)], with <SOS>, <EOS>, <PAD> appended
        """
        # Generate prompts
        prompts = self.generate_prompts(task="a2m", num_prompts=len(a_tokens))
        
        # Tokenize the prompts
        prompt_tokens = []
        prompt_mask = []
        for pro in prompts:
            tokenizer_outputs = self.tokenizer(pro, return_tensors="pt")
            tokens = tokenizer_outputs.input_ids.to(self.device)
            mask = tokenizer_outputs.attention_mask.long().to(self.device)
            prompt_tokens.append(tokens)
            prompt_mask.append(mask)
        prompt_tokens = torch.cat(prompt_tokens, dim=0)
        prompt_mask = torch.cat(prompt_mask, dim=0)
        
        # Get embeddings from prompt tokens
        input_embeddings = self.llama_model.get_input_embeddings()
        prompt_emb = input_embeddings(prompt_tokens)
        
        # Get embeddings from audio tokens
        audio_emb, audio_mask = self.prepare_training_embeddings(
            inp_tokens=a_tokens, 
            sos=self.get_special_token_id("sos", is_learnable=True), 
            eos=self.get_special_token_id("eos", is_learnable=True), 
            pad=self.get_special_token_id("pad", is_learnable=True), 
            token_type="audio", padding_type="bidirectional", 
            targ_len=a_tokens.size(-1), 
            shifting_token=-(self.conf["tokens"]["motion"]+3)
        )
        
        # Get embeddings from motion tokens
        motion_emb, motion_mask = self.prepare_training_embeddings(
            inp_tokens=m_tokens if cond_m_tokens is None else torch.cat([cond_m_tokens, m_tokens], dim=-1), 
            token_type="motion_a", 
            sos=self.get_special_token_id("sos", is_learnable=True), 
            eos=self.get_special_token_id("eos", is_learnable=True), 
            pad=self.get_special_token_id("pad", is_learnable=True), 
            padding_type="bidirectional", 
            targ_len=m_tokens.size(-1), 
            shifting_token=-3   # We need to shift the tokens by -3 because we have shifted it by +3 previously.
        )

        # Get input embeddings and attention mask
        embeddings = torch.cat([prompt_emb, audio_emb, motion_emb], dim=1)
        attn_masks = torch.cat([prompt_mask, audio_mask, motion_mask], dim=1)
        if cond_m_tokens is None:
            cond_len = prompt_emb.size(1) + audio_emb.size(1)
        else:
            cond_len = prompt_emb.size(1) + audio_emb.size(1) + cond_m_tokens.size(1)

        outputs = self.llama_model(
            inputs_embeds=embeddings, 
            attention_mask=attn_masks, 
            output_hidden_states=True, 
            output_attentions=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # logits
        cond_logits = outputs.logits[:, :cond_len, :]
        pred_logits = self.lm_heads["motion_a"](last_hidden_state[:, cond_len:])
        
        # Calc losses.
        loss_fn = nn.CrossEntropyLoss()
        
        losses = {}
        accuracy = {}
        # Condition loss
        if "cond" in loss_type:
            pass
            
        # Predction loss
        if "pred" in loss_type:
            pred_shift_logits = pred_logits[..., :-1, :]
            pred_shift_pred = pred_shift_logits.argmax(dim=-1)
            pred_shift_labels = self.generate_a2m_labels(m_tokens[..., 1:], mask=None)
            losses["pred"] = loss_fn(
                pred_shift_logits.contiguous().view(-1, self.conf["tokens"]["motion"]+3), 
                pred_shift_labels.contiguous().view(-1))
            accuracy["pred"] = self.calc_prediction_accuracy(
                pred_shift_pred, pred_shift_labels, 
                self.get_special_token_id("pad", is_learnable=True), 
                model_dtype=self.model_dtype)
        
        return {"losses": losses, "accuracy": accuracy, "pred_tokens": pred_shift_pred, "target_tokens": pred_shift_labels}
    
    def motion_to_audio(self, a_tokens, m_tokens, loss_type=["pred"]):
        """
        :param a_tokens: [batch_size, seq_len(audio)] or [batch_size, n_channels, seq_len(audio)]
        :param m_tokens: [batch_size, seq_len(motion)]
        """
        n_channels = self.conf.get("n_encodec_channels_for_m2a", 1) # Number of channels used for motion-to-audio
        
        # Generate prompts
        prompts = self.generate_prompts(task="m2a", num_prompts=len(a_tokens))
        
        # Tokenize the prompts
        prompt_tokens = []
        prompt_mask = []
        for pro in prompts:
            tokenizer_outputs = self.tokenizer(pro, return_tensors="pt")
            tokens = tokenizer_outputs.input_ids.to(self.device)
            mask = tokenizer_outputs.attention_mask.long().to(self.device)
            prompt_tokens.append(tokens)
            prompt_mask.append(mask)
        prompt_tokens = torch.cat(prompt_tokens, dim=0)
        prompt_mask = torch.cat(prompt_mask, dim=0)
        
        # Get embeddings from prompt tokens
        input_embeddings = self.llama_model.get_input_embeddings()
        prompt_emb = input_embeddings(prompt_tokens)
        
        # Get embeddings from motion tokens
        motion_emb, motion_mask = self.prepare_training_embeddings(
            inp_tokens=m_tokens, token_type="motion_a", 
            sos=self.get_special_token_id("sos", is_learnable=True), 
            eos=self.get_special_token_id("eos", is_learnable=True), 
            pad=self.get_special_token_id("pad", is_learnable=True), 
            padding_type="bidirectional", 
            targ_len=m_tokens.size(-1), 
            shifting_token=-3   # We need to shift the tokens by -3 because we have shifted it by +3 previously.
        )
        
        # Get embeddings from audio tokens
        audio_emb, audio_mask = self.prepare_training_embeddings(
            inp_tokens=a_tokens if a_tokens.dim() == 2 else a_tokens[:, :n_channels], 
            sos=self.get_special_token_id("sos", is_learnable=True), 
            eos=self.get_special_token_id("eos", is_learnable=True), 
            pad=self.get_special_token_id("pad", is_learnable=True), 
            token_type="audio", padding_type="bidirectional", 
            targ_len=a_tokens.size(-1), 
            shifting_token=-(self.conf["tokens"]["motion"]+3)
        )
        
        # Get input embeddings and attention mask
        embeddings = torch.cat([prompt_emb, motion_emb, audio_emb], dim=1)
        attn_masks = torch.cat([prompt_mask, motion_mask, audio_mask], dim=1)
        cond_len = prompt_emb.size(1) + motion_emb.size(1)
        
        outputs = self.llama_model(
            inputs_embeds=embeddings, 
            attention_mask=attn_masks, 
            output_hidden_states=True, 
            output_attentions=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # logits
        cond_logits = outputs.logits[:, :cond_len, :]
        if hasattr(self.lm_heads["audio"], "num_residual_layers"):
            pred_logits = self.lm_heads["audio"](last_hidden_state[:, cond_len:], 
                                                 targ=self.generate_m2a_labels(a_tokens.clone(), mask=None))
        else:
            pred_logits = self.lm_heads["audio"](last_hidden_state[:, cond_len:])   # [B, T, C] or [B, N, T, C]

        loss_fn = nn.CrossEntropyLoss()
        
        losses = {}
        accuracy = {}
        
        # Condition loss
        if "cond" in loss_type:
            pass
        
        # Prediction loss
        if "pred" in loss_type:
            pred_shift_logits = pred_logits[..., :-1, :]
            pred_shift_pred = pred_shift_logits.argmax(dim=-1)
            pred_shift_labels = self.generate_m2a_labels(a_tokens[..., 1:], mask=None)
            if pred_logits.dim() == 3:
                losses["pred"] = loss_fn(
                    pred_shift_logits.contiguous().view(-1, self.conf["tokens"]["audio"]+3), 
                    pred_shift_labels[:, 0].contiguous().view(-1))
                accuracy["pred"] = self.calc_prediction_accuracy(
                    pred_shift_pred.contiguous().view(-1), 
                    pred_shift_labels[:, 0].contiguous().view(-1), 
                    self.get_special_token_id("pad", is_learnable=True), 
                    model_dtype=self.model_dtype)
            elif pred_logits.dim() == 4:
                n_q = pred_logits.size(1)
                losses["pred"] = loss_fn(
                    pred_shift_logits.contiguous().view(-1, self.conf["tokens"]["audio"]+3), 
                    pred_shift_labels[:, :n_q].contiguous().view(-1))
                accuracy["pred"] = self.calc_prediction_accuracy(
                    pred_shift_pred.contiguous().view(-1), 
                    pred_shift_labels[:, :n_q].contiguous().view(-1), 
                    self.get_special_token_id("pad", is_learnable=True), 
                    model_dtype=self.model_dtype)
        
        return {"losses": losses, "accuracy": accuracy, "pred_tokens": pred_shift_pred, "target_tokens": pred_shift_labels}
    
    def text_to_text(self, inp_text, targ_text, loss_type=["pred"]):
        pass
    
    def generate_text_to_motion(self, texts, topk=1, max_num_tokens=50):
        # Generate prompts
        prompts = self.generate_prompts(task="t2m", num_prompts=len(texts))
        texts_w_propmts = [t1 + t2 for (t1, t2) in zip(prompts, texts)]
        
        # Tokenize the input text prompts.
        text_tokenizer_outputs = self.tokenizer.batch_encode_plus(texts_w_propmts, **self.conf["tokenizer_config"])
        text_tokens = text_tokenizer_outputs["input_ids"]
        text_mask = text_tokenizer_outputs["attention_mask"].long().to(self.device)

        # Get embeddings from text tokens
        input_embeddings = self.llama_model.get_input_embeddings()
        text_emb = input_embeddings(text_tokens.long().to(self.device))
        
        # Get motion token <SOS>
        start_token = torch.tensor(np.array([[self.get_special_token_id("sos", is_learnable=True)]])).long().to(self.device)
        motion_emb = self.token_embed(start_token)
        motion_mask = torch.ones_like(start_token).long()
        
        # Get input embeddings and attention masks
        embeddings = torch.cat([text_emb, motion_emb], dim=1)
        attn_masks = torch.cat([text_mask, motion_mask], dim=1)
        pred_tokens = [start_token]
        while True and len(pred_tokens) < max_num_tokens:
            outputs = self.llama_model(
                inputs_embeds=embeddings, 
                attention_mask=attn_masks, 
                output_hidden_states=True, 
                output_attentions=True)
            last_hidden_state = outputs.hidden_states[-1]
            pred_logits = self.lm_heads["motion_t"](last_hidden_state[:, -1:])
            if topk == 1:
                pred_logits = F.softmax(pred_logits, dim=-1)
                pred_token = pred_logits.argmax(dim=-1)
            else:
                pred_logits = top_k_logits(pred_logits, k=topk)
                pred_logits = F.softmax(pred_logits, dim=-1)
                pred_token = torch.multinomial(pred_logits[:, 0], num_samples=1)
            
            pred_tokens.append(pred_token)
            if pred_token.item() != self.get_special_token_id("eos", is_learnable=True):
                pred_emb, pred_mask = self.prepare_training_embeddings(
                    pred_token, token_type="motion_t", 
                    sos=self.get_special_token_id("sos", is_learnable=True), 
                    eos=self.get_special_token_id("eos", is_learnable=True), 
                    pad=self.get_special_token_id("pad", is_learnable=True), 
                    padding_type="none", 
                    targ_len=1, 
                    shifting_token=-3)
                embeddings = torch.cat([embeddings, pred_emb], dim=1)
                attn_masks = torch.cat([attn_masks, pred_mask], dim=1)
            else:
                break
        
        pred_tokens = torch.cat(pred_tokens, dim=0)
        return pred_tokens.squeeze(dim=0)
        
    def generate_motion_to_text(
        self, m_tokens, topk=1, max_num_tokens=10
    ):
        # Generate prompts
        prompts = self.generate_prompts(task="m2t", num_prompts=len(m_tokens))
        # 1. Tokenize prompts
        prompt_tokens = []
        prompt_mask = []
        for prompt in prompts:
            tokenizer_outputs = self.tokenizer(prompt, return_tensors="pt")
            prompt_tokens.append(tokenizer_outputs["input_ids"])
            prompt_mask.append(tokenizer_outputs["attention_mask"])
        prompt_tokens = torch.cat(prompt_tokens, dim=0).long().to(self.device)
        prompt_mask = torch.cat(prompt_mask, dim=0).long().to(self.device)
        
        # 2. Get text and prompts embedding
        input_embeddings = self.llama_model.get_input_embeddings()
        prompt_emb = input_embeddings(prompt_tokens)
        
        # 3. Get embeddings from motion tokens
        motion_emb, motion_mask = self.prepare_training_embeddings(
            inp_tokens=m_tokens, token_type="motion_t", 
            sos=self.get_special_token_id("sos", is_learnable=True), 
            eos=self.get_special_token_id("eos", is_learnable=True), 
            pad=self.get_special_token_id("pad", is_learnable=True), 
            padding_type="bidirectional", 
            targ_len=m_tokens.size(-1), 
            shifting_token=-3   # We need to shift the tokens by -3 because we have shifted it by +3 previously.
        )
        
        # 4. Get <SOS> token and its embedding
        start_token = torch.from_numpy(np.array([[self.get_special_token_id("sos", is_learnable=False)]])).long().to(self.device)
        text_emb = input_embeddings(start_token)
        text_mask = torch.ones_like(start_token).long()
        
        # Get input embeddings and attention mask
        embeddings = torch.cat([prompt_emb, motion_emb, text_emb], dim=1)
        attn_masks = torch.cat([prompt_mask, motion_mask, text_mask], dim=1)
        pred_tokens = [start_token]
        while True and len(pred_tokens) < max_num_tokens:
            outputs = self.llama_model(
                inputs_embeds=embeddings, 
                attention_mask=attn_masks, 
                output_hidden_states=True, 
                output_attentions=True)
            pred_logits = outputs.logits[:, -1:, :]
    
            if topk == 1:
                pred_logits = F.softmax(pred_logits, dim=-1)
                pred_token = pred_logits.argmax(dim=-1)
            else:
                pred_logits = top_k_logits(pred_logits, k=topk)
                pred_logits = F.softmax(pred_logits, dim=-1)
                pred_token = torch.multinomial(pred_logits[:, 0], num_samples=1)
                
            pred_tokens.append(pred_token)
            if pred_token.item() != self.get_special_token_id("eos", is_learnable=False):
                pred_emb = input_embeddings(pred_token)
                pred_mask = torch.ones_like(pred_token).long()
                embeddings = torch.cat([embeddings, pred_emb], dim=1)
                attn_masks = torch.cat([attn_masks, pred_mask], dim=1)
            else:
                break
        
        pred_tokens = torch.cat(pred_tokens, dim=0)
        pred_text = self.tokenizer.decode(pred_tokens.squeeze(), skip_special_tokens=True)
        return pred_text
        
    def generate_audio_to_motion_autoregressive(
        self, a_tokens, cond_m_tokens=None, 
        topk=1, max_num_tokens=10, window_size=10, 
        step_size=1, cond_m_len=7
    ):
        """Generate motion tokens from audio tokens auto-regressively.
        NOTICE: 1 motion token corresponds to 10 audio tokens, and 3200 audio features.
        :param a_tokens: [batch_size, seq_len] or [batch_size, n_channels, seq_len], no <SOS>, <EOS>, <PAD> are appended
        :param cond_m_tokens: (optional) [batch_size, seq_len], no <SOS>, <EOS>, <PAD> are appended
        :param window_size: length of audio token sequence of each segment condition.
        :param step_size: step size of audio token sequence when running auto-regressive generation.
        :param cond_m_len: length of initial conditional motion token sequence.
        """
        sos_token = self.get_special_token_id("sos", is_learnable=True)
        eos_token = self.get_special_token_id("eos", is_learnable=True)
        pad_token = self.get_special_token_id("pad", is_learnable=True)
        
        m_tokens = cond_m_tokens    # Keep API compatibility
        if m_tokens is None:
            audio_tokens_len = a_tokens.size(-1)
            audio_feat_window_size = 48000
            audio_token_window_size = audio_feat_window_size // 320
            audio_feat_step_size = 3200
            audio_token_step_size = audio_feat_step_size // 320
            cond_motion_token_len = 0
            pred_motion_token_len = audio_token_window_size // 10
        else:
            audio_tokens_len = a_tokens.size(-1)
            audio_feat_window_size = 24000
            audio_token_window_size = audio_feat_window_size // 320
            audio_feat_step_size = 3200
            audio_token_step_size = audio_feat_step_size // 320
            cond_motion_token_len = m_tokens.gt(eos_token).sum().item()
            pred_motion_token_len = audio_token_window_size // 10
        
        # Prepare initial motion tokens
        if m_tokens is None:
            pred_tokens = torch.empty(0, 1).long().to(self.device)
            inp_motion_tokens = None
        else:
            pred_tokens = self.update_predicted_tokens(m_tokens.permute(1, 0), pad_token, shifting_token=None)  # [T, 1]
            inp_motion_tokens = m_tokens.clone()
            inp_motion_tokens = self.prepare_generation_tokens(inp_motion_tokens.permute(1, 0), # [T, 1] 
                                                               sos_token, eos_token, pad_token, 
                                                               targ_length=pred_tokens.size(0)+1, 
                                                               preparation_type="pre", 
                                                               shifting_token=None)
            inp_motion_tokens = inp_motion_tokens.permute(1, 0)
        
        # Generate the first segment of the motion tokens
        inp_audio_tokens = a_tokens[..., :audio_token_window_size]    # [1, T] or [1, N, T]
        inp_audio_tokens = self.prepare_generation_tokens(inp_audio_tokens.permute(1, 0) if inp_audio_tokens.dim() == 2 else inp_audio_tokens.permute(2, 1, 0), 
                                                          sos_token, eos_token, pad_token, 
                                                          targ_length=inp_audio_tokens.size(-1), 
                                                          preparation_type="bidirectional", 
                                                          shifting_token=None)
        inp_audio_tokens = inp_audio_tokens.permute(1, 0) if inp_audio_tokens.dim() == 2 else inp_audio_tokens.permute(2, 1, 0) # [1, T] or [1, N, T]
        
        # Generate motion tokens
        cur_pred_tokens = self.generate_audio_to_motion_once(a_tokens=inp_audio_tokens, 
                                                            m_tokens=inp_motion_tokens, 
                                                            topk=topk, max_num_tokens=(inp_audio_tokens.size(-1)-2)//10+1)
        cur_pred_tokens = self.update_predicted_tokens(cur_pred_tokens, pad_token, shifting_token=None)
        pred_tokens = torch.cat([pred_tokens, cur_pred_tokens], dim=0)  # [T, 1]
        
        # Start to generate motion tokens auto-regressively
        for i in range(10, audio_tokens_len - audio_token_window_size, audio_token_step_size):
            # Prepare input conditional audio tokens
            inp_audio_tokens = a_tokens[..., i:i+audio_token_window_size]   # [1, T] or [1, N, T]
            inp_audio_tokens = self.prepare_generation_tokens(inp_audio_tokens.permute(1, 0) if inp_audio_tokens.dim() == 2 else inp_audio_tokens.permute(2, 1, 0), 
                                                              sos_token, eos_token, pad_token, 
                                                              targ_length=inp_audio_tokens.size(-1), 
                                                              preparation_type="bidirectional", 
                                                              shifting_token=None)
            inp_audio_tokens = inp_audio_tokens.permute(1, 0) if inp_audio_tokens.dim() == 2 else inp_audio_tokens.permute(2, 1, 0) # [1, T] or [1, N, T]
            # Prepare input conditional motion tokens
            inp_motion_tokens = pred_tokens[-(pred_motion_token_len-1):]    # [T, 1]
            inp_motion_tokens = self.prepare_generation_tokens(inp_motion_tokens,  
                                                                sos_token, eos_token, pad_token, 
                                                                targ_length=inp_motion_tokens.size(1)+1, 
                                                                preparation_type="pre", 
                                                                shifting_token=None)
            inp_motion_tokens = inp_motion_tokens.permute(1, 0) # [1, T]
            # Predict next token
            cur_pred_tokens = self.generate_audio_to_motion_once(a_tokens=inp_audio_tokens, 
                                                                m_tokens=inp_motion_tokens, 
                                                                topk=topk, max_num_tokens=2)
            # Update the condition motion tokens
            cur_pred_tokens = self.update_predicted_tokens(cur_pred_tokens, pad_token, shifting_token=None)
            pred_tokens = torch.cat([pred_tokens, cur_pred_tokens], dim=0)  # [T, 1]
        
        return pred_tokens
    
    def generate_motion_to_audio_autoregressive(
        self, m_tokens, cond_a_tokens=None, 
        topk=1, max_num_tokens=10, 
        window_size=10, step_size=1
    ):
        """Generate audio tokens from motion tokens auto-regressively.
        NOTICE: 1 motion token corresponds to 10 audio tokens, and 3200 audio features.
        :param m_tokens: [batch_size, seq_len], no <SOS>, <EOS>, <PAD> are appended.
        :param cond_a_tokens: (optional) [batch_size, seq_len] or [batch_size, n_channels, seq_len], no <SOS>, <EOS>, <PAD> are appended.
        :param topk: integer
        :param max_num_tokens: integer
        """
        sos_token = self.get_special_token_id("sos", is_learnable=True)
        eos_token = self.get_special_token_id("eos", is_learnable=True)
        pad_token = self.get_special_token_id("pad", is_learnable=True)
        
        n_channels = self.conf.get("n_encodec_channels_for_m2a", 1) # Number of channels used for motion-to-audio
        a_tokens = cond_a_tokens    # Keep API compatibility
        if a_tokens.dim() == 3:
            a_tokens = a_tokens[:, :n_channels]
        if n_channels == 1:
            a_tokens = a_tokens.squeeze(dim=1)
        
        motion_tokens_len = m_tokens.size(1)
        motion_feat_window_size = 60
        motion_token_window_size = motion_feat_window_size // 4
        motion_token_step_size = 1
        if a_tokens is None:
            cond_audio_token_len = 0
        else:
            cond_audio_token_len = cond_a_tokens.size(-1)
        pred_audio_token_len = motion_token_window_size * 10 - cond_audio_token_len
        audio_feat_window_size = 48000
        audio_token_window_size = audio_feat_window_size // 320
        
        # Prepare motion tokens
        inp_motion_tokens = self.prepare_generation_tokens(m_tokens[:, :motion_token_window_size].permute(1, 0), 
                                                            sos_token, eos_token, pad_token, 
                                                            targ_length=self.conf.get("m_token_len", 51), 
                                                            preparation_type="bidirectional", 
                                                            shifting_token=None)
        inp_motion_tokens = inp_motion_tokens.permute(1, 0)
        
        # Prepare audio tokens
        if a_tokens is None:
            inp_audio_tokens = None
        else:
            pred_tokens = self.update_predicted_tokens(a_tokens.permute(1, 0) if a_tokens.dim() == 2 else a_tokens.permute(2, 1, 0), 
                                                       pad_token, shifting_token=None)
            inp_audio_tokens = self.prepare_generation_tokens(a_tokens.permute(1, 0) if a_tokens.dim() == 2 else a_tokens.permute(2, 1, 0), 
                                                              sos_token, eos_token, pad_token, 
                                                              targ_length=a_tokens.size(-1), 
                                                              preparation_type="pre", 
                                                              shifting_token=None)
            inp_audio_tokens = inp_audio_tokens.permute(1, 0) if inp_audio_tokens.dim() == 2 else inp_audio_tokens.permute(2, 1, 0)
        
        # Generate initial segment of audio tokens
        cur_pred_tokens = self.generate_motion_to_audio_once(m_tokens=inp_motion_tokens, 
                                                            a_tokens=inp_audio_tokens, 
                                                            topk=topk, max_num_tokens=pred_audio_token_len+1)
        cur_pred_tokens = self.update_predicted_tokens(cur_pred_tokens, pad_token, shifting_token=-(self.conf["tokens"]["motion"]))
        pred_tokens = torch.cat([pred_tokens, cur_pred_tokens], dim=0)  # [T, 1] or [T, N, 1]
        
        # Start to generate audio tokens auto-regressively
        for i in range(1, motion_tokens_len - motion_token_window_size, motion_token_step_size):
            # Prepare input motion tokens
            inp_motion_tokens = m_tokens[:, i:i+motion_token_window_size]
            inp_motion_tokens = self.prepare_generation_tokens(inp_motion_tokens.permute(1, 0), 
                                                            sos_token, eos_token, pad_token, 
                                                            targ_length=self.conf.get("m_token_len", 51), 
                                                            preparation_type="bidirectional", 
                                                            shifting_token=None)
            inp_motion_tokens = inp_motion_tokens.permute(1, 0)
            # Prepare input audio tokens
            inp_audio_tokens = pred_tokens[-(audio_token_window_size-10):]  # [T, 1]
            inp_audio_tokens = self.prepare_generation_tokens(inp_audio_tokens, 
                                                              sos_token, eos_token, pad_token, 
                                                              targ_length=inp_audio_tokens.size(0), 
                                                              preparation_type="pre", 
                                                              shifting_token=None)
            inp_audio_tokens = inp_audio_tokens.permute(1, 0) if inp_audio_tokens.dim() == 2 else inp_audio_tokens.permute(2, 1, 0)
            # Predict next motion token
            cur_pred_tokens = self.generate_motion_to_audio_once(m_tokens=inp_motion_tokens, 
                                                                a_tokens=inp_audio_tokens, 
                                                                topk=topk, max_num_tokens=11)
            cur_pred_tokens = self.update_predicted_tokens(cur_pred_tokens, pad_token, 
                                                           shifting_token=-(self.conf["tokens"]["motion"]))
            pred_tokens = torch.cat([pred_tokens, cur_pred_tokens], dim=0)  # [T, 1] or [T, N, 1]
            
        return pred_tokens      # [T, 1] or [T, N, 1]
    
if __name__ == "__main__":
    
    import yaml
    import importlib
    import time
    
    lora_arch = "lora_llama_7b"
    exp_name = "exp7"
    
    with open("configs/lora_llama/{:s}/config_{:s}_{:s}.yaml".format(lora_arch, lora_arch, exp_name), "r") as f:
        conf = yaml.safe_load(f)
        
    model = AvatarGPT(conf["models"]["gpt"])
    model = model.to(model.device)
    # model.load_model("logs/avatar_gpt/lora_llama/exp01/pretrained-0719/checkpoints/gpt/AvatarGPT_E0060")
    
    quantizer = importlib.import_module(conf["models"]["vqvae"]["t2m"]["body"]["quantizer"]["arch_path"], package="networks").__getattribute__(
        conf["models"]["vqvae"]["t2m"]["body"]["quantizer"]["arch_name"])(**conf["models"]["vqvae"]["t2m"]["body"]["quantizer"]).to(model.device)
    encodec = importlib.import_module(conf["models"]["encodec"]["arch_path"], package="networks").__getattribute__(
            conf["models"]["encodec"]["arch_name"]).encodec_model_24khz(**conf["models"]["encodec"]).to(model.device)
    encodec.set_target_bandwidth(conf["models"]["encodec"].get("target_bandwidth", 1.5))
    
    model.set_quantizer(quantizer, type="motion_t")
    model.set_quantizer(quantizer, type="motion_a")
    model.set_quantizer(encodec.quantizer, type="audio")
    
    texts = [
        "a person is running forward."
    ] * 2
    
    a_tokens = torch.randint(0, 1024, (2, 4, 75)).to(model.device)
    sos = torch.tensor(model.get_special_token_id("sos", is_learnable=True)).view(1, 1, 1).repeat(2, 4, 1).to(model.device)
    eos = torch.tensor(model.get_special_token_id("eos", is_learnable=True)).view(1, 1, 1).repeat(2, 4, 1).to(model.device)
    pos = torch.tensor(model.get_special_token_id("pad", is_learnable=True)).view(1, 1, 1).repeat(2, 4, 1).to(model.device)
    a_tokens += (conf["models"]["gpt"]["tokens"]["motion"]+ 3)
    a_tokens = torch.cat([sos, a_tokens, eos], dim=-1)
    
    sos = model.get_special_token_id("sos", is_learnable=True)
    eos = model.get_special_token_id("eos", is_learnable=True)
    sos = torch.tensor(sos).view(1, 1).repeat(2, 1).to(model.device)
    eos = torch.tensor(eos).view(1, 1).repeat(2, 1).to(model.device)
    m_tokens = torch.randint(0, 1024, size=(2, 50)).to(model.device)
    m_tokens += 3
    m_tokens = torch.cat([sos, m_tokens, eos], dim=-1)
    
    model.audio_to_motion(a_tokens=a_tokens, m_tokens=m_tokens[:, 26:], cond_m_tokens=m_tokens[:, :26])
    model.generate_audio_to_motion_autoregressive(a_tokens=a_tokens, max_num_tokens=20)
    
    start_time = time.time()
    for _ in range(100):
        print("Start to test the runtime")
        model.text_to_motion(texts, m_tokens)
        end_time = time.time()
        duration = end_time - start_time
        print('Time duration is', duration)
    exit(0)
    # a_tokens = torch.randint(0, 1024, size=(1, 500)).long().to(model.device)
    # output = model.generate_audio_to_motion_autoregressive(a_tokens, m_tokens, 
    #                                                         topk=10, max_num_tokens=50, 
    #                                                         window_size=150, step_size=10, cond_m_len=7)
    # print(output.shape)
    m_tokens = torch.randint(0, 2048, size=(1, 100)).to(model.device)
    m_tokens += 3
    a_tokens = torch.randint(0, 1024, size=(1, 75)).long().to(model.device)
    a_tokens += (2048 + 3)
    output = model.generate_motion_to_audio_autoregressive(m_tokens, 
                                                            cond_a_tokens=a_tokens, 
                                                            topk=10, max_num_tokens=50, 
                                                            window_size=15, step_size=1)