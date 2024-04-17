import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from typing import Optional, Tuple, Union
import warnings
import random
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel
)
from networks.utils.model_utils import *
# from networks.encodec import EncodecModel
# from networks.encodec.utils import convert_audio

def print_log(logger, log_str):
    if logger is None:
        print(log_str)
    else:
        logger.info(log_str)

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

def minimize_special_token_logits(logits, k, first_k=True):
    """
    :param logits: [num_seq, num_dim]
    """
    dim = logits.dim()
    if dim == 3:
        logits = logits.squeeze(dim=1)
    out = logits.clone()
    if first_k:
        out[..., :k] = -float('Inf')
    else:
        out[..., -k:] = -float('Inf')
    if dim == 3:
        out = out.unsqueeze(dim=1)
    return out

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
    print_log(logger, "{:.3f}% pretrained parameters loaded!".format(100. * len(valid_params) / len(loaded_params)))
    return valid_params

class AvatarGPT(nn.Module):
    def __init__(
        self, conf, logger=None, m_quantizer=None, a_quantizer=None
    ):
        super(AvatarGPT, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.conf = conf
        self.model_type = conf.get("model_type", "gpt")
        self.max_length = conf["tokenizer_config"].get("n_motion_tokens", 256)
        self.noise_density = conf.get("noise_density", 0.15)
        self.mean_noise_span_length = conf.get("mean_noise_span_length", 3)
        self.m_codebook_size = conf.get("n_motion_tokens", 512)
        # Instantiate language model
        self.build_llm(conf=conf, logger=logger)
        self.build_tokenizer(conf=conf, logger=logger)
        self.build_trainables(conf=conf, logger=logger)
        self.load_instruction_templates()
    
    def load_instruction_templates(self):
        import json
        with open("networks/llama/instruction_template.json", "r") as f:
            self.instruction_template = json.load(f)
            
    def build_llm(self, conf, logger=None):
        print_log(logger=logger, log_str="Build language model")
        if self.model_type == "gpt":
            self.llm_model = GPT2LMHeadModel.from_pretrained(conf["model"])
            self.lm_type = "dec" # encoder-decoder
        elif self.model_type == "llama":
            pass
        
    def build_tokenizer(self, conf, logger=None):
        print_log(logger=logger, log_str="Build tokenizer")
        if self.model_type == "gpt":
            self.tokenizer = GPT2Tokenizer.from_pretrained(conf["tokenizer"], legacy=True)
        elif self.model_type == "llama":
            pass
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        if conf.get("add_motion_token_type", "token") == "token":
            # Append motion vocabulary to LLM vocabulary
            print_log(logger=logger, log_str="Resize the toke embeddings from {:d} to {:d}".format(
                len(self.tokenizer), len(self.tokenizer)+conf.get("n_motion_tokens", 512)+3))
            # self.tokenizer.pad_token = "<|padoftext|>"
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id + 1
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # self.tokenizer.add_tokens(["<|padoftext|>"])
            self.tokenizer.add_tokens(
                ["<motion_id_{:d}>".format(i) for i in range(conf.get("n_motion_tokens", 512)+3)]
            )
            self.llm_model.resize_token_embeddings(len(self.tokenizer))
        elif conf.get("add_motion_token_type", "token") == "mlp":
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llm_model.resize_token_embeddings(len(self.tokenizer))
            self.motion_embeddings = nn.Embedding(
                num_embeddings=3, 
                embedding_dim=conf.get("d_motion_embeds", 512))
        print_log(logger=logger, log_str='Special token <EOS>: {:d}'.format(self.tokenizer.eos_token_id))
        print_log(logger=logger, log_str='Special token <PAD>: {:d}'.format(self.tokenizer.pad_token_id))
        
    def build_trainables(self, conf, logger=None):
        print_log(logger=logger, log_str="Build other trainable headers")
        if conf.get("add_motion_token_type", "token") == "mlp":
            # If use quantizer's embedding, we need to project them to the same dimension as embedding in LLM.
            self.projection = nn.Linear(conf.get("d_motion_embeds", 512), 
                                        conf.get("d_model", 1024), bias=False)
        
        if conf.get("head_type", "shared") == "shared":
            # If we use 'shared' head, we use the LLM's head
            pass
        elif conf.get("head_type", "shared") == "separate":
            # If we use 'separate head, we train a separate head for motion token prediction
            self.head = nn.Linear(conf.get("d_model", 1024), conf.get("n_motion_tokens", 512)+3, bias=False)

    def set_quantizer(self, quantizer, type="motion"):
        self.quantizer = copy.deepcopy(quantizer)
        for p in self.quantizer.parameters():
            p.requires_grad = False
            
    def train(self):
        self.llm_model.train()
        if hasattr(self, "projection"):
            self.projection.train()
        if hasattr(self, "head"):
            self.head.train()
        if hasattr(self, "motion_embeddings"):
            self.motion_embeddings.train()
        if hasattr(self, "quantizer"):
            self.quantizer.eval()
    
    def eval(self):
        self.llm_model.eval()
        if hasattr(self, "projection"):
            self.projection.eval()
        if hasattr(self, "head"):
            self.head.eval()
        if hasattr(self, "motion_embeddings"):
            self.motion_embeddings.eval()
        if hasattr(self, "quantizer"):
            self.quantizer.eval()
            
    def get_trainable_parameters(self):
        state_dict = {}
        for key, param in super().state_dict().items():
            if "quantizer" not in key:
                state_dict[key] = param
        return state_dict
    
    def save_model(self, output_path):
        trainable_parameters = self.get_trainable_parameters()
        torch.save(trainable_parameters, os.path.join(output_path, "trainable.pth"))
        
    def load_model(self, input_path, logger=None, strict=True):
        learnable_param_ckpt = torch.load(os.path.join(input_path, "trainable.pth"), map_location=self.device)
        valid_learnable_param = load_partial_parameters(self, learnable_param_ckpt, logger=logger)
        super().load_state_dict(valid_learnable_param,strict=False)
        print_log(logger=logger, log_str='Trainable parameters loaded from {:s} successfully.'.format(
            os.path.join(input_path, "trainable.pth")))
        
    def forward(self, **kwargs):
        pass
    
    def get_special_token_id(self, token, is_learnable=True):
        assert token in ["sos", "eos", "pad"]
        if token == "sos":
            # if is_learnable: return self.tokenizer.added_tokens_encoder["<motion_id_0>"]
            if is_learnable: return 0
            else: return self.tokenizer.bos_token_id
        elif token == "eos":
            # if is_learnable: return self.tokenizer.added_tokens_encoder["<motion_id_1>"]
            if is_learnable: return 1
            else: return self.tokenizer.eos_token_id
        elif token == "pad":
            # if is_learnable: return self.tokenizer.added_tokens_encoder["<motion_id_1>"]
            if is_learnable: return 2
            else:
                return self.tokenizer.pad_token_id
    
    @staticmethod
    def decompose_input_text(inp_texts, mode="input"):
        assert mode in ["input", "output", "scene", "current"]
        decomposers = {
            "input": ["[scene] ", "[current action]: "], 
            "output": ["[next action] "], 
            "scene": ["[scene] "], 
            "current": ["[current action]: "]
        }
        
        out_texts = []
        for inp in inp_texts:
            if mode == "output":
                out_texts.append(inp.replace(decomposers["output"][0], ""))
            elif mode == "input":
                inp1 = inp.split(decomposers["input"][1])[0].replace(decomposers["input"][0], "")
                inp2 = inp.split(decomposers["input"][1])[1]
                out_texts.append((inp1, inp2))
            elif mode == "scene":
                out_texts.append(inp.replace(decomposers["scene"][0], ""))
            elif mode == "current":
                out_texts.append(inp.replace(decomposers["current"][0], ""))
        return out_texts
    
    @staticmethod
    def calc_prediction_accuracy(pred, target, ignore_cls):
        acc_mask = pred.eq(target).float()
        valid_mask = target.ne(ignore_cls).float()
        accuracy = acc_mask.sum() / valid_mask.sum()
        return accuracy
    
    def calculate_loss(self, pred_logits, targ_labels):
        """
        :param pred_logits: [batch_size, seq_len, num_dim]
        :param targ_labels: [batch_size, seq_len]
        """
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        losses = {}
        accuracy = {}
        shift_logits = pred_logits[..., :-1, :].contiguous()
        shift_labels = targ_labels[..., 1:].contiguous()
        pred_tokens = shift_logits.argmax(dim=-1)
        losses["pred"] = loss_fct(
            shift_logits.contiguous().view(-1, pred_logits.size(-1)), 
            shift_labels.contiguous().view(-1))
        accuracy["pred"] = self.calc_prediction_accuracy(
            pred_tokens, shift_labels, 
            ignore_cls=-100)
        
        results = {
            "losses": losses,
            "accuracy": accuracy, 
            "pred_tokens": pred_tokens, 
            "target_tokens": shift_labels
        }
        
        return results
    
    def generate_prompts(self, task, num_prompts=1):
        """
        :param task: 
            1) t2m: text-to-motion (middle-level generation)
            2) m2t: motion-to-text (middle-level understanding)
            3) a2m: audio-to-motion (middle-level generation)
            4) m2a: motion-to-audio(dumped)
            5) t2t: text-to-text (high-level decision)
            6) se: scene-estimation (high-level)
            7) s2t: scene-to-text (high-level decision)
            8) m2m: motion-to-motion (middle-level prediction)
        """
        prompts = [self.instruction_template[task]["main"]] * num_prompts
        return prompts
    
    @torch.no_grad()
    def get_input_prompts(self, prompts, batch, task="ct2t"):
        if task == "ct2t":
            output = prompts.format(batch["scene"], batch["cur_task"])
        elif task == "cs2s":
            output = prompts.format(batch["scene"], batch["cur_steps"])
        elif task == "ct2s":
            output = prompts.format(batch["scene"], batch["cur_task"])
        elif task == "cs2t":
            output = prompts.format(batch["scene"], batch["cur_steps"])
        elif task == "t2c":
            output = prompts.format(batch["cur_task"])
        elif task == "s2c":
            output = prompts.format(batch["cur_steps"])
        elif task == "t2s":
            output = prompts.format(batch["cur_task"])
        elif task == "s2t":
            output = prompts.format(batch["cur_steps"])
        return output
    
    @torch.no_grad()
    def get_target_texts(self, batch, task="ct2t"):
        if task == "ct2t":
            output = batch["next_task"]
        elif task == "cs2s":
            output = batch["next_steps"]
        elif task == "ct2s":
            output = batch["next_steps"]
        elif task == "cs2t":
            output = batch["next_task"]
        elif task == "t2c":
            output = batch["scene"]
        elif task == "s2c":
            output = batch["scene"]
        elif task == "t2s":
            output = batch["cur_steps"]
        elif task == "s2t":
            output = batch["cur_task"]
        return output
        
    @torch.no_grad()
    def convert_motion_string_to_token(self, m_string):
        """
        :param m_string: list of strings
        """
        sos_id = self.get_special_token_id("sos", is_learnable=True)
        eos_id = self.get_special_token_id("eos", is_learnable=True)
        pad_id = self.get_special_token_id("pad", is_learnable=True)
        m_tokens = []
        for i in range(len(m_string)):
            string_list = m_string[i].split(">")
            tokens = []
            for string in string_list:
                if "<motion_id_" not in string:
                    continue
                try:
                    tok = torch.tensor(int(string.replace("<motion_id_", ""))).long().to(self.device)
                    if tok == sos_id:
                        continue
                    if tok == eos_id or tok == pad_id:
                        break
                    tokens.append(tok)
                except:
                    pass
            try:
                tokens = torch.stack(tokens)
                m_tokens.append(tokens)
            except:
                pass
        # Deal with exceptions
        if len(m_tokens) == 0:
            m_tokens.append(None)
        return m_tokens
    
    @torch.no_grad()
    def generate_motion_tokens_from_text(
        self, input_attn_mask, input_embeds, 
        topk=1, max_num_tokens=50
    ):
        sos_id = self.get_special_token_id("sos", is_learnable=True)
        eos_id = self.get_special_token_id("eos", is_learnable=True)
        pad_id = self.get_special_token_id("pad", is_learnable=True)
        
        pred_tokens = []
        while len(pred_tokens) < max_num_tokens:
            # Predict next token
            outputs = self.llm_model(
                inputs_embeds=input_embeds, 
                attention_mask=input_attn_mask, 
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1][:, -1:]
            raw_pred_logit = self.head(last_hidden_state)
            if topk == 1:
                # Sample the token with highest probability
                pred_logit = F.softmax(raw_pred_logit.clone(), dim=-1)
                pred_token = pred_logit.argmax(dim=-1)
            else:
                # Sample one token from tokens with top-k probability
                pred_logit = top_k_logits(raw_pred_logit.clone(), k=topk)
                pred_logit = F.softmax(pred_logit, dim=-1)
                pred_token = torch.multinomial(pred_logit[:, 0], num_samples=1)
            if pred_token.item() > pad_id:
                pred_tokens.append(pred_token)
                pred_emb = self.projection(self.quantizer.get_codebook_entry(pred_token-3))
                attn_mask = torch.ones(1, 1).long().to(self.device)
                input_embeds = torch.cat([input_embeds, pred_emb], dim=1)
                input_attn_mask = torch.cat([input_attn_mask, attn_mask], dim=1)
            else:
                if len(pred_tokens) == 0:
                    while len(pred_tokens) == 0:
                        pred_logit = minimize_special_token_logits(raw_pred_logit.clone(), k=3)
                        pred_logit = F.softmax(pred_logit, dim=-1)
                        pred_token = torch.multinomial(pred_logit[:, 0], num_samples=1)
                        pred_tokens.append(pred_token)
                        pred_emb = self.projection(self.quantizer.get_codebook_entry(pred_token-3))
                        attn_mask = torch.ones(1, 1).long().to(self.device)
                        input_embeds = torch.cat([input_embeds, pred_emb], dim=1)
                        input_attn_mask = torch.cat([input_attn_mask, attn_mask], dim=1)
                else:
                    break
                
        return torch.cat(pred_tokens, dim=1).squeeze(dim=0) # [T]
        
    @torch.no_grad()
    def generate_text_tokens_from_motion(
        self, input_attn_mask, input_embeds, 
        topk=1, max_num_tokens=50
    ):
        eos_id = self.get_special_token_id("eos", is_learnable=False)
        pad_id = self.get_special_token_id("pad", is_learnable=False)
        
        pred_tokens = []
        while len(pred_tokens) < max_num_tokens:
            # Predict next token
            outputs = self.llm_model(
                inputs_embeds=input_embeds, 
                attention_mask=input_attn_mask,
                output_hidden_states=True
            )
            raw_pred_logit = outputs.logits[:, -1:]
            if topk == 1:
                # Sample the token with highest probability
                pred_logit = F.softmax(raw_pred_logit.clone(), dim=-1)
                pred_token = pred_logit.argmax(dim=-1)
            else:
                # Sample one token from tokens with top-k probability
                pred_logit = top_k_logits(raw_pred_logit.clone(), k=topk)
                pred_logit = F.softmax(pred_logit, dim=-1)
                pred_token = torch.multinomial(pred_logit[:, 0], num_samples=1)
            if pred_token.item() < eos_id:
                pred_tokens.append(pred_token)
                pred_emb = self.get_llm_embedding(pred_token)
                attn_mask = torch.ones(1, 1).long().to(self.device)
                input_embeds = torch.cat([input_embeds, pred_emb], dim=1)
                input_attn_mask = torch.cat([input_attn_mask, attn_mask], dim=1)
            else:
                if len(pred_tokens) == 0:
                    pred_logit = minimize_special_token_logits(raw_pred_logit.clone(), k=2, first_k=False)
                    pred_logit = F.softmax(pred_logit, dim=-1)
                    pred_token = torch.multinomial(pred_logit[:, 0], num_samples=1)
                    pred_tokens.append(pred_token)
                    pred_emb = self.get_llm_embedding(pred_token)
                    attn_mask = torch.ones(1, 1).long().to(self.device)
                    input_embeds = torch.cat([input_embeds, pred_emb], dim=1)
                    input_attn_mask = torch.cat([input_attn_mask, attn_mask], dim=1)
                else:
                    break
        
        return torch.cat(pred_tokens, dim=1)
        
    @torch.no_grad()
    def generate_motion_tokens_from_motion_primitives(
        self, input_attn_mask, input_embeds, 
        topk=1, max_num_tokens=50
    ):
        return self.generate_motion_tokens_from_text(
            input_attn_mask=input_attn_mask, 
            input_embeds=input_embeds, 
            topk=topk, 
            max_num_tokens=max_num_tokens)
        
    def tokenize(self, inp_string, device, output_type="ids"):
        tokenize_output = self.tokenizer(
            inp_string, 
            padding="max_length", 
            max_length=self.max_length, 
            truncation=True, 
            # return_special_tokens=True, 
            add_special_tokens=True, 
            return_tensors="pt")
        attn_mask = tokenize_output.attention_mask.to(device)
        ids = tokenize_output.input_ids.to(device)
        return attn_mask, ids
    
    def tokenize_valid(self, inp_string, device, output_type="ids", add_eos=False):
        raw_attn_mask, raw_input_ids = self.tokenize(
            inp_string=inp_string, 
            device=device, 
            output_type=output_type)
        attn_mask, input_ids = [], []
        for (mask, ids) in zip(raw_attn_mask, raw_input_ids):
            num_valid = mask.sum()
            valid_mask = mask[:num_valid]
            valid_ids = ids[:num_valid]
            if add_eos:
                eos_id = self.get_special_token_id("eos", is_learnable=False)
                eos_tok = torch.tensor(eos_id).view(1).long().to(self.device)
                eos_mask = torch.ones(1).long().to(self.device)
                valid_mask = torch.cat([valid_mask, eos_mask], dim=0)
                valid_ids = torch.cat([valid_ids, eos_tok], dim=0)
            attn_mask.append(valid_mask)
            input_ids.append(valid_ids)
        return attn_mask, input_ids
    
    def get_llm_embedding(self, tokens):
        """Get the LLaMA embedding from input tokens.
        :param tokens: [batch_size, seq_len] or [seq_len]
        """
        llm_embeddings = self.llm_model.get_input_embeddings()
        return llm_embeddings(tokens)

    def get_valid_motion_token(self, m_token):
        """Get the valid token from input tokens.
        :param m_token: [seq_len]
        """
        sos_id = self.get_special_token_id("sos", is_learnable=True)
        sos_token = torch.tensor(sos_id).long().view(1).to(m_token.device)
        eos_id = self.get_special_token_id("eos", is_learnable=True)
        eos_token = torch.tensor(eos_id).long().view(1).to(m_token.device)
        pad_id = self.get_special_token_id("pad", is_learnable=True)
        pad_token = torch.tensor(pad_id).long().view(1).to(m_token.device)
        
        mask = m_token.gt(pad_id)
        valid_m_token = m_token[mask]
        
        return valid_m_token
    
    def convert_motion_token_to_string(self, m_token):
        """Convert motion tokens to motion strings.
        :param m_token: [seq_len]
        """
        sos_id = self.get_special_token_id("sos", is_learnable=True)
        sos_token = torch.tensor(sos_id).long().view(1).to(m_token.device)
        eos_id = self.get_special_token_id("eos", is_learnable=True)
        eos_token = torch.tensor(eos_id).long().view(1).to(m_token.device)
        pad_id = self.get_special_token_id("pad", is_learnable=True)
        pad_token = torch.tensor(pad_id).long().view(1).to(m_token.device)
        
        mask = m_token.gt(pad_id)
        valid_m_token = m_token[mask]
        
        padded_m_token = torch.cat([sos_token, valid_m_token, eos_token], dim=0)
        cvt_m_token = padded_m_token.cpu().tolist()
        m_string = "".join("<motion_id_{:d}>".format(i) for i in cvt_m_token)
        return m_string
    
    def convert_motion_token_to_embeds(self, m_tokens):
        """Convert motion tokens to motion embeddings.
        :param m_token: [batch_size, seq_len]
        """
        sos_id = self.get_special_token_id("sos", is_learnable=True)
        sos_token = torch.tensor(sos_id).long().view(1).to(m_tokens.device)
        eos_id = self.get_special_token_id("eos", is_learnable=True)
        eos_token = torch.tensor(eos_id).long().view(1).to(m_tokens.device)
        pad_id = self.get_special_token_id("pad", is_learnable=True)
        pad_token = torch.tensor(pad_id).long().view(1).to(m_tokens.device)
        
        embeds = []
        attn_masks = []
        for m_token in m_tokens:
            mask = m_token.gt(pad_id)
            valid_m_token = m_token[mask]
            valid_m_token -= 3

            with torch.no_grad():
                valid_m_embed = self.quantizer.get_codebook_entry(valid_m_token)        # [T, D]
            valid_m_embed = self.projection(valid_m_embed)                              # [T, D]

            sos_embed = self.projection(self.motion_embeddings(sos_token))
            eos_embed = self.projection(self.motion_embeddings(eos_token))
            padded_m_embed = torch.cat([sos_embed, valid_m_embed, eos_embed], dim=0)

            # Pad if necessary
            padded_m_len = padded_m_embed.size(0)
            padding_m_len = self.max_length - padded_m_len
            if padding_m_len > 0:
                pad_embed = self.projection(self.motion_embeddings(pad_token))
                pad_embed = pad_embed.repeat(padding_m_len, 1)
                padded_m_embed = torch.cat([padded_m_embed, pad_embed], dim=0)

            # Generate the attention mask
            attn_mask = torch.zeros(self.max_length).to(self.device)
            attn_mask[:padded_m_len] = 1
            
            embeds.append(padded_m_embed)
            attn_masks.append(attn_mask.long())
        
        attn_masks = torch.stack(attn_masks, dim=0)
        embeds = torch.stack(embeds, dim=0)            
        return attn_masks, embeds
    
    def convert_motion_token_to_embeds_valid(self, m_tokens, add_eos=False):
        raw_attn_mask, raw_input_ids = self.convert_motion_token_to_embeds(m_tokens=m_tokens)
        attn_mask, input_ids = [], []
        for (mask, ids) in zip(raw_attn_mask, raw_input_ids):
            num_valid = mask.sum()
            if add_eos:
                eos_id = self.get_special_token_id("eos", is_learnable=False)
                eos_tok = torch.tensor(eos_id).view(1).long().to(self.device)
                eos_mask = torch.ones(1).long().to(self.device)
                valid_mask = torch.cat([mask[:num_valid], eos_mask], dim=0)
                valid_ids = torch.cat([ids[:num_valid], eos_tok], dim=0)
            else:
                valid_mask = mask[:num_valid]
                valid_ids = ids[:num_valid]
            attn_mask.append(valid_mask)
            input_ids.append(valid_ids)
        return attn_mask, input_ids
    
    def pretrain(self, texts, m_tokens, loss_type=["pred"]):
        # Tokenize text prompts
        if self.conf.get("add_motion_token_type", "token") == "token":
            pass
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            tex_attn_mask, tex_ids = self.tokenize_valid(
                inp_string=texts, device=m_tokens.device, add_eos=False)
            tex_embeds = [self.get_llm_embedding(ids) for ids in tex_ids]
        # Convert motion tokens to motion strings
        if self.conf.get("add_motion_token_type", "token") == "token":
            motion_strings = [self.convert_motion_token_to_string(m_token=m_tok) for m_tok in m_tokens]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            # Convert motion tokens to motion embedding
            mot_ids = [self.get_valid_motion_token(m_token=m_tok) for m_tok in m_tokens]
            mot_embeds = [self.projection(self.quantizer.get_codebook_entry(tok.unsqueeze(0)-3)).squeeze(0) for tok in mot_ids]
            mot_attn_mask = [torch.ones(tok.size(0)).long().to(self.device) for tok in mot_ids]
        
        if self.conf.get("add_motion_token_type", "token") == "token":
            eos_id = self.get_special_token_id("eos", is_learnable=False)
            pad_id = self.get_special_token_id("pad", is_learnable=False)
            eos_tok = torch.tensor(eos_id).view(1).long().to(self.device)
            pad_tok = torch.tensor(pad_id).view(1).long().to(self.device)
            
            attn_mask = []
            input_ids = []
            targ_labels = []
            for (tex, mot) in zip(texts, motion_strings):
                if random.random() < 0.5:
                    inp_text = tex
                    ful_text = tex + mot
                else:
                    inp_text = mot
                    ful_text = mot + tex
                # Tokenize the full texts and get the target labels
                inp_attn_mask, inp_ids = self.tokenize_valid(inp_string=[inp_text], device=self.device, add_eos=False)
                ful_attn_mask, ful_ids = self.tokenize_valid(inp_string=[ful_text], device=self.device, add_eos=True)
                targ_lbls = ful_ids[0].clone()
                targ_lbls[:inp_ids[0].size(0)] = -100
                # Pad at right
                pad_len = self.max_length - targ_lbls.size(0)
                pad_attn_mask = torch.zeros(pad_len).long().to(self.device)
                pad_ids = pad_tok.repeat(pad_len)
                pad_lbls = -100 * torch.ones(pad_len).long().to(self.device)
            
                attn_mask.append(torch.cat([ful_attn_mask[0], pad_attn_mask], dim=0))
                input_ids.append(torch.cat([ful_ids[0], pad_ids], dim=0))
                targ_labels.append(torch.cat([targ_lbls, pad_lbls], dim=0))
            
            attn_mask = torch.stack(attn_mask, dim=0)
            input_ids = torch.stack(input_ids, dim=0)
            targ_labels = torch.stack(targ_labels, dim=0)
            
            outputs = self.llm_model(
                input_ids=input_ids, 
                attention_mask=attn_mask, 
                labels=targ_labels, 
                output_hidden_states=True
            )
            logits = outputs.logits
            results = self.calculate_loss(pred_logits=logits, targ_labels=targ_labels)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            
            def get_special(token, is_learnable=True):
                id = self.get_special_token_id(token, is_learnable=is_learnable)
                tok = torch.tensor(id).view(1).long().to(self.device)
                if is_learnable:
                    emb = self.projection(self.motion_embeddings(tok))
                else:
                    emb = self.get_llm_embedding(tok)
                if token == "eos":
                    mask = torch.ones(1).long().to(self.device)     # [1]
                else:
                    mask = torch.zeros(1).long().to(self.device)    # [1]
                return tok, emb, mask
            
            t_eos_id, t_eos_emb, t_eos_mask = get_special("eos", is_learnable=False)
            t_pad_id, t_pad_emb, t_pad_mask = get_special("pad", is_learnable=False)
            m_eos_id, m_eos_emb, m_eos_mask = get_special("eos", is_learnable=True)
            m_pad_id, m_pad_emb, m_pad_mask = get_special("pad", is_learnable=True)
            
            attn_mask = []
            input_embeds = []
            targ_labels = []
            label_tags = []
            for (tex_mask, tex_id, tex_emb, mot_mask, mot_id, mot_emb) in zip(tex_attn_mask, tex_ids, tex_embeds, mot_attn_mask, mot_ids, mot_embeds):
                if random.random() < 0.5:
                    inp_ids = tex_id.clone()
                    ful_ids = torch.cat([tex_id, mot_id, m_eos_id], dim=0)
                    ful_mask = torch.cat([tex_mask, mot_mask, m_eos_mask], dim=0)
                    ful_emb = torch.cat([tex_emb, mot_emb, m_eos_emb], dim=0)
                    label_tags.append("motion")
                    # Pad right
                    pad_len = self.max_length - ful_emb.size(0)
                    ful_mask = torch.cat([ful_mask, m_pad_mask.repeat(pad_len)], dim=0)
                    ful_emb = torch.cat([ful_emb, m_pad_emb.repeat(pad_len, 1)], dim=0)
                else:
                    inp_ids = mot_id.clone()
                    ful_ids = torch.cat([mot_id, tex_id, t_eos_id], dim=0)
                    ful_mask = torch.cat([mot_mask, tex_mask, t_eos_mask], dim=0)
                    ful_emb = torch.cat([mot_emb, tex_emb, t_eos_emb], dim=0)
                    label_tags.append("text")
                    # Pad right
                    pad_len = self.max_length - ful_emb.size(0)
                    ful_mask = torch.cat([ful_mask, t_pad_mask.repeat(pad_len)], dim=0)
                    ful_emb = torch.cat([ful_emb, t_pad_emb.repeat(pad_len, 1)], dim=0)
                
                # Get target labels
                targ_ids = -100 * torch.ones(self.max_length).long().to(self.device)
                targ_ids[:ful_ids.size(0)] = ful_ids
                targ_ids[:inp_ids.size(0)] = -100
                # Append
                attn_mask.append(ful_mask)
                input_embeds.append(ful_emb)
                targ_labels.append(targ_ids)
            
            attn_mask = torch.stack(attn_mask, dim=0)
            input_embeds = torch.stack(input_embeds, dim=0)
            targ_labels = torch.stack(targ_labels, dim=0)
                    
            outputs = self.llm_model(
                inputs_embeds=input_embeds, 
                attention_mask=attn_mask, 
                output_hidden_states=True
            )
                
            logits = outputs.logits
            last_hidden_state = outputs.hidden_states[-1]
            
            motion_ids, text_ids = [], []
            for i, t in enumerate(label_tags):
                if t == "motion": motion_ids.append(i)
                elif t == "text": text_ids.append(i)
            results = {
                "losses": {"pred": 0},
                "accuracy": {"pred": 0}, 
                "pred_tokens": [], 
                "target_tokens": []
            }
                
            def update_result(src, targ, src_num, total_num):
                targ["losses"]["pred"] += src["losses"]["pred"] * (src_num / total_num)
                targ["accuracy"]["pred"] += src["accuracy"]["pred"] * (src_num / total_num)
                targ["pred_tokens"].append(src["pred_tokens"])
                targ["target_tokens"].append(src["target_tokens"])
                
            if len(motion_ids) > 0:
                results_mot = self.calculate_loss(
                    self.head(last_hidden_state[motion_ids]), 
                    targ_labels[motion_ids])
                update_result(src=results_mot, targ=results, src_num=len(motion_ids), total_num=len(label_tags))
            if len(text_ids) > 0:
                results_tex = self.calculate_loss(logits[text_ids], targ_labels[text_ids])
                update_result(src=results_tex, targ=results, src_num=len(text_ids), total_num=len(label_tags))
            results["pred_tokens"] = torch.cat(results["pred_tokens"], dim=0)
            results["target_tokens"] = torch.cat(results["target_tokens"], dim=0)
        
        return results
    
    def text_to_motion(self, texts, m_tokens, loss_type=["pred"]):
        """[Training] Text-to-Motion, this is a middle-level generation task.
        :param texts: list of strings.
        :param m_tokens: [batch_size, seq_len] with <SOS>, <EOS>, and <PAD> appended.
        """
        # Generate prompts
        prompts = self.generate_prompts(task="t2m", num_prompts=len(texts))
        # Fill in the prompts
        input_texts = [p.format(t) for (p, t) in zip(prompts, texts)]
        if self.conf.get("add_motion_token_type", "token") == "token":
            # Convert motion tokens to motion strings
            motion_strings = [self.convert_motion_token_to_string(m_token=m_tok) for m_tok in m_tokens]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            # Convert motion tokens to motion embedding
            mot_ids = [self.get_valid_motion_token(m_token=m_tok) for m_tok in m_tokens]
            mot_embeds = [self.projection(self.quantizer.get_codebook_entry(tok.unsqueeze(0)-3)).squeeze(0) for tok in mot_ids]
            mot_attn_mask = [torch.ones(tok.size(0)).long().to(self.device) for tok in mot_ids]
        # Tokenize the input and targets
        # 1. Tokenize into input_ids
        if self.conf.get("add_motion_token_type", "token") == "token":
            eos_id = self.get_special_token_id("eos", is_learnable=False)
            pad_id = self.get_special_token_id("pad", is_learnable=False)
            eos_tok = torch.tensor(eos_id).view(1).long().to(self.device)
            pad_tok = torch.tensor(pad_id).view(1).long().to(self.device)
            # Get full text
            full_texts = [t + m for (t, m) in zip(input_texts, motion_strings)]
            
            attn_mask = []
            input_ids = []
            targ_labels = []
            for (inp_tex, ful_tex) in zip(input_texts, full_texts):
                inp_attn_mask, inp_ids = self.tokenize_valid(inp_string=[inp_tex], device=m_tokens.device, add_eos=False)
                ful_attn_mask, ful_ids = self.tokenize_valid(inp_string=[ful_tex], device=m_tokens.device, add_eos=True)
            
                targ_lbls = ful_ids[0].clone()
                targ_lbls[:inp_ids[0].size(0)] = -100
                # Pad at right
                pad_len = self.max_length - targ_lbls.size(0)
                pad_attn_mask = torch.zeros(pad_len).long().to(self.device)
                pad_ids = pad_tok.repeat(pad_len)
                pad_lbls = -100 * torch.ones(pad_len).long().to(self.device)
                
                attn_mask.append(torch.cat([ful_attn_mask[0], pad_attn_mask], dim=0))
                input_ids.append(torch.cat([ful_ids[0], pad_ids], dim=0))
                targ_labels.append(torch.cat([targ_lbls, pad_lbls], dim=0))
                
            attn_mask = torch.stack(attn_mask, dim=0)
            input_ids = torch.stack(input_ids, dim=0)
            targ_labels = torch.stack(targ_labels, dim=0)
            
            outputs = self.llm_model(
                input_ids=input_ids, 
                attention_mask=attn_mask, 
                labels=targ_labels, 
                output_hidden_states=True
            )
            logits = outputs.logits
            results = self.calculate_loss(pred_logits=logits, targ_labels=targ_labels)
        
        # 2. Tokenize into input_embeds
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            
            def get_special(token, is_learnable=True):
                id = self.get_special_token_id(token, is_learnable=is_learnable)
                tok = torch.tensor(id).view(1).long().to(self.device)
                if is_learnable:
                    emb = self.projection(self.motion_embeddings(tok))
                else:
                    emb = self.get_llm_embedding(tok)
                if token == "eos":
                    mask = torch.ones(1).long().to(self.device)     # [1]
                else:
                    mask = torch.zeros(1).long().to(self.device)    # [1]
                return tok, emb, mask
            
            m_eos_id, m_eos_emb, m_eos_mask = get_special("eos", is_learnable=True)
            m_pad_id, m_pad_emb, m_pad_mask = get_special("pad", is_learnable=True)
            
            # Tokenize input texts to embeddings
            tex_attn_mask, tex_ids = self.tokenize_valid(inp_string=input_texts, device=self.device, add_eos=False)
            tex_embeds = [self.get_llm_embedding(ids) for ids in tex_ids]
            # 
            attn_mask = []
            input_embeds = []
            targ_labels = []
            for (tex_mask, tex_id, tex_emb, mot_mask, mot_id, mot_emb) in zip(tex_attn_mask, tex_ids, tex_embeds, mot_attn_mask, mot_ids, mot_embeds):
                inp_ids = tex_id.clone()
                ful_ids = torch.cat([tex_id, mot_id, m_eos_id], dim=0)
                ful_mask = torch.cat([tex_mask, mot_mask, m_eos_mask], dim=0)
                ful_emb = torch.cat([tex_emb, mot_emb, m_eos_emb], dim=0)
                # Pad right
                pad_len = self.max_length - ful_emb.size(0)
                ful_mask = torch.cat([ful_mask, m_pad_mask.repeat(pad_len)], dim=0)
                ful_emb = torch.cat([ful_emb, m_pad_emb.repeat(pad_len, 1)], dim=0)
                # Get target labels
                targ_ids = -100 * torch.ones(self.max_length).long().to(self.device)
                targ_ids[:ful_ids.size(0)] = ful_ids
                targ_ids[:inp_ids.size(0)] = -100
                # Append
                attn_mask.append(ful_mask)
                input_embeds.append(ful_emb)
                targ_labels.append(targ_ids)
            
            attn_mask = torch.stack(attn_mask, dim=0)
            input_embeds = torch.stack(input_embeds, dim=0)
            targ_labels = torch.stack(targ_labels, dim=0)
                    
            outputs = self.llm_model(
                inputs_embeds=input_embeds, 
                attention_mask=attn_mask, 
                output_hidden_states=True
            )
            
            last_hidden_state = outputs.hidden_states[-1]
            logits = self.head(last_hidden_state)
            results = self.calculate_loss(pred_logits=logits, targ_labels=targ_labels)
            
        return results
    
    def motion_to_text(self, texts, m_tokens, loss_type=["pred"]):
        """[Training] Text-to-Motion, this is a middle-level generation task.
        :param texts: list of strings.
        :param m_tokens: [batch_size, seq_len] with <SOS>, <EOS>, and <PAD> appended.
        """
        # Generate prompts
        prompts = self.generate_prompts(task="m2t", num_prompts=len(texts))
        if self.conf.get("add_motion_token_type", "token") == "token":
            # Convert motion tokens to motion strings
            motion_strings = [self.convert_motion_token_to_string(m_token=m_tok) for m_tok in m_tokens]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            # # Convert motion tokens to motion embedding
            mot_ids = [self.get_valid_motion_token(m_token=m_tok) for m_tok in m_tokens]
            mot_embeds = [self.projection(self.quantizer.get_codebook_entry(tok.unsqueeze(0)-3)).squeeze(0) for tok in mot_ids]
            mot_attn_mask = [torch.ones(tok.size(0)).long().to(self.device) for tok in mot_ids]
        
        # Tokenize the input and targets
        # 1. Tokenize into input_ids
        if self.conf.get("add_motion_token_type", "token") == "token":
            eos_id = self.get_special_token_id("eos", is_learnable=False)
            pad_id = self.get_special_token_id("pad", is_learnable=False)
            eos_tok = torch.tensor(eos_id).view(1).long().to(self.device)
            pad_tok = torch.tensor(pad_id).view(1).long().to(self.device)
            # Get input texts and full texts
            input_texts = [p.format(m) for (p, m) in zip(prompts, motion_strings)]
            full_texts = [p + t for (p, t) in zip(input_texts, texts)]
            
            attn_mask = []
            input_ids = []
            targ_labels = []
            for (inp_tex, ful_tex) in zip(input_texts, full_texts):
                inp_attn_mask, inp_ids = self.tokenize_valid(inp_string=[inp_tex], device=m_tokens.device, add_eos=False)
                ful_attn_mask, ful_ids = self.tokenize_valid(inp_string=[ful_tex], device=m_tokens.device, add_eos=True)
            
                targ_lbls = ful_ids[0].clone()
                targ_lbls[:inp_ids[0].size(0)] = -100
                # Pad at right
                pad_len = self.max_length - targ_lbls.size(0)
                pad_attn_mask = torch.zeros(pad_len).long().to(self.device)
                pad_ids = pad_tok.repeat(pad_len)
                pad_lbls = -100 * torch.ones(pad_len).long().to(self.device)
                
                attn_mask.append(torch.cat([ful_attn_mask[0], pad_attn_mask], dim=0))
                input_ids.append(torch.cat([ful_ids[0], pad_ids], dim=0))
                targ_labels.append(torch.cat([targ_lbls, pad_lbls], dim=0))
                
            attn_mask = torch.stack(attn_mask, dim=0)
            input_ids = torch.stack(input_ids, dim=0)
            targ_labels = torch.stack(targ_labels, dim=0)
            
            outputs = self.llm_model(
                input_ids=input_ids, 
                attention_mask=attn_mask, 
                labels=targ_labels, 
                output_hidden_states=True
            )
            logits = outputs.logits
            results = self.calculate_loss(pred_logits=logits, targ_labels=targ_labels)
        
        # 2. Tokenize into input_embeds
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            
            def get_special(token, is_learnable=True):
                id = self.get_special_token_id(token, is_learnable=is_learnable)
                tok = torch.tensor(id).view(1).long().to(self.device)
                if is_learnable:
                    emb = self.projection(self.motion_embeddings(tok))
                else:
                    emb = self.get_llm_embedding(tok)
                if token == "eos":
                    mask = torch.ones(1).long().to(self.device)     # [1]
                else:
                    mask = torch.zeros(1).long().to(self.device)    # [1]
                return tok, emb, mask
            
            def tokenize_string_to_embedding(string, device):
                """The output skips <EOS>"""
                tokenization = self.tokenizer([string], return_tensors="pt")
                attn_mask = tokenization.attention_mask[:, :-1].to(device)
                ids = tokenization.input_ids[:, :-1].to(device)
                embeds = self.get_llm_embedding(ids)
                return attn_mask[0], ids[0], embeds[0]
            
            t_eos_id, t_eos_emb, t_eos_mask = get_special("eos", is_learnable=False)
            t_pad_id, t_pad_emb, t_pad_mask = get_special("pad", is_learnable=False)
                                    
            attn_mask = []
            input_embeds = []
            targ_labels = []
            for (prompt, text, mot_mask, mot_id, mot_emb) in zip(prompts, texts, mot_attn_mask, mot_ids, mot_embeds):
                # Decompose and tokenize the prompts
                ins_mask, ins_ids, ins_emb = tokenize_string_to_embedding(prompt.split("\n[Input]")[0], device=self.device)
                ipt_mask, ipt_ids, ipt_emb = tokenize_string_to_embedding("\n[Input] ", device=self.device)
                res_mask, res_ids, res_emb = tokenize_string_to_embedding("\n[Response] ", device=self.device)
                # Tokenize the target texts
                tex_mask, tex_ids, tex_emb = tokenize_string_to_embedding(text, device=self.device)
                # 
                inp_ids = torch.cat([ins_ids, ipt_ids, mot_id, res_ids], dim=0)
                ful_ids = torch.cat([ins_ids, ipt_ids, mot_id, res_ids, tex_ids, t_eos_id], dim=0)
                ful_mask = torch.cat([ins_mask, ipt_mask, mot_mask, res_mask, tex_mask, t_eos_mask], dim=0)
                ful_emb = torch.cat([ins_emb, ipt_emb, mot_emb, res_emb, tex_emb, t_eos_emb], dim=0)
                # Pad right
                pad_len = self.max_length - ful_emb.size(0)
                ful_mask = torch.cat([ful_mask, t_pad_mask.repeat(pad_len)], dim=0)
                ful_emb = torch.cat([ful_emb, t_pad_emb.repeat(pad_len, 1)], dim=0)
                # Get target labels
                targ_ids = -100 * torch.ones(self.max_length).long().to(self.device)
                targ_ids[:ful_ids.size(0)] = ful_ids
                targ_ids[:inp_ids.size(0)] = -100
                # Append
                attn_mask.append(ful_mask)
                input_embeds.append(ful_emb)
                targ_labels.append(targ_ids)
            
            attn_mask = torch.stack(attn_mask, dim=0)
            input_embeds = torch.stack(input_embeds, dim=0)
            targ_labels = torch.stack(targ_labels, dim=0)
                    
            outputs = self.llm_model(
                inputs_embeds=input_embeds, 
                attention_mask=attn_mask, 
                output_hidden_states=True
            )
            
            logits = outputs.logits
            results = self.calculate_loss(pred_logits=logits, targ_labels=targ_labels)
            
        return results
    
    def motion_to_motion(self, m_tokens, loss_type=["pred"]):
        """[Training] Motion-to-Motion, this is a middle-level motion-in-between task.
        :param m_tokens: [batch_size, seq_len] with <SOS>, <EOS>, and <PAD> appended.
        """
            
        def tokenize_string_to_embedding(string, device):
            """The output skips <EOS>"""
            tokenization = self.tokenizer([string], return_tensors="pt")
            attn_mask = tokenization.attention_mask[:, :-1].to(device)
            ids = tokenization.input_ids[:, :-1].to(device)
            embeds = self.get_llm_embedding(ids)
            return attn_mask, embeds
        
        # Generate prompts
        prompts = self.generate_prompts(task="m2m", num_prompts=len(m_tokens))
        # Get valid motion tokens from input motion tokens
        valid_m_tokens = [self.get_valid_motion_token(m_token=m_tok) for m_tok in m_tokens]
        # Convert motion tokens to motion strings or motion embeddings
        motion_conversion_dict = {"start": [], "end": [], "targ": [], "label": []}
        for m_tok in valid_m_tokens:
            m_len = len(m_tok)
            s_len = (m_len * 2) // 5    # Length of starting primitive
            e_len = (m_len * 2) // 5    # Length of ending primitive
            if self.conf.get("add_motion_token_type", "token") == "token":
                motion_conversion_dict["start"].append(self.convert_motion_token_to_string(m_token=m_tok[:s_len]))
                motion_conversion_dict["end"].append(self.convert_motion_token_to_string(m_token=m_tok[-e_len:]))
                motion_conversion_dict["targ"].append(self.convert_motion_token_to_string(m_token=m_tok[s_len:-e_len]))
            elif self.conf.get("add_motion_token_type", "token") == "mlp":
                motion_conversion_dict["start"].append(self.projection(self.quantizer.get_codebook_entry(m_tok[:s_len]-3)))
                motion_conversion_dict["end"].append(self.projection(self.quantizer.get_codebook_entry(m_tok[-e_len:]-3)))
                motion_conversion_dict["targ"].append(self.projection(self.quantizer.get_codebook_entry(m_tok[s_len:-e_len]-3)))
                motion_conversion_dict["label"].append(m_tok[s_len:-e_len])
        # Fill in the prompts and tokenize the filled prompts
        if self.conf.get("add_motion_token_type", "token") == "token":
            eos_id = self.get_special_token_id("eos", is_learnable=False)
            pad_id = self.get_special_token_id("pad", is_learnable=False)
            eos_tok = torch.tensor(eos_id).view(1).long().to(self.device)
            pad_tok = torch.tensor(pad_id).view(1).long().to(self.device)
            # Fill in the input and full prompts
            inp_texts = [p.format(s, e) for (p, s, e) in zip(prompts, motion_conversion_dict["start"], motion_conversion_dict["end"])]
            ful_texts = [p + m for (p, m) in zip(inp_texts, motion_conversion_dict["targ"])]
            attn_mask = []
            input_ids = []
            targ_labels = []
            for (inp_tex, ful_tex) in zip(inp_texts, ful_texts):
                inp_attn_mask, inp_ids = self.tokenize_valid(inp_string=[inp_tex], device=m_tokens.device, add_eos=False)
                ful_attn_mask, ful_ids = self.tokenize_valid(inp_string=[ful_tex], device=m_tokens.device, add_eos=True)
            
                targ_lbls = ful_ids[0].clone()
                targ_lbls[:inp_ids[0].size(0)] = -100
                # Pad at right
                pad_len = self.max_length - targ_lbls.size(0)
                pad_attn_mask = torch.zeros(pad_len).long().to(self.device)
                pad_ids = pad_tok.repeat(pad_len)
                pad_lbls = -100 * torch.ones(pad_len).long().to(self.device)
                
                attn_mask.append(torch.cat([ful_attn_mask[0], pad_attn_mask], dim=0))
                input_ids.append(torch.cat([ful_ids[0], pad_ids], dim=0))
                targ_labels.append(torch.cat([targ_lbls, pad_lbls], dim=0))
                
            attn_mask = torch.stack(attn_mask, dim=0)
            input_ids = torch.stack(input_ids, dim=0)
            targ_labels = torch.stack(targ_labels, dim=0)
            
            outputs = self.llm_model(
                input_ids=input_ids, 
                attention_mask=attn_mask, 
                labels=targ_labels, 
                output_hidden_states=True
            )
            logits = outputs.logits
            results = self.calculate_loss(pred_logits=logits, targ_labels=targ_labels)
            
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            
            def get_special(token, is_learnable=True):
                id = self.get_special_token_id(token, is_learnable=is_learnable)
                tok = torch.tensor(id).view(1).long().to(self.device)
                if is_learnable:
                    emb = self.projection(self.motion_embeddings(tok))
                else:
                    emb = self.get_llm_embedding(tok)
                if token == "eos":
                    mask = torch.ones(1).long().to(self.device)     # [1]
                else:
                    mask = torch.zeros(1).long().to(self.device)    # [1]
                return tok, emb, mask
            
            m_eos_id, m_eos_emb, m_eos_mask = get_special("eos", is_learnable=True)
            m_pad_id, m_pad_emb, m_pad_mask = get_special("pad", is_learnable=True)
            
            attn_mask, input_embeds, targ_labels = [], [], []
            for (prompt, mot_sta_emb, mot_end_emb, mot_targ_emb, mot_targ_label) in zip(
                prompts, motion_conversion_dict["start"], 
                motion_conversion_dict["end"], 
                motion_conversion_dict["targ"], 
                motion_conversion_dict["label"]):
                
                ins_attn_mask, ins_embed = tokenize_string_to_embedding(prompt.split("\n[Starting]")[0], device=self.device)
                sta_attn_mask, sta_embed = tokenize_string_to_embedding("\n[Starting] ", device=self.device)
                end_attn_mask, end_embed = tokenize_string_to_embedding("\n[Ending] ", device=self.device)
                res_attn_mask, res_embed = tokenize_string_to_embedding("\n[Response] ", device=self.device)

                sta_mot_attn_mask = torch.ones(1, mot_sta_emb.size(0)).long().to(self.device)
                end_mot_attn_mask = torch.ones(1, mot_end_emb.size(0)).long().to(self.device)
                targ_mot_attn_mask = torch.ones(1, mot_targ_emb.size(0)).long().to(self.device)
                                
                inp_embed = torch.cat([
                    ins_embed, sta_embed, mot_sta_emb.unsqueeze(dim=0), 
                    end_embed, mot_end_emb.unsqueeze(dim=0), res_embed
                ], dim=1)   # [1, T, C]
                ful_attn_mask = torch.cat([
                    ins_attn_mask, sta_attn_mask, sta_mot_attn_mask, 
                    end_attn_mask, end_mot_attn_mask, res_attn_mask, 
                    targ_mot_attn_mask, m_eos_mask.unsqueeze(dim=0)
                ], dim=1)   # [1, T]
                ful_embed = torch.cat([
                    ins_embed, sta_embed, mot_sta_emb.unsqueeze(dim=0), 
                    end_embed, mot_end_emb.unsqueeze(dim=0), res_embed, 
                    mot_targ_emb.unsqueeze(dim=0), m_eos_emb.unsqueeze(dim=0)
                ], dim=1)   # [1, T, C]
                # Pad right
                pad_len = self.max_length - ful_embed.size(1)
                pad_attn_mask = m_pad_mask.repeat(pad_len).unsqueeze(dim=0)
                pad_embed = m_pad_emb.repeat(pad_len, 1).unsqueeze(dim=0)
                # Get target labels
                targ_lbls = -100 * torch.ones(1, self.max_length).long().to(self.device)
                targ_lbls[:, inp_embed.size(1):ful_embed.size(1)] = torch.cat([mot_targ_label, m_eos_id], dim=0)
                
                attn_mask.append(torch.cat([ful_attn_mask, pad_attn_mask], dim=1))
                input_embeds.append(torch.cat([ful_embed, pad_embed], dim=1))
                targ_labels.append(targ_lbls)
                
            attn_mask = torch.cat(attn_mask, dim=0)
            input_embeds = torch.cat(input_embeds, dim=0)
            targ_labels = torch.cat(targ_labels, dim=0)
            
            outputs = self.llm_model(
                inputs_embeds=input_embeds, 
                attention_mask=attn_mask, 
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1]
            logits = self.head(last_hidden_state)
            results = self.calculate_loss(pred_logits=logits, targ_labels=targ_labels)
        
        return results
    
    def planning(self, batch, task=None, loss_type=["pred"]):
        """[Training] Decision Making tasks.
        :param batch: dictionary containing following items:
            1. scene: textual description of the scene information.
            2. cur_task: textual description of current task.
            3. cur_steps: textual description of executable steps corresponding to current task.
            4. next_task: textual description of next task.
            5. next_steps: textual description of executable steps corresponding to next task.
        """
        tasks = ["ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]
        batch_size = len(batch["scene"])
        input_texts = []
        full_texts = []
        for i in range(batch_size):
            if task is None:
                # Select a task
                task = random.choice(tasks)
            # Generate instruction prompts
            prompts = self.generate_prompts(task=task, num_prompts=1)
            # Get batch for current task
            cur_batch = {key: val[i] for key, val in batch.items()}
            # Fill out the input prompts
            inp_texts = self.get_input_prompts(prompts=prompts[0], batch=cur_batch, task=task)
            input_texts.append(inp_texts)
            # Get the target texts
            targ_texts = self.get_target_texts(batch=cur_batch, task=task)
            full_texts.append(inp_texts + targ_texts)
        # Tokenize the inputs and targets
        inp_attn_mask, inp_ids = self.tokenize_valid(inp_string=input_texts, device=self.device, add_eos=False)
        ful_attn_mask, ful_ids = self.tokenize_valid(inp_string=full_texts, device=self.device, add_eos=True)

        # Get inputs and targets
        input_ids, input_attn_mask, targ_labels = [], [], []
        for (inp_id, ful_id, ful_mask) in zip(inp_ids, ful_ids, ful_attn_mask):
            pad_len = self.max_length - ful_id.size(0)
            if pad_len > 0:
                pad_id = self.get_special_token_id("pad", is_learnable=False)
                pad_tok = torch.tensor(pad_id).view(1).long().to(self.device).repeat(pad_len)
                pad_mask = torch.zeros(pad_len).long().to(self.device)
                targ_lbls = -100 * torch.ones(self.max_length).long().to(self.device)
                targ_lbls[:ful_id.size(0)] = ful_id
                targ_lbls[:inp_id.size(0)] = -100
            else:
                targ_lbls = -100 * torch.ones(self.max_length).long().to(self.device)
                targ_lbls[:self.max_length] = ful_id[:self.max_length]
                targ_lbls[:inp_id.size(0)] = -100
            # Append
            if pad_len > 0:
                input_ids.append(torch.cat([ful_id, pad_tok], dim=0))
                input_attn_mask.append(torch.cat([ful_mask, pad_mask], dim=0))
                targ_labels.append(targ_lbls)
            else:
                input_ids.append(ful_id[:self.max_length])
                input_attn_mask.append(ful_mask[:self.max_length])
                targ_labels.append(targ_lbls)
        
        input_ids = torch.stack(input_ids, dim=0)
        input_attn_mask = torch.stack(input_attn_mask, dim=0)
        targ_labels = torch.stack(targ_labels, dim=0)
        
        outputs = self.llm_model(
            input_ids=input_ids, 
            attention_mask=input_attn_mask, 
            labels=targ_labels, 
            output_hidden_states=True
        )
        
        # Caculate the loss
        logits = outputs.logits
        results = self.calculate_loss(logits, targ_labels)
        
        return results
    
    def text_to_text(self, inp_text, targ_text, loss_type=["pred"]):
        """[Training] Text-to-Text, this is a high-level decision task.
        :param inp_text: list of text descriptions.
        :param
        """
        # Generate prompts
        prompts = self.generate_prompts(task="t2t", num_prompts=len(inp_text))
        # Decompose input texts
        dec_inp_text = self.decompose_input_text(inp_texts=inp_text, mode="input")
        scene_texts, cur_texts = [], []
        for t in dec_inp_text:
            scene_texts.append(t[0])
            cur_texts.append(t[1])
        # Decompose target texts
        next_texts = self.decompose_input_text(inp_texts=targ_text, mode="output")
        # Fill out the input texts
        input_texts = [p.format(s, c) for (p, s, c) in zip(prompts, scene_texts, cur_texts)]
        full_texts = [p + t for (p, t) in zip(input_texts, next_texts)]
        # Tokenize the inputs and targets
        inp_attn_mask, inp_ids = self.tokenize_valid(inp_string=input_texts, device=self.device, add_eos=False)
        ful_attn_mask, ful_ids = self.tokenize_valid(inp_string=full_texts, device=self.device, add_eos=True)
        # Get inputs and targets
        input_ids, input_attn_mask, targ_labels = [], [], []
        for (inp_id, ful_id, ful_mask) in zip(inp_ids, ful_ids, ful_attn_mask):
            pad_len = self.max_length - ful_id.size(0)
            if pad_len > 0:
                pad_id = self.get_special_token_id("pad", is_learnable=False)
                pad_tok = torch.tensor(pad_id).view(1).long().to(self.device).repeat(pad_len)
                pad_mask = torch.zeros(pad_len).long().to(self.device)
                targ_lbls = -100 * torch.ones(self.max_length).long().to(self.device)
                targ_lbls[:ful_id.size(0)] = ful_id
                targ_lbls[:inp_id.size(0)] = -100
            else:
                targ_lbls = -100 * torch.ones(self.max_length).long().to(self.device)
                targ_lbls[:self.max_length] = ful_id[:self.max_length]
                targ_lbls[:inp_id.size(0)] = -100
            # Append
            if pad_len > 0:
                input_ids.append(torch.cat([ful_id, pad_tok], dim=0))
                input_attn_mask.append(torch.cat([ful_mask, pad_mask], dim=0))
                targ_labels.append(targ_lbls)
            else:
                input_ids.append(ful_id[:self.max_length])
                input_attn_mask.append(ful_mask[:self.max_length])
                targ_labels.append(targ_lbls)
        
        input_ids = torch.stack(input_ids, dim=0)
        input_attn_mask = torch.stack(input_attn_mask, dim=0)
        targ_labels = torch.stack(targ_labels, dim=0)
        
        outputs = self.llm_model(
            input_ids=input_ids, 
            attention_mask=input_attn_mask, 
            labels=targ_labels, 
            output_hidden_states=True
        )
        
        # Caculate the loss
        logits = outputs.logits
        results = self.calculate_loss(logits, targ_labels)
        
        return results
    
    def scene_estimation(self, inp_text, targ_text, loss_type=["pred"]):
        """[Training] Scene-Estimation, this is a high-level decision task.
        :param inp_text: list of text descriptions.
        :param targ_text: list of text descriptions.
        """
        # Generate prompts
        prompts = self.generate_prompts(task="se", num_prompts=len(inp_text))
        # Decompose input texts
        dec_inp_text = self.decompose_input_text(inp_texts=inp_text, mode="input")
        scene_texts, cur_texts = [], []
        for t in dec_inp_text:
            scene_texts.append(t[0])
            cur_texts.append(t[1])
        # Decompose target texts
        next_texts = self.decompose_input_text(inp_texts=targ_text, mode="output")
        # Fill out the input texts
        input_texts = [p.format(c) for (p, c) in zip(prompts, cur_texts)]
        full_texts = [p + t for (p, t) in zip(input_texts, next_texts)]
        # Tokenize the inputs and targets
        inp_attn_mask, inp_ids = self.tokenize_valid(inp_string=input_texts, device=self.device, add_eos=False)
        ful_attn_mask, ful_ids = self.tokenize_valid(inp_string=full_texts, device=self.device, add_eos=True)
        # Get inputs and targets
        input_ids, input_attn_mask, targ_labels = [], [], []
        for (inp_id, ful_id, ful_mask) in zip(inp_ids, ful_ids, ful_attn_mask):
            pad_len = self.max_length - ful_id.size(0)
            if pad_len > 0:
                pad_id = self.get_special_token_id("pad", is_learnable=False)
                pad_tok = torch.tensor(pad_id).view(1).long().to(self.device).repeat(pad_len)
                pad_mask = torch.zeros(pad_len).long().to(self.device)
                targ_lbls = -100 * torch.ones(self.max_length).long().to(self.device)
                targ_lbls[:ful_id.size(0)] = ful_id
                targ_lbls[:inp_id.size(0)] = -100
            else:
                targ_lbls = -100 * torch.ones(self.max_length).long().to(self.device)
                targ_lbls[:self.max_length] = ful_id[:self.max_length]
                targ_lbls[:inp_id.size(0)] = -100
            # Append
            if pad_len > 0:
                input_ids.append(torch.cat([ful_id, pad_tok], dim=0))
                input_attn_mask.append(torch.cat([ful_mask, pad_mask], dim=0))
                targ_labels.append(targ_lbls)
            else:
                input_ids.append(ful_id[:self.max_length])
                input_attn_mask.append(ful_mask[:self.max_length])
                targ_labels.append(targ_lbls)
        
        input_ids = torch.stack(input_ids, dim=0)
        input_attn_mask = torch.stack(input_attn_mask, dim=0)
        targ_labels = torch.stack(targ_labels, dim=0)
        
        outputs = self.llm_model(
            input_ids=input_ids, 
            attention_mask=input_attn_mask, 
            labels=targ_labels, 
            output_hidden_states=True
        )
        
        # Caculate the loss
        logits = outputs.logits
        results = self.calculate_loss(logits, targ_labels)
        
        return results
        
    @torch.no_grad()
    def generate_text_to_motion(
        self, texts, topk=1, 
        min_num_tokens=10, 
        max_num_tokens=50, 
        use_semantic_sampling=False, 
        temperature=1.0
    ):
        """[Generation] Text-to-Motion, a middle-level generation task.
        """
        # Generate prompts
        prompts = self.generate_prompts(task="t2m", num_prompts=len(texts))
        # Fill in the prompts
        input_texts = [p.format(t) for (p, t) in zip(prompts, texts)]
        # Tokenize the input texts
        # 1. Tokenize into input_ids
        if self.conf.get("add_motion_token_type", "token") == "token":
            input_attn_mask, input_ids = self.tokenize_valid(inp_string=input_texts, device=self.device, add_eos=False)
            input_attn_mask = torch.stack(input_attn_mask, dim=0)
            input_ids = torch.stack(input_ids, dim=0)
        # 2. Tokenize into input_embeds
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            input_attn_mask, input_ids = self.tokenize_valid(inp_string=input_texts, device=self.device, add_eos=False)
            input_ids = torch.stack(input_ids, dim=0)
            input_attn_mask = torch.stack(input_attn_mask, dim=0)
            input_embeds = self.get_llm_embedding(input_ids)
        # Start to generate
        if self.conf.get("add_motion_token_type", "token") == "token":
            outputs = self.llm_model.generate(
                input_ids=input_ids, 
                attention_mask=input_attn_mask, 
                max_length=max_num_tokens,
                num_beams=1,
                do_sample=True if topk > 1 else False,
                bad_word_ids=None
            )                
            outputs = outputs[:, input_ids.size(1):]    # We only keep the predicted part
            pred_strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_tokens = self.convert_motion_string_to_token(m_string=pred_strings)
            pred_tokens = pred_tokens[0]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            pred_tokens = self.generate_motion_tokens_from_text(
                input_attn_mask=input_attn_mask, 
                input_embeds=input_embeds, 
                topk=topk, 
                max_num_tokens=max_num_tokens)
        
        return pred_tokens  # [T]
    
    @torch.no_grad()
    def generate_motion_to_text(
        self, m_tokens, topk=1, 
        max_num_tokens=50, 
        temperature=1.0
    ):
        """[Generation] Motion-to-Text, a middle-level understanding task.
        """
        # Generate prompts
        prompts = self.generate_prompts(task="m2t", num_prompts=len(m_tokens))
        if self.conf.get("add_motion_token_type", "token") == "token":
            # Convert motion tokens to motion strings
            motion_strings = [self.convert_motion_token_to_string(m_token=m_tok) for m_tok in m_tokens]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            # Convert motion tokens to motion embedding
            mot_ids = [self.get_valid_motion_token(m_token=m_tok) for m_tok in m_tokens]
            mot_embeds = [self.projection(self.quantizer.get_codebook_entry(tok.unsqueeze(0)-3)).squeeze(0) for tok in mot_ids]
            mot_attn_mask = [torch.ones(tok.size(0)).long().to(self.device) for tok in mot_ids]
        
        if self.conf.get("add_motion_token_type", "token") == "token":
            # Fill in the prompts
            input_texts = [p.format(m) for (p, m) in zip(prompts, motion_strings)]
            # Tokenize the input and targets
            input_attn_mask, input_ids = self.tokenize_valid(inp_string=input_texts, device=self.device)
            input_attn_mask = torch.stack(input_attn_mask, dim=0)
            input_ids = torch.stack(input_ids, dim=0)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            
            def tokenize_string_to_embedding(string, device):
                """The output skips <EOS>"""
                tokenization = self.tokenizer([string], return_tensors="pt")
                attn_mask = tokenization.attention_mask[:, :-1].to(device)
                ids = tokenization.input_ids[:, :-1].to(device)
                embeds = self.get_llm_embedding(ids)
                return attn_mask[0], ids[0], embeds[0]
            
            input_attn_mask = []
            input_embeds = []
            for (prompt, mot_mask, mot_id, mot_emb) in zip(prompts, mot_attn_mask, mot_ids, mot_embeds):
                # Decompose and tokenize the prompts
                ins_mask, ins_ids, ins_emb = tokenize_string_to_embedding(prompt.split("\n[Input]")[0], device=self.device)
                ipt_mask, ipt_ids, ipt_emb = tokenize_string_to_embedding("\n[Input] ", device=self.device)
                res_mask, res_ids, res_emb = tokenize_string_to_embedding("\n[Response] ", device=self.device)
                
                inp_emb = torch.cat([ins_emb, ipt_emb, mot_emb, res_emb], dim=0)
                inp_mask = torch.cat([ins_mask, ipt_mask, mot_mask, res_mask], dim=0)
                
                input_attn_mask.append(inp_mask)
                input_embeds.append(inp_emb)
                
            input_attn_mask = torch.stack(input_attn_mask, dim=0)
            input_embeds = torch.stack(input_embeds, dim=0)
            
        # Start to generate
        if self.conf.get("add_motion_token_type", "token") == "token":
            outputs = self.llm_model.generate(
                input_ids=input_ids, 
                attention_mask=input_attn_mask, 
                max_length=max_num_tokens,
                num_beams=1,
                do_sample=True if topk > 1 else False,
                bad_word_ids=None
            )
            outputs = outputs[:, input_ids.size(1):]    # Only keep the predicted part
            pred_strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_strings = pred_strings[0]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            outputs = self.generate_text_tokens_from_motion(
                input_attn_mask=input_attn_mask, 
                input_embeds=input_embeds, 
                topk=topk, 
                max_num_tokens=max_num_tokens)
            pred_strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_strings = pred_strings[0]
        
        return pred_strings
    
    @torch.no_grad()
    def generate_motion_to_motion(
        self, m_start_tokens, m_end_tokens, topk=1, 
        max_num_tokens=5, 
        use_semantic_sampling=False, 
        temperature=1.0
    ):
        """[Generation] Motion-to-Motion, a middle-level motion-in-between task.
        """
        def tokenize_string_to_embedding(string, device):
            """The output skips <EOS>"""
            tokenization = self.tokenizer([string], return_tensors="pt")
            attn_mask = tokenization.attention_mask[:, :-1].to(device)
            ids = tokenization.input_ids[:, :-1].to(device)
            embeds = self.get_llm_embedding(ids)
            return attn_mask, embeds
        
        # Generate prompts
        prompts = self.generate_prompts(task="m2m", num_prompts=len(m_start_tokens))
        # Get valid motion tokens from input motion tokens
        valid_m_start_tokens = [self.get_valid_motion_token(m_token=m_tok) for m_tok in m_start_tokens]
        valid_m_end_tokens = [self.get_valid_motion_token(m_token=m_tok) for m_tok in m_end_tokens]
        
        if self.conf.get("add_motion_token_type", "token") == "token":
            # Convert motion tokens to motion strings
            motion_conversion_dict = {"start": [], "end": []}
            for (m_sta_tok, m_end_tok) in zip(valid_m_start_tokens, valid_m_end_tokens):
                motion_conversion_dict["start"].append(self.convert_motion_token_to_string(m_sta_tok))
                motion_conversion_dict["end"].append(self.convert_motion_token_to_string(m_end_tok))
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            # Convert motion tokens to motion embeddings
            motion_conversion_dict = {"start": [], "end": []}
            for (m_sta_tok, m_end_tok) in zip(valid_m_start_tokens, valid_m_end_tokens):
                motion_conversion_dict["start"].append(self.projection(self.quantizer.get_codebook_entry(m_sta_tok-3)))
                motion_conversion_dict["end"].append(self.projection(self.quantizer.get_codebook_entry(m_end_tok-3)))
        # Fill in the inputs
        if self.conf.get("add_motion_token_type", "token") == "token":
            input_texts = [p.format(s, e) for (p, s, e) in zip(prompts, motion_conversion_dict["start"], motion_conversion_dict["end"])]
            # Tokenize the input and targets
            input_attn_mask, input_ids = self.tokenize_valid(inp_string=input_texts, device=m_tokens.device)
            input_attn_mask = torch.stack(input_attn_mask, dim=0)
            input_ids = torch.stack(input_ids, dim=0)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            input_attn_mask, input_embeds = [], []
            for (prompt, mot_sta_emb, mot_end_emb) in zip(prompts, motion_conversion_dict["start"], motion_conversion_dict["end"]):
                ins_attn_mask, ins_embed = tokenize_string_to_embedding(prompt.split("\n[Starting]")[0], device=self.device)
                sta_attn_mask, sta_embed = tokenize_string_to_embedding("\n[Starting] ", device=self.device)
                end_attn_mask, end_embed = tokenize_string_to_embedding("\n[Ending] ", device=self.device)
                res_attn_mask, res_embed = tokenize_string_to_embedding("\n[Response] ", device=self.device)

                sta_mot_attn_mask = torch.ones(1, mot_sta_emb.size(0)).long().to(self.device)
                end_mot_attn_mask = torch.ones(1, mot_end_emb.size(0)).long().to(self.device)
                
                inp_emb = torch.cat([ins_embed, sta_embed, mot_sta_emb.unsqueeze(dim=0), end_embed, mot_end_emb.unsqueeze(dim=0), res_embed], dim=1)
                inp_mask = torch.cat([ins_attn_mask, sta_attn_mask, sta_mot_attn_mask, end_attn_mask, end_mot_attn_mask, res_attn_mask], dim=1)
                
                input_attn_mask.append(inp_mask)
                input_embeds.append(inp_emb)
            
            input_attn_mask = torch.cat(input_attn_mask, dim=0)
            input_embeds = torch.cat(input_embeds, dim=0)
            
        # Start to generate
        if self.conf.get("add_motion_token_type", "token") == "token":
            outputs = self.llm_model.generate(
                input_ids=input_ids, 
                max_length=max_num_tokens,
                num_beams=1,
                do_sample=True if topk > 1 else False,
                bad_word_ids=None
            )
            outputs = outputs[:, input_ids.size(1):]    # We only keep the predicted part
            pred_strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_tokens = self.convert_motion_string_to_token(m_string=pred_strings)
            pred_tokens = pred_tokens[0]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            pred_tokens = self.generate_motion_tokens_from_motion_primitives(
                input_attn_mask=input_attn_mask, 
                input_embeds=input_embeds, 
                topk=topk, 
                max_num_tokens=max_num_tokens)
        
        return pred_tokens  # [T]
    
    @torch.no_grad()
    def generate_planning(
        self, batch, task="ct2t", 
        topk=1, max_num_tokens=50, 
        temperature=1.0
    ):
        """[Generation] Decision Making tasks.
        :param batch: dictionary containing following items:
            1. scene: textual description of the scene information.
            2. cur_task: textual description of current task.
            3. cur_steps: textual description of executable steps corresponding to current task.
            4. next_task: textual description of next task.
            5. next_steps: textual description of executable steps corresponding to next task.
        """
        # Generate prompts
        prompts = self.generate_prompts(task=task, num_prompts=1)
        # Fill out the input prompts
        cur_batch = {key: val[0] for key, val in batch.items()}
        inp_texts = self.get_input_prompts(prompts=prompts[0], batch=cur_batch, task=task)
        # Tokenize the inputs and targets
        input_attn_mask, input_ids = self.tokenize_valid(inp_string=inp_texts, device=self.device, add_eos=False)
        input_attn_mask = torch.stack(input_attn_mask, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        
        # Start to generate
        outputs = self.llm_model.generate(
            input_ids=input_ids, 
            max_length=max_num_tokens,
            num_beams=1,
            do_sample=True if topk > 1 else False,
            bad_word_ids=None
        )
        outputs = outputs[:, input_ids.size(1):]
        pred_strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred_strings = pred_strings[0]
        
        return pred_strings
        
    @torch.no_grad()
    def generate_text_to_text(
        self, inp_text, topk=1, 
        max_num_tokens=50
    ):
        # Generate prompts
        prompts = self.generate_prompts(task="t2t", num_prompts=len(inp_text))
        # Decompose input texts
        dec_inp_text = self.decompose_input_text(inp_texts=inp_text, mode="input")
        scene_texts, cur_texts = [], []
        for t in dec_inp_text:
            scene_texts.append(t[0])
            cur_texts.append(t[1])
        # Fill out the input texts
        input_texts = [p.format(s, c) for (p, s, c) in zip(prompts, scene_texts, cur_texts)]
        # Tokenize the inputs
        input_attn_mask, input_ids = self.tokenize_valid(inp_string=input_texts, device=self.device, add_eos=False)
        input_attn_mask = torch.stack(input_attn_mask, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        
        # Start to generate
        outputs = self.llm_model.generate(
            input_ids=input_ids, 
            max_length=max_num_tokens,
            num_beams=1,
            do_sample=True if topk > 1 else False,
            bad_word_ids=None
        )
        pred_strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        pred_strings = pred_strings[0]
        
        return pred_strings
    
    @torch.no_grad()
    def generate_scene_estimation(
        self, inp_text, topk=1, 
        max_num_tokens=50
    ):
        """[Generation] Scene-Estimation, a high-level decision task.
        """
        # Generate prompts
        prompts = self.generate_prompts(task="se", num_prompts=len(inp_text))
        # Decompose input texts
        dec_inp_text = self.decompose_input_text(inp_texts=inp_text, mode="input")
        scene_texts, cur_texts = [], []
        for t in dec_inp_text:
            scene_texts.append(t[0])
            cur_texts.append(t[1])
        # Fill out the input texts
        input_texts = [p.format(c) for (p, c) in zip(prompts, cur_texts)]
        # Tokenize the inputs
        input_attn_mask, input_ids = self.tokenize_valid(inp_string=input_texts, device=self.device, add_eos=False)
        input_attn_mask = torch.stack(input_attn_mask, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        
        # Start to generate
        outputs = self.llm_model.generate(
            input_ids=input_ids, 
            max_length=max_num_tokens,
            num_beams=1,
            do_sample=True if topk > 1 else False,
            bad_word_ids=None
        )
        pred_strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        pred_strings = pred_strings[0]
        
        return pred_strings
    
if __name__ == "__main__":
    import yaml, importlib
    with open("configs/llm_gpt/gpt_large/config_gpt_large_exp1_pretrain.yaml", "r") as f:
        conf = yaml.safe_load(f)
    
    model = AvatarGPT(conf["models"]["gpt"])
    
    for v, k in model.named_parameters():
        print(v, k.shape)
        # if not k.requires_grad: print(v, k.shape)
        # print(v, "|", k.shape)
        
    quantizer = importlib.import_module(conf["models"]["vqvae"]["t2m"]["body"]["quantizer"]["arch_path"], package="networks").__getattribute__(
        conf["models"]["vqvae"]["t2m"]["body"]["quantizer"]["arch_name"])(**conf["models"]["vqvae"]["t2m"]["body"]["quantizer"]).to(model.device)
    model.set_quantizer(quantizer=quantizer)
        
    prompts = [
        "The man crouches down remarkably low before springing up in a disappointing leap.",
        "With exceptional flexibility, the man lowers himself into a deep squat and then launches himself upwards in a lackluster jump.",
        "In an astonishing display of agility, the man squats down incredibly low only to disappoint with a feeble jump.",
        "With an extraordinary level of flexibility, the man drops down into a deep squat and quickly rises up in an unsatisfying jump.",
        "The man impresses with his ability to squat exceptionally low, but his subsequent jump leaves much to be desired.",
    ]
    
    m_tokens = torch.randint(3, 515, size=(len(prompts), 50))
    
    # model.pretrain(texts=prompts, m_tokens=m_tokens)
    # model.text_to_motion(texts=prompts, m_tokens=m_tokens)
    # model.motion_to_text(texts=prompts, m_tokens=m_tokens)
    # model.motion_to_motion(m_tokens=m_tokens)
    
    model.generate_motion_to_motion(m_start_tokens=m_tokens[:1, :20], m_end_tokens=m_tokens[:, -20:])