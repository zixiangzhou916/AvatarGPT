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
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    T5Model, 
    T5Config, 
    GPT2Tokenizer, 
    GPT2LMHeadModel
)
from transformers.modeling_outputs import (
    Seq2SeqLMOutput, 
    BaseModelOutput
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

def minimize_special_token_logits(logits, k):
    """
    :param logits: [num_seq, num_dim]
    """
    dim = logits.dim()
    if dim == 3:
        logits = logits.squeeze(dim=1)
    out = logits.clone()
    out[..., :k] = -float('Inf')
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
        self.model_type = conf.get("model_type", "t5")
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
        with open("networks/llm/instruction_template.json", "r") as f:
            self.instruction_template = json.load(f)
        
    def build_llm(self, conf, logger=None):
        print_log(logger=logger, log_str="Build language model")
        if self.model_type == "t5":
            self.llm_model = T5ForConditionalGeneration.from_pretrained(conf["model"])
            self.lm_type = "encdec" # encoder-decoder
        elif self.model_type == "gpt":
            self.llm_model = GPT2LMHeadModel.from_pretrained(conf["model"])
            self.lm_type = "dec" # encoder-decoder
        elif self.model_type == "llama":
            pass
        
    def build_tokenizer(self, conf, logger=None):
        print_log(logger=logger, log_str="Build tokenizer")
        if self.model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(conf["tokenizer"], legacy=True)
        elif self.model_type == "gpt":
            self.tokenizer = GPT2Tokenizer.from_pretrained(conf["tokenizer"], legacy=True)
        elif self.model_type == "llama":
            pass
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        if conf.get("add_motion_token_type", "token") == "token":
            # Append motion vocabulary to LLM vocabulary
            print_log(logger=logger, log_str="Resize the toke embeddings from {:d} to {:d}".format(
                len(self.tokenizer), len(self.tokenizer)+conf.get("n_motion_tokens", 512)+3))
            self.tokenizer.add_tokens(
                ["<motion_id_{:d}>".format(i) for i in range(conf.get("n_motion_tokens", 512)+3)]
            )
            self.llm_model.resize_token_embeddings(len(self.tokenizer)+conf.get("n_motion_tokens", 512)+3)
        elif conf.get("add_motion_token_type", "token") == "mlp":
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
                                        conf.get("d_model", 768), bias=False)
        
        if conf.get("head_type", "shared") == "shared":
            # If we use 'shared' head, we use the LLM's head
            pass
        elif conf.get("head_type", "shared") == "separate":
            # If we use 'separate head, we train a separate head for motion token prediction
            self.head = nn.Linear(conf.get("d_model", 768), conf.get("n_motion_tokens", 512)+3, bias=False)
            
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
    
    def shift_right(self, input):
        sos_id = 0
        if input.dim() == 2:    # [B, T]
            sos_tok = torch.tensor(sos_id).view(1, 1).long().to(self.device)
            sos_tok = sos_tok.repeat(input.size(0), 1)      # [B, T]
            output = torch.cat([sos_tok, input[:, :-1]], dim=1)
        elif input.dim() == 3:  # [B, T, C]
            sos_tok = torch.tensor(sos_id).view(1, 1).long().to(self.device)
            sos_emb = self.get_llm_embedding(sos_tok)
            sos_emb = sos_emb.repeat(input.size(0), 1, 1)   # [B, T, C]
            output = torch.cat([sos_emb, input[:, :-1]], dim=1)
        return output
    
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
    
    def convert_input_of_motion_to_text_task_to_embeds(self, prompts, m_tokens, device):
        """Convert the inputs of motion-to-text task(prompts and motion tokens) to embeddings.
        :param prompts: list of string.
        :param m_token: [batch_size, seq_len]
        """
        sos_id = self.get_special_token_id("sos", is_learnable=True)
        sos_token = torch.tensor(sos_id).long().view(1).to(m_tokens.device)
        eos_id = self.get_special_token_id("eos", is_learnable=True)
        eos_token = torch.tensor(eos_id).long().view(1).to(m_tokens.device)
        pad_id = self.get_special_token_id("pad", is_learnable=True)
        pad_token = torch.tensor(pad_id).long().view(1).to(m_tokens.device)
        
        def tokenize_string_to_embedding(string, device):
            """The output skips <EOS>"""
            tokenization = self.tokenizer([string], return_tensors="pt")
            attn_mask = tokenization.attention_mask[:, :-1].to(device)
            ids = tokenization.input_ids[:, :-1].to(device)
            embeds = self.get_llm_embedding(ids)
            return attn_mask, embeds
        
        def tokenize_token_to_embedding(token, device):
            """The """
            mask = token.gt(pad_id)
            valid_token = token[mask]
            valid_token -= 3
            
            with torch.no_grad():
                valid_embed = self.quantizer.get_codebook_entry(valid_token)
            valid_embed = self.projection(valid_embed)
            sos_embed = self.projection(self.motion_embeddings(sos_token))
            eos_embed = self.projection(self.motion_embeddings(eos_token))
            padded_embed = torch.cat([sos_embed, valid_embed, eos_embed], dim=0)
            attn_mask = torch.ones(padded_embed.size(0)).long().to(device)
            return attn_mask.unsqueeze(dim=0), padded_embed.unsqueeze(dim=0)
        
        attn_masks = []
        input_embeds = []
        for (prompt, m_token) in zip(prompts, m_tokens):
            ins_attn_mask, ins_embed = tokenize_string_to_embedding(prompt.split("\n[Input]")[0], device=device)
            inp_attn_mask, inp_embed = tokenize_string_to_embedding("\n[Input] ", device=device)
            res_attn_mask, res_embed = tokenize_string_to_embedding("\n[Response] ", device=device)
            mot_attn_mask, mot_embed = tokenize_token_to_embedding(m_token, device=device)
            eos_attn_mask = torch.ones(1, 1).long().to(device)
            eos_ids_ = torch.tensor(self.get_special_token_id("eos", is_learnable=False)).long().view(1, 1).to(device)
            eos_embed = self.get_llm_embedding(tokens=eos_ids_)
            
            attn_mask = torch.cat([ins_attn_mask, inp_attn_mask, mot_attn_mask, res_attn_mask, eos_attn_mask], dim=1)
            input_embed = torch.cat([ins_embed, inp_embed, mot_embed, res_embed, eos_embed], dim=1)
            
            pad_len = self.max_length - input_embed.size(1)
            pad_attn_mask = torch.zeros(1, pad_len).long().to(device)
            pad_ids_ = torch.tensor(self.get_special_token_id("pad", is_learnable=False)).long().view(1, 1).to(device)
            pad_embed = self.get_llm_embedding(tokens=pad_ids_.repeat(1, pad_len))
            
            attn_masks.append(torch.cat([attn_mask, pad_attn_mask], dim=1))
            input_embeds.append(torch.cat([input_embed, pad_embed], dim=1))

        return torch.cat(attn_masks, dim=0), torch.cat(input_embeds, dim=0)
            
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
        
    def tokenize(self, inp_string, device, output_type="ids"):
        tokenize_output = self.tokenizer(
            inp_string, 
            padding="max_length", 
            max_length=self.max_length, 
            truncation=True, 
            # return_special_tokens=True, 
            return_tensors="pt")
        attn_mask = tokenize_output.attention_mask.to(device)
        ids = tokenize_output.input_ids.to(device)
        return attn_mask, ids
    
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
        pred_tokens = pred_logits.argmax(dim=-1)
        losses["pred"] = loss_fct(
            pred_logits.contiguous().view(-1, pred_logits.size(-1)), 
            targ_labels.contiguous().view(-1))
        accuracy["pred"] = self.calc_prediction_accuracy(
            pred_tokens, targ_labels, 
            ignore_cls=-100)
        
        results = {
            "losses": losses,
            "accuracy": accuracy, 
            "pred_tokens": pred_tokens, 
            "target_tokens": targ_labels
        }
        
        return results
    
    @torch.no_grad()
    def random_spans_noise_mask(self, length):
        # From https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens,
                                                  num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                     num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length, ), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
    
    @torch.no_grad()
    def create_sentinel_ids(self, mask_indices):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        start_indices = mask_indices - np.roll(mask_indices, 1,
                                               axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0,
                                np.cumsum(start_indices, axis=-1),
                                start_indices)
        sentinel_ids = np.where(sentinel_ids != 0,
                                (len(self.tokenizer) - sentinel_ids - (self.m_codebook_size + 3)), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    @torch.no_grad()
    def filter_input_ids(self, input_ids, sentinel_ids):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids,
                                  input_ids.to('cpu'))

        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape(
            (batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1),
                        self.tokenizer.eos_token_id,
                        dtype=np.int32),
            ],
            axis=-1,
        )

        input_ids = torch.tensor(input_ids, device=self.device)

        return input_ids
    
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
    def generate_motion_tokens_from_text(
        self, input_attn_mask, input_embeds, 
        topk=1, max_num_tokens=50, temperature=1.0
    ):
        sos_id = self.get_special_token_id("sos", is_learnable=True)
        eos_id = self.get_special_token_id("eos", is_learnable=True)
        pad_id = self.get_special_token_id("pad", is_learnable=True)
        sos_tok = torch.tensor(sos_id).view(1, 1).long().to(self.device)    # [1, 1]
        # sos_emb = self.get_llm_embedding(sos_tok)                           # [1, 1, D]
        # sos_emb = self.projection(self.motion_embeddings(sos_tok))                           # [1, 1, D]
        """
        FIXME: Because we prepended two sos_emb at the decoder_inputs_embeds during training, 
        we need to initialize the pred_embeds with two sos_emb.
        """
        sos_emb = torch.cat([
            self.get_llm_embedding(sos_tok), 
            self.projection(self.motion_embeddings(sos_tok))
        ], dim=1)
        
        pred_embeds = sos_emb.clone()
        pred_tokens = []
        pred_attn_mask = torch.ones(1, 2).long().to(self.device)
        while len(pred_tokens) < max_num_tokens:
            # Predict next token
            outputs = self.llm_model(
                inputs_embeds=input_embeds, 
                attention_mask=input_attn_mask, 
                decoder_inputs_embeds=pred_embeds, 
                decoder_attention_mask=pred_attn_mask, 
                output_hidden_states=True
            )
            last_hidden_state = outputs.decoder_hidden_states[-1][:, -1:]
            raw_pred_logit = self.head(last_hidden_state)
            if topk == 1:
                # Sample the token with highest probability
                pred_logit = F.softmax(raw_pred_logit.clone(), dim=-1)
                pred_token = pred_logit.argmax(dim=-1)
            else:
                # Sample one token from tokens with top-k probability
                pred_logit = top_k_logits(raw_pred_logit.clone(), k=topk)
                pred_logit = F.softmax(pred_logit / temperature, dim=-1)
                pred_token = torch.multinomial(pred_logit[:, 0], num_samples=1)
            # print('--- predicted token: ', pred_token)
            if pred_token.item() > pad_id:
                pred_tokens.append(pred_token)
                pred_emb = self.projection(self.quantizer.get_codebook_entry(pred_token-3))
                attn_mask = torch.ones(1, 1).long().to(self.device)
                pred_embeds = torch.cat([pred_embeds, pred_emb], dim=1)
                pred_attn_mask = torch.cat([pred_attn_mask, attn_mask], dim=1)
            else:
                if len(pred_tokens) == 0:
                    pred_logit = minimize_special_token_logits(raw_pred_logit.clone(), k=3)
                    pred_logit = top_k_logits(pred_logit, k=topk)
                    pred_logit = F.softmax(pred_logit / temperature, dim=-1)
                    pred_token = torch.multinomial(pred_logit[:, 0], num_samples=1)
                    pred_tokens.append(pred_token)
                    pred_emb = self.projection(self.quantizer.get_codebook_entry(pred_token-3))
                    attn_mask = torch.ones(1, 1).long().to(self.device)
                    pred_embeds = torch.cat([pred_embeds, pred_emb], dim=1)
                    pred_attn_mask = torch.cat([pred_attn_mask, attn_mask], dim=1)
                else:
                    break
            
        return torch.cat(pred_tokens, dim=1).squeeze(dim=0) # [T]
    
    @torch.no_grad()
    def generate_motion_tokens_from_motion_primitives(
        self, input_attn_mask, input_embeds, 
        topk=1, max_num_tokens=50, temperature=1.0
    ):
        return self.generate_motion_tokens_from_text(
            input_attn_mask=input_attn_mask, 
            input_embeds=input_embeds, 
            topk=topk, 
            max_num_tokens=max_num_tokens, 
            temperature=temperature)
    
    @torch.no_grad()
    def generate_text_tokens_from_motion(
        self, input_attn_mask, input_embeds, 
        topk=1, max_num_tokens=50, temperature=1.0
    ):
        sos_id = 0
        eos_id = self.get_special_token_id("eos", is_learnable=False)
        pad_id = self.get_special_token_id("pad", is_learnable=False)
        sos_tok = torch.tensor(sos_id).view(1, 1).long().to(self.device)    # [1, 1]
        sos_emb = self.get_llm_embedding(sos_tok)                           # [1, 1, D]
        
        pred_embeds = sos_emb.clone()
        pred_tokens = []
        pred_attn_mask = torch.ones(1, 1).long().to(self.device)
        while len(pred_tokens) < max_num_tokens:
            # Predict next token
            outputs = self.llm_model(
                inputs_embeds=input_embeds, 
                attention_mask=input_attn_mask, 
                decoder_inputs_embeds=pred_embeds, 
                decoder_attention_mask=pred_attn_mask, 
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
                pred_logit = F.softmax(pred_logit / temperature, dim=-1)
                pred_token = torch.multinomial(pred_logit[:, 0], num_samples=1)
            if pred_token.item() > eos_id:
                pred_tokens.append(pred_token)
                pred_emb = self.get_llm_embedding(pred_token)
                attn_mask = torch.ones(1, 1).long().to(self.device)
                pred_embeds = torch.cat([pred_embeds, pred_emb], dim=1)
                pred_attn_mask = torch.cat([pred_attn_mask, attn_mask], dim=1)
            else:
                break
        
        return torch.cat(pred_tokens, dim=1)
    
    def pretrain(self, texts, m_tokens, loss_type=["pred"]):
        # Tokenize text prompts
        if self.conf.get("add_motion_token_type", "token") == "token":
            tex_attn_mask, tex_ids = self.tokenize(
                inp_string=texts, device=m_tokens.device)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            tex_attn_mask, tex_ids = self.tokenize(
                inp_string=texts, device=m_tokens.device)
            tex_embeds = self.get_llm_embedding(tex_ids)
        
        # Tokenize motion tokens
        if self.conf.get("add_motion_token_type", "token") == "token":
            # Convert motion tokens to motion strings
            motion_strings = [self.convert_motion_token_to_string(m_token=m_tok) for m_tok in m_tokens]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            # Convert motion tokens to motion embedding
            mot_attn_mask, mot_embeds = self.convert_motion_token_to_embeds(m_tokens=m_tokens)

        # condition = random.choice(["text", "motion", "supervised", "supervised", "supervised"])
        condition = random.choice(["supervised", "supervised", "supervised"])
        if condition == "text":
            inputs = texts
            outputs = texts
        elif condition == "motion":
            inputs = motion_strings
            outputs = motion_strings
        else:
            if self.conf.get("add_motion_token_type", "token") == "token":
                inputs, outputs = [], []
                for (t, m) in zip(texts, motion_strings):
                    if random.random() < 0.5:
                        inputs.append(t)
                        outputs.append(m)
                    else:
                        inputs.append(m)
                        outputs.append(t)
                
                input_attn_mask, input_ids = self.tokenize(
                    inp_string=inputs, device=m_tokens.device)
                lables_attention_mask, labels_input_ids = self.tokenize(
                    inp_string=outputs, device=m_tokens.device)
                ignore_id = self.get_special_token_id("pad", is_learnable=False)
                labels_input_ids[labels_input_ids == ignore_id] = -100
                outputs = self.llm_model(
                    input_ids=input_ids, 
                    attention_mask=None, 
                    labels=labels_input_ids, 
                    decoder_attention_mask=None, 
                    output_hidden_states=True
                )
                
                logits = outputs.logits
                # last_hidden_state = outputs.decoder_hidden_states[-1]
                # Caculate the loss
                results = self.calculate_loss(logits, labels_input_ids)
                
            elif self.conf.get("add_motion_token_type", "token") == "mlp":
                inputs, outputs, labels, label_tags = [], [], [], []
                sos_id = 0  # Decoder start token ID
                sos_tok = torch.tensor(sos_id).view(1).long().to(self.device)
                sos_emb = self.get_llm_embedding(sos_tok)
                for (te, me, tl, ml) in zip(tex_embeds, mot_embeds, tex_ids, m_tokens):
                    if random.random() < 0.5:
                        inputs.append(te)                                           # Text embedding
                        outputs.append(torch.cat([sos_emb, me[:-1]], dim=0))        # Motion embedding, manually shift right
                        ignore_id = self.get_special_token_id("pad", is_learnable=True)
                        ignore_tok = torch.tensor(ignore_id).view(1).long().to(self.device)
                        lbl = ml.clone()
                        lbl = torch.cat([lbl, ignore_tok.repeat(self.max_length-lbl.size(0))], dim=0)
                        lbl[lbl == ignore_id] = -100
                        labels.append(lbl)
                        label_tags.append("motion")
                    else:
                        inputs.append(me)                                       # Motion embedding
                        outputs.append(torch.cat([sos_emb, te[:-1]], dim=0))    # Text embedding, manuall shift right
                        ignore_id = self.get_special_token_id("pad", is_learnable=False)
                        lbl = tl.clone()
                        lbl[tl == ignore_id] = -100
                        labels.append(lbl)
                        label_tags.append("text")
                inputs = torch.stack(inputs, dim=0)
                outputs = torch.stack(outputs, dim=0)
                labels = torch.stack(labels, dim=0)
                outputs = self.llm_model(
                    inputs_embeds=inputs, 
                    attention_mask=None, 
                    decoder_inputs_embeds=outputs, 
                    decoder_attention_mask=None, 
                    output_hidden_states=True
                )
                
                logits = outputs.logits
                last_hidden_state = outputs.decoder_hidden_states[-1]
            
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
                        labels[motion_ids])
                    update_result(src=results_mot, targ=results, src_num=len(motion_ids), total_num=len(label_tags))
                if len(text_ids) > 0:
                    results_tex = self.calculate_loss(logits[text_ids], labels[text_ids])
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
            targ_attn_mask, targ_embeds = self.convert_motion_token_to_embeds(m_tokens=m_tokens)
            targ_embeds = self.shift_right(input=targ_embeds)
        # Tokenize the input and targets
        # 1. Tokenize the inputs
        if self.conf.get("add_motion_token_type", "token") == "token":
            input_attn_mask, input_ids = self.tokenize(inp_string=input_texts, device=m_tokens.device)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            input_attn_mask, input_ids = self.tokenize(inp_string=input_texts, device=m_tokens.device)
            input_embeds = self.get_llm_embedding(tokens=input_ids)
        # 2. Tokenize the targets
        if self.conf.get("add_motion_token_type", "token") == "token":
            targ_attn_mask, targ_ids = self.tokenize(inp_string=motion_strings, device=m_tokens.device)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            pass
        # Generate target labels
        if self.conf.get("add_motion_token_type", "token") == "token":
            ignore_id = self.get_special_token_id("pad", is_learnable=False)
            targ_labels = targ_ids.clone()
            targ_labels[targ_ids == ignore_id] = -100
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            targ_labels = -100 * torch.ones(len(texts), self.max_length).long().to(self.device)
            for b, (mask, label) in enumerate(zip(targ_attn_mask, m_tokens)):
                valid_len = min(mask.sum().item(), label.size(0))
                targ_labels[b, :valid_len] = label[:valid_len]
        
        if self.conf.get("add_motion_token_type", "token") == "token":
            outputs = self.llm_model(
                input_ids=input_ids, 
                attention_mask=input_attn_mask, 
                labels=targ_labels, 
                decoder_attention_mask=targ_attn_mask, 
                output_hidden_states=True
            )
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            outputs = self.llm_model(
                inputs_embeds=input_embeds, 
                attention_mask=input_attn_mask, 
                decoder_inputs_embeds=targ_embeds, 
                decoder_attention_mask=targ_attn_mask, 
                output_hidden_states=True
            )
        
        if self.conf.get("head_type", "shared") == "shared":
            # If we use 'shared' head, we use the LLM's head
            # loss = outputs.loss
            logits = outputs.logits
            # last_hidden_state = outputs.decoder_hidden_states[-1]
        elif self.conf.get("head_type", "shared") == "separate":
            last_hidden_state = outputs.decoder_hidden_states[-1]
            logits = self.head(last_hidden_state)
        
        # Caculate the loss
        results = self.calculate_loss(logits, targ_labels)
        
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
            # mot_attn_mask, mot_embeds = self.convert_motion_token_to_embeds(m_tokens=m_tokens)
            pass
        # Tokenize the input and targets
        # 1. Tokenize the inputs
        if self.conf.get("add_motion_token_type", "token") == "token":
            input_texts = [p.format(m) for (p, m) in zip(prompts, motion_strings)]
            input_attn_mask, input_ids = self.tokenize(inp_string=input_texts, device=m_tokens.device)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            input_attn_mask, input_embeds = self.convert_input_of_motion_to_text_task_to_embeds(
                prompts=prompts, m_tokens=m_tokens, device=m_tokens.device)
        # 2. Tokenize the targets
        if self.conf.get("add_motion_token_type", "token") == "token":
            targ_attn_mask, targ_ids = self.tokenize(inp_string=texts, device=m_tokens.device)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            targ_attn_mask, targ_ids = self.tokenize(inp_string=texts, device=m_tokens.device)
            targ_embeds = self.get_llm_embedding(tokens=targ_ids)
            targ_embeds = self.shift_right(input=targ_embeds)
        # Generate target labels
        if self.conf.get("add_motion_token_type", "token") == "token":
            ignore_id = self.get_special_token_id("pad", is_learnable=False)
            targ_labels = targ_ids.clone()
            targ_labels[targ_ids == ignore_id] = -100
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            targ_labels = -100 * torch.ones(len(texts), self.max_length).long().to(self.device)
            for b, (mask, label) in enumerate(zip(targ_attn_mask, targ_ids)):
                valid_len = min(mask.sum().item(), label.size(0))
                targ_labels[b, :valid_len] = label[:valid_len]
            
        if self.conf.get("add_motion_token_type", "token") == "token":
            outputs = self.llm_model(
                input_ids=input_ids, 
                attention_mask=input_attn_mask, 
                labels=targ_labels, 
                decoder_attention_mask=targ_attn_mask, 
                output_hidden_states=True
            )
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            outputs = self.llm_model(
                inputs_embeds=input_embeds, 
                attention_mask=input_attn_mask, 
                decoder_inputs_embeds=targ_embeds, 
                decoder_attention_mask=targ_attn_mask, 
                output_hidden_states=True
            )
        
        # For motion-to-text task, we always use LLM head
        logits = outputs.logits        
        # Caculate the loss
        results = self.calculate_loss(logits, targ_labels)
        
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
        
        def get_inputs(prompts, inp_start, inp_end, device):
            """Generate input embeddings and input attention masks."""
            attn_masks = []
            input_embeds = []
            for (p, sta_mot_embed, end_mot_embed) in zip(prompts, inp_start, inp_end):
                ins_attn_mask, ins_embed = tokenize_string_to_embedding(p.split("\n[Starting]")[0], device=device)
                sta_attn_mask, sta_embed = tokenize_string_to_embedding("\n[Starting] ", device=device)
                end_attn_mask, end_embed = tokenize_string_to_embedding("\n[Ending] ", device=device)
                res_attn_mask, res_embed = tokenize_string_to_embedding("\n[Response] ", device=device)
                sta_mot_attn_mask = torch.ones(1, sta_mot_embed.size(0)).long().to(device)
                end_mot_attn_mask = torch.ones(1, end_mot_embed.size(0)).long().to(device)
                
                eos_attn_mask = torch.ones(1, 1).long().to(device)
                eos_ids_ = torch.tensor(self.get_special_token_id("eos", is_learnable=False)).long().view(1, 1).to(device)
                eos_embed = self.get_llm_embedding(tokens=eos_ids_)
                
                attn_mask = torch.cat([
                    ins_attn_mask, sta_attn_mask, sta_mot_attn_mask, 
                    end_attn_mask, end_mot_attn_mask, res_attn_mask, eos_attn_mask
                ], dim=1)   # [1, T]
                input_embed = torch.cat([
                    ins_embed, sta_embed, sta_mot_embed.unsqueeze(dim=0), 
                    end_embed, end_mot_embed.unsqueeze(dim=0), res_embed, eos_embed
                ], dim=1)   # [1, T, C]
                
                pad_len = self.max_length - input_embed.size(1)
                pad_attn_mask = torch.zeros(1, pad_len).long().to(device)
                pad_ids_ = torch.tensor(self.get_special_token_id("pad", is_learnable=False)).long().view(1, 1).to(device)
                pad_embed = self.get_llm_embedding(tokens=pad_ids_.repeat(1, pad_len))
                
                attn_masks.append(torch.cat([attn_mask, pad_attn_mask], dim=1))
                input_embeds.append(torch.cat([input_embed, pad_embed], dim=1))
            
            return torch.cat(attn_masks, dim=0), torch.cat(input_embeds, dim=0)
        
        def get_targets(inp_embeds, inp_labels, device):
            """Get target embeddings, target attention masks, and target labels."""
            sos_id = 0
            sos_tok = torch.tensor(sos_id).view(1).long().to(device)
            sos_emb = self.get_llm_embedding(sos_tok)
            eos_id = self.get_special_token_id("eos", is_learnable=True)
            eos_tok = torch.tensor(eos_id).view(1).long().to(device)
            eos_emb = self.get_llm_embedding(eos_tok)
            pad_id = self.get_special_token_id("pad", is_learnable=True)
            pad_tok = torch.tensor(pad_id).view(1).long().to(device)
                
            targ_attn_masks, targ_embeds, targ_labels = [], [], []
            for (emb, lbl) in zip(inp_embeds, inp_labels):
                pad_len = self.max_length - emb.size(0)
                pad_emb = self.get_llm_embedding(pad_tok).repeat(pad_len, 1)
                mask = torch.zeros(self.max_length).long().to(device)
                mask[:emb.size(0)+1] = 1
                embeds = torch.cat([emb, eos_emb, pad_emb[:-1]], dim=0)
                targ_attn_masks.append(mask)
                targ_embeds.append(torch.cat([sos_emb, embeds[:-1]], dim=0))    # Right shift
                labels = -100 * torch.ones(self.max_length).long().to(device)
                labels[:lbl.size(0)] = lbl
                labels[lbl.size(0)] = eos_id
                targ_labels.append(labels)
            targ_attn_masks = torch.stack(targ_attn_masks, dim=0)
            targ_embeds = torch.stack(targ_embeds, dim=0)
            targ_labels = torch.stack(targ_labels, dim=0)
            return targ_attn_masks, targ_embeds, targ_labels
        
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
                # motion_conversion_dict["start"].append(self.get_llm_embedding(tokens=m_tok[:s_len]))
                # motion_conversion_dict["end"].append(self.get_llm_embedding(tokens=m_tok[-e_len:]))
                # motion_conversion_dict["targ"].append(self.get_llm_embedding(tokens=m_tok[s_len:-e_len]))
                motion_conversion_dict["start"].append(self.projection(self.quantizer.get_codebook_entry(m_tok[:s_len]-3)))
                motion_conversion_dict["end"].append(self.projection(self.quantizer.get_codebook_entry(m_tok[-e_len:]-3)))
                motion_conversion_dict["targ"].append(self.projection(self.quantizer.get_codebook_entry(m_tok[s_len:-e_len]-3)))
                motion_conversion_dict["label"].append(m_tok[s_len:-e_len])
        # Fill in the prompts
        if self.conf.get("add_motion_token_type", "token") == "token":
            input_texts = [p.format(s, e) for (p, s, e) in zip(prompts, motion_conversion_dict["start"], motion_conversion_dict["end"])]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            input_attn_mask, input_embeds = get_inputs(
                prompts=prompts, 
                inp_start=motion_conversion_dict["start"], 
                inp_end=motion_conversion_dict["end"], 
                device=self.device)
        # Tokenize the input and targets
        # 1. Tokenize the input
        if self.conf.get("add_motion_token_type", "token") == "token":
            input_attn_mask, input_ids = self.tokenize(inp_string=input_texts, device=m_tokens.device)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            pass
        
        # 2. Tokenize the target
        if self.conf.get("add_motion_token_type", "token") == "token":
            targ_attn_mask, targ_ids = self.tokenize(inp_string=motion_conversion_dict["targ"], device=m_tokens.device)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            targ_attn_mask, targ_embeds, targ_labels = get_targets(
                motion_conversion_dict["targ"], 
                motion_conversion_dict["label"], 
                device=self.device)
        
        # Generate target labels
        if self.conf.get("add_motion_token_type", "token") == "token":
            ignore_id = self.get_special_token_id("pad", is_learnable=False)
            targ_labels = targ_ids.clone()
            targ_labels[targ_ids == ignore_id] = -100
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            pass
        
        if self.conf.get("add_motion_token_type", "token") == "token":
            outputs = self.llm_model(
                input_ids=input_ids, 
                attention_mask=input_attn_mask, 
                labels=targ_labels, 
                decoder_attention_mask=targ_attn_mask, 
                output_hidden_states=True
            )
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            outputs = self.llm_model(
                inputs_embeds=input_embeds, 
                decoder_inputs_embeds=targ_embeds, 
                decoder_attention_mask=targ_attn_mask, 
                output_hidden_states=True
            )
        
        if self.conf.get("head_type", "shared") == "shared":
            # If we use 'shared' head, we use the LLM's head
            # loss = outputs.loss
            logits = outputs.logits
            # last_hidden_state = outputs.decoder_hidden_states[-1]
        elif self.conf.get("head_type", "shared") == "separate":
            last_hidden_state = outputs.decoder_hidden_states[-1]
            logits = self.head(last_hidden_state)
        
        # Caculate the loss
        results = self.calculate_loss(logits, targ_labels)
        
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
        target_texts = []
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
            target_texts.append(targ_texts)
        # Tokenize the inputs and targets
        input_attn_mask, input_ids = self.tokenize(inp_string=input_texts, device=self.device, output_type="ids")
        targ_attn_mask, targ_ids = self.tokenize(inp_string=target_texts, device=self.device)
        # Generate target labels
        ignore_id = self.get_special_token_id("pad", is_learnable=False)
        targ_labels = targ_ids.clone()
        targ_labels[targ_ids == ignore_id] = -100
        
        outputs = self.llm_model(
            input_ids=input_ids, 
            attention_mask=input_attn_mask, 
            labels=targ_labels, 
            decoder_attention_mask=targ_attn_mask, 
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
        # Tokenize the input and targets
        if self.conf.get("add_motion_token_type", "token") == "token":
            input_attn_mask, input_ids = self.tokenize(inp_string=input_texts, device=self.device)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            input_attn_mask, input_ids = self.tokenize(inp_string=input_texts, device=self.device)
            input_embeds = self.get_llm_embedding(tokens=input_ids)
        # Start to generate
        if self.conf.get("add_motion_token_type", "token") == "token":
            outputs = self.llm_model.generate(
                input_ids=input_ids, 
                max_length=max_num_tokens,
                num_beams=1,
                do_sample=True if topk > 1 else False,
                bad_word_ids=None
            )                
            pred_strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_tokens = self.convert_motion_string_to_token(m_string=pred_strings)
            pred_tokens = pred_tokens[0]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            pred_tokens = self.generate_motion_tokens_from_text(
                input_attn_mask=input_attn_mask, 
                input_embeds=input_embeds, 
                topk=topk, 
                max_num_tokens=max_num_tokens, 
                temperature=temperature)
            
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
            pass
            
        if self.conf.get("add_motion_token_type", "token") == "token":
            # Fill in the prompts
            input_texts = [p.format(m) for (p, m) in zip(prompts, motion_strings)]
            # Tokenize the input and targets
            input_attn_mask, input_ids = self.tokenize(inp_string=input_texts, device=self.device)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            input_attn_mask, input_embeds = self.convert_input_of_motion_to_text_task_to_embeds(
                prompts=prompts, m_tokens=m_tokens, device=self.device)
        
        # Start to generate
        if self.conf.get("add_motion_token_type", "token") == "token":
            outputs = self.llm_model.generate(
                input_ids=input_ids, 
                max_length=max_num_tokens,
                num_beams=1,
                do_sample=True if topk > 1 else False,
                bad_word_ids=None
            )
            pred_strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_strings = pred_strings[0]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            outputs = self.generate_text_tokens_from_motion(
                input_attn_mask=input_attn_mask, 
                input_embeds=input_embeds, 
                topk=topk, 
                max_num_tokens=max_num_tokens, 
                temperature=temperature)
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
        
        def get_inputs(prompts, inp_start, inp_end, device):
            """Generate input embeddings and input attention masks."""
            attn_masks = []
            input_embeds = []
            for (p, sta_mot_embed, end_mot_embed) in zip(prompts, inp_start, inp_end):
                ins_attn_mask, ins_embed = tokenize_string_to_embedding(p.split("\n[Starting]")[0], device=device)
                sta_attn_mask, sta_embed = tokenize_string_to_embedding("\n[Starting] ", device=device)
                end_attn_mask, end_embed = tokenize_string_to_embedding("\n[Ending] ", device=device)
                res_attn_mask, res_embed = tokenize_string_to_embedding("\n[Response] ", device=device)
                sta_mot_attn_mask = torch.ones(1, sta_mot_embed.size(0)).long().to(device)
                end_mot_attn_mask = torch.ones(1, end_mot_embed.size(0)).long().to(device)
                
                eos_attn_mask = torch.ones(1, 1).long().to(device)
                eos_ids_ = torch.tensor(self.get_special_token_id("eos", is_learnable=False)).long().view(1, 1).to(device)
                eos_embed = self.get_llm_embedding(tokens=eos_ids_)
                
                attn_mask = torch.cat([
                    ins_attn_mask, sta_attn_mask, sta_mot_attn_mask, 
                    end_attn_mask, end_mot_attn_mask, res_attn_mask, eos_attn_mask
                ], dim=1)   # [1, T]
                input_embed = torch.cat([
                    ins_embed, sta_embed, sta_mot_embed.unsqueeze(dim=0), 
                    end_embed, end_mot_embed.unsqueeze(dim=0), res_embed, eos_embed
                ], dim=1)   # [1, T, C]
                
                pad_len = self.max_length - input_embed.size(1)
                pad_attn_mask = torch.zeros(1, pad_len).long().to(device)
                pad_ids_ = torch.tensor(self.get_special_token_id("pad", is_learnable=False)).long().view(1, 1).to(device)
                pad_embed = self.get_llm_embedding(tokens=pad_ids_.repeat(1, pad_len))
                
                attn_masks.append(torch.cat([attn_mask, pad_attn_mask], dim=1))
                input_embeds.append(torch.cat([input_embed, pad_embed], dim=1))
            
            return torch.cat(attn_masks, dim=0), torch.cat(input_embeds, dim=0)
        
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
            input_attn_mask, input_ids = self.tokenize(inp_string=input_texts, device=self.device)
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            input_attn_mask, input_embeds = get_inputs(
                prompts=prompts, 
                inp_start=motion_conversion_dict["start"], 
                inp_end=motion_conversion_dict["end"], 
                device=self.device)
        
        # Start to generate
        if self.conf.get("add_motion_token_type", "token") == "token":
            outputs = self.llm_model.generate(
                input_ids=input_ids, 
                max_length=max_num_tokens,
                num_beams=1,
                do_sample=True if topk > 1 else False,
                bad_word_ids=None
            )
            pred_strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_tokens = self.convert_motion_string_to_token(m_string=pred_strings)
            pred_tokens = pred_tokens[0]
        elif self.conf.get("add_motion_token_type", "token") == "mlp":
            pred_tokens = self.generate_motion_tokens_from_motion_primitives(
                input_attn_mask=input_attn_mask, 
                input_embeds=input_embeds, 
                topk=topk, 
                max_num_tokens=max_num_tokens, 
                temperature=temperature)
        
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
        # Generate instruction prompts
        prompts = self.generate_prompts(task=task, num_prompts=1)
        # Fill out the input prompts
        cur_batch = {key: val[0] for key, val in batch.items()}
        inp_texts = self.get_input_prompts(prompts=prompts[0], batch=cur_batch, task=task)
        # Tokenize the inputs and targets
        input_attn_mask, input_ids = self.tokenize(inp_string=inp_texts, device=self.device, output_type="ids")
        
        # Generate responses
        # if temperature > 1.0:
        #     sos_id = 0
        #     eos_id = self.get_special_token_id("eos", is_learnable=False)
        #     sos_tok = torch.tensor(sos_id).view(1, 1).long().to(self.device)    # [1, 1]
        #     sos_emb = self.get_llm_embedding(sos_tok) 
        #     pred_embeds = sos_emb.clone()
        #     pred_attn_mask = torch.ones(1, 1).long().to(self.device)
        #     pred_tokens = []
        #     while len(pred_tokens) < max_num_tokens:
        #         outputs = self.llm_model(
        #             input_ids=input_ids, 
        #             attention_mask=input_attn_mask, 
        #             decoder_inputs_embeds=pred_embeds, 
        #             decoder_attention_mask=pred_attn_mask, 
        #             output_hidden_states=True
        #         )
        #         raw_pred_logit = outputs.logits[:, -1:]
        #         pred_logit = top_k_logits(raw_pred_logit.clone(), k=topk)
                                
        #         pred_logit = F.softmax(pred_logit / temperature, dim=-1)   # Make the probability distribution more smooth
        #         # np.savetxt("logit.txt", pred_logit[0,0].data.cpu().numpy(), fmt="%.8f")
        #         pred_token = torch.multinomial(pred_logit[:, 0], num_samples=100, replacement=True)   # [1, num_sample]
        #         # print(pred_token)
        #         random_sample = np.random.randint(0, 100)
        #         pred_token = pred_token[:, random_sample:random_sample+1]

        #         if pred_token.item() > eos_id:
        #             pred_tokens.append(pred_token)
        #             pred_emb = self.get_llm_embedding(pred_token)
        #             attn_mask = torch.ones(1, 1).long().to(self.device)
        #             pred_embeds = torch.cat([pred_embeds, pred_emb], dim=1)
        #             pred_attn_mask = torch.cat([pred_attn_mask, attn_mask], dim=1)
        #         else:
        #             break
        #     outputs = torch.cat(pred_tokens, dim=1)
        # else:
        #     outputs = self.llm_model.generate(
        #         input_ids=input_ids, 
        #         max_length=max_num_tokens,
        #         num_beams=1,
        #         do_sample=True if topk > 1 else False,
        #         bad_word_ids=None, 
        #         top_k=10
        #     )
        outputs = self.llm_model.generate(
            input_ids=input_ids, 
            max_length=max_num_tokens,
            num_beams=1,
            do_sample=True if topk > 1 else False,
            bad_word_ids=None, 
            top_k=topk
        )
        pred_strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred_strings = pred_strings[0]
                
        return pred_strings