import argparse, os, sys
sys.path.append(os.getcwd())
from collections import defaultdict

import codecs as cs
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import transformers
from packaging import version
from transformers import (AutoModel, AutoTokenizer, GPT2Tokenizer)
from transformers import __version__ as trans_version

class BertScore(nn.Module):
    def __init__(self, args):
        super(BertScore, self).__init__()
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # token_path = os.path.join(cur_path, "motion_to_text", "roberta-large-pretrained", "tokenizer")
        # roberta_path = os.path.join(cur_path, "motion_to_text", "roberta-large-pretrained", "model")
        print("Load pretrained RoBERTa tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print("Load pretrained RoBERTa checkpoint")
        self.model = AutoModel.from_pretrained(args.model).to(self.device)
        self.model.eval()
        if hasattr(self.model, "decoder") and hasattr(self.model, "encoder"):
            self.model = self.model.encoder
        baseline_path = args.baseline_path
        self.baselines = torch.from_numpy(
            pd.read_csv(baseline_path).iloc[args.num_layers].to_numpy()
        )[1:].float().to(self.device)
        
        # drop unused layers
        if hasattr(self.model, "n_layers"):  # XLM
            assert (
                    0 <= args.num_layers <= self.model.n_layers
                ), f"Invalid num_layers: num_layers should be between 0 and {self.model.n_layers}"
        elif hasattr(self.model, "layer"):   # XLNet
            assert (
                0 <= args.num_layers <= len(self.model.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(self.model.layers)}"
            self.model.layer = torch.nn.ModuleList(
                [layer for layer in self.model.layer[:args.num_layers]]
            )
        elif hasattr(self.model, "encoder"): # Albert
            if hasattr(self.model.encoder, "albert_layer_groups"):
                assert (
                    0 <= args.num_layers <= self.model.encoder.config.num_hidden_layers
                ), f"Invalid num_layers: num_layers should be between 0 and {self.model.encoder.config.num_hidden_layers}"
                self.model.encoder.config.num_hidden_layers = args.num_layers
            elif hasattr(self.model.encoder, "block"):       # T5
                assert (
                    0 <= args.num_layers <= len(self.model.encoder.block)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(self.model.encoder.block)}"
                self.model.encoder.block = torch.nn.ModuleList(
                    [layer for layer in self.model.encoder.block[:self.num_layers]]
                )
            else:   # Bert, Roberta
                assert (
                    0 <= args.num_layers <= len(self.model.encoder.layer)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(self.model.encoder.layer)}"
                self.model.encoder.layer = torch.nn.ModuleList(
                    [layer for layer in self.model.encoder.layer[:args.num_layers]]
                )
        else:
            raise NotImplementedError
        
        # load IDF
        self.idf_dict = defaultdict(lambda: 1.0)
        self.idf_dict[self.tokenizer.sep_token_id] = 0
        self.idf_dict[self.tokenizer.cls_token_id] = 0
        
        # print(self.model)
        
    def to(self, device):
        self.model = self.model.to(device)
    
    @torch.no_grad()  
    def forward(self, refs, cands, batch_size, verbose=False, 
                all_layers=False, 
                rescale_with_baseline=True, 
                baseline_path=None):
        """Compute BERTScore.
        :param refs: (list of str), reference sentences
        :param cands: (list of str), candidate sentences
        :param batch_size: (int), BERT score processing batch size
        :param verbose: (bool) turn on intermediate status update
        :param all_layers: (bool)
        """
        
        # Reorg texts
        ref_group_boundaries = None
        if not isinstance(refs[0], str):
            ref_group_boundaries = []
            ori_cands, ori_refs = cands, refs
            cands, refs = [], []
            count = 0
            for cand, ref_group in zip(ori_cands, ori_refs):
                cands += [cand] * len(ref_group)
                refs += ref_group
                ref_group_boundaries.append((count, count + len(ref_group)))
                count += len(ref_group)
        
        preds = []
        
        # Compute BERT embedding
        sentences = self.dedup_and_sort(refs + cands)
        embds = []
        iter_range = range(0, len(sentences), batch_size)
        if verbose:
            print("computing bert embedding...")
            iter_range = tqdm(iter_range)
        stats_dict = dict()
        for batch_start in iter_range:
            sen_batch = sentences[batch_start:batch_start+batch_size]
            embds, masks, padded_idf = self.get_bert_embedding(
                sen_batch, all_layers=all_layers
            )
            for i, sen in enumerate(sen_batch):
                sequence_len = masks[i].sum().item()
                emb = embds[i, :sequence_len]
                idf = padded_idf[i, :sequence_len]
                stats_dict[sen] = (emb, idf)
            
        iter_range = range(0, len(refs), batch_size)
        if verbose:
            print("computing greedy matching")
            iter_range = tqdm(iter_range)
            
        for batch_start in iter_range:
            batch_refs = refs[batch_start:batch_start+batch_size]
            batch_cands = cands[batch_start:batch_start+batch_size]
            ref_stats = self.pad_batch_stats(batch_refs, stats_dict)
            cands_stats = self.pad_batch_stats(batch_cands, stats_dict)
            
            P, R, F1 = self.greedy_cos_idf(*ref_stats, *cands_stats, all_layers)
            preds.append(torch.stack((P, R, F1), dim=-1))
        
        all_preds = torch.cat(preds, dim=1 if all_layers else 0)
        
        # Get BERT score
        if ref_group_boundaries is not None:
            max_preds = []
            for beg, end in ref_group_boundaries:
                max_preds.append(all_preds[beg:end].max(dim=0)[0])
            all_preds = torch.stack(max_preds, dim=0)
        
        # Rescale the BERT score to adjust the score range
        use_custom_baseline = baseline_path is not None
        if rescale_with_baseline:
            all_preds = (all_preds - self.baselines) / (1 - self.baselines)
            
        P, R, F = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]
        
        return P, R, F
    
    def pad_batch_stats(self, sen_batch, stats_dict):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        emb = [e.to(self.device) for e in emb]
        idf = [i.to(self.device) for i in idf]
        lens = [e.size(0) for e in emb]
        emb_pad = torch.nn.utils.rnn.pad_sequence(emb, batch_first=True, padding_value=2.0)
        idf_pad = torch.nn.utils.rnn.pad_sequence(idf, batch_first=True)
        
        pad_mask = self.length_to_mask(lens).to(self.device)
        return emb_pad, pad_mask, idf_pad
        
    def dedup_and_sort(self, l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)
    
    def get_bert_embedding(self, all_sens, batch_size=-1, all_layers=False):
        """Compute BERT embedding in batches.
        """
        padded_sens, padded_idf, lens, mask = self.collate_idf(all_sens)
        if batch_size == -1:
            batch_size = len(all_sens)
        
        embeddings = []
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = self.bert_encode(x=padded_sens[i:i+batch_size], 
                                               attention_mask=mask[i:i+batch_size], 
                                               all_layers=all_layers)
            embeddings.append(batch_embedding)
            del batch_embedding
        
        total_embedding = torch.cat(embeddings, dim=0)
        return total_embedding, mask, padded_idf
        
    def collate_idf(self, arr):
        """Helper function that pads a list of sentences to have the same length and 
        loads idf score for words in the sentences.
        """
        arr = [self.sent_encode(a) for a in arr]
        idf_weights = [[self.idf_dict[i] for i in a] for a in arr]
        pad_token = self.tokenizer.pad_token_id
        
        padded, lens, mask = self.padding(arr, pad_token, dtype=torch.long)
        padded_idf, *_ = self.padding(idf_weights, 0, dtype=torch.float)
        
        padded = padded.to(self.device)
        mask = mask.to(self.device)
        lens = lens.to(self.device)
        
        return padded, padded_idf, lens, mask
        
    def sent_encode(self, sent):
        sent = sent.strip()
        if sent == "":
            return self.tokenizer.build_inputs_with_special_tokens([])
        elif isinstance(self.tokenizer, GPT2Tokenizer):
            # for RoBERTa and GPT-2
            if version().parse(trans_version) >= version.parse("4.0.0"):
                return self.tokenizer.encode(
                    sent, 
                    add_special_tokens=True, 
                    add_prefix_space=True, 
                    max_length=self.tokenizer.model_max_length, 
                    truncation=True, 
                )
            elif version.parse(trans_version) >= version.parse("3.0.0"):
                return self.tokenizer.encode(
                    sent, 
                    add_special_tokens=True, 
                    add_prefix_space=True, 
                    max_length=self.tokenizer.max_len, 
                    truncation=True, 
                )
            elif version.parse(trans_version) >= version.parse("2.0.0"):
                return self.tokenizer.encode(
                    sent, 
                    add_special_tokens=True, 
                    add_prefix_space=True, 
                    max_length=self.tokenizer.max_len, 
                )
            else:
                raise NotImplementedError(
                    f"transformers version{trans_version} is not supported"
                )
        else:
            if version.parse(trans_version) >= version.parse("4.0.0"):
                return self.tokenizer.encode(
                    sent, 
                    add_special_tokens=True, 
                    max_length=self.tokenizer.model_max_length, 
                    truncation=True, 
                )
            elif version.parse(trans_version) >= version.parse("3.0.0"):
                return self.tokenizer.encode(
                    sent, 
                    add_special_tokens=True, 
                    max_length=self.tokenizer.max_len, 
                    truncation=True, 
                )
            elif version.parse(trans_version) >= version.parse("2.0.0"):
                return self.tokenizer.encode(
                    sent, 
                    add_special_tokens=True, 
                    max_length=self.tokenizer.max_len
                )
            else:
                raise NotImplementedError(
                    f"transformers version {trans_version} is not supported"
                )
    
    @staticmethod
    def padding(arr, pad_token, dtype=torch.long):
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=dtype)
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, :lens[i]] = 1
        return padded, lens, mask
    
    def bert_encode(self, x, attention_mask, all_layers=False):
        out = self.model(x, attention_mask=attention_mask, output_hidden_states=all_layers)
        if all_layers:
            emb = torch.stack(out[-1], dim=2)
        else:
            emb = out[0]
        return emb
    
    @staticmethod
    def length_to_mask(lens):
        lens = torch.tensor(lens, dtype=torch.long)
        max_len = max(lens)
        base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
        return base < lens.unsqueeze(1)
    
    @staticmethod
    def greedy_cos_idf( 
        ref_embedding, 
        ref_masks, 
        ref_idf, 
        hyp_embedding, 
        hyp_masks, 
        hyp_idf, 
        all_layers=False, 
    ):
        """
        Compute greedy matching based on cosing similarity.
        :pram ref_embedding: 
        :pram ref_masks: 
        :pram ref_idf: 
        :pram hyp_embedding: 
        :pram hyp_idf: 
        :pram all_layers: 
        """
        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
        
        if all_layers:
            B, _, L, D = hyp_embedding.size()
            hyp_embedding = (
                hyp_embedding.transpose(1, 2)
                .transpose(0, 1)
                .contiguous()
                .view(L*B, hyp_embedding.size(1), D)
            )
            ref_embedding = (
                ref_embedding.transpose(1, 2)
                .transpose(0, 1)
                .contiguous()
                .view(L*B, ref_embedding.size(1), D)
            )
        batch_size = ref_embedding.size(0)
        sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
        masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
        if all_layers:
            masks = masks.unsqueeze(0).expand(L, -1, -1, -1).contiguous().view_as(sim)
        else:
            masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
        
        masks = masks.float().to(sim.device)
        sim = sim * masks
        
        word_precision = sim.max(dim=2)[0]
        word_recall = sim.max(dim=1)[0]
        
        hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
        ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
        precision_scale = hyp_idf.to(word_precision.device)
        recall_scale = ref_idf.to(word_recall.device)
        if all_layers:
            precision_scale = (
                precision_scale.unsqueeze(0)
                .expand(L, B, -1)
                .contiguous()
                .view_as(word_precision)
            )
            recall_scale = (
                recall_scale.unsqueeze(0)
                .expand(L, B, -1)
                .contiguous()
                .view_as(word_recall)
            )
        P = (word_precision * precision_scale).sum(dim=1)
        R = (word_recall * recall_scale).sum(dim=1)
        F = 2 * P * R / (P + R)
        
        hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
        ref_zero_mask = ref_masks.sum(dim=1).eq(2)
        
        if all_layers:
            P = P.view(L, B)
            R = R.view(L, B)
            F = F.view(L, B)
            
        if torch.any(hyp_zero_mask):
            P = P.masked_fill(hyp_zero_mask, 0.0)
            R = R.masked_fill(hyp_zero_mask, 0.0)
            
        if torch.any(ref_zero_mask):
            P = P.masked_fill(ref_zero_mask, 0.0)
            R = R.masked_fill(ref_zero_mask, 0.0)
            
        F = F.masked_fill(torch.isnan(F), 0.0)
        
        return P, R, F

def prepare_refs_and_cands(refs_input_dir, refs_split_dir, gt_lists, pred_lists):
    
    refs_files = []
    with cs.open(refs_split_dir, "r") as f:
        for line in f.readlines():
            refs_files.append(line.strip())
    
    texts = []
    files = []
    files_and_texts = {}
    for file in tqdm(refs_files, desc="Loading refs"):
        with cs.open(os.path.join(refs_input_dir, file+".txt")) as f:
            lines = f.readlines()
            for line in lines:
                try:
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    texts.append(caption)
                    files.append(file)
                    if file not in files_and_texts.keys():
                        files_and_texts[file] = [caption]
                    else:
                        files_and_texts[file].append(caption)
                except:
                    pass
    
    # 
    refs_list = []
    cand_list = []
    for (gt, pred) in zip(gt_lists, pred_lists):
        try:
            index = texts.index(gt)
            file = files[index]
            refs_list.append(files_and_texts[file])
            cand_list.append([pred])
        except:
            pass
    return refs_list, cand_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", type=str, 
        default="../UDE2.0/networks/perception/motion_to_text/roberta-large-pretrained/tokenizer", 
        help="")
    parser.add_argument(
        "--model", type=str, 
        default="../UDE2.0/networks/perception/motion_to_text/roberta-large-pretrained/model", 
        help="")
    parser.add_argument(
        "--baseline_path", type=str, 
        default="../UDE2.0/networks/perception/motion_to_text/roberta-large-pretrained/rescale_baseline/en/roberta-large.tsv", 
        help="")
    parser.add_argument("--num_layers", type=int, default=17, help="")
    parser.add_argument(
        "--input_dir", type=str, 
        default="logs/avatar_gpt/eval/gpt_large/exp3/output/planning_se2_p0.json", 
        help="")
    args = parser.parse_args()
    return args
    
def quick_proc(input_text):
    if isinstance(input_text, list):
        return input_text[0]
    else:
        return input_text
    
def estimate(data):
    from scipy.stats import norm
    mu, std = norm.fit(data)
    interval = norm.interval(0.95, loc=mu, scale=std*.5)
    return mu, std, interval

def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def run_bert_score(score, cand_list, ref_list, task="step"):
    Precision = []
    Recall = []
    Final = []
    for (gt, pred) in tqdm(zip(ref_list, cand_list), desc="Evaluating BertScore"):
        # Calculate the BertScore
        P, R, F = score(cands=[pred] * len(gt), refs=gt, batch_size=1, verbose=False, all_layers=False)
        Precision.append(P.max().item())
        Recall.append(R.max().item())
        Final.append(R.max().item())

    p_mu, p_std, p_interval = estimate(np.asarray(Precision))
    r_mu, r_std, r_interval = estimate(np.asarray(Recall))
    f_mu, f_std, f_interval = estimate(np.asarray(Final))
    
    print("Task: {:s} | Precision | mu = {:.5f} | std = {:.5f} | interval = {:.5f}".format(task, p_mu, p_std, abs(p_mu - p_interval[0])))
    print("Task: {:s} | Recall | mu = {:.5f} | std = {:.5f} | interval = {:.5f}".format(task, r_mu, r_std, abs(r_mu - r_interval[0])))
    print("Task: {:s} | Final | mu = {:.5f} | std = {:.5f} | interval = {:.5f}".format(task, f_mu, f_std, abs(f_mu - f_interval[0]))) 

def main_planning_reverse(args):
    import json
    
    with open(args.input_dir, "r") as f:
        input_json = json.load(f)
    
    gt_steps, pred_steps = [], []
    for data in input_json.values():
        gt_steps_ = []
        for step in data["steps"]:
            gt_steps_.append(step["gt"])
            pred_steps.append(step["pred"])
        for _ in range(len(gt_steps_)):
            gt_steps.append(gt_steps_)
            
    gt_tasks, pred_tasks = [], []
    for data in input_json.values():
        gt_tasks_ = []
        for task in data["tasks"]:
            gt_tasks_.append(task["gt"])
            pred_tasks.append(task["pred"])
        for _ in range(len(gt_tasks_)):
            gt_tasks.append(gt_tasks_)
            
    score = BertScore(args=args)
    run_bert_score(score, pred_steps, gt_steps, task="Steps")
    run_bert_score(score, pred_tasks, gt_tasks, task="Tasks")
        
    # p_mu, p_conf = get_metric_statistics(Ps, replication_times=len(Ps))
    # r_mu, r_conf = get_metric_statistics(Rs, replication_times=len(Rs))
    # f_mu, f_conf = get_metric_statistics(Fs, replication_times=len(Fs))
    
    # print('=' * 50)
    # print("Precision | mu = {:.5f} | conf = {:.5f}".format(p_mu, p_conf))
    # print("Recall | mu = {:.5f} | conf = {:.5f}".format(r_mu, r_conf))
    # print("Final | mu = {:.5f} | conf = {:.5f}".format(f_mu, f_conf))

if __name__ == "__main__":
    
    args = parse_args()    
    main_planning_reverse(args=args)