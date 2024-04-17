import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print("Unable to import tensorboard")
import torch.optim as optim
from datetime import datetime
import importlib
from tqdm import tqdm
from scipy import ndimage
import yaml
import random
import json
from funcs.logger import setup_logger
from funcs.comm_utils import get_rank
from modules.training_utils import *

class AvatarGPTTrainer(object):
    def __init__(self, args, opt):

        self.opt = opt
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.timestep = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        if not self.opt["train"].get("grad_accumulation", False):
            self.loss_scalar = 1
        else:
            self.loss_scalar = self.opt["train"]["num_accumulation_steps"] * len(self.opt["train"]["tasks"])
        self.training_folder = os.path.join(self.args.training_folder, self.args.training_name, self.timestep)
        if not os.path.exists(self.training_folder): 
            os.makedirs(self.training_folder)
        self.logger = setup_logger('AvatarGPT', self.training_folder, 0, filename='avatar_gpt_training_log.txt')
        try:
            self.writer = SummaryWriter(log_dir=self.training_folder)
        except:
            pass
        with open(os.path.join(self.training_folder, "config_avatar_gpt.yaml"), 'w') as outfile:
            yaml.dump(self.opt, outfile, default_flow_style=False)
        self.epoch = 0
        self.global_step = 0
        
        self.training_outputs = TrainingOutput()    # 
        self.setup_models()
        self.load_checkpoints()
        self.print_learnable()
        self.setup_loaders()
        
    def setup_loaders(self):
        self.logger.info("Setup loaders")
        self.train_loader, self.train_dataset = importlib.import_module(
            ".loader", package="data").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                  self.opt["data"]["loader"]["train"], 
                                  meta_dir=self.training_folder)
        self.val_loader, _ = importlib.import_module(
            ".loader", package="data").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                  self.opt["data"]["loader"]["vald"], 
                                  meta_dir=None)
        self.logger.info("Loaders setup done")
    
    def build_vqvae_models(self, model_conf, ckpt_path, part_name, device):
        """We load pretrained VQ-VAE model."""
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        models = {}
        for key, conf in model_conf.items():
            models[key] = importlib.import_module(conf["arch_path"], package="networks").__getattribute__(
                conf["arch_name"])(**conf).to(device)
            models[key].load_state_dict(checkpoint["{:s}_{:s}".format(part_name, key)], strict=True)
            
        sos = torch.tensor(self.models["gpt"].get_special_token_id("sos", is_learnable=True)).to(device)
        eos = torch.tensor(self.models["gpt"].get_special_token_id("eos", is_learnable=True)).to(device)
        pos = torch.tensor(self.models["gpt"].get_special_token_id("pad", is_learnable=True)).to(device)
        self.logger.info("<SOS> = {:d} | <EOS> = {:d} | <POS> = {:d}".format(sos.item(), eos.item(), pos.item()))
        return models, sos, eos, pos
    
    def build_encodec_model(self, model_conf, device):
        model = importlib.import_module(model_conf["arch_path"], package="networks").__getattribute__(
            model_conf["arch_name"]).encodec_model_24khz(**model_conf)
        model.set_target_bandwidth(model_conf.get("target_bandwidth", 1.5))
        # Freeze the parameters
        for p in model.parameters(): 
            p.requires_grad = False
        sos = torch.tensor(self.models["gpt"].get_special_token_id("sos", is_learnable=True)).to(device)
        eos = torch.tensor(self.models["gpt"].get_special_token_id("eos", is_learnable=True)).to(device)
        pos = torch.tensor(self.models["gpt"].get_special_token_id("pad", is_learnable=True)).to(device)
        return model.to(device).eval(), sos, eos, pos
            
    def setup_models(self):
        self.sos = {}   # SOS token
        self.eos = {}   # EOS token
        self.pos = {}   # PAD token
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
                        
        # Build AvatarGPT model
        torch.cuda.empty_cache()
        self.models["gpt"] = build_gpt_model(self.opt["models"]["gpt"], device=self.device, logger=self.logger)
        if self.opt["train"].get("optimizer", "Adam") == "Adam":
            self.optimizers["gpt"] = optim.Adam(filter(lambda p: p.requires_grad, self.models["gpt"].parameters()), 
                                                lr=self.opt["train"].get("lr", 0.00001), 
                                                betas=self.opt["train"].get("betas", (0.9, 0.999)), 
                                                weight_decay=self.opt["train"].get("weight_decay", 0.0))
        else:
            self.optimizers["gpt"] = optim.AdamW(params=filter(lambda p: p.requires_grad, self.models["gpt"].parameters()), 
                                                 lr=self.opt["train"].get("lr", 0.00001), 
                                                 betas=self.opt["train"].get("betas", (0.9, 0.999)), 
                                                 weight_decay=self.opt["train"].get("weight_decay", 0.0))
        self.schedulers["gpt"] = optim.lr_scheduler.StepLR(self.optimizers["gpt"], 
                                                           step_size=self.opt["train"].get("step_lr", 100), 
                                                           gamma=self.opt["train"].get("gamma", 0.1))
        self.logger.info("AvatarGPT model built successfully")
        
        # Build VQ-VAE models
        for cat_name, model_confs in self.opt["models"]["vqvae"].items():
            for part_name, part_confs in model_confs.items():
                ckpt_path = self.opt["train"]["checkpoints"]["vqvae"][cat_name][part_name]
                models, sos, eos, pos = self.build_vqvae_models(part_confs, ckpt_path, part_name, self.device)
                for key, model in models.items():
                    self.models["{:s}_{:s}_{:s}".format(cat_name, part_name, key)] = model
                    self.logger.info("VQVAE {:s}_{:s}_{:s} model built and checkpoint resumed from {:s} successfully".format(
                        cat_name, part_name, key, ckpt_path))
                self.sos["{:s}_{:s}".format(cat_name, part_name)] = sos[None]   # [1]
                self.eos["{:s}_{:s}".format(cat_name, part_name)] = eos[None]   # [1]
                self.pos["{:s}_{:s}".format(cat_name, part_name)] = pos[None]   # [1]
         
        # Build EnCodec model
        if self.opt["models"].get("encodec", None) is not None:
            self.models["encodec"], sos, eos, pos = self.build_encodec_model(self.opt["models"]["encodec"], self.device)
            self.sos["encodec"] = sos[None]   # [1]
            self.eos["encodec"] = eos[None]   # [1]
            self.pos["encodec"] = pos[None]   # [1]
        
        if "all_body_quantizer" in self.models.keys():
            self.models["gpt"].set_quantizer(quantizer=self.models["all_body_quantizer"], type="motion")
        else:
            if "t2m_body_quantizer" in self.models.keys():
                self.models["gpt"].set_quantizer(quantizer=self.models["t2m_body_quantizer"], type="motion_t")
            if "a2m_body_quantizer" in self.models.keys():
                self.models["gpt"].set_quantizer(quantizer=self.models["a2m_body_quantizer"], type="motion_a")
        if self.models.get("encodec", None) is not None:
            self.models["gpt"].set_quantizer(quantizer=self.models["encodec"].quantizer, type="audio")
    
    def print_learnable(self):
        for name, param in self.models["gpt"].named_parameters():
            if param.requires_grad: 
                shape = ', '.join(map(str, param.size()))
                log_str = "{:s} | parameter range: ({:.5f}, {:.5f}) | shape: {:s}".format(
                    name, param.min().item(), param.max().item(), shape)
                self.logger.info(log_str)
    
    def load_checkpoints(self):
        if self.opt["train"]["checkpoints"]["gpt"] is not None:
            self.models["gpt"].load_model(self.opt["train"]["checkpoints"]["gpt"], 
                                        logger=self.logger, strict=False)

    def save_checkpoints(self, epoch, name):
        checkpoint_list = ["gpt"]
        for key in checkpoint_list:
            if key not in self.models.keys(): continue
            save_dir = os.path.join(self.training_folder, "checkpoints", key, name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.models[key].save_model(save_dir)
                    
    def motion_preprocess(self, motion, part_name):
        """Because our vqvae takes different types of input, we need to preprocess the 
        motion sequence accordingly.
        """
        # We don't conduct auxilliary process to left and right hand poses
        if part_name != "body":
            return {"inp": motion}

        if motion.size(-1) == 263:
            return {"inp": motion}
        
        trans = motion[..., :3].clone()
        pose = motion[..., 3:].clone()
        offsets = trans[:, 1:] - trans[:, :-1]
        zero_trans = torch.zeros(motion.size(0), 1, 3).float().to(motion.device)
        offsets = torch.cat([zero_trans, offsets], dim=1)
        inputs = torch.cat([offsets, pose], dim=-1) # root set to offsets relative previous frame
        return {"inp": inputs, "trans": trans}
    
    def motion_postprocess(self, motion, aux, part_name):
        """Because our vqvae takes different types of input, we need to postprocess the 
        motion sequence accordingly.
        :param motion: [B, T, C], C = 75 for strategy is ['naive', 'orien_pose'], and C = 69 for strategy is 'pose'.
        :param aux: [B, T, C], C = 3 for strategy is ['orien_pose'], and C = 6 for strategy is 'pose'.
        """
        # We don't conduct auxilliary postprocess to left and right hand poses
        if part_name != "body":
            return motion
        
    def quantize_motion(self, motion, cat_name, part_name, lengths=None, padding=True):
        """Tokenize the motion sequence to token sequence. 
        Our model is performed on token space.
        :param cat_name: category name, [t2m, a2m].
        :param part_name: body part name, [body]
        """
        outputs = self.motion_preprocess(motion, part_name)
        
        if lengths is None:
            lengths = [x.size(0) for x in motion]
        
        m_token_len = self.opt["models"]["gpt"].get("m_token_len", 52)
        motion_tokens = []
        for b in range(motion.size(0)):
            processed_motion = self.motion_preprocess(motion[b:b+1, :lengths[b]], part_name)["inp"]
            motion_emb = self.models["{:s}_{:s}_vqencoder".format(cat_name, part_name)](processed_motion)
            motion_token = self.models["{:s}_{:s}_quantizer".format(cat_name, part_name)].map2index(motion_emb)
            motion_token += 3   # Because <SOS> = 1, <EOS> = 2, <PAD> = 0
            sos = self.sos["{:s}_{:s}".format(cat_name, part_name)]
            eos = self.eos["{:s}_{:s}".format(cat_name, part_name)]
            # motion_token = torch.cat([motion_token, eos], dim=0)              # Comment this
            motion_token = torch.cat([sos, motion_token, eos], dim=0)           # Uncomment this
            if padding: # If padding is specified
                pad_len = m_token_len - motion_token.size(0)
                if pad_len > 0:
                    pos = self.pos["{:s}_{:s}".format(cat_name, part_name)].repeat(pad_len)
                    # motion_token = torch.cat([sos, motion_token, pos], dim=0) # Comment this
                    motion_token = torch.cat([motion_token, pos], dim=0)        # Uncomment this
                else:
                    # motion_token = torch.cat([sos, motion_token], dim=0)      # Comment this
                    pass                                                        # Uncomment this
            motion_tokens.append(motion_token)
        motion_tokens = torch.stack(motion_tokens, dim=0)   # [B, T]
        return motion_tokens
    
    def decode_motion(self, input, cat_name, part_name):
        """Decode the motion token sequences back to motion sequence.
        :param input: [batch_size, nframes] token sequences or [batch_size, nframes, dim] token distribution sequence.
        :param cat_name: category name, [t2m, a2m, s2m]
        :param part_name: body part name, [body, left, right]
        """
        return_pose = False

        if input.dim() == 2:
            if "{:s}_{:s}_quantizer".format(cat_name, part_name) in self.models.keys():
                # Input is token sequence.
                # 1. We get latent from codebook directly.
                latents = self.models["{:s}_{:s}_quantizer".format(cat_name, part_name)].get_codebook_entry(input.contiguous())
                # 2. We decode the motion from the latents.
                if return_pose:
                    motion, aux = self.models["{:s}_{:s}_vqdecoder".format(cat_name, part_name)](latents, return_pose=return_pose)
                    motion = self.motion_postprocess(motion, aux, part_name)
                else:
                    motion = self.models["{:s}_{:s}_vqdecoder".format(cat_name, part_name)](latents)
            else:
                size = (input.size(0), input.size(1)*4, 75 if part_name == "body" else 12)
                motion = torch.zeros(size).float().to(self.device)
        elif input.dim() == 3:
            # Input is token distribution.
            if "{:s}_{:s}_quantizer".format(cat_name, part_name) in self.models.keys():
                # 1. We use gumble softmax to convert distribution to a soft one-hot distribution.
                input_gumble = F.gumbel_softmax(input, tau=1.0, hard=True, dim=-1)
                # 2. Then we can use this soft one-hot distribution to sample a discrete latent from the codebook.
                latents = torch.matmul(input_gumble, self.models["{:s}_{:s}_quantizer".format(cat_name, part_name)].embedding.weight)
                # 3. We decode the motion from the latents.
                if return_pose:
                    motion, out = self.models["{:s}_{:s}_vqdecoder".format(cat_name, part_name)](latents, return_pose=return_pose)
                    motion = self.motion_postprocess(motion, aux, part_name)
                else:
                    motion = self.models["{:s}_{:s}_vqdecoder".format(cat_name, part_name)](latents)
            else:
                size = (input.size(0), input.size(1)*4, 75 if part_name == "body" else 12)
                motion = torch.zeros(size).float().to(self.device)
        else:
            raise ValueError("Wrong input dimension, expect dim=2 or dim=3, but input is dim={:d}".format(input.dim()))
        return motion
    
    def collect_log_info(self, losses_info, accuracy_info, epoch, total_epoch, cur_step, total_step, task="t2m"):
        log_str = "Epoch: [{:d}/{:d}] | Iter: [{:d}/{:d}] | Task: {:s}".format(epoch, total_epoch, cur_step, total_step, TASK_MAP[task])
        if losses_info is not None:
            total_loss = 0
            for key, val in losses_info.items():
                log_str += " | {:s}(loss): {:.5f}".format(key, val.item())
                total_loss += val.item()
            log_str += " | total(loss): {:.5f}".format(total_loss)
        if accuracy_info is not None:
            for key, val in accuracy_info.items():
                log_str += " | {:s}(acc): {:.3f}%".format(key, val*100.0)
        self.logger.info(log_str)      

    def collect_tensorboard_info(self, losses_info, accuracy_info, step, task="t2m", type="train"):
        type_map = {
            "t2m": "Text-to-Motion", 
            "m2t": "Motion-to-Text", 
            "m2m": "Motion-to-Motion", 
            "dm": "Decision-Making", 
            "pre": "Pre-Train"
        }
        if losses_info is not None:
            for key, val in losses_info.items():
                try:
                    self.writer.add_scalar("{:s}/{:s}/{:s}(loss)".format(type, TASK_MAP[task], key), val, step)
                except:
                    pass
        if accuracy_info is not None:
            for key, val in accuracy_info.items():
                try:
                    self.writer.add_scalar("{:s}/{:s}/{:s}(acc)".format(type, TASK_MAP[task], key), val, step)
                except:
                    pass

    def collect_monitoring_info(self, pred_tokens, target_tokens, epoch, total_epoch, task="t2m"):
        log_str = "Epoch: [{:d}/{:d}] | Task: {:s}".format(epoch, total_epoch, TASK_MAP[task])
        if task == "t2m" or task == "m2m":
            target_t = target_tokens[0].data.cpu().numpy().tolist()
            target_token_str = ", ".join(map(str, target_t))
            pred_t = pred_tokens[0].data.cpu().numpy().tolist()
            pred_token_str = ", ".join(map(str, pred_t))
        elif task == "m2t":
            target_t = target_tokens[0]
            target_mask = target_t.ne(-100)
            pred_t = pred_tokens[0]
            pred_mask = pred_t.ne(-100)
            target_token_str = self.models["gpt"].tokenizer.decode(target_t[target_mask], skip_special_tokens=True)
            pred_token_str = self.models["gpt"].tokenizer.decode(pred_t[pred_mask], skip_special_tokens=True)
        elif task in ["dm", "ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]:
            target_t = target_tokens[0]
            target_mask = target_t.ne(-100)
            pred_t = pred_tokens[0]
            pred_mask = pred_t.ne(-100)
            target_token_str = self.models["gpt"].tokenizer.decode(target_t[target_mask], skip_special_tokens=True)
            pred_token_str = self.models["gpt"].tokenizer.decode(pred_t[pred_mask], skip_special_tokens=True)
        elif task == "pre":
            return
        target_token_str = "Epoch: [{:d}/{:d}] | Task: {:s} | GT tokens: {:s}".format(epoch, total_epoch, TASK_MAP[task], target_token_str)
        pred_token_str = "Epoch: [{:d}/{:d}] | Task: {:s} | Pred tokens: {:s}".format(epoch, total_epoch, TASK_MAP[task], pred_token_str)
        self.logger.info(target_token_str)
        self.logger.info(pred_token_str)
    
    def calc_total_loss(self, losses):
        total_loss = 0.0
        for key, val in losses.items():
            weight = self.opt["losses"].get(key, 1.0)
            total_loss += weight * val
        
        return total_loss
    
    def run_pretrain_one_step(self, batch, epoch, stage="train"):
        
        losses_info = {}
        accuracy_info = {}
        motion = batch["body"].detach().to(self.device).float()
        caption = batch["text"]
        lengths = batch["length"].data.cpu().numpy().tolist() if "length" in batch.keys() else None
        batch_size = motion.size(0)
        
        tokens = {}
        with torch.no_grad():
            cat_name = "t2m" if "t2m" in self.opt["models"]["vqvae"].keys() else "all"
            tokens["body"] = self.quantize_motion(
                motion=motion, cat_name=cat_name, 
                part_name="body", lengths=lengths, padding=True)
        
        if stage == "train":
            if not self.opt["train"].get("grad_accumulation", False):  # Check whether to accumulate gradiants
                self.optimizers["gpt"].zero_grad()
            output = self.models["gpt"].pretrain(texts=caption, m_tokens=tokens["body"])
            loss = self.calc_total_loss(output["losses"]) * self.opt["lambda"].get("t2m", 1.0) / self.loss_scalar
            loss.backward()
            if self.opt["train"].get("clip_grad", False):
                clip_grad_norm_(self.models["gpt"].parameters(), 0.5)
            if not self.opt["train"].get("grad_accumulation", False):  # Check whether to accumulate gradients
                self.optimizers["gpt"].step()
        else:
            pass
        return output
    
    def run_text_to_motion_one_step(self, batch, epoch, stage="train"):
        """Train one text-to-motion step.(Low-level motion generation)"""
        losses_info = {}
        accuracy_info = {}
        motion = batch["body"].detach().to(self.device).float()
        caption = batch["text"]                     # Manually annotated textual description
        augmented_caption = batch["aug_text"]       # ChatGPT annotated textual description
        lengths = batch["length"].data.cpu().numpy().tolist() if "length" in batch.keys() else None
        batch_size = motion.size(0)
        
        if len(augmented_caption) != 0:
            cat_caption = [[t1, t2] for (t1, t2) in zip(caption, augmented_caption)]
            caption = [random.choice(ts) for ts in cat_caption]
        
        tokens = {}
        with torch.no_grad():
            cat_name = "t2m" if "t2m" in self.opt["models"]["vqvae"].keys() else "all"
            tokens["body"] = self.quantize_motion(
                motion=motion, cat_name=cat_name, 
                part_name="body", lengths=lengths, padding=True)
        
        if stage == "train":
            if not self.opt["train"].get("grad_accumulation", False):  # Check whether to accumulate gradiants
                self.optimizers["gpt"].zero_grad()
            output = self.models["gpt"].text_to_motion(texts=caption, m_tokens=tokens["body"])
            loss = self.calc_total_loss(output["losses"]) * self.opt["lambda"].get("t2m", 1.0) / self.loss_scalar
            loss.backward()
            if self.opt["train"].get("clip_grad", False):
                clip_grad_norm_(self.models["gpt"].parameters(), 0.5)
            if not self.opt["train"].get("grad_accumulation", False):  # Check whether to accumulate gradients
                self.optimizers["gpt"].step()
        else:
            pass
            
        return output
    
    def run_motion_to_text_one_step(self, batch, epoch, stage="train"):
        """Train one motion-to-text step.(Low-level motion understanding)"""
        losses_info = {}
        accuracy_info = {}
        motion = batch["body"].detach().to(self.device).float()
        caption = batch["text"]
        lengths = batch["length"].data.cpu().numpy().tolist() if "length" in batch.keys() else None
        batch_size = motion.size(0)
        
        tokens = {}
        with torch.no_grad():
            cat_name = "t2m" if "t2m" in self.opt["models"]["vqvae"].keys() else "all"
            tokens["body"] = self.quantize_motion(
                motion=motion, cat_name=cat_name, 
                part_name="body", lengths=lengths, padding=True)
        
        if stage == "train":
            if not self.opt["train"].get("grad_accumulation", False):  # Check whether to accumulate gradiants
                self.optimizers["gpt"].zero_grad()
            output = self.models["gpt"].motion_to_text(texts=caption, m_tokens=tokens["body"])
            loss = self.calc_total_loss(output["losses"]) * self.opt["lambda"].get("m2t", 1.0) / self.loss_scalar
            loss.backward()
            if self.opt["train"].get("clip_grad", False):
                clip_grad_norm_(self.models["gpt"].parameters(), 0.5)
            if not self.opt["train"].get("grad_accumulation", False):  # Check whether to accumulate gradiants
                self.optimizers["gpt"].step()
        else:
            pass

        return output
    
    def run_motion_to_motion_one_step(self, batch, epoch, stage="train"):
        """Train one motion-to-motion step.(Low-level motion-in-between)"""
        losses_info = {}
        accuracy_info = {}
        motion = batch["body"].detach().to(self.device).float()
        lengths = batch["length"].data.cpu().numpy().tolist() if "length" in batch.keys() else None
        batch_size = motion.size(0)
        
        tokens = {}
        with torch.no_grad():
            cat_name = "t2m" if "t2m" in self.opt["models"]["vqvae"].keys() else "all"
            tokens["body"] = self.quantize_motion(
                motion=motion, cat_name=cat_name, 
                part_name="body", lengths=lengths, padding=True)
        
        if stage == "train":
            if not self.opt["train"].get("grad_accumulation", False):  # Check whether to accumulate gradiants
                self.optimizers["gpt"].zero_grad()
            output = self.models["gpt"].motion_to_motion(m_tokens=tokens["body"])
            loss = self.calc_total_loss(output["losses"]) * self.opt["lambda"].get("m2m", 1.0) / self.loss_scalar
            loss.backward()
            if self.opt["train"].get("clip_grad", False):
                clip_grad_norm_(self.models["gpt"].parameters(), 0.5)
            if not self.opt["train"].get("grad_accumulation", False):  # Check whether to accumulate gradients
                self.optimizers["gpt"].step()
        else:
            pass
            
        return output
    
    def run_planning_one_step(self, batch, epoch, task=None, stage="train"):
        """Train one high-level decision making step.(Random combination of any high-level tasks)"""
        losses_info = {}
        accuracy_info = {}
        scene = batch["scene"]                  # List of strings
        duration = batch["duration"]            # List of strings
        next_duration = batch["next_duration"]  # List of strings
        cur_action = batch["cur_action"]        # List of strings
        next_action = batch["next_action"]      # List of strings
        cur_steps = batch["cur_steps"]          # List of lisr of strings
        next_steps = batch["next_steps"]        # List of list of strings
        
        # Preprocess the textual descriptions
        scene = [t.replace("[scene] ", "") for t in scene]
        cur_action = [t.replace("[current action]: ", "") for t in cur_action]
        
        proc_batch = {
            "scene": [t.replace("[scene] ", "") for t in scene], 
            "cur_task": [t.replace("[current action]: ", "") for t in cur_action], 
            "next_task": next_action, 
            "cur_steps": cur_steps, 
            "next_steps": next_steps
        }
        
        if stage == "train":
            if not self.opt["train"].get("grad_accumulation", False):  # Check whether to accumulate gradiants
                self.optimizers["gpt"].zero_grad()
            output = self.models["gpt"].planning(batch=proc_batch, task=task, loss_type=["pred"])
            loss = self.calc_total_loss(output["losses"]) * self.opt["lambda"].get("dm", 1.0) / self.loss_scalar
            loss.backward()
            if self.opt["train"].get("clip_grad", False):
                clip_grad_norm_(self.models["gpt"].parameters(), 0.5)
            if not self.opt["train"].get("grad_accumulation", False):  # Check whether to accumulate gradiants
                self.optimizers["gpt"].step()
        else:
            pass
        return output
    
    def train_one_step(self, batch, epoch, step, total_step):
        train_tasks = self.opt["train"].get("tasks", ["t2m", "m2t", "m2m", "dm"])
        
        for cat_name, batch_per_cat in batch.items():           
            if cat_name == "t2m":
                # Pretrain
                if "pre" in train_tasks:
                    output = self.run_pretrain_one_step(batch_per_cat, epoch, stage="train")
                    self.training_outputs.update(input_dict=output, mode="pre")
                # Text-to-Motion
                if "t2m" in train_tasks:
                    output = self.run_text_to_motion_one_step(batch_per_cat, epoch, stage="train")
                    self.training_outputs.update(input_dict=output, mode="t2m")
                # Motion-to-Text
                if "m2t" in train_tasks:
                    output = self.run_motion_to_text_one_step(batch_per_cat, epoch, stage="train")
                    self.training_outputs.update(input_dict=output, mode="m2t")
                # Motion-to-Motion
                if "m2m" in train_tasks:
                    output = self.run_motion_to_motion_one_step(batch_per_cat, epoch, stage="train")
                    self.training_outputs.update(input_dict=output, mode="m2m")
            elif cat_name == "t2t":
                if "dm" in train_tasks:
                    # for task in ["ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]:
                    output = self.run_planning_one_step(batch_per_cat, epoch, task=None, stage="train")
                    self.training_outputs.update(input_dict=output, mode="dm")
            
        self.training_outputs.update_num()  
        # If gradients accumulation is On, update the parameters and clear the accumulated gradients and the training information
        if self.opt["train"].get("grad_accumulation", False) and self.global_step % self.opt["train"].get("num_accumulation_steps", 1) == 0:
            self.optimizers["gpt"].step()
            self.optimizers["gpt"].zero_grad()
            
            for task in ["t2m", "m2t", "m2m", "dm", "pre", "ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]:
                if self.training_outputs.has_losses(mode=task) and self.training_outputs.has_accuracy(mode=task):
                    if self.global_step % 10 == 0:
                        self.collect_log_info(
                            losses_info=self.training_outputs.get_losses(mode=task), 
                            accuracy_info=self.training_outputs.get_accuracy(mode=task), 
                            epoch=epoch, total_epoch=self.opt["train"]["num_epochs"], 
                            cur_step=step, total_step=total_step, task=task)
                    self.collect_tensorboard_info(
                        losses_info=self.training_outputs.get_losses(mode=task), 
                        accuracy_info=self.training_outputs.get_accuracy(mode=task), 
                        step=self.global_step, task=task, type="train")
                if self.training_outputs.has_pred_tokens(mode=task) and self.training_outputs.has_target_tokens(mode=task) and self.global_step % 100 == 0:
                    self.collect_monitoring_info(
                        pred_tokens=self.training_outputs.get_pred_tokens(mode=task), 
                        target_tokens=self.training_outputs.get_target_tokens(mode=task), 
                        epoch=epoch, total_epoch=self.opt["train"]["num_epochs"], task=task)
            self.training_outputs.reset()
            
        # If gradients accumulation is Off, update the training information only
        if not self.opt["train"].get("grad_accumulation", False):
            for task in ["t2m", "m2t", "m2m", "dm", "pre", "ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]:
                if self.training_outputs.has_losses(mode=task) and self.training_outputs.has_accuracy(mode=task):
                    if self.global_step % 10 == 0:
                        self.collect_log_info(
                            losses_info=self.training_outputs.get_losses(mode=task), 
                            accuracy_info=self.training_outputs.get_accuracy(mode=task), 
                            epoch=epoch, total_epoch=self.opt["train"]["num_epochs"], 
                            cur_step=step, total_step=total_step, task=task)
                    self.collect_tensorboard_info(
                        losses_info=self.training_outputs.get_losses(mode=task), 
                        accuracy_info=self.training_outputs.get_accuracy(mode=task), 
                        step=self.global_step, task="t2t", type="train")
                if self.training_outputs.has_pred_tokens(mode=task) and self.training_outputs.has_target_tokens(mode=task) and self.global_step % 1000 == 0:
                    self.collect_monitoring_info(
                        pred_tokens=self.training_outputs.get_pred_tokens(mode=task), 
                        target_tokens=self.training_outputs.get_target_tokens(mode=task), 
                        epoch=epoch, total_epoch=self.opt["train"]["num_epochs"], task=task)
            self.training_outputs.reset()
                        
        self.global_step += 1
        
    def eval_one_step(self, batch, epoch, step, total_step):
        for cat_name, batch_per_cat in batch.items():
            if cat_name == "t2m":
                t2m_outputs = self.run_text_to_motion_one_step(batch_per_cat, epoch, stage="eval")
                m2t_outputs = self.run_motion_to_text_one_step(batch_per_cat, epoch, stage="eval")
                save_eval_results(t2m_outputs, self.training_folder, epoch, step, "t2m")
                save_eval_results(m2t_outputs, self.training_folder, epoch, step, "m2t")
            elif cat_name == "a2m":
                # a2m_outputs = self.run_audio_to_motion_continual_one_step(batch_per_cat, epoch, stage="eval")
                a2m_outputs = self.run_audio_to_motion_one_step(batch_per_cat, epoch, stage="eval")
                m2a_outputs = self.run_motion_to_audio_one_step(batch_per_cat, epoch, stage="eval")
                save_eval_results(a2m_outputs, self.training_folder, epoch, step, "a2m")
                save_eval_results(m2a_outputs, self.training_folder, epoch, step, "m2a")
                
    def train_one_epoch(self, epoch, loader):
        for key in self.models.keys():
            if key in self.opt["train"]["model_to_train"]:
                self.models[key].train()
            else:
                self.models[key].eval()
        
        for iter, batch in enumerate(loader):
            self.train_one_step(batch, epoch=epoch, step=iter, total_step=len(loader))
        
        for key in self.schedulers.keys():
            self.schedulers[key].step()
            
    def eval_one_epoch(self, epoch, loader):
        for key in self.models.keys():
            self.models[key].eval()
        
        for iter, batch in enumerate(loader):
            self.eval_one_step(batch, epoch, step=iter, total_step=len(loader))
            
    def train(self):
        for epoch in range(self.epoch, self.opt["train"]["num_epochs"], 1):
            self.train_one_epoch(epoch, self.train_loader)
            if epoch % self.opt["train"]["save_per_epoch"] == 0:
                self.save_checkpoints(epoch, "AvatarGPT_E{:04d}".format(epoch))
            if epoch % self.opt["train"]["eval_per_epoch"] == 0 and epoch != 0:
                self.eval_one_epoch(epoch, self.val_loader)
        self.save_checkpoints(epoch, "AvatarGPT_final")
