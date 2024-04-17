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
import pickle
from funcs.logger import setup_logger
from funcs.comm_utils import get_rank
from modules.training_utils import *
from modules.generation_utils import *
from networks.roma.utils import rotvec_slerp

class AvatarGPTEvaluator(object):
    def __init__(self, args, opt):
        self.opt = opt
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.output_dir = os.path.join(self.args.eval_folder, self.args.eval_name, "output")
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        self.logger = setup_logger('AvatarGPT', self.output_dir, get_rank(), filename='avatar_gpt_eval_log.txt')
        self.setup_loaders()
        self.setup_models()

    def setup_loaders(self):
        self.eval_loader, self.eval_dataset = importlib.import_module(
            ".loader", package="data").__getattribute__(
                "get_dataloader")(self.opt["data"]["dataset"], 
                                self.opt["data"]["loader"]["test"], 
                                meta_dir=self.output_dir)
    
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
                    
    def setup_models(self):
        self.sos = {}   # SOS token
        self.eos = {}   # EOS token
        self.pos = {}   # PAD token
        self.models = {}

        # Build AvatarGPT model
        torch.cuda.empty_cache()
        self.models["gpt"] = build_gpt_model(self.opt["models"]["gpt"], device=self.device, logger=self.logger)
        self.models["gpt"].load_model(self.opt["eval"]["checkpoints"]["gpt"], logger=self.logger, strict=True)
        self.logger.info("AvatarGPT model built successfully")

        # Build VQ-VAE models
        for cat_name, model_confs in self.opt["models"]["vqvae"].items():
            for part_name, part_confs in model_confs.items():
                ckpt_path = self.opt["eval"]["checkpoints"]["vqvae"][cat_name][part_name]
                models, sos, eos, pos = self.build_vqvae_models(part_confs, ckpt_path, part_name, self.device)
                for key, model in models.items():
                    self.models["{:s}_{:s}_{:s}".format(cat_name, part_name, key)] = model
                    self.logger.info("VQVAE {:s}_{:s}_{:s} model built and checkpoint resumed from {:s} successfully".format(
                        cat_name, part_name, key, ckpt_path))
                self.sos["{:s}_{:s}".format(cat_name, part_name)] = sos[None]   # [1]
                self.eos["{:s}_{:s}".format(cat_name, part_name)] = eos[None]   # [1]
                self.pos["{:s}_{:s}".format(cat_name, part_name)] = pos[None]   # [1]
                
        if "all_body_quantizer" in self.models.keys():
            self.models["gpt"].set_quantizer(quantizer=self.models["all_body_quantizer"], type="motion")
        else:
            if "t2m_body_quantizer" in self.models.keys():
                self.models["gpt"].set_quantizer(quantizer=self.models["t2m_body_quantizer"], type="motion_t")
            if "a2m_body_quantizer" in self.models.keys():
                self.models["gpt"].set_quantizer(quantizer=self.models["a2m_body_quantizer"], type="motion_a")
        if "encodec" in self.models.keys():
            self.models["gpt"].set_quantizer(quantizer=self.models["encodec"].quantizer.eval(), type="audio")

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
        return motion

    @staticmethod
    def merge_motion_segments(segments, seg_len):
        merged = []
        for i in range(1, len(segments), 1):
            clip_1 = segments[i-1][0]  # [T, D]
            clip_2 = segments[i][0]    # [T, D]
            
            len_1 = clip_1.size(0)
            len_2 = clip_2.size(0)
        
            init_transl = clip_2[:1, :3]
            last_transl = clip_1[seg_len:seg_len+1, :3]
            offset = init_transl - last_transl
            clip_2[:, :3] -= offset
            
            num_rotvec = (clip_1.size(1) - 3) // 3
            num_steps = len_1 - seg_len
            interp_rotvecs = []
            interp_transl = []
            for j in range(0, num_steps, 1):
                steps = torch.Tensor([j / num_steps]).float().to(clip_1.device)
                # Slerp the rotvecs
                rotvecs_interp = rotvec_slerp(rotvec0=clip_1[j+seg_len, 3:].view(-1, 3), 
                                              rotvec1=clip_2[j, 3:].view(-1, 3), steps=steps)[0]
                interp_rotvecs.append(rotvecs_interp)
                # Lerp the transl
                transl_interp = torch.lerp(clip_1[j+seg_len:j+seg_len+1, :3], clip_2[j:j+1, :3], weight=steps)
                interp_transl.append(transl_interp)  
            
            interp_rotvecs = torch.stack(interp_rotvecs, dim=0).view(num_steps, -1)
            interp_transl = torch.cat(interp_transl, dim=0)  
            
            if len(merged) == 0:
                merged.append(clip_1[:seg_len])
            # else:
            #     merged.append(clip_1[num_steps:seg_len])
            merged.append(torch.cat([interp_transl, interp_rotvecs], dim=-1))
            
        merged.append(clip_2[seg_len:])
    
        merged = torch.cat(merged, dim=0).unsqueeze(dim=0)
        return merged
    
    @torch.no_grad()
    def encode_motion(self, motion, cat_name, part_name, lengths=None):
        """Tokenize the motion sequence to token sequence. 
        Our model is performed on token space.
        :param cat_name: category name, [t2m, a2m].
        :param part_name: body part name, [body]
        """
        outputs = self.motion_preprocess(motion, part_name)
        
        if lengths is None:
            lengths = [x.size(0) for x in motion]
        
        m_token_len = self.opt["models"]["gpt"].get("m_token_len", 51)
        motion_tokens = []
        for b in range(motion.size(0)):
            processed_motion = self.motion_preprocess(motion[b:b+1, :lengths[b]], part_name)["inp"]
            motion_emb = self.models["{:s}_{:s}_vqencoder".format(cat_name, part_name)](processed_motion)
            motion_token = self.models["{:s}_{:s}_quantizer".format(cat_name, part_name)].map2index(motion_emb)
            motion_token += 3   # Because <SOS> = 1, <EOS> = 2, <PAD> = 0
            sos = self.sos["{:s}_{:s}".format(cat_name, part_name)]
            eos = self.eos["{:s}_{:s}".format(cat_name, part_name)]
            # motion_token = torch.cat([motion_token, eos], dim=0)          # Comment this
            motion_token = torch.cat([sos, motion_token, eos], dim=0)       # Uncomment this
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

    @torch.no_grad()
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

    @torch.no_grad()
    def reconstruct_motion_from_tokens(self, tokens, init_trans, repr="d263", cat_name="t2m"):
        # Decode the tokens
        token_mask = tokens.gt(self.eos["{:s}_body".format(cat_name)].item())
        tokens = tokens[token_mask]
        tokens -= 3    # Convert back to original token space
        if tokens.dim() == 1: 
            tokens = tokens.unsqueeze(dim=0)
        
        if repr == "d263":
            # 1. Motion representation is: d263
            motion = self.decode_motion(tokens, cat_name, "body")
        else:
            pass
        motion = motion[:, :-1]   # It is a bit wierd that the last pose is always presenting artifacts, so we skip it.
        return motion
    
    @torch.no_grad()
    def generate_text_to_motion(self, batch):
        gt_motion = batch["body"]
        lengths = batch["length"]
        cat_name = "t2m" if "t2m" in self.opt["models"]["vqvae"].keys() else "all"
        
        # Tokenize gt motion sequence, shape is [1, T]
        gt_tokens = self.encode_motion(
            motion=gt_motion.float().to(self.device), 
            cat_name=cat_name, 
            part_name="body", 
            lengths=lengths)
        gt_motion = torch.stack([x[:l] for (x, l) in zip(gt_motion, lengths)], dim=0)
        if gt_motion.size(-1) == 263:
            gt_motion = apply_inverse_transform(gt_motion, data_obj=self.eval_dataset)
        # texts = batch["text"]
        text_list = batch["text_list"]

        results = []
        for _ in range(self.args.repeat_times):
            texts = random.choice(text_list)  # Sample one text description
            # Predict the motion tokens
            min_num_tokens = lengths[0] // 4  # Minimum token lengths, it's 1/4 of the length of gt motion sequence.
            max_num_tokens = 256
            pred_tokens = self.models["gpt"].generate_text_to_motion(
                texts, topk=self.args.topk, 
                # min_num_tokens=gt_tokens.size(-1)-2, 
                min_num_tokens=10, 
                max_num_tokens=max_num_tokens, 
                use_semantic_sampling=self.args.use_semantic_sampling, 
                temperature=self.args.temperature)
            # Deal with exceptions!!!
            if pred_tokens is None: 
                continue
            
            self.logger.info("[Text-to-Motion] Prompt: {:s} | GT motion token length: {:d} | Generated motion token length: {:d}".format(
                texts[0], lengths[0] // 4, pred_tokens.shape[0]))
            # self.logger.info("[Motion-to-Text] GT text: {:s}. Generated text: {:s}".format(texts[0], pred_texts))
                
            # Decode the tokens
            pred_motion = self.reconstruct_motion_from_tokens(
                tokens=pred_tokens, 
                init_trans=gt_motion[:, :1, :3].float().to(self.device), 
                repr="d263" if gt_motion.size(-1) == 263 else "d75", 
                cat_name=cat_name)
            
            # Apply inverse transform (optional)
            if pred_motion.size(-1) == 263:
                pred_motion = apply_inverse_transform(pred_motion, data_obj=self.eval_dataset)
                
            # output
            output = {
                "gt": {"body": gt_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                "pred": {"body": pred_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                "caption": texts, 
                # "pred_caption": [pred_texts]
            }
            results.append(output)
        return results

    @torch.no_grad()
    def generate_motion_to_text(self, batch):
        gt_motion = batch["body"]
        lengths = batch["length"]
        text_list = batch["text_list"]
        cat_name = "t2m" if "t2m" in self.opt["models"]["vqvae"].keys() else "all"
        
        # Tokenize gt motion sequence, shape is [1, T]
        gt_tokens = self.encode_motion(
            motion=gt_motion.float().to(self.device), 
            cat_name=cat_name, 
            part_name="body", 
            lengths=lengths)
        gt_motion = torch.stack([x[:l] for (x, l) in zip(gt_motion, lengths)], dim=0)
        if gt_motion.size(-1) == 263:
                gt_motion = apply_inverse_transform(gt_motion, data_obj=self.eval_dataset)
        results = []
        for _ in range(self.args.repeat_times):
            texts = random.choice(text_list)  # Sample one text description
            pred_texts = self.models["gpt"].generate_motion_to_text(
                gt_tokens, topk=self.args.topk, 
                max_num_tokens=256, 
                temperature=self.args.temperature)
            self.logger.info("[Motion-to-Text] GT text: {:s} | Generated text: {:s}".format(texts[0], pred_texts))
            
            # output
            output = {
                "gt": {"body": gt_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                "pred": {"body": gt_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                "caption": texts, 
                "pred_caption": [pred_texts]
            }
            results.append(output)
        return results
    
    @torch.no_grad()
    def generate_motion_to_motion(self, batch):
        
        gt_motion = batch["body"].detach().to(self.device).float()
        lengths = batch["length"]
        cat_name = "t2m" if "t2m" in self.opt["models"]["vqvae"].keys() else "all"
        texts = batch["text"]
        
        # Tokenize gt motion sequence
        # TODO: there is a bug in here!!!
        gt_tokens = self.encode_motion(motion=gt_motion.float().to(self.device), cat_name=cat_name, part_name="body", lengths=lengths)
        gt_motion = torch.stack([x[:l] for (x, l) in zip(gt_motion, lengths)], dim=0)
        
        valid_token_len = lengths[0] // 4
        start_tokens = gt_tokens[:, 1:(valid_token_len*2)//5]
        end_tokens = gt_tokens[:, -(valid_token_len*2)//5-2:-2]
        
        results = []
        max_num_tokens = 256
        for _ in range(self.args.repeat_times):
            pred_tokens = self.models["gpt"].generate_motion_to_motion(
                m_start_tokens=start_tokens, 
                m_end_tokens=end_tokens, 
                topk=self.args.topk, 
                max_num_tokens=max_num_tokens, 
                use_semantic_sampling=self.args.use_semantic_sampling, 
                temperature=self.args.temperature)
            # Deal with exceptions!!!
            if pred_tokens is None: 
                continue
            pred_tokens = torch.cat([start_tokens, pred_tokens.unsqueeze(dim=0), end_tokens], dim=1)    # [1, T]
            self.logger.info("[Motion-to-Motion] GT motion token length: {:d} | Interpolated motion token length: {:d}".format(
                valid_token_len, pred_tokens.size(-1)))
            
            # Decode the tokens
            token_mask = pred_tokens.gt(self.pos["{:s}_body".format(cat_name)].item())
            pred_tokens = pred_tokens[token_mask]
            pred_tokens -= 3    # Convert back to original token space
            if pred_tokens.dim() == 1: pred_tokens = pred_tokens.unsqueeze(dim=0)
                
            if gt_motion.size(-1) == 263:
                # 1. Motion representation is: d263
                pred_motion = self.decode_motion(pred_tokens, cat_name, "body")
            else:
                pass
            
            # Apply inverse transform (optional)
            if gt_motion.size(-1) == 263:
                gt_motion = apply_inverse_transform(gt_motion, data_obj=self.eval_dataset)
            if pred_motion.size(-1) == 263:
                pred_motion = apply_inverse_transform(pred_motion, data_obj=self.eval_dataset)
            
            # Generate color labels of predicted motion sequence
            color_labels = np.zeros((pred_motion.size(1),))
            color_labels[start_tokens.size(-1)*4:-end_tokens.size(-1)*4] = 1
            # output
            output = {
                "gt": {"body": gt_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                "pred": {"body": pred_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                "caption": texts, 
                "color_labels": color_labels
            }
            results.append(output)
        return results
    
    @torch.no_grad()
    def generate_planning(self, batch, task="ct2t"):
        
        # scene = batch["scene"]                  # List of strings
        scene_list = batch.get("scene_list", None)
        cur_action_list = batch.get("cur_action_list", None)
        next_action_list = batch.get("next_action_list", None)
        cur_steps_list = batch.get("cur_steps_list", None)
        next_steps_list = batch.get("next_steps_list", None)
        name = batch.get("name", None)
        
        def process_descriptions(text_list, text_to_dump=None):
            if text_list is None:
                return None
            # Random select one text from the input list
            text = random.choice(text_list)
            if text_to_dump is not None:
                text = text.replace(text_to_dump, "")
            return text
        
        results = []
        for _ in range(self.args.repeat_times):
            # Process the texutal descriptions
            scene = process_descriptions(scene_list, text_to_dump=None)
            cur_task = process_descriptions(cur_action_list, text_to_dump=None)
            cur_steps = process_descriptions(cur_steps_list, text_to_dump=None)
            next_task = process_descriptions(next_action_list, text_to_dump=None)
            next_steps = process_descriptions(next_steps_list, text_to_dump=None)
            inp_batch = {
                "scene": scene, "cur_task": cur_task, "cur_steps": cur_steps, 
                "next_task": next_task, "next_steps": next_steps
            }
            pred_text = self.models["gpt"].generate_planning(
                batch=inp_batch, task=task, 
                topk=self.args.topk, 
                max_num_tokens=256)
            
            output = {}
            output.update(inp_batch)
            output["task"] = task
            output["pred"] = [pred_text]
            self.logger.info(print_generation_info(inp_batch=inp_batch, predicted=pred_text, task=task))
            results.append(output)
        return results
    
    def generate_on_testset(self):
        """Generate results on testset."""
        for batch_id, batch in enumerate(tqdm(self.eval_loader)):
            modality = batch["modality"][0]
            if modality == "t2m":
                if "t2m" in self.args.eval_task:
                    self.logger.info("Generate one motion sequence from text description.")
                    output = self.generate_text_to_motion(batch)
                    save_generation_results(results=output, output_path=self.output_dir, modality="t2m", batch_id=batch_id)
                if "m2m" in self.args.eval_task:
                    self.logger.info("Generate one motion sequence between starting and ending motion.")
                    output = self.generate_motion_to_motion(batch)
                    save_generation_results(results=output, output_path=self.output_dir, modality="m2m", batch_id=batch_id)
                if "m2t" in self.args.eval_task:
                    self.logger.info("Generate one text description from given motion sequence.")
                    output = self.generate_motion_to_text(batch)
                    save_generation_results(results=output, output_path=self.output_dir, modality="m2t", batch_id=batch_id)
            elif modality == "t2t":
                tasks = ["ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]
                for task in tasks:
                    if task in self.args.eval_task:
                        output = self.generate_planning(batch, task=task)
                        save_generation_results(results=output, output_path=self.output_dir, modality=task, batch_id=batch_id)
    
    @torch.no_grad()
    def generate_long_motion_sequence_from_scene(self):
        """Scenario 1: given a scene description, 
        1. generate series of action descriptions, 
        2. synthesize corresponding motion sequence, 
        3. interpolate them into a long motion sequence.
        """
        text_descriptions = load_scene_information_decompose(
            input_path=self.args.demo_data, 
            split_file=self.args.demo_list)
        # random.shuffle(text_descriptions)
        cat_name = "t2m" if "t2m" in self.opt["models"]["vqvae"].keys() else "all"
        for batch_id, batch in enumerate(text_descriptions):            
            results = []
            for tid in range(self.args.repeat_times):
                self.logger.info("[{:d}/{:d}] Generated {:s}".format(batch_id+1, len(text_descriptions), "=" * 50))
                if self.args.eval_pipeline == "p0":
                    scene, token_segments, action_descriptions = self.generate_scenario_one_pipeline_zero(batch=batch, temperature=1.0)
                if self.args.eval_pipeline == "p1":
                    pass
                elif self.args.eval_pipeline == "p2":
                    pass
                elif self.args.eval_pipeline == "p3":
                    pass
                elif self.args.eval_pipeline == "p4":
                    pass

                # Generate the motion tokens in between
                self.logger.info("Generate the motion tokens in between segments")
                interp_token_segments = self.__merge_token_segments(token_segments=token_segments)
                
                # Combine the motion tokens
                self.logger.info("Combine motion tokens")
                combined_tokens = self.__combine_token_segments(token_segments=token_segments, interp_token_segments=interp_token_segments)

                # Decode the tokens
                self.logger.info("Decode motion tokens")
                pred_motion = self.__decode_motion(input_tokens=combined_tokens, cat_name=cat_name)
                pred_motion = apply_inverse_transform(pred_motion, data_obj=self.eval_dataset)

                # Generate color labels of predicted motion sequence (visualization only)
                start_i = 0
                color_labels = np.ones((pred_motion.size(1),))
                for i in range(len(interp_token_segments)):
                    start_i += token_segments[i].size(-1) * 4       # Update the start_index to current position
                    end_i = start_i + interp_token_segments[i].size(-1) * 4
                    color_labels[start_i:end_i] = 0
                    start_i = end_i                                 # Update the end_index
                                
                output = {
                    "gt": {"body": pred_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                    "pred": {"body": pred_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                    "caption": [scene], 
                    "actions": action_descriptions, 
                    "color_labels": color_labels
                }
                results.append(output)
            save_generation_results(results=results, output_path=self.output_dir, batch_id=batch_id, 
                                    modality="se1_{:s}".format(self.args.eval_pipeline))

    @torch.no_grad()
    def generate_text_from_long_motion_sequence(self):
        planning_dir = os.path.join(self.output_dir, "planning_se1_p0.json")
        planning_task = load_task_planning_results(planning_dir)
        
        idx = 0
        total_cnt = len(planning_task)
        for filename, task_item in planning_task.items():
            gen_data = np.load(os.path.join(self.output_dir, "se1_{:s}".format(self.args.eval_pipeline), filename), allow_pickle=True).item()
            inp_motion = apply_transform(inp_motion=gen_data["pred"]["body"].transpose((0, 2, 1)), data_obj=self.eval_dataset)
            num_steps = len(task_item["steps"])
            num_tasks = len(task_item["tasks"])
            step_cnt = task_item["step_cnt"]
            inp_motion_seg = torch.chunk(inp_motion, num_steps, dim=1)
            pred_steps = []
            for (gt_text, motion_seg) in zip(task_item["steps"], inp_motion_seg):
                pred_step = self.__generate_motion_text(inp_motion=motion_seg, max_num_tokens=256)
                self.logger.info("[{:d}/{:d}] | [Motion-to-Text] GT text: {:s} | Pred text: {:s}".format(idx+1, total_cnt, gt_text, pred_step))
                pred_steps.append(pred_step)
            
            pred_tasks = []
            for i, index in enumerate(step_cnt):
                gt_task = task_item["tasks"][i]
                pred_steps_ = [pred_steps[k] for k in index]
                pred_steps_reorg = ["{:d}. {:s}".format(i+1, text) for i, text in enumerate(pred_steps_)]
                inp_batch = {"cur_steps": ["\n".join(pred_steps_reorg)]}
                pred_task = self.models["gpt"].generate_planning(
                    batch=inp_batch, task="s2t", topk=self.args.topk, 
                    max_num_tokens=256, temperature=self.args.temperature)
                self.logger.info("[{:d}/{:d}] | [Steps-to-Task] GT task: {:s} | Pred task: {:s}".format(idx+1, total_cnt, gt_task, pred_task))
                pred_tasks.append(pred_task)
            
            output_item = {}
            output_item.update(task_item)
            output_item["pred_steps"] = pred_steps
            output_item["pred_task"] = pred_tasks
            
            output_dir = os.path.join(self.output_dir, "se2_{:s}".format(self.args.eval_pipeline))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(os.path.join(output_dir, filename.replace(".npy", ".json")), "w") as f:
                json.dump(output_item, f)
            
            idx += 1
    
    @torch.no_grad()
    def generate_scenario_two(self):
        """Scenario 2: given a motion sequence, 
        1. understand the motion sequence and generate the text description, 
        2. estimate the scene information based on action description, 
        3. generate series of action descriptions, 
        4. synthesize corresponding motion sequence, 
        5. interpolate them into a long motion sequence.
        """
        for batch_id, batch in enumerate(self.eval_loader):
            modality = batch["modality"][0]
            if modality != "t2m": 
                continue
            
            gt_motion = batch["body"]
            lengths = batch["length"]
            cat_name = "t2m" if "t2m" in self.opt["models"]["vqvae"].keys() else "all"

            # Tokenize gt motion sequence, shape is [1, T]
            gt_tokens = self.encode_motion(
                motion=gt_motion.float().to(self.device), 
                cat_name=cat_name, 
                part_name="body", 
                lengths=lengths)
            gt_motion = apply_inverse_transform(gt_motion[:, :lengths[0]], data_obj=self.eval_dataset)
            
            # Understand the motion as task description
            pred_task = self.models["gpt"].generate_motion_to_text(
                gt_tokens, topk=self.args.topk, max_num_tokens=256)
            inp_batch = {"cur_task": [pred_task]}
            pred_steps = self.models["gpt"].generate_planning(batch=inp_batch, task="t2s", topk=10, max_num_tokens=256)
            # Estimate scene information from task description
            inp_batch = {"cur_task": [pred_task]}
            pred_scene = self.models["gpt"].generate_planning(batch=inp_batch, task="t2c", topk=10, max_num_tokens=256)
            inp_batch = {"scene": [pred_scene], "action": [{"executable_steps": pred_steps, "executable_simplified": [pred_task]}]}
            
            results = []
            for tid in range(self.args.repeat_times):
                if self.args.eval_pipeline == "p1":
                    scene, token_segments, action_descriptions = self.generate_scenario_one_pipeline_one(batch=inp_batch, temperature=1.0)
                elif self.args.eval_pipeline == "p2":
                    scene, token_segments, action_descriptions = self.generate_scenario_one_pipeline_two(batch=inp_batch, temperature=5.0)
                elif self.args.eval_pipeline == "p3":
                    scene, token_segments, action_descriptions = self.generate_scenario_one_pipeline_three(batch=inp_batch, temperature=1.0)
                elif self.args.eval_pipeline == "p4":
                    scene, token_segments, action_descriptions = self.generate_scenario_one_pipeline_four(batch=inp_batch, temperature=1.0)
                
                # Generate the motion tokens in between
                self.logger.info("Generate the motion tokens in between segments")
                interp_token_segments = self.__merge_token_segments(token_segments=token_segments)
                
                # Combine the motion tokens
                self.logger.info("Combine motion tokens")
                combined_tokens = self.__combine_token_segments(token_segments=token_segments, interp_token_segments=interp_token_segments)

                # Decode the tokens
                pred_motion = self.__decode_motion(input_tokens=combined_tokens, cat_name=cat_name)
                pred_motion = apply_inverse_transform(pred_motion, data_obj=self.eval_dataset)

                # Generate color labels of predicted motion sequence (visualization only)
                start_i = 0
                color_labels = np.ones((pred_motion.size(1),))
                for i in range(len(interp_token_segments)):
                    start_i += token_segments[i].size(-1) * 4       # Update the start_index to current position
                    end_i = start_i + interp_token_segments[i].size(-1) * 4
                    color_labels[start_i:end_i] = 0
                    start_i = end_i                                 # Update the end_index
                                
                output = {
                    "gt": {"body": gt_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                    "pred": {"body": pred_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                    "caption": [pred_scene], 
                    "actions": action_descriptions, 
                    "color_labels": color_labels
                }
                results.append(output)
            save_generation_results(results=results, output_path=self.output_dir, batch_id=batch_id, 
                                    modality="se2_{:s}".format(self.args.eval_pipeline))
    
    @torch.no_grad()
    def generate_user_input(self):
        with open(self.args.user_input, "r") as f:
            user_input_descriptions = json.load(f)
        total_sements = []  # Store tokens corresponding to all tasks descriptions
        total_captions = [] # Store captioned texts corresponding to all tasks descriptions
        total_tasks = []    # Store predicted tasks descriptions
        for batch_id, texts in enumerate(user_input_descriptions):
            segments = []   # Store tokens corresponding to same task description
            captions = []   # Store captioned texts corresponding to same task description
            for text in texts:
                # Generate motion tokens from text description
                pred_tokens = self.__generate_next_motion(inp_text=text, max_num_tokens=256)
                print("[{:d}/{:d}] | [Text] {:s} | Motion Token Length {:d}".format(
                    batch_id+1, len(user_input_descriptions), text, pred_tokens.size(-1)))
                # Understand the motion as task description
                pred_text = self.models["gpt"].generate_motion_to_text(
                    pred_tokens, topk=self.args.topk, max_num_tokens=256)
                print("[{:d}/{:d}] | [Captioned Text] {:s}".format(
                    batch_id+1, len(user_input_descriptions), pred_text))
                segments.append(pred_tokens)
                captions.append(pred_text)
                total_sements.append(pred_tokens)
                total_captions.append(pred_text)
            # Predict task description from steps descriptions
            inp_batch = {"cur_steps": ["\n".join(["{:d}. {:s}".format(i+1, t) for (i, t) in enumerate(captions)])]}
            pred_task = self.models["gpt"].generate_planning(
                batch=inp_batch, task="s2t", topk=10, 
                max_num_tokens=256, temperature=1.0)
            total_tasks.append(pred_task)
            print("[{:d}/{:d}] | [Predicted Task] {:s}".format(
                batch_id+1, len(user_input_descriptions), pred_task))
            # Generate the motion tokens in between corresponding to same task description
            self.logger.info("Generate the motion tokens in between segments")
            interp_token_segments = self.__merge_token_segments(token_segments=segments)
            # Combine the motion tokens corresponding to same task description
            self.logger.info("Combine motion tokens")
            combined_tokens = self.__combine_token_segments(token_segments=segments, interp_token_segments=interp_token_segments)
            # Decode the tokens corresponding to same task description
            self.logger.info("Decode motion tokens")
            pred_motion = self.__decode_motion(input_tokens=combined_tokens, cat_name="t2m")
            pred_motion = apply_inverse_transform(pred_motion, data_obj=self.eval_dataset)
            
            # Generate color labels of predicted motion sequence (visualization only)
            start_i = 0
            color_labels = np.ones((pred_motion.size(1),))
            for i in range(len(interp_token_segments)):
                start_i += segments[i].size(-1) * 4       # Update the start_index to current position
                end_i = start_i + interp_token_segments[i].size(-1) * 4
                color_labels[start_i:end_i] = 0
                start_i = end_i                                 # Update the end_index
                        
            output = {
                "gt": {"body": pred_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                "pred": {"body": pred_motion.permute(0, 2, 1).data.cpu().numpy()}, 
                "caption": texts,       # Input steps descriptions
                "actions": captions,    # Captioned steps descriptions
                "task": [pred_task], 
                "color_labels": color_labels
            }
            save_generation_results(results=[output], output_path=self.output_dir, batch_id=batch_id, modality="usr_inp")
        
        # Generate the motion tokens in between corresponding to all tasks descriptions
        self.logger.info("Generate the motion tokens in between segments")
        interp_token_segments = self.__merge_token_segments(token_segments=total_sements)
        # Combine the motion tokens corresponding to all tasks descriptions
        self.logger.info("Combine motion tokens")
        combined_tokens = self.__combine_token_segments(token_segments=total_sements, interp_token_segments=interp_token_segments)
        # Decode the tokens corresponding to all tasks descriptions
        self.logger.info("Decode motion tokens")
        pred_motion = self.__decode_motion(input_tokens=combined_tokens, cat_name="t2m")
        pred_motion = apply_inverse_transform(pred_motion, data_obj=self.eval_dataset)
        
        # Generate color labels of predicted motion sequence (visualization only)
        start_i = 0
        color_labels = np.ones((pred_motion.size(1),))
        for i in range(len(interp_token_segments)):
            start_i += total_sements[i].size(-1) * 4       # Update the start_index to current position
            end_i = start_i + interp_token_segments[i].size(-1) * 4
            color_labels[start_i:end_i] = 0
            start_i = end_i                                 # Update the end_index
        
        output = {
            "gt": {"body": pred_motion.permute(0, 2, 1).data.cpu().numpy()}, 
            "pred": {"body": pred_motion.permute(0, 2, 1).data.cpu().numpy()}, 
            "caption": texts,       # Input steps descriptions
            "actions": total_captions,      # Captioned steps descriptions
            "color_labels": color_labels
        }
        save_generation_results(results=[output], output_path=self.output_dir, batch_id=9999, modality="usr_inp")
    
    @torch.no_grad()
    def generate_scenario_one_pipeline_zero(self, batch, temperature=1.0):
        """Forward process, start from scene information, generate task plannings, and generation motion sequences.
        """
        results = []
        scene = random.choice(batch["scene"])
        action = random.choice(batch["action"])
        task = random.choice(action["executable_simplified"])
        # steps = action["executable_steps"]
        
        token_segments = []
        text_descriptions = []
        # Generate task descriptions and corresponding token sequence
        while len(token_segments) < 15: # Conduct 3 rounds of planning
            # Scene-Task-to-Task
            inp_batch = {"scene": [scene], "cur_task": [task]}
            output_task = self.models["gpt"].generate_planning(
                batch=inp_batch, task="ct2t", topk=10, 
                max_num_tokens=256, temperature=temperature)
            self.logger.info("[Decision Making] scene: {:s} | generated task: {:s}".format(scene, output_task))
            # Decompose task to steps
            inp_batch = {"scene": [scene], "cur_task": [output_task]}
            output_steps = self.models["gpt"].generate_planning(
                batch=inp_batch, task="t2s", topk=self.args.topk, 
                max_num_tokens=256, temperature=temperature)
            self.logger.info("[Decision Making] task: {:s} | generated steps: {:s}".format(output_task, output_steps))
            steps = []
            for i in range(5, 0, -1):
                try:
                    steps.append(output_steps.split("{:d}. ".format(i))[1])
                    output_steps = output_steps.split("{:d}. ".format(i))[0]
                except:
                    pass
            if len(steps) == 0:
                continue
            steps = steps[::-1]
            # Task-to-Motion
            step_id = 0
            while step_id < len(steps):
                output_token = self.__generate_next_motion(inp_text=steps[step_id], max_num_tokens=256)
                if output_token is None or output_token.size(-1) <= 0:
                    continue
                else:
                    self.logger.info("[Synthesis] task: {:s} | generated token length: {:d}".format(steps[step_id], output_token.size(-1)))
                    token_segments.append(output_token)
                    text_descriptions.append({"task": output_task, "step": steps[step_id]})
                    step_id += 1
            task = output_task
    
        return scene, token_segments, text_descriptions
                    
    @torch.no_grad()
    def __generate_next_motion(self, inp_text, max_num_tokens=10):
        pred_tokens = self.models["gpt"].generate_text_to_motion(
            [inp_text], topk=self.args.topk, 
            max_num_tokens=max_num_tokens, 
            temperature=self.args.temperature)
        if pred_tokens is not None:
            return pred_tokens.unsqueeze(dim=0)
        else:
            return None
        
    @torch.no_grad()
    def __generate_motion_text(self, inp_motion, max_num_tokens=10):
        # Tokenize gt motion sequence, shape is [1, T]
        lengths = [inp_motion.size(1)]
        gt_tokens = self.encode_motion(
            motion=inp_motion.float().to(self.device), 
            cat_name="t2m", 
            part_name="body", 
            lengths=lengths)
        pred_texts = self.models["gpt"].generate_motion_to_text(
            gt_tokens, topk=self.args.topk, 
            max_num_tokens=256, 
            temperature=self.args.temperature)
        return pred_texts
        
    @torch.no_grad()
    def __merge_token_segments(self, token_segments):
        interp_token_segments = []
        for i in range(1, len(token_segments), 1):
            start_len = (token_segments[i-1].size(1) * 2) // 5
            end_len = (token_segments[i].size(1) * 2) // 5
            start_tokens = token_segments[i-1][:, -start_len:]
            end_tokens = token_segments[i][:, :end_len]
            interp_tokens = self.models["gpt"].generate_motion_to_motion(
                m_start_tokens=start_tokens, 
                m_end_tokens=end_tokens, 
                topk=self.args.topk, 
                max_num_tokens=50, 
                use_semantic_sampling=self.args.use_semantic_sampling)
            interp_token_segments.append(interp_tokens.unsqueeze(dim=0))
        return interp_token_segments
    
    @torch.no_grad()
    def __combine_token_segments(self, token_segments, interp_token_segments):
        combined_tokens = []
        for i in range(len(interp_token_segments)):
            combined_tokens.append(token_segments[i])
            combined_tokens.append(interp_token_segments[i])
        combined_tokens.append(token_segments[-1])
        combined_tokens = torch.cat(combined_tokens, dim=-1)    # [1, T]
        return combined_tokens
    
    @torch.no_grad()
    def __decode_motion(self, input_tokens, cat_name="t2m"):
        token_mask = input_tokens.gt(self.eos["{:s}_body".format(cat_name)].item())
        input_tokens = input_tokens[token_mask]
        input_tokens -= 3    # Convert back to original token space
        if input_tokens.dim() == 1: 
            input_tokens = input_tokens.unsqueeze(dim=0)
        pred_motion = self.decode_motion(input_tokens, cat_name, "body")
        return pred_motion
    
    def generate(self):
        for key, model in self.models.items():
            model.eval()
        
        # We generate motion from text descriptions, the text descriptions are generated from text-to-text mode.
        if self.args.eval_mode == "test":
            self.generate_on_testset()
        elif self.args.eval_mode == "usr_inp":
            self.generate_user_input()
        elif self.args.eval_mode == "se1":
            self.generate_long_motion_sequence_from_scene()
        elif self.args.eval_mode == "se2":
            self.generate_text_from_long_motion_sequence()
            # self.generate_scenario_two()
                
        self.logger.info("Done")
