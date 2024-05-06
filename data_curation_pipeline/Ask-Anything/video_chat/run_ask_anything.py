import os, sys, json, argparse
sys.path.append(os.getcwd())
from utils.config import Config
import math
from models.videochat import VideoChat
from utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from models.video_transformers import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms

def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret

def get_context_emb(conv, model, img_list):
    prompt = get_prompt(conv)
    print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    seg_tokens = [
        model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
        # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]
    seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs

def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])
    
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
def answer(
    conv, model, img_list, 
    max_new_tokens=200, 
    num_beams=1, 
    min_length=1, 
    top_p=0.9,
    repetition_penalty=1.0, 
    length_penalty=1, 
    temperature=1.0, 
    do_sample=True
):
    stop_words_ids = [
        torch.tensor([835]).to("cuda:0"),
        torch.tensor([2277, 29937]).to("cuda:0")]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], None])
    embs = get_context_emb(conv, model, img_list)
    outputs = model.llama_model.generate(
        inputs_embeds=embs,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        num_beams=num_beams,
        do_sample=do_sample,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        temperature=temperature,
    )
    
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    conv.messages[-1][1] = output_text
    return output_text, output_token.cpu().numpy()

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def get_video_info(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    num_frames = len(vr)
    return math.ceil(fps), num_frames

def crop_video_into_segments(segment_len, fps, num_frames, num_valid_frames_per_seg):
    """
    :param segment_len: length of each video segment (second)
    :param fps: the frame-per-second of input video
    :param num_frames: number of frames of input video
    :param num_valid_frames_per_seg: number of frames we wand to keep in each segment
    """
    num_frames_per_seg = fps * segment_len
    stepsize = num_frames_per_seg // num_valid_frames_per_seg
    index_per_segment = []
    for i in range(0, num_frames, num_frames_per_seg):
        index = [j for j in range(i, i+num_frames_per_seg, stepsize) if j < num_frames]
        index_per_segment.append(index)
    return index_per_segment

def load_video(video_path, num_segments=8, return_msg=False):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = 224
    scale_size = 224
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs

def load_frames_from_video(vid_obj, frame_index):
    # transform
    crop_size = 224
    scale_size = 224
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    msgs = []
    for index in frame_index:
        img = Image.fromarray(vid_obj[index].asnumpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    
    fps = float(vid_obj.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in frame_index])
    msg = f"The video contains {len(frame_index)} frames sampled at {sec} seconds."
        
    return torch_imgs, msg

def build_model(cfg):
    
    model = VideoChat(config=cfg.model)
    model = model.to(torch.device(cfg.device))
    model = model.eval()
    
    return model

def main_simple_answer(model, config_file, vid_path, num_segments_list=[], prompt_list=[], json_file=""):
    
    json_results = {}
        
    fps, num_frames = get_video_info(vid_path)
    print(fps, num_frames)
    segment_len = 10
    index_per_segment = crop_video_into_segments(segment_len=segment_len, fps=fps, num_frames=num_frames, num_valid_frames_per_seg=16)
    
    vid_obj = VideoReader(vid_path, ctx=cpu(0))
    
    output_list = {
        "video_name": vid_path, 
        "video_fps": fps, 
        "caption_list": []
    }
    
    # 1. Get the video descriptions
    vid, msg = load_video(vid_path, num_segments=16, return_msg=True)
    # The model expects inputs of shape: T x C x H x W
    TC, H, W = vid.shape
    video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")
    img_list = []
    image_emb, _ = model.encode_img(video)
    img_list.append(image_emb)
    
    descriptions = []
    for prompt in prompt_list:
        
        chat = EasyDict({
        #     "system": "You are an AI assistant. A human gives an image or a video and asks some questions. You should give helpful, detailed, and polite answers.\n",
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
    
        chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
        ask(prompt, chat)
        for _ in range(5):
            llm_message = answer(conv=chat, model=model, img_list=img_list, max_new_tokens=1000)[0]
            descriptions.append(llm_message.replace("###Assistant:", ""))
        
    output_list["description"] = descriptions
    
    
    # 2. Get description of each video segment
    for i, index in enumerate(index_per_segment):
        start_time = i * segment_len                  # second
        start_frame = start_time * fps
        end_time = (i+1) * segment_len    # second
        end_frame = end_time * fps          # second
        try:
            vid, msg = load_frames_from_video(vid_obj=vid_obj, frame_index=index)
            print(start_time, start_frame, end_time, end_frame, vid.shape)

            # The model expects inputs of shape: T x C x H x W
            TC, H, W = vid.shape
            video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")

            img_list = []
            image_emb, _ = model.encode_img(video)
            img_list.append(image_emb)

            captions = []
            for prompt in prompt_list:

                chat = EasyDict({
                #     "system": "You are an AI assistant. A human gives an image or a video and asks some questions. You should give helpful, detailed, and polite answers.\n",
                    "system": "",
                    "roles": ("Human", "Assistant"),
                    "messages": [],
                    "sep": "###"
                })

                chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
                ask(prompt, chat)
                for _ in range(5):
                    llm_message = answer(conv=chat, model=model, img_list=img_list, max_new_tokens=1000)[0]
                    captions.append(llm_message.replace("###Assistant:", ""))

            output_list["caption_list"].append(
                {
                    "start_time": start_time, "start_frame": start_frame, 
                    "end_time": end_time, "end_frame": end_frame, 
                    "captions": captions
                }
            )
        except:
            print("Failed:", start_time, start_frame, end_time, end_frame, vid.shape)
            pass
    
    with open(json_file, "w") as f:
        json.dump(output_list, f)

def main_detail_answer(model, config_file, vid_path, max_new_tokens, num_segments_list=[], prompt_list=[], json_file=""):
    
    json_results = {}
        
    fps, num_frames = get_video_info(vid_path)
    print(fps, num_frames)
    segment_len = 10
    index_per_segment = crop_video_into_segments(
        segment_len=segment_len, fps=fps, 
        num_frames=num_frames, 
        num_valid_frames_per_seg=8)
    
    vid_obj = VideoReader(vid_path, ctx=cpu(0))
    
    output_list = {
        "video_name": vid_path, 
        "video_fps": fps, 
        "caption_list": []
    }
    
    """Encode the video."""
    vid, msg = load_video(vid_path, num_segments=16, return_msg=True)
    # The model expects inputs of shape: T x C x H x W
    TC, H, W = vid.shape
    video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")
    img_list = []
    image_emb, _ = model.encode_img(video)
    img_list.append(image_emb)
    
    """Get the context description of the input video."""
    descriptions = []
    for prompt in prompt_list:
        
        chat = EasyDict({
        #     "system": "You are an AI assistant. A human gives an image or a video and asks some questions. You should give helpful, detailed, and polite answers.\n",
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
    
        chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
        ask(prompt, chat)
        for _ in range(3):
            llm_message = answer(conv=chat, model=model, img_list=img_list, max_new_tokens=max_new_tokens)[0]
            descriptions.append(llm_message.replace("###Assistant:", ""))
        
    output_list["description"] = descriptions
    
    
    """Get description of each video segment."""
    for i, index in enumerate(index_per_segment):
        start_time = i * segment_len                  # second
        start_frame = start_time * fps
        end_time = (i+1) * segment_len    # second
        end_frame = end_time * fps          # second
        
        try:
            vid, msg = load_frames_from_video(vid_obj=vid_obj, frame_index=index)
            print(start_time, start_frame, end_time, end_frame, vid.shape)

            # The model expects inputs of shape: T x C x H x W
            TC, H, W = vid.shape
            video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")

            img_list = []
            image_emb, _ = model.encode_img(video)
            img_list.append(image_emb)

            captions = []
            for prompt in prompt_list:

                chat = EasyDict({
                #     "system": "You are an AI assistant. A human gives an image or a video and asks some questions. You should give helpful, detailed, and polite answers.\n",
                    "system": "",
                    "roles": ("Human", "Assistant"),
                    "messages": [],
                    "sep": "###"
                })

                chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
                ask(prompt, chat)
                for _ in range(3):
                    llm_message = answer(
                        conv=chat, 
                        model=model, 
                        img_list=img_list, 
                        max_new_tokens=500  # Default: 1000
                    )[0]
                    captions.append(llm_message.replace("###Assistant:", ""))

            output_list["caption_list"].append(
                {
                    "start_time": start_time, "start_frame": start_frame, 
                    "end_time": end_time, "end_frame": end_frame, 
                    "captions": captions
                }
            )
        except:
            print("Failed:", start_time, start_frame, end_time, end_frame, vid.shape)
            raise ValueError
    # with open("test_detail_answer.json", "w") as f:
    #     json.dump(descriptions, f)
    with open(json_file, "w") as f:
        json.dump(output_list, f)

def main_part_answer(model, config_file, vid_path, max_new_tokens, num_segments_list=[], prompt_list=[], json_file=""):
    
    json_results = {}
        
    fps, num_frames = get_video_info(vid_path)
    print(fps, num_frames)
    segment_len = 10
    index_per_segment = crop_video_into_segments(
        segment_len=segment_len, fps=fps, 
        num_frames=num_frames, 
        num_valid_frames_per_seg=16)
    
    vid_obj = VideoReader(vid_path, ctx=cpu(0))
    
    output_list = {
        "video_name": vid_path, 
        "video_fps": fps, 
        "caption_list": []
    }
    
    """Encode the video."""
    vid, msg = load_video(vid_path, num_segments=16, return_msg=True)
    # The model expects inputs of shape: T x C x H x W
    TC, H, W = vid.shape
    video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")
    img_list = []
    image_emb, _ = model.encode_img(video)
    img_list.append(image_emb)
    
    """Get the context description of the input video."""
    descriptions = []
    for prompt in prompt_list["context"]:
        chat = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
        ask(prompt, chat)
        for _ in range(3):
            llm_message = answer(conv=chat, model=model, img_list=img_list, max_new_tokens=max_new_tokens)[0]
            descriptions.append(llm_message.replace("###Assistant:", ""))
        
    output_list["description"] = descriptions
    
    """Get description of each video segment."""
    for i, index in enumerate(index_per_segment):
        start_time = i * segment_len                  # second
        start_frame = start_time * fps
        end_time = (i+1) * segment_len    # second
        end_frame = end_time * fps          # second
        
        vid, msg = load_frames_from_video(vid_obj=vid_obj, frame_index=index)
        print(start_time, start_frame, end_time, end_frame, vid.shape)
        
        # The model expects inputs of shape: T x C x H x W
        TC, H, W = vid.shape
        video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")
        
        img_list = []
        image_emb, _ = model.encode_img(video)
        img_list.append(image_emb)
        
        captions = []
        
        results = {key: [] for key in prompt_list["content"].keys()}
        for key, prompt in prompt_list["content"].items():
                
            chat = EasyDict({
                "system": "",
                "roles": ("Human", "Assistant"),
                "messages": [],
                "sep": "###"
            })

            chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
            for _ in range(3):
                try:
                    llm_message = answer(conv=chat, model=model, img_list=img_list, max_new_tokens=1000)[0]
                    # captions.append(llm_message.replace("###Assistant:", ""))
                    results[key].append(llm_message.replace("###Assistant:", ""))
                except:
                    pass
            
        captions.append(results)
        
        output_list["caption_list"].append(
            {
                "start_time": start_time, "start_frame": start_frame, 
                "end_time": end_time, "end_frame": end_frame, 
                "captions": captions
            }
        )
            
    # with open("test_detail_answer.json", "w") as f:
    #     json.dump(descriptions, f)
    with open(json_file, "w") as f:
        json.dump(output_list, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="part", help="1. simple, 2. detail, 3. part")
    parser.add_argument('--input_video_dir', type=str, default="example/pamela_videos", help='')
    parser.add_argument('--output_video_dir', type=str, default="example/pamela_videos", help='')
    parser.add_argument('--max_new_tokens', type=int, default=10, help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    os.makedirs(args.output_video_dir, exist_ok=True)
    config_file = "configs/config.json"
    
    cfg = Config.from_file(config_file)
    print("--- Build the Ask-Anything model")
    model = build_model(cfg=cfg)
    # model = None
    print("--- Model built successfully!")
        
    num_segments_list=[1, 4, 8, 16, 32]
    
    prompt_simple_list = [
        "Please describe the content of the video in detail, especially the actions performed by the person in the video.", 
        "Please describe the content of the video in detail."
    ]
    
    prompt_detail_list = [
        "Please describe the content of the video in detail, especially the actions performed by the person in the video.", 
        "Please describe the content of the video in detail."
    ]
    
    prompt_part_list = {
        "content": {
            "LH": "Please describe the movements of the persons's left arm in the video. The answer should be short and accurate.", 
            "RH": "Please describe the movements of the persons's right arm in the video. The answer should be short and accurate.", 
            "LF": "Please describe the movements of the persons's left leg in the video. The answer should be short and accurate.", 
            "RF": "Please describe the movements of the persons's right leg in the video. The answer should be short and accurate.", 
            "FB": "Please describe the body movements of the person in the video. The answer should be short and accurate.", 
            "LH": "What's the person's left arm movements?", 
            "RH": "What's the person's right arm movements?", 
            "LF": "What's the person's left leg movements?", 
            "RF": "What's the person's right leg movements?", 
            "FB": "Please describe how the person move his/her body in the video."
        }, 
        "context": "Please describe the content of the video in detail, especially the actions performed by the person in the video."
    }
    
    PROMPTS = {
        "simple": prompt_simple_list, 
        "detail": prompt_detail_list, 
        "part": prompt_part_list
    }
    
    # json_file = "../example/Ab_Development_Diary_Flat_Tummy_Eliminate_Lower_Belly_Fat_BeginnerFriendly.json"
    
    # base_path = "../example/test_videos"
    base_path = args.input_video_dir
    vid_files = [f for f in os.listdir(base_path) if ".mp4" in f]

    for file in vid_files:
        print('=' * 20, file, "=" * 20)
        vid_path = os.path.join(base_path, file)
        json_file = os.path.join(args.output_video_dir, file.replace(".mp4", ".json"))
        
        if os.path.exists(json_file): 
            continue
        
        if args.task == "simple":
            main_simple_answer(
                model=model, 
                config_file=config_file, 
                vid_path=vid_path, 
                num_segments_list=num_segments_list, 
                prompt_list=PROMPTS[args.task], 
                json_file=json_file)
        elif args.task == "detail":
            main_detail_answer(
                model=model, 
                config_file=config_file, 
                vid_path=vid_path, 
                max_new_tokens=args.max_new_tokens, 
                num_segments_list=num_segments_list, 
                prompt_list=PROMPTS[args.task], 
                json_file=json_file)
        elif args.task == "part":
            main_part_answer(
                model=model, 
                config_file=config_file, 
                vid_path=vid_path, 
                max_new_tokens=args.max_new_tokens, 
                num_segments_list=num_segments_list, 
                prompt_list=PROMPTS[args.task], 
                json_file=json_file)
        
    print("Done")
