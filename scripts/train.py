import argparse, os, sys
sys.path.append(os.getcwd())
import importlib
import yaml

# sys.path.append("/mnt/user/zhouzixiang/projects/workspace/UDE2.0")
# os.environ["NODE_RANK"] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CURL_CA_BUNDLE'] = '' # SSLError: HTTPSConnectionPool(host='huggingface.co', port=443)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='AvatarGPTTrainerV2',  
                        help='task, choose from [AvatarGPTTrainer, AudioGenTrainer]')
    parser.add_argument('--config', type=str, default='configs/llm_gpt/gpt_large/config_gpt_large_exp2.yaml', help='path to the config file')
    parser.add_argument('--dataname', type=str, default='HumanML3D', help='name of dataset, choose from [AMASS, AMASS-single, HumanML3D')
    parser.add_argument('--training_folder', type=str, default='logs/avatar_gpt/gpt_large/', help='path of training folder')
    parser.add_argument('--training_name', type=str, default='exp2-instruct', help='name of the training')
    parser.add_argument('--a2m_trg_start_index', type=int, default=8, help='number of primitive motion tokens for audio-to-motion task')
    parser.add_argument('--training_mode', type=str, default="t2m", help='mode of training')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    
    Agent = importlib.import_module(r".trainer", package="modules").__getattribute__(args.task)(args, config)
    Agent.train()
