import argparse, os, sys
sys.path.append(os.getcwd())
import importlib
import yaml
# os.environ['CURL_CA_BUNDLE'] = '' # SSLError: HTTPSConnectionPool(host='huggingface.co', port=443)
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='AvatarGPTEvaluator', help='task, choose from [AvatarGPTEvaluator, ...]')
    parser.add_argument('--config', type=str, default='configs/llm_t5/config_t5_large.yaml', help='path to the config file')
    parser.add_argument('--eval_folder', type=str, default='demo_outputs/', help='path of training folder')
    parser.add_argument('--eval_name', type=str, default='demo', help='')
    parser.add_argument('--repeat_times', type=int, default='1', help='number of repeat times per caption')
    parser.add_argument('--eval_mode', type=str, default="se1", help='choose from 1) test, 2) se1, 3) se2, 4) usr_inp')
    parser.add_argument('--eval_task', type=str, default="t2m,m2m,m2t,ct2t,cs2s,ct2s,cs2t,t2c,s2c,t2s,s2t", help="")
    parser.add_argument('--eval_pipeline', type=str, default="p0", help='evaluation pipeline, only valid then eval_mode is one of [se1, se2]')
    parser.add_argument('--topk', type=int, default=1, help='sample top-k from the output probability distribution')
    parser.add_argument('--use_semantic_sampling', type=str2bool, default=False, help='whether use semantic-aware sampling')
    parser.add_argument('--temperature', type=float, default=1.0, help='')
    parser.add_argument('--user_input', type=str, default="demo_inputs/text_descriptions.json", help='user input text')
    parser.add_argument('--demo_list', type=str, default="demo_inputs/test.txt", help='user input text')
    parser.add_argument('--demo_data', type=str, default="demo_inputs/samples", help='user input text')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    Agent = importlib.import_module(r".evaluator", package="modules").__getattribute__(args.task)(args, config)
    Agent.generate()
    
