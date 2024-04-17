import argparse, os, sys
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
from nlgeval import NLGEval, compute_metrics
import codecs as cs

def _strip(s):
    return s.strip()

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

class NLGEval_(NLGEval):
    glove_metrics = {
        'EmbeddingAverageCosineSimilarity',
        'VectorExtremaCosineSimilarity',
        'GreedyMatchingScore',
    }

    valid_metrics = {
                        # Overlap
                        'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
                        'METEOR',
                        'ROUGE_L',
                        'CIDEr',

                        # Skip-thought
                        'SkipThoughtCS',
                    } | glove_metrics
    
    def __init__(self, no_overlap=False, no_skipthoughts=False, no_glove=False,
                 metrics_to_omit=None):
        super(NLGEval_, self).__init__(no_overlap, no_skipthoughts, no_glove, metrics_to_omit)
        
    def compute_metrics(self, refs_list, hyps_list):
        refs = {idx: ref for idx, ref in enumerate(refs_list)}
        hyps = {idx: hyp for idx, hyp in enumerate(hyps_list)}
        assert len(refs) == len(hyps)
        
        ret_scores = {}
        if not self.no_overlap:
            for scorer, method in self.scorers:
                score, scores = scorer.compute_score(refs, hyps)
                if isinstance(method, list):
                    for sc, scs, m in zip(score, scores, method):
                        ret_scores[m] = sc
                else:
                    ret_scores[method] = score

        if not self.no_skipthoughts:
            vector_hyps = self.skipthought_encoder.encode([h.strip() for h in hyp_list], verbose=False)
            ref_list_T = self.np.array(ref_list).T.tolist()
            vector_refs = map(lambda refl: self.skipthought_encoder.encode([r.strip() for r in refl], verbose=False), ref_list_T)
            cosine_similarity = list(map(lambda refv: self.cosine_similarity(refv, vector_hyps).diagonal(), vector_refs))
            cosine_similarity = self.np.max(cosine_similarity, axis=0).mean()
            ret_scores['SkipThoughtCS'] = cosine_similarity

        if not self.no_glove:
            glove_hyps = [h.strip() for h in hyp_list]
            ref_list_T = self.np.array(ref_list).T.tolist()
            glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
            scores = self.eval_emb_metrics(glove_hyps, glove_refs, emb=self.glove_emb)
            scores = scores.split('\n')
            for score in scores:
                name, value = score.split(':')
                value = float(value.strip())
                ret_scores[name] = value

        return ret_scores

def Bleu_Rouge_Cider(nlg_eval, pred_texts, gt_texts, task="Task"):
    
    scores = nlg_eval.compute_metrics(refs_list=gt_texts, hyps_list=pred_texts)
    bleu_score = np.array([scores['Bleu_1'],scores['Bleu_2'],scores['Bleu_3'],scores['Bleu_4']])
    rouge_score = scores['ROUGE_L']
    cider_score = scores['CIDEr']
    
    print("Task: {:s} | ROUGE_L = {:.5f}".format(task, scores["ROUGE_L"]))
    print("Task: {:s} | CIDER = {:.5f}".format(task, scores["CIDEr"]))
    for i in range(4):
        print("Task: {:s} | Bleu_{:d} = {:.5f}".format(task, i+1, scores["Bleu_{:d}".format(i+1)]))
        
    return scores

def quick_proc(input_text):
    if isinstance(input_text, list):
        return input_text[0]
    else:
        return input_text

def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def main_planning_reverse(args):
    import json
    with open(args.input_dir, "r") as f:
        input_json = json.load(f)
    
    gt_steps, pred_steps = [], []
    for data in input_json.values():
        gt_steps_ = []
        for step in data["steps"]:
            gt_steps_.append(step["gt"])
            pred_steps.append([step["pred"]])
        for _ in range(len(gt_steps_)):
            gt_steps.append(gt_steps_)
            
    gt_tasks, pred_tasks = [], []
    for data in input_json.values():
        gt_tasks_ = []
        for task in data["tasks"]:
            gt_tasks_.append(task["gt"])
            pred_tasks.append([task["pred"]])
        for _ in range(len(gt_tasks_)):
            gt_tasks.append(gt_tasks_)
            
    print("Start to calculate Blue, Rouge, Cider")
    nlg_eval = NLGEval_(
        metrics_to_omit=[
            'METEOR',
            'EmbeddingAverageCosineSimilarity' ,
            'SkipThoughtCS',
            'VectorExtremaCosineSimilarity',
            'GreedyMatchingScore'
        ]
    )
    scores = Bleu_Rouge_Cider(
        nlg_eval, 
        pred_texts=pred_steps, 
        gt_texts=gt_steps, 
        task="Steps"
    )
    scores = Bleu_Rouge_Cider(
        nlg_eval, 
        pred_texts=pred_tasks, 
        gt_texts=gt_tasks, 
        task="Tasks"
    )
    
    # print("ROUGE_L: mean = {:.5f}".format(scores["ROUGE_L"]))
    # print("CIDEr: mean = {:.5f}".format(scores["CIDEr"]))
    # for i in range(4):
    #     bleus[i].append(scores["Bleu_{:d}".format(i+1)])
    #     print("Bleu_{:d}: mean = {:.5f}".format(i+1, scores["Bleu_{:d}".format(i+1)]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, 
        default="logs/avatar_gpt/eval/flan_t5_large/exp7/output/planning_se2_p0.json", 
        help="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main_planning_reverse(args=args)