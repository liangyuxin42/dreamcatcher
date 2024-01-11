import argparse
import jsonlines
import torch
import gc
import yaml
from tqdm import tqdm
from scorer import UnigramScorer,SimilarityScorer,RewardModelScorer,ProbeScorer,UnigramScorer_en
from ranker import LenRanker,SimpleRanker

def add_args(args_parser):
    args_parser.add_argument("--config_path", type=str)
    return args_parser

def load_data(config):
    data = []
    with jsonlines.open(config["input_data_path"]) as reader:
        for obj in reader:
            data.append(obj)
    print(f"Loaded {len(data)} data",flush=True)
    print("data[0]",data[0],flush=True)
    for i,d in enumerate(data):
        for j,gen in enumerate(d["generation"]):
            if "gpt_judgement" in gen:
                d["generation"][j]["gpt_judgement_score"] = 1 if gen["gpt_judgement"]=="正确" else 0
    print("data[0]",data[0],flush=True)

    return data

def score_data(data,config):
    scorers = {
        # "gpt_scorer":GPTScorer,
        "unigram_scorer":UnigramScorer,
        "unigram_scorer_en":UnigramScorer_en, # for english
        "similarity_scorer":SimilarityScorer,
        "probe_scorer":ProbeScorer,
        "rm_scorer":RewardModelScorer,
        }

    for scorer_name in config:
        print(f"Scoring with {scorer_name}",flush=True)
        score_config = config[scorer_name]
        Scorer = scorers[scorer_name]
        scorer = Scorer(score_config)
        for d in tqdm(data):
            scorer.score(d)
        print(f"Done scoring with {scorer_name}",flush=True)
        print(data[0],flush=True)
        del scorer
        torch.cuda.empty_cache()
        gc.collect()

    return data

def rank_data(data,config=None):
    rankers = {
        "SimpleRanker":SimpleRanker,
        "LenRanker":LenRanker,
    }
    Ranker = rankers[config["ranker_name"]]
    ranker = Ranker(all_data=data,config=config)
    ranker.rank()
    
    data = ranker.data
    ranked_data = ranker.ranked_data

    return data,ranked_data

def main(args):
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config,flush=True)
    
    data = load_data(config["data"])
    print("load data ",data[0],flush=True)
    if config["scorer"] is not None:
        data = score_data(data,config["scorer"])
        print("score data ",data[0],flush=True)

    with jsonlines.open(config["data"]["output_data_path"],"w") as writer:
        for d in data:
            writer.write(d)
            
    ranked_data = None
    if config["ranker"] is not None:
        data,ranked_data = rank_data(data,config["ranker"])
        print("rank data ",data[0],ranked_data[0],flush=True)

    with jsonlines.open(config["data"]["output_data_path"],"w") as writer:
        for d in data:
            writer.write(d)
            
    if ranked_data is not None:
        with jsonlines.open(config["data"]["output_rank_data_path"],"w") as writer:
            for d in ranked_data:
                writer.write(d)

    print("Done!")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser = add_args(args_parser)
    args = args_parser.parse_args()
    print("args",args,flush=True)
    main(args=args)