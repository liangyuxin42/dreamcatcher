from .base_ranker import BaseRanker
import numpy as np

SCORE_NAEMS = ["unigram_overlap_with_answer_score","top_k_avg_probs_score","sim2answer_score","gpt_judgement_score"]

class LenRanker(BaseRanker):
    def __init__(self,short_diff=80,all_data=None,score_names=SCORE_NAEMS):
        super().__init__()
        self.short_diff = short_diff
        self.data = all_data
        self.score_names = score_names
        self.threshold = {}
        self.ranked_data = []
    
    def rank(self):
        self.add_lendiff_label()
        self.normalize_score()
        self.label()
        self.rank_by_label()

    def add_lendiff_label(self):
        for i,data in enumerate(self.data):
            for j,gen in enumerate(data["generation"]):
                len_diff = len(gen["text"]) - len(data["answer"])
                if len_diff < self.short_diff:
                    self.data[i]["generation"][j]["lendiff"] = "short"
                else:
                    self.data[i]["generation"][j]["lendiff"] = "long"
    
    def normalize_score(self):
        for name in self.score_names:
            score = [gen[name] for data in self.data for gen in data["generation"] if name in gen]
            max_score = max(score)
            min_score = min(score)
            for i,data in enumerate(self.data):
                for j,gen in enumerate(data["generation"]):
                    if name in gen:
                        self.data[i]["generation"][j][name] = (gen[name]-min_score)/(max_score-min_score)
            score = [gen[name] for data in self.data for gen in data["generation"] if name in gen]
            self.threshold[name] = np.median(score)
        
        self.threshold["gpt_judgement_score"] = 0.5
        print("threshold:", self.threshold)
    
    def label(self):
        # 长度差小 -> unigram overlap with answer
        # 长度差大 -> 计算gpt+similarity_to_answer_score+unigram_score+unigram_overlap
        short_threshold = self.threshold["unigram_overlap_with_answer_score"]
        long_threshold = self.threshold["gpt_judgement_score"]+self.threshold["sim2answer_score"]+self.threshold["unigram_overlap_with_answer_score"]+self.threshold["top_k_avg_probs_score"]

        for i,data in enumerate(self.data):
            for j,gen in enumerate(data["generation"]):
                if gen["type"] == "uncertainty":
                    continue
                if gen["lendiff"] == "short":
                    score = gen["unigram_overlap_with_answer_score"]
                    self.data[i]["generation"][j]["label"] = 1 if score > short_threshold else 0
                else:
                    score = gen["gpt_judgement_score"]+gen["sim2answer_score"]+gen["unigram_overlap_with_answer_score"]+gen["top_k_avg_probs_score"]
                    self.data[i]["generation"][j]["label"] = 1 if score > long_threshold else 0

    def rank_by_label(self):
        # 根据generation的label分类：all-right,all-wrong,partial-right
        # 根据分类，对generation进行排序：right>uncertainty>wrong
        for i,data in enumerate(self.data):
            right_count = sum([gen["label"] for gen in data["generation"] if gen["type"]=="normal"])
            normal_count = sum([gen["type"]=="normal" for gen in data["generation"]])
            best_right = max([gen for gen in data["generation"] if gen["type"]=="normal" and gen["label"]==1],key=lambda x:x["rm_score"]+x["sim2answer_score"], default=None)
            best_wrong = max([gen for gen in data["generation"] if gen["type"]=="normal" and gen["label"]==0],key=lambda x:x["rm_score"]-x["sim2answer_score"], default=None)
            best_uncertainty = max([gen for gen in data["generation"] if gen["type"]=="uncertainty"],key=lambda x:x["rm_score"])

            if right_count == normal_count:
                self.data[i]["category"] = "all-right"
                self.data[i]["rank"] = [best_right,best_uncertainty]
            elif right_count == 0:
                self.data[i]["category"] = "all-wrong"
                self.data[i]["rank"] = [best_uncertainty,best_wrong]
            else:
                self.data[i]["category"] = "partial-right"
                self.data[i]["rank"] = [best_right,best_uncertainty,best_wrong]
            self.ranked_data.append({
                "query":[{"role":"human","text":data["question"]}],
                "preference":[g["text"] for g in data["rank"]],
                "task":data["category"]
                })
            