from .base_ranker import BaseRanker
import numpy as np

SCORE_NAEMS = ["unigram_overlap_with_answer_score","sim2answer_score","probe_score","sim2gen_score"] # ,"top_k_avg_probs_score"

class SimpleRanker(BaseRanker):
    def __init__(self, all_data,config, score_names=SCORE_NAEMS):
        super().__init__()
        self.data = all_data
        self.score_names = score_names
        self.up_percentile = config["up_percentile"]
        self.down_percentile = config["down_percentile"]
        self.threshold = {}
        self.threshold_up = {}
        self.threshold_down = {}
        self.ranked_data = []

    def rank(self):
        self.normalize_score()
        self.rank_by_score()

    def normalize_score(self):
        for name in self.score_names:
            score = [gen[name] for data in self.data for gen in data["generation"] if name in gen]
            if len(score)==0:
                continue
            max_score = max(score)
            min_score = min(score)
            for i,data in enumerate(self.data):
                for j,gen in enumerate(data["generation"]):
                    if name in gen:
                        self.data[i]["generation"][j][name] = (gen[name]-min_score)/(max_score-min_score)
            score = [gen[name] for data in self.data for gen in data["generation"] if name in gen]
            self.threshold[name] = np.median(score)
            self.threshold_up[name] = np.percentile(score,self.up_percentile)
            self.threshold_down[name] = np.percentile(score,self.down_percentile)
        
        self.threshold["gpt_judgement_score"] = 0.5
        print("threshold:", self.threshold)
    
    def rank_by_score(self):
        judge = ["probe_score","unigram_overlap_with_answer_score","sim2answer_score","sim2gen_score"]
        for i,data in enumerate(self.data):
            # sort by probe_score + unigram_overlap_with_answer_score + sim2answer_score + sim2gen_score
            judge_i=[]
            for j in judge:
                if any([(j not in gen) for gen in data["generation"] if gen["type"]=="normal"]):
                    continue    
                judge_i.append(j)
            threshold_up_i = sum([self.threshold_up[j] for j in judge_i])
            threshold_down_i = sum([self.threshold_down[j] for j in judge_i])

            rank = []
            category = []

            best_right = max([gen for gen in data["generation"] if gen["type"]=="normal"],key=lambda x:sum([x[j] for j in judge_i]), default=None)
            best_wrong = max([gen for gen in data["generation"] if gen["type"]=="normal"],key=lambda x:sum([-1*x[j] for j in judge_i]), default=None)
            best_uncertainty = max([gen for gen in data["generation"] if gen["type"]=="uncertainty"],key=lambda x:x["rm_score"])


            if sum([best_right[j] for j in judge]) > threshold_up_i:
                rank.append(best_right)
                category.append("right")
            rank.append(best_uncertainty)
            if sum([best_wrong[j] for j in judge]) < threshold_down_i:
                rank.append(best_wrong)
                category.append("wrong")

            if "wrong" not in category:
                self.data[i]["category"] = "all-right"
            elif "right" not in category:
                self.data[i]["category"] = "all-wrong"
            else:
                self.data[i]["category"] = "partial-right"

            self.data[i]["rank"] = rank

            if len(rank)>1:
                self.ranked_data.append({
                    "query":[{"role":"human","text":data["question"]}],
                    "preference":[g["text"] for g in data["rank"]],
                    "task":self.data[i]["category"]
                    })
