from .base_scorer import BaseScorer
import jieba
import jieba.analyse
import spacy
import math
very_small_value = 0.000001

class UnigramScorer(BaseScorer):
    def __init__(self,config):
        super().__init__()
        self.top_k = config["unigram_top_k"]
    
    def score(self, data):
        right_answer = data["answer"]
        onegram_probs,top_k_avg_probs = self.get_onegram_prob(right_answer,data["generation"]) 
        for j,generation in enumerate(data["generation"]):
            if generation["type"]=="normal":
                generated_answer = generation["text"]
                score = self.unigram_overlap(right_answer,generated_answer)
                data["generation"][j]["unigram_overlap_with_answer_score"] = score
                # data["generation"][j]["onegram_probs"] = onegram_probs[j]
                data["generation"][j]["top_k_avg_probs_score"] = top_k_avg_probs[j]

        return data
    
    def unigram_overlap(self,answer,generation):
        if not isinstance(answer,str):
            print("answer is not str: ",answer)
            answer = " ".join(answer)
        unigram_answer = [g for g in jieba.cut(answer)]
        unigram_gen = jieba.analyse.extract_tags(generation, topK=100, withWeight=False, allowPOS=())
        count = 0
        for uni in unigram_answer:
            if uni in unigram_gen:
                count+=1
        score = count/(len(unigram_answer)+very_small_value)
        return score   

    def get_onegram_prob(self,answer,generations):
        onegram_count = {}
        if not isinstance(answer,str):
            print("answer is not str: ",answer)
            answer = " ".join(answer)

        for s in jieba.analyse.extract_tags(answer, topK=20, withWeight=False, allowPOS=()):
            onegram_count[s] = onegram_count[s]+1 if s in onegram_count else 1
        generation_cuts = []
        for g in generations:
            cuts  = [s for s in jieba.analyse.extract_tags(g["text"], topK=20, withWeight=False, allowPOS=())]
            generation_cuts.append(cuts)
            for s in cuts:
                onegram_count[s] = onegram_count[s]+1 if s in onegram_count else 1
        all_count = sum([v for k,v in onegram_count.items()])
        onegram_probs = []
        top_k_avg_probs = []
        for cuts in generation_cuts:
            probs = [onegram_count[c]/(all_count+very_small_value) for c in cuts]
            probs = [-1*math.log(p) for p in probs]
            onegram_probs.append([{c:p} for c,p in zip(cuts,probs)])
            probs.sort()
            top_k_avg_probs.append(sum(probs[:self.top_k])/(len(probs[:self.top_k])+very_small_value))

        return onegram_probs,top_k_avg_probs

class UnigramScorer_en(BaseScorer):
    def __init__(self,config):
        super().__init__()
        self.top_k = config["unigram_top_k"]
        self.nlp = spacy.load("en_core_web_sm")
    
    def score(self, data):
        right_answer = data["answer"]
        if not isinstance(right_answer,str):
            print("answer is not str: ",right_answer,flush=True)
            right_answer = str(right_answer)
        onegram_probs,top_k_avg_probs = self.get_onegram_prob(right_answer,data["generation"]) 
        for j,generation in enumerate(data["generation"]):
            if generation["type"]=="normal":
                generated_answer = generation["text"]
                score = self.unigram_overlap(right_answer,generated_answer)
                data["generation"][j]["unigram_overlap_with_answer_score"] = score
                # data["generation"][j]["onegram_probs"] = onegram_probs[j]
                data["generation"][j]["top_k_avg_probs_score"] = top_k_avg_probs[j]
        return data
    
    def unigram_overlap(self,answer,generation):
        unigram_answer = [token.text.lower() for token in self.nlp(answer)]
        unigram_gen = set([token.text.lower() for token in self.nlp(generation)])

        count = 0
        for uni in unigram_answer:
            if uni in unigram_gen:
                count+=1
        score = count/(len(unigram_answer)+very_small_value)
        return score   

    def get_onegram_prob(self,answer,generations):
        onegram_count = {}
        for s in answer.split(" "):
            onegram_count[s] = onegram_count[s]+1 if s in onegram_count else 1
        generation_cuts = []
        for g in generations:
            cuts = [s.text.lower() for s in self.nlp(g["text"]) if not s.is_stop]
            generation_cuts.append(cuts)
            for s in cuts:
                onegram_count[s] = onegram_count[s]+1 if s in onegram_count else 1
        all_count = sum([v for k,v in onegram_count.items()])
        onegram_probs = []
        top_k_avg_probs = []
        for cuts in generation_cuts:
            probs = [onegram_count[c]/(all_count+very_small_value) for c in cuts]
            probs = [-1*math.log(p) for p in probs]
            onegram_probs.append([{c:p} for c,p in zip(cuts,probs)])
            probs.sort()
            top_k_avg_probs.append(sum(probs[:self.top_k])/(len(probs[:self.top_k])+very_small_value))

        return onegram_probs,top_k_avg_probs
