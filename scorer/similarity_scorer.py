from .base_scorer import BaseScorer
import torch
from transformers import AutoTokenizer, AutoModel

class SimilarityScorer(BaseScorer):
    def __init__(self,config):
        super().__init__()
        self.embedding_model = AutoModel.from_pretrained(config["embedding_model_path"]).cuda().eval()
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(config["embedding_tokenizer_path"])
        self.lang = "zh" if "zh" in config["embedding_model_path"] else "en"
        self.use_answer_only = True if "use_answer_only" in config else False
        print("use_answer_only:",self.use_answer_only)

    @torch.no_grad()
    def score(self, data):
        question = data["question"]
        right_answer = data["answer"]
        text1 = self.add_prefix(question,right_answer)

        right_answer_embedding = self.get_embedding([text1])
        text2 = [self.add_prefix(question,generation["text"]) for generation in data["generation"] if generation["type"]=="normal"]
        gen_embedding = self.get_embedding(text2)
        sim2answer = right_answer_embedding @ gen_embedding.T
        sim2answer = sim2answer[0].tolist()
        sim2gen = gen_embedding @ gen_embedding.T
        sim2gen = (sim2gen.sum(dim=0)-1)/(gen_embedding.shape[0]-1)
        sim2gen = sim2gen.tolist()
        for j,generation in enumerate(data["generation"]):
            if generation["type"]=="normal":
                data["generation"][j]["sim2answer_score"] = sim2answer[j]
                data["generation"][j]["sim2gen_score"] = sim2gen[j]

        # right_answer_embedding = self.get_embedding(text1)
        # gen_embeddings = []
        # for j,generation in enumerate(data["generation"]):
        #     if generation["type"]=="normal":
        #         generated_answer = generation["text"]
        #         text2 = self.add_prefix(question,generated_answer)
        #         gen_embedding = self.get_embedding(text2)
        #         similarity = right_answer_embedding @ gen_embedding.T
        #         data["generation"][j]["sim2answer_score"] = similarity.item()
        #         gen_embeddings.append(gen_embedding)
                
        # #similarity between generations
        # for j,generation in enumerate(data["generation"]):
        #     if generation["type"]=="normal":
        #         gen_embedding = gen_embeddings[j]
        #         similarity = torch.stack([gen_embedding @ g.T for g in gen_embeddings]).mean()
        #         data["generation"][j]["sim2gen_score"] = similarity.item()

        return data

    # def get_embedding(self,text):
        # encoded_input = self.embedding_tokenizer([text], padding=True, truncation=True, return_tensors='pt',max_length=512)
        # encoded_input = {k:v.to(self.embedding_model.device) for k,v in encoded_input.items()}
        # model_output = self.embedding_model(**encoded_input)
        # embeddings = model_output[0][:, 0]
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # return embeddings[0]

    def get_embedding(self,text):
        encoded_input = self.embedding_tokenizer(text, padding=True, truncation=True, return_tensors='pt',max_length=512)
        encoded_input = {k:v.to(self.embedding_model.device) for k,v in encoded_input.items()}
        model_output = self.embedding_model(**encoded_input)
        embeddings = model_output[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


    def add_prefix(self,question,answer):
        if self.use_answer_only:
            if isinstance(answer,str):
                return answer
            print("answer:",answer)
            return str(answer)
        if self.lang=="zh":
            return f"问题：{question}\n回答：{answer}"
        else:
            return f"Question: {question}\nAnswer: {answer}"
    

