import torch
from typing import Callable, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, LlamaConfig, LlamaModel, AutoTokenizer
from .base_scorer import BaseScorer

class LlamaHFRewardModel(PreTrainedModel):
    # SAMPLE-LEVEL REWARD MODEL
    config_class =LlamaConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.value_head = torch.nn.Linear(config.hidden_size, 1) 
        # self.granularity = granularity
    
    def forward(self,
                input_ids: torch.LongTensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids,attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        if attention_mask is None:
            last_hidden_states = hidden_states[:, -1]
        else:
            last_index =  torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attention_mask], dtype=torch.int64)
            last_hidden_states = hidden_states[torch.arange(hidden_states.shape[0]), last_index]
        values = self.value_head(last_hidden_states).squeeze(-1)
        return values

def build_llama2_13B_tokenizer(tokenizer_path:str) -> AutoTokenizer:
    special_token_dict = {'pad_token': '</s>'}
    llama_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    llama_tokenizer.add_special_tokens(special_token_dict)
    return llama_tokenizer

def build_llama2_en_13B_tokenizer(tokenizer_path:str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class RewardModelScorer(BaseScorer):
    def __init__(self, config):
        super().__init__()

        self.model = LlamaHFRewardModel.from_pretrained(config["rm_model_path"]).to(torch.bfloat16).cuda().eval()
        self.model_type = config["model_type"]
        if config["model_type"]=="llama2_13b_en":
            self.tokenizer = build_llama2_en_13B_tokenizer(config["rm_tokenizer_path"])
        else:
            self.tokenizer = build_llama2_13B_tokenizer(config["rm_tokenizer_path"])

    def score(self, data):
        all_text = []
        for j,generation in enumerate(data["generation"]):
            question = data["question"]
            task = generation["type"] if "type" in  generation else generation["task"]
            question = f"请回答下列{task}问题：" + question
            response = generation["text"]
            if generation["type"]=="uncertainty" and self.model_type!="llama2_13b_en":
                question = f"{question}\n对于这个问题，请简短地回答你由于缺乏对应的知识或信息，因此不知道或不确定答案。"
            elif generation["type"]=="uncertainty" and self.model_type=="llama2_13b_en":
                question = f"{question}\nFor this question, please answer that you are unsure of the answer due to lack of corresponding knowledge or information."
            
            # text = f"<Human Round-1>:{question}\n<Assistant Round-1>:{response}"+self.tokenizer.eos_token
            #TODO add more model_type
            if self.model_type=="llama2_13b_en":
                text  = f"[INST] <<SYS>><</SYS>>{question} [/INST] {response}"+self.tokenizer.eos_token
            else:
                text = f"<human>:{question}\n<bot>:{response}"+self.tokenizer.eos_token
            all_text.append(text)

        d = self.tokenizer(all_text,return_tensors="pt",padding=True,truncation=True,max_length=2048)
        with torch.no_grad():
            outputs = self.model(input_ids=d["input_ids"].to(self.model.device),attention_mask=d["attention_mask"].to(self.model.device))
        outputs = outputs.cpu()
        for j in range(len(data["generation"])):
            data["generation"][j]["rm_score"] = outputs[j].item()

        return data
        