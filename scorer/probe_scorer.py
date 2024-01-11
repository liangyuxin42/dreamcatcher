import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM
from tokenizers import AddedToken

from .base_scorer import BaseScorer

class LinearProbe(nn.Module):
    def __init__(self, hidden_size: int,):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        
    def forward(self,state):
        out = torch.sigmoid(self.linear(state))
        return out

class ProbeScorer(BaseScorer):
    def __init__(self,config):
        super().__init__()
        self.probe = LinearProbe(config["hidden_size"])
        if config["probe_model_path"] is not None:
            self.probe.load_state_dict(torch.load(config["probe_model_path"]))
        self.state_key = config["probe_state_key"]
        self.probe_layer = config["probe_layer"]
        self.load_model(config)
        self.set_tokenizer(config)

    def load_model(self,config):
        # self.model = LlamaForCausalLM.from_pretrained(config["model_path"]).to(torch.bfloat16).cuda().eval()
        self.model = AutoModelForCausalLM.from_pretrained(config["model_path"], trust_remote_code=True).to(torch.bfloat16).cuda().eval()

    def set_tokenizer(self, config):
        tk_path = config["tokenizer_path"]

        if config["model_type"] == "ziya":
            _SPECIAL_TOKENS_DICT = {'pad_token': '</s>'}
            human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
            bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)
            llama_tokenizer = AutoTokenizer.from_pretrained(tk_path,use_fast=False)
            llama_tokenizer.add_special_tokens(_SPECIAL_TOKENS_DICT)
            llama_tokenizer.add_special_tokens({"additional_special_tokens":[human, bot]})
            self.tokenizer = llama_tokenizer
        elif config["model_type"] == "qwen":
            self.tokenizer =  AutoTokenizer.from_pretrained(
                    tk_path,
                    pad_token='<|extra_0|>',
                    eos_token='<|endoftext|>',
                    # padding_side='left',
                    trust_remote_code=True
                )
        elif config["model_type"] == "llama2_13b_en":
            tokenizer = AutoTokenizer.from_pretrained(tk_path)
            tokenizer.pad_token = tokenizer.eos_token
            self.tokenizer = tokenizer
        else:
            raise ValueError("model_type not supported, got %s"%config["model_type"])

    @torch.no_grad()
    def score(self, data):
        question = data["question"]
        state = self.get_activation(question)
        out = self.probe(state)
        for j in range(len(data["generation"])):
            data["generation"][j]["probe_score"] = out.item()
        return data

    def get_activation(self,text,answer=None):
        input_ids = self.tokenizer(text,return_tensors="pt",padding=True,truncation=True,max_length=2048).input_ids
        with torch.no_grad():
            output = self.model(input_ids.to(self.model.device), output_hidden_states = True)
        hidden_states = [s.cpu()[0] for s in output.hidden_states]
        state = hidden_states[self.probe_layer][-1].to(torch.float)
        return state
