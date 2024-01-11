from .base_generator import BaseGenerator
from transformers import LlamaForCausalLM, AutoTokenizer,LlamaTokenizer
from tokenizers import AddedToken
import torch
import random
# from typing import List

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class LlamaGenerator(BaseGenerator):
    def __init__(self,config):
        super().__init__()
        self.load_model(config)
        self.set_tokenizer(config)

    def load_model(self,config):
        model_path = config["model_path"]
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            return_dict=True,
        ).half().cuda()

    def set_tokenizer(self,config):
        tokenizer_path = config["tokenizer_path"]
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    
    def format_tokens(self, queries, system_prompt=None):
        if system_prompt is None:
            queries = [B_INST+q+E_INST for q in queries]
        else:
            queries = [B_INST+B_SYS+system_prompt+E_SYS+q+E_INST for q in queries]
        inputs = self.tokenizer(queries, padding='longest',return_tensors="pt")
        return inputs
    
    @torch.no_grad()
    def generate(self, queries, k = 5, generation_type="normal", **kwargs):
        generate_kwargs = {
            "do_sample": True,
            "top_p": 1.0,   
            "top_k": 80,
            "max_length": 2048,
            "repetition_penalty": 1.0,
            "temperature": 0.8,
        }
        generate_kwargs.update(kwargs)
        responses = []
        for query in queries:
            if generation_type == "normal":
                tokens = self.format_tokens([query]*k)
            elif generation_type == "uncertainty":
                # system_prompt = "For the following factual questions, give an answer indicating that you are unsure or don't know because of a lack of relevant knowledge."
                system_prompt = random.choice([
                        "For the following questions, acknowledge your lack of relevant information in your answer.",
                        "For the following questions, acknowledge in your answer that you are unaware of the information and therefore cannot answer.",
                        "For the following questions, give an answer indicating that you are not aware of the correct answer because of a lack of relevant knowledge.",
                        "For the following questions, apologize for your lack of sufficient information to answer them correctly, and provide other relevant information that may be helpful.",
                        "For the following questions, apologize for your lack of sufficient information to answer them correctly, and ask for further information to clarify of the question."
                    ])
                tokens = self.format_tokens([query]*k,system_prompt=system_prompt)
            else:
                raise ValueError("generation_type must be normal or uncertainty, but got %s"%generation_type)
            
            outputs = self.model.generate(
                input_ids=tokens.input_ids.to(self.model.device),
                attention_mask=tokens.attention_mask.to(self.model.device),
                **generate_kwargs
                )
            output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            response = [text.split(E_INST)[1].strip() for text in output_text]
            responses.append(response)

        return responses
    
class LlamaGenerator_vllm():
    def __init__(self,config):
        super().__init__()
        from vllm import LLM, SamplingParams
        self.model = LLM(
            model=config["model_path"],
            tokenizer=config["tokenizer_path"],
            gpu_memory_utilization=0.8,
        )

        self.sampling_params = SamplingParams(
            top_p = 1.0,
            top_k = 80,
            max_tokens = 2048,
            repetition_penalty = 1.0,
            temperature=0.8, )

    def format_tokens(self, queries, system_prompt=None):
        if system_prompt is None:
            queries = [B_INST+q+E_INST for q in queries]
        else:
            queries = [B_INST+B_SYS+system_prompt+E_SYS+q+E_INST for q in queries]
        return queries

    def generate(self, queries, k = 5, generation_type="normal", **kwargs):
        all_queries = []
        for query in queries:
            if generation_type == "normal":
                tokens = self.format_tokens([query]*k)
            elif generation_type == "uncertainty":
                system_prompt = random.choice([
                        "For the following questions, acknowledge your lack of relevant information in your answer.",
                        "For the following questions, acknowledge in your answer that you are unaware of the information and therefore cannot answer.",
                        "For the following questions, give an answer indicating that you are not aware of the correct answer because of a lack of relevant knowledge.",
                        "For the following questions, apologize for your lack of sufficient information to answer them correctly, and provide other relevant information that may be helpful.",
                        "For the following questions, apologize for your lack of sufficient information to answer them correctly, and ask for further information to clarify of the question."
                    ])
                tokens = self.format_tokens([query]*k,system_prompt=system_prompt)
            else:
                raise ValueError("generation_type must be normal or uncertainty, but got %s"%generation_type)      
            all_queries.extend(tokens)
        print(len(all_queries))
        outputs = self.model.generate(all_queries, self.sampling_params)
        outputs = [o.outputs[0].text.strip() for o in outputs]
        outputs = [outputs[i:i+k] for i in range(0,len(outputs),k)]
        return outputs