import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from .base_generator import BaseGenerator
import random
import sys
sys.path.append("/cognitive_comp/songzhuoyang/models/modelscope/Qwen-14B-Chat/")
from qwen_generation_utils import make_context, decode_tokens

class QwenGenerator(BaseGenerator):
    def __init__(self,config):
        super().__init__()
        self.set_tokenizer(config)
        self.load_model(config)
    
    def load_model(self,config):
        path = config["model_path"]
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            pad_token_id=self.tokenizer.pad_token_id,
            device_map="auto",
            trust_remote_code=True,
            bf16=True
        ).eval().cuda()
        self.model.generation_config = GenerationConfig.from_pretrained(path, pad_token_id=self.tokenizer.pad_token_id)

    def set_tokenizer(self,config):
        path = config["tokenizer_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            padding_side='left',
            trust_remote_code=True
        )

    def batch_generation_qwen(self, all_raw_text, system_prompt="You are a helpful assistant."):
        batch_raw_text = []
        for q in all_raw_text:
            raw_text, _ = make_context(
                self.tokenizer,
                q,
                system=system_prompt,
                max_window_size=self.model.generation_config.max_window_size,
                chat_format=self.model.generation_config.chat_format,
            )
            batch_raw_text.append(raw_text)

        batch_input_ids = self.tokenizer(batch_raw_text, padding='longest')
        batch_input_ids = torch.LongTensor(batch_input_ids['input_ids']).to(self.model.device)
        batch_out_ids = self.model.generate(
            batch_input_ids,
            return_dict_in_generate=False,
            generation_config=self.model.generation_config
        )
        padding_lens = [batch_input_ids[i].eq(self.tokenizer.pad_token_id).sum().item() for i in range(batch_input_ids.size(0))]

        batch_response = [
            decode_tokens(
                batch_out_ids[i][padding_lens[i]:],
                self.tokenizer,
                raw_text_len=len(batch_raw_text[i]),
                context_length=(batch_input_ids[i].size(0)-padding_lens[i]),
                chat_format="chatml",
                verbose=False,
                errors='replace'
            ) for i in range(len(all_raw_text))
        ]
        return batch_response

    def generate(self, queries, k = 5, generation_type="normal", **kwargs):
        responses = []
        for query in queries:
            if generation_type=="normal":
                response = self.batch_generation_qwen([query]*k)
            elif generation_type=="uncertainty":
                system_prompt="You are a helpful assistant. 对于下面的事实性问题，请给出由于缺乏相关知识，因此你不确定或不知道的回答。"
                response = self.batch_generation_qwen([query]*k,system_prompt=system_prompt)
            else:
                print("generation_type not supported: ",generation_type)
                raise NotImplementedError
            responses.append(response)
        return responses


class QwenGenerator_vllm(BaseGenerator):
    def __init__(self,config):
        super().__init__()
        from vllm import LLM, SamplingParams
        self.model = LLM(
            model=config["model_path"],
            tokenizer=config["tokenizer_path"],
            gpu_memory_utilization=0.8,
            trust_remote_code = True,
        )

        self.sampling_params = SamplingParams(
            top_p = 0.8,
            top_k = -1,
            max_tokens = 2048,
            repetition_penalty = 1.1,
            stop_token_ids=[151643])

    def format_tokens(self,queries,system_prompt="You are a helpful assistant."):
        return_queries = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n" for q in queries]
        return return_queries
    
    def generate(self, queries, k = 5, generation_type="normal", **kwargs):
        all_queries = []
        for query in queries:
            if generation_type == "normal":
                tokens = self.format_tokens([query]*k)
            elif generation_type == "uncertainty":
                system_prompt=random.choice(["You are a helpful assistant. 对于下面的事实性问题，请给出由于缺乏相关知识，因此你不确定或不知道的回答。",])
                print(system_prompt)
                tokens = self.format_tokens([query]*k,system_prompt=system_prompt)
            else:
                raise ValueError("generation_type must be normal or uncertainty, but got %s"%generation_type)      
            all_queries.extend(tokens)
        print(len(all_queries))
        outputs = self.model.generate(all_queries, self.sampling_params)
        outputs = [o.outputs[0].text.split("<|im_end|>")[0] for o in outputs]
        outputs = [outputs[i:i+k] for i in range(0,len(outputs),k)]
        return outputs