from .base_generator import BaseGenerator
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
from tokenizers import AddedToken
import torch.nn.functional as F


class ZiyaGenerator(BaseGenerator):
    def __init__(self,config):
        super().__init__()
        self.load_model(config)
        self.set_tokenizer(config)
    
    def load_model(self,config):
        self.model = LlamaForCausalLM.from_pretrained(config["model_path"]).to(torch.bfloat16).cuda().eval()

    def set_tokenizer(self,config):
        tk_path = config["tokenizer_path"]
        _SPECIAL_TOKENS_DICT = {'pad_token': '</s>'}
        human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
        bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)
        llama_tokenizer = AutoTokenizer.from_pretrained(tk_path)
        llama_tokenizer.add_special_tokens(_SPECIAL_TOKENS_DICT)
        llama_tokenizer.add_special_tokens({"additional_special_tokens":[human, bot]})
        self.tokenizer = llama_tokenizer

    @torch.no_grad()
    def generate(self, queries, k = 5, generation_type="normal", **kwargs):
        generate_kwargs = {
                "do_sample": True,
                "top_p": 0.95,   
                "top_k": 0,
                "max_length": 2048,
                "repetition_penalty": 1.0,
                "temperature": 0.8,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
        responses = []
        if generation_type == "uncertainty":
            queries = [f"{q}\n对于上述问题，请简短地回答你由于缺乏对应的具体知识或信息，因此不知道或不确定答案。" for q in queries]
        for query in queries:
            response = self.generate_batch([query]*k, **generate_kwargs)
            responses.append(response)
        return responses

    def generate_batch(self, queries, **generate_kwargs):
        def _apply_prefix(query):
            return f"<human>:{query.strip()}\n<bot>:"

        def _tokenizing(queries):

            input_ids = []
            for query in queries:
                query = _apply_prefix(query)
                input_ids.append(torch.tensor(self.tokenizer(query).input_ids))
            inputs = self.zero_pad_sequences(input_ids, side="left", padding_value=self.tokenizer.pad_token_id)
            return inputs
        # print(queries)
        input_ids = _tokenizing(queries).to(self.model.device)
        pad_token_id = self.tokenizer.pad_token_id
        input_attention_mask = input_ids.not_equal(pad_token_id).to(dtype=torch.bool, device=self.model.device)
        sequences = self.model.generate(
            input_ids.to(0), attention_mask=input_attention_mask, **generate_kwargs)
        output = []
        for seq in sequences:
            out_text = self.tokenizer.decode(seq.tolist()[len(input_ids[0]):], skip_special_tokens=False)
            # print(out_text)
            output.append(out_text.replace('<s>','').replace('</s>',''))
            # print(out_text)
        return output
    
    def zero_pad_sequences(self, sequences, side = 'left', padding_value: int = 0) -> torch.Tensor:
        assert side in ('left', 'right')
        max_len = max(seq.size(0) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            pad_len = max_len - seq.size(0)
            padding = (pad_len, 0) if side == 'left' else (0, pad_len)
            padded_sequences.append(F.pad(seq, padding, value=padding_value))
        return torch.stack(padded_sequences, dim=0)

class ZiyaGenerator_vllm(BaseGenerator):
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
            top_p = 0.95,
            top_k = -1,
            max_tokens = 2048,
            repetition_penalty = 1.0,
            temperature=0.8, )

    def generate(self, queries, k = 5, generation_type="normal", **kwargs):
        responses = []
        if generation_type == "uncertainty":
            queries = [f"{q}\n对于上述问题，请简短地回答你由于缺乏对应的具体知识或信息，因此不知道或不确定答案。" for q in queries]
        all_queries = []
        for q in queries:
            all_queries.extend([f"<human>:{q.strip()}\n<bot>:"]*k)
        
        outputs = self.model.generate(all_queries, self.sampling_params)
        outputs = [o.outputs[0].text.strip() for o in outputs]
        outputs = [outputs[i:i+k] for i in range(0,len(outputs),k)]
        return outputs