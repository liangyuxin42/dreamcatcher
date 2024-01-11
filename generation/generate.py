import argparse
import pandas as pd
import jsonlines
import yaml
from generators import LlamaGenerator, QwenGenerator, ZiyaGenerator, LlamaGenerator_vllm,QwenGenerator_vllm,ZiyaGenerator_vllm

GENERATOR = {
    "llama2": LlamaGenerator,
    "qwen": QwenGenerator,
    "ziya": ZiyaGenerator,
    "llama2_vllm": LlamaGenerator_vllm,
    "qwen_vllm": QwenGenerator_vllm,
    "ziya_vllm": ZiyaGenerator_vllm,
}

def generate(config):
    print(config["input_data_path"])
    df = pd.read_json(config["input_data_path"],lines=True)[:5].reset_index(drop=True)
    print(df.shape)
    if "generation" not in df.columns:
        df["generation"] = [[] for _ in range(len(df))]
    queries = df["question"].tolist()
    generator = GENERATOR[config["model_name"]](config)
    for generation_type in config["generation_type"]:
        print("generation_type:",generation_type)
        responses = generator.generate(queries,generation_type=generation_type)
        print(len(responses))
        with jsonlines.open(config["output_data_path"],mode="w") as writer:
            for i,response in enumerate(responses):
                generation = df["generation"].tolist()[i]
                generation.extend([{"text":r,"type":generation_type} for r in response])
                d = {
                    "question":queries[i],
                    "generation":generation,
                }
                for key in ["answer","source","task","benchmark",]:
                    if key in df.columns:
                        d[key] = df[key].tolist()[i]
                writer.write(d)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config_path", type=str)
    args = args_parser.parse_args()
    print("args",args)
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    generate(config)
    print("Done!")