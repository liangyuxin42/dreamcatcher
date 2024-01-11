from .base_scorer import BaseScorer

import openai
openai.api_key = "your key"

TEMPLATE = "请你扮演批改作业的老师，对比正确答案，判断学生答案是否正确。\
    \n下面给出一个问题和对应的正确答案：\
    \n问题：{question}\n正确答案：{right_answer}\
    \n\n下面是学生的答案：\
    \n答案：{generated_answer}\
    \n\n请根据与正确答案的一致程度来判断学生的答案是否正确。\
    \n如果学生的答案中包含与正确答案不一致的事实，如时间错误，人物错误等，需要判断为错误。\
    \n如果学生的答案缺少问题中被问到的，并且包括在正确答案中的关键信息，也需要判断为错误。\
    \n如果学生的答案是对正确答案的扩展，或语义与正确答案一致，则判断为正确。\
    \n按照如下格式回答：{\"reason\":判断理由,\"answer\":正确/错误}"


class GPTScorer(BaseScorer):
    def __init__(self,template=TEMPLATE):
        super().__init__()
        self.template = template

    def gpt_api(self,text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}]
            )

        return response['choices'][0]["message"]['content']
    
    def score(self, data):
        question = data["question"]
        right_answer = data["answer"]
        # source = data["source"]
        for j,generation in enumerate(data["generation"]):
            if generation["type"]=="uncertainty":
                continue
            print(generation)
            generated_answer = generation["text"][:512]
            query = self.template.replace("{question}",question).replace("{right_answer}",right_answer).replace("{generated_answer}",generated_answer)
            gpt_judgemented = "gpt_judgement" in generation
            error_count = 0
            while not gpt_judgemented and error_count<3:
                if "type" in generation and generation["type"]=="uncertainty":
                    continue
                try:
                    g = eval(self.gpt_api(query))
                    data["generation"][j]["gpt_judgement"] = g["answer"]
                    data["generation"][j]["gpt_judgement_reason"] = g["reason"]
                    data["generation"][j]["gpt_judgement_score"] = 1 if "正确" in g["answer"] else 0
                    gpt_judgemented = True
                except:
                    print("error")
                    error_count+=1
                    continue
            if not gpt_judgemented:
                data["generation"][j]["gpt_judgement"] = "error"
                data["generation"][j]["gpt_judgement_reason"] = "error"
                data["generation"][j]["gpt_judgement_score"] = 0.5
                
        return data