<h1 style="text-align: center;">Dreamcatcher :star: :spider_web: :star: </h1>

Read this in [Chinese](readme-zh.md).


Dreamcacher is an automatic hallucination annotation tool that uses a combination of tools to annotate and rank the degree of hallucination of model responses.

Methods of assessing hallucinations include:

1. Consistency with the correct answer

    If the correct answer exists, consistency with the correct answer can be checked to determine whether the answer is correct. Consistency can be judged by cosine similarity of text embedding and unigram overlap between texts.

2. self-consistency

    Several research works have shown that self-consistency can be used for hallucination estimation, i.e., if the model is generating a hallucination, the self-consistency will be lower. Consistency is also judged by the similarity method.

3. Model Knowledge Detection

    The hidden state in the model generation process can be used to determine whether the model has corresponding knowledge.

Dreamcacher is labeled in the order of scorer score -> ranker rank, or it can be scored alone without ranking.

---

## Usage

    sh run.sh 

The run.sh specifies the config file to use, and the config specifies the data to be used, the path and parameters of the scorer and ranker.

### scorer

The scoring is done in dreamcatcher.py by calling the scorers specified in config, each of which adds a score item to each sample.

Currently available scorers include:

1. similarity_scorer
    - sim2answer_score: calculates the similarity between the generated answer and the correct answer using embedding_model
    - sim2gen_score: computes the average similarity of a generated answer to other generated answers using embedding_model

2. unigram_scorer
    - top_k_avg_probs_score: unigram overlap with correct answers

3. probe_scorer
    - Score input questions with a **pre-trained** knowledge probe to determine if the question is within the model's knowledge.

4. gpt_scorer
    - (not recommended, gpt calling time is too long, if necessary, you can preprocess the data in addition to call gpt judgment)

5. rm_scorer
    - Use generic rm to score the quality of the resulting language. (Only in the sorting of the selection of uncertainty samples need, if only scoring you can not use this item.)

### ranker

Currently available rankers include:

1. SimpleRanker
    Use the sum of scoring results:["probe_score", "unigram_overlap_with_answer_score", "sim2answer_score", "sim2gen_score"] to make a judgment, above the threshold is judged to be correct, below the threshold is judged to be incorrect.

    Sort the results according to [correct, uncertain, wrong], if there is no correct/wrong result above the threshold in the generated results, then omit the corresponding item. For example, if there are no answers above the threshold judged as correct in the five results generated, the ordering is [Uncertain, Wrong].

    The threshold is determined by the incoming up_percentile and down_percentile parameters, samples with a sort number < 2 are filtered, and half of the samples are typically filtered when up_percentile=70 and down_percentile=30. (up_percentile=70 means that those scored above 70% are judged as correct, and down_percentile=30 means that those scored below 30% are judged as incorrect. Since it is judged by percentage, it may not be accurate when the amount of data is small.)

### 输入&输出格式

Input: refer to data/test_generated.jsonl, "type": "normal"/"uncertainty" corresponds to the normal generation of results and uncertainty in the generation of results, only scoring is not sorted if you do not need uncertainty in the generation of results.

Output: reference /data/test_autolabeled.jsonl and /test_autoranked.jsonl


### 模型知识探测

To train the above probe_scorer, we need to label about 1k data with other scorers first.

Example: /data/test_labeled_for_probe.jsonl,Probe experiments need data including "question" and "category",category $\in$ ["all-right", "all-wrong", "partial-right"]

Knowledge probing code:
    /probe/probing.ipynb
