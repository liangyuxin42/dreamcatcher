<h1 style="text-align: center;">Dreamcatcher :spider_web: :feather: </h1>

Dreamcacher是一个幻觉自动标注工具，使用多种工具组合对模型回答是否存在幻觉进行标注和排序。

对幻觉的判断方法包括：

1. 与正确答案的一致性

    如果存在正确答案，可以检查与正确答案的一致性来判断回答是否正确。一致性可以通过文本embedding的cosine similarity 和 文本间的unigram overlap来判断。

2. self-consistency

    多个研究工作表明，多次生成间的一致性可以用于幻觉判断，即：如果模型在生成幻觉则多次生成间一致性会较低。一致性判断同样通过相似度和unigram overlap两种方法判断。

3. 模型知识探测结果

    模型生成过程中的hidden state可以用于判断模型是否具有对应知识。

Dreamcacher按照 scorer打分 -> ranker排序的顺序标注，也可以单独打分不进行排序。

---

## Usage

    sh run.sh 

run.sh中指定使用的config文件，config中指定需要使用的数据，scorer和ranker的路径和参数。

### scorer

在dreamcatcher.py中调用config中指定的scorer打分，每个scorer为每条样本增加一个分数项。

目前可用的scorer包括：

1. similarity_scorer
    - sim2answer_score：用embedding_model计算生成答案与正确答案的相似度
    - sim2gen_score：用embedding_model计算生成答案与其他生成答案的平均相似度

2. unigram_scorer
    - unigram_overlap_with_answer_score: 与其他生成结果的unigram overlap
    - top_k_avg_probs_score：与正确答案的unigram overlap

3. probe_scorer
    - 用**预先训练好的**事实性探测模型对输入问题打分，判断问题是否在模型知识范围内。

4. gpt_scorer
    - (不建议使用，调用gpt时间太长，需要的话可以预处理数据时另外调用gpt判断)

5. rm_scorer
    - 用通用rm为生成结果语言质量打分。（仅在排序时选择不确定性样本需要，如果只打分可以不用这一项。）

### ranker

目前可用的ranker包括：

1. SimpleRanker
    使用打分结果:["probe_score","unigram_overlap_with_answer_score","sim2answer_score","sim2gen_score"]之和做判断，高于阈值判断为正确，低于阈值判断为错误。

    按照[正确，不确定，错误]排序，如果生成结果中没有超过阈值的正确/错误结果，则省略对应项。比如生成的五条结果中没有高于阈值被判断为正确的答案，则排序为[不确定，错误]。

    阈值由传入的up_percentile和down_percentile参数决定，排序数<2的样本会被过滤，当up_percentile=70，down_percentile=30时一般会有一半的样本被过滤。（up_percentile=70即打分高于70%的被判断为正确，down_percentile=30即打分低于30%的被判断为错误。由于按照百分比判断，数据少的时候可能不准确。）

### 输入&输出格式

输入：参考data/test_generated.jsonl, "type":"normal"/"uncertainty"分别对应正常生成结果和不确定性生成结果，只打分不排序的话不需要不确定性生成结果。

    {"question":"南方农村报是哪个省的农业报纸?","answer":"广东省","type":"地理","generation":[{"text":"南方农村报是广东省的农业报纸。它是由广东省农业厅主办的，主要面向广东省的农民和农业从业者，报道农业政策、农业技术、农业市场等方面的信息。","type":"normal"},{"text":"南方农村报是广东省的农业报纸。","type":"normal"},{"text":"《南方农村报》是广东省的农业报纸。","type":"normal"},{"text":"南方农村报是广东省的农业报纸。它是一份全国性的报纸，主要面向农村读者，以及从事农业相关行业的人士。它以深度报道和专业分析为主，涵盖了农业政策、市场动态、技术创新等多个方面的内容。同时，南方农村报还积极关注农村经济、社会发展等问题，为农民提供实用的信息和服务。","type":"normal"},{"text":"南方农村报是广东省的农业报纸。","type":"normal"},{"text":"对不起，我无法回答这个问题，因为我没有足够的信息来确定南方农村报所属的省份。可以请问更多关于这份报纸的信息吗？","type":"uncertainty"},{"text":"对不起，我无法回答这个问题，因为我的知识库中没有相关信息。","type":"uncertainty"},{"text":"很抱歉，我不知道南方农村报是哪个省的农业报纸。","type":"uncertainty"},{"text":"对不起，我无法回答这个问题，因为我无法确定南方农村报是哪个省的农业报纸。如果您能提供更多信息，我将尽力回答您的问题。","type":"uncertainty"},{"text":"对不起，我无法回答这个问题，因为我没有足够的信息来确定南方农村报是哪个省的农业报纸。建议您查阅相关资料或咨询相关机构以获取更多信息。","type":"uncertainty"}]}

输出：参考/data/test_autolabeled.jsonl 和 /test_autoranked.jsonl


### 模型知识探测
训练上述的probe_scorer，需要先使用其他scorer标注大约1k条数据。

示例：/data/test_labeled_for_probe.jsonl,探测实验需要数据包括"question"和"category",category $\in$ ["all-right","all-wrong","partial-right"]

知识探测代码：
    /probe/probing.ipynb

