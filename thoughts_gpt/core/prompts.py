# flake8: noqa
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.stuff_prompt import template # this template
from thoughts_gpt.core.const import DOCUMENT_SUMMARIES_CONTENT
from thoughts_gpt.core.const import DOCUMENT_SUMMARIES_SOURCE

## Use a shorter template to reduce the number of tokens in the prompt
# https://smith.langchain.com/hub/rlm/rag-prompt

default_template = """使用提供的文档摘录（排名不分先后）作为来源，创建最终答案并推理出{suggested_questions_limit}个联想问题。始终根据文档的摘录推理个{suggested_questions_limit}联想问题。始终在您的答案中包含“来源”部分，仅引用回答问题所需的最少来源。如果您无法回答问题，只需说明您没有足够的信息来回答问题，并将“来源”部分留空。仅使用提供的文件，不要试图编造答案。你应该始终分析并推理出答案背后的{suggested_questions_limit}个相关的联想问题问题。始终在您的答案中包含“联想问题”部分

---------
问题: “一带一路”是什么?
=========
内容: 一带一路倡议（BRI，或 B&R[1]），在中国被称为“一带一路”（中文：一带一路；拼音：Yīdài Yīlù）或简称 OBOR/1B1R，[2] 是一项全球基础设施发展中国政府2013年采取的战略是对150多个国家和国际组织进行投资。
来源: 1-32
内容: 总部位于伦敦的经济与商业研究咨询中心表示，到 2040 年，“一带一路”倡议可能会使世界 GDP 每年增加 7.1 万亿美元，而且随着基础设施的改善减少“阻碍世界贸易的摩擦”，这种好处将是“广泛的”。 CEBR 还得出结论，如果全球基础设施倡议取得进展并获得动力，该项目可能会吸引更多国家加入。
来源: 1-33
内容: 支持者称赞“一带一路”倡议有潜力提高全球GDP，特别是发展中国家的GDP。然而，也有人批评侵犯人权和环境影响，以及对债务陷阱外交导致新殖民主义和经济帝国主义的担忧。这些不同的观点是激烈辩论的主题。
来源: 1-30
=========
最终答案: 一带一路倡议（BRI，或 B&R[1]），在中国被称为“一带一路”（中文：一带一路；拼音：Yīdài Yīlù）或简称 OBOR/1B1R，[2] 是一项全球基础设施发展中国政府2013年采取的战略是对150多个国家和国际组织进行投资。
来源: 1-32
联想问题: 
- “一带一路”的成果是什么？
- “一带一路的挑战是什么”？
- “一带一路”发起国家是哪个?

---------
问题: {question}
=========
{summaries}
=========
最终答案: 

"""

qtype_prompt = """你只能回答两个词qa和summarization, 你需要分析问题推理意图，如果是问题是寻求答案请输出“qa”，如果问题是希望总结下内容请输出"summarization"？ 你永远输出这两个单词的一个， 不要做任何内容输出。

比如
========
问题: 文中的作者结婚了嘛
输出: qa

问题: 帮我总结下这篇内容
输出: summarization

问题: What is this document about?
输出: summarization

问题: 这篇文档主要讲了什么?
输出: summarization

问题: 大概内容是什么?
输出: summarization

========
问题: {question}
输出: 

"""

summarization_template = """使用下面提供的文档摘录（前面的内容优先）作为来源，将这些内容提炼成主要问题的最终综合摘要并推理出{suggested_questions_limit}个联想问题。始终根据文档的摘录推理个{suggested_questions_limit}联想问题。你应该始终详细的分析并推理出答案背后的{suggested_questions_limit}个相关的联想问题问题。始终在您的答案中包含“联想问题”部分，始终根据提供的摘要详细的分析出总结，并通过列表组织这些摘要. 

举个例子:

=========
摘录部分:
孙悟空三次打死了白骨精，拯救了师父，师父对他的态度由怀疑到肯定。

主题是刘备、关羽和张飞三人联合起义，救汉室于危亡。

主题是兄弟情义和勇敢的斗争精神。

根据此文档列表，主题可能是武大郎与猛虎的战斗。

主题是关于武大郎与一只猛虎的斗争。

问题: 总结下刘备、关于和张飞的故事

答案：
- 刘备、关羽和张飞三人联合起义，救汉室于危亡，展现了兄弟情义和勇敢的斗争精神。
- 刘备、关羽和张飞三人的故事，他们联合起义，拯救了汉室于危亡之中，展现了兄弟情义和勇敢的斗争精神。
- 他们以拯救国家为使命，共同抵抗外敌，积极参与战斗，表现出了坚定的决心和奉献精神。
- 他们之间的兄弟情义也在各种困难和考验中得到了体现，彼此相互支持和帮助。
联想问题:
1. 这个故事中的三位英雄是如何展现勇敢的斗争精神的？
2. 兄弟情义对于他们的联合起义起到了什么样的作用？
3. 这个故事中的主题与现实生活中的英雄故事有哪些相似之处？

=========
摘录部分: 
{text}
问题: {question}
答案:
"""

def get_summarization_prompt(question, suggested_questions_limit=5, ):
    return  PromptTemplate.from_template(
        summarization_template.format(
            suggested_questions_limit=suggested_questions_limit, 
            question=question,
            text='{text}'
        )
    )

DOCUMENT_PROMPT = PromptTemplate(
    template= DOCUMENT_SUMMARIES_CONTENT \
    + ": {page_content}\n" \
    + DOCUMENT_SUMMARIES_SOURCE + ": {source}",
    input_variables=["page_content", "source"],
)

ENGLISH_DOCUMENT_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}", # if english prompt
    input_variables=["page_content", "source"],
)

STUFF_PROMPT = PromptTemplate(
    template=default_template, 
    input_variables=["summaries", "question", "suggested_questions_limit"]
)

QTYPE_PROMPT = PromptTemplate(
    template=qtype_prompt, 
    input_variables=["question"]
)

def get_prompt(qtype: str = "qa", **kwargs) -> PromptTemplate:
    if qtype == "qa":
        return STUFF_PROMPT
    else:
        return get_summarization_prompt(**kwargs)