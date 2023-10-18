# flake8: noqa
from langchain.prompts import PromptTemplate
from langchainhub import Client as HubClient

## Use a shorter template to reduce the number of tokens in the prompt
# https://smith.langchain.com/hub/rlm/rag-prompt

hub = HubClient()

prompt = hub.pull("rlm/rag-prompt")

default_template = """使用提供的文档摘录（排名不分先后）作为来源，创建最终答案并推理出三个联想问题。始终根据文档的摘录推理三个联想问题。始终在您的答案中包含“来源”部分，仅引用回答问题所需的最少来源。如果您无法回答问题，只需说明您没有足够的信息来回答问题，并将“来源”部分留空。仅使用提供的文件，不要试图编造答案。你应该始终分析并推理出答案背后的三个相关的联想问题问题。始终在您的答案中包含“联想问题”部分

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

STUFF_PROMPT = PromptTemplate(
    template=default_template, input_variables=["summaries", "question"]
)


def get_prompt(template: str=default_template) -> PromptTemplate:
    return PromptTemplate(
        template=template, input_variables=["summaries", "question"]
    )