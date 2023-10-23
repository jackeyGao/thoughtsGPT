from typing import List
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel
from thoughts_gpt.core.embedding import FolderIndex
from thoughts_gpt.core.const import SUGGESTED_QUESTION_PREFIX
from thoughts_gpt.core.qa import get_suggested_questions

map_template = """以下是一组文件
{text}
根据此文档列表，请确定主题
答案："""

map_prompt = PromptTemplate.from_template(map_template)

reduce_template = """使用下面提供的文档摘录（排名不分先后）作为来源，将这些内容提炼成主要主题的最终综合摘要并推理出{suggested_questions_limit}个联想问题。始终根据文档的摘录推理个{suggested_questions_limit}联想问题。你应该始终分析并推理出答案背后的{suggested_questions_limit}个相关的联想问题问题。始终在您的答案中包含“联想问题”部分

举个例子:

=========
摘录部分
主题是刘备、关羽和张飞三人联合起义，救汉室于危亡。

主题是兄弟情义和勇敢的斗争精神。

根据此文档列表，主题可能是武大郎与猛虎的战斗。

主题是关于武大郎与一只猛虎的斗争。
------------
答案：刘备、关羽和张飞三人联合起义，救汉室于危亡，展现了兄弟情义和勇敢的斗争精神。
综合摘要: 这些文档摘录描绘了刘备、关羽和张飞三人的故事，他们联合起义，拯救了汉室于危亡之中，展现了兄弟情义和勇敢的斗争精神。他们以拯救国家为使命，共同抵抗外敌，积极参与战斗，表现出了坚定的决心和奉献精神。他们之间的兄弟情义也在各种困难和考验中得到了体现，彼此相互支持和帮助。
联想问题:
1. 这个故事中的三位英雄是如何展现勇敢的斗争精神的？
2. 兄弟情义对于他们的联合起义起到了什么样的作用？
3. 这个故事中的主题与现实生活中的英雄故事有哪些相似之处？
=========
摘录部分: 
{text}
=========
答案:
"""

def get_reduce_template(suggested_questions_limit=5):
    return  PromptTemplate.from_template(
        reduce_template.format(
            suggested_questions_limit=suggested_questions_limit, 
            text='{text}'
        )
    )


class SummarizerWithSources(BaseModel):
    answer: str
    suggested_questions: List[str]


def summarizer_folder(
    folder_index: FolderIndex,
    llm: BaseChatModel,
    suggested_questions_limit: int = 5,
) -> SummarizerWithSources:
    """Summarizer a folder index

    Args:
        query (str): The query to search for.
        folder_index (FolderIndex): The folder index to search.
        return_all (bool): Whether to return all the documents from the embedding or
        just the sources for the answer.
        model (str): The model to use for the answer generation.
        **model_kwargs (Any): Keyword arguments for the model.

    Returns:
        SummarizerWithSources: The answer and the source documents.
    """

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        verbose=True,
        map_prompt=map_prompt,
        combine_prompt=get_reduce_template(suggested_questions_limit),
    )

    relevant_docs = folder_index.index.similarity_search(" ")

    summaries = chain.run(relevant_docs)

    suggested_questions = get_suggested_questions(summaries)

    answer = summaries.split(f"{SUGGESTED_QUESTION_PREFIX}:")[0]

    return SummarizerWithSources(
        answer=answer,
        suggested_questions=suggested_questions
    )

