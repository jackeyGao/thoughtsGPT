from typing import List, Dict
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.summarize import load_summarize_chain
from thoughts_gpt.core.prompts import STUFF_PROMPT
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel
from thoughts_gpt.core.embedding import FolderIndex
from thoughts_gpt.core.prompts import DOCUMENT_PROMPT
from thoughts_gpt.core.const import SUGGESTED_QUESTION_PREFIX
from thoughts_gpt.core.const import DOCUMENT_SUMMARIES_SOURCE
from thoughts_gpt.core.prompts import get_summarization_prompt


class AnswerWithSources(BaseModel):
    answer: str
    sources: List[Document]
    match_sources: List[Document]
    suggested_questions: List[str]
    variables: Dict
    prompt_length: int
    original_anwser: str


def query_folder(
    qtype: str,
    query: str,
    folder_index: FolderIndex,
    llm: BaseChatModel,
    k: int = 5,
    stuff_prompt: PromptTemplate=STUFF_PROMPT,
    suggested_questions_limit: int = 5,
) -> AnswerWithSources:
    """Queries a folder index for an answer.

    Args:
        query (str): The query to search for.
        folder_index (FolderIndex): The folder index to search.
        return_all (bool): Whether to return all the documents from the embedding or
        just the sources for the answer.
        model (str): The model to use for the answer generation.
        **model_kwargs (Any): Keyword arguments for the model.

    Returns:
        AnswerWithSources: The answer and the source documents.
    """

    relevant_docs = folder_index.index.similarity_search(query, k=k)

    if qtype == 'qa':
        chain = load_qa_with_sources_chain(
            llm=llm,
            chain_type="stuff",
            prompt=stuff_prompt,
            document_prompt=DOCUMENT_PROMPT
        )

        result = chain(
            {
                "input_documents": relevant_docs, 
                "question": query,
                "suggested_questions_limit": suggested_questions_limit,
            }, return_only_outputs=True
        )
        answer_res = result["output_text"]
    else:
        chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",
            verbose=False,
            # map_prompt=map_prompt,
            # combine_prompt=get_summarization_prompt(suggested_questions_limit),
            prompt=get_summarization_prompt(suggested_questions_limit),
        )

        answer_res = chain.run(relevant_docs)

    variables = chain._get_inputs(relevant_docs)
    prompt_length = chain.prompt_length(
        relevant_docs, question=query, 
        suggested_questions_limit=suggested_questions_limit
    )

    match_sources = get_sources(answer_res, folder_index)

    # convert
    _source_map = { s.metadata["source"]: s for s in match_sources }
    for source in relevant_docs:
        if source.metadata["source"] in _source_map:
            source.metadata["source"] = f"🔥 {source.metadata['source']}"

    suggested_questions = get_suggested_questions(answer_res)
    answer = answer_res.split(f"{SUGGESTED_QUESTION_PREFIX}:")[0]

    return AnswerWithSources(
        answer=answer, 
        sources=relevant_docs, 
        match_sources=match_sources, 
        suggested_questions=suggested_questions,
        variables=variables,
        prompt_length=prompt_length,
        original_anwser=answer_res,
    )

def get_sources_key(line: str) -> List[str]:
    prefix_list = [f"{DOCUMENT_SUMMARIES_SOURCE}: "]
    keys = []
    for prefix in prefix_list:
        if not line.startswith(prefix):
            continue

        for sec in line.replace(prefix, '').split(','):
            sec = sec.strip()
            if '-' not in sec:
                continue

            keys.append(sec.strip())
    return keys

def get_sources(answer: str, folder_index: FolderIndex) -> List[Document]:
    """Retrieves the docs that were used to answer the question the generated answer."""

    source_keys = []
    for line in answer.split('\n'):
        source_keys += get_sources_key(line)

    source_docs = []
    for file in folder_index.files:
        for doc in file.docs:
            if doc.metadata["source"] in source_keys:
                source_docs.append(doc)
    return source_docs


def get_suggested_questions(answer: str) -> List[str]:
    prefix_list = [f"{SUGGESTED_QUESTION_PREFIX}:"]

    maybe_questions = []
    for prefix in prefix_list:
        if prefix not in answer:
            continue

        for line in answer.split(prefix)[1].split('\n'):
            if line.startswith('- '):
                maybe_questions.append(line.replace('- ', ''))
            else:
                maybe_questions.append(line)

    return [ q.strip() for q in maybe_questions if q ]
        

