from typing import List
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from thoughts_gpt.core.prompts import STUFF_PROMPT
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel
from thoughts_gpt.core.embedding import FolderIndex
from thoughts_gpt.core.prompts import DOCUMENT_PROMPT
from thoughts_gpt.core.const import SUGGESTED_QUESTION_PREFIX
from thoughts_gpt.core.const import DOCUMENT_SUMMARIES_SOURCE


class AnswerWithSources(BaseModel):
    answer: str
    sources: List[Document]
    match_sources: List[Document]
    suggested_questions: List[str]
    summaries: str
    prompt_length: int
    original_anwser: str


def query_folder(
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

    chain = load_qa_with_sources_chain(
        llm=llm,
        chain_type="stuff",
        prompt=stuff_prompt,
        document_prompt=DOCUMENT_PROMPT
    )

    relevant_docs = folder_index.index.similarity_search(query, k=k)

    summaries = chain._get_inputs(relevant_docs)
    prompt_length = chain.prompt_length(
        relevant_docs, question=query, 
        suggested_questions_limit=suggested_questions_limit
    )

    result = chain(
        {
            "input_documents": relevant_docs, 
            "question": query,
            "suggested_questions_limit": suggested_questions_limit,
        }, return_only_outputs=True
    )

    sources = relevant_docs

    match_sources = get_sources(result["output_text"], folder_index)

    # convert
    _source_map = { s.metadata["source"]: s for s in match_sources }
    for source in sources:
        if source.metadata["source"] in _source_map:
            source.metadata["source"] = f"🔥 {source.metadata['source']}"

    answer = result["output_text"].split("SOURCES: ")[0]

    suggested_questions = get_suggested_questions(answer)
    answer = answer.split(f"{SUGGESTED_QUESTION_PREFIX}:")[0]

    return AnswerWithSources(
        answer=answer, 
        sources=sources, 
        match_sources=match_sources, 
        suggested_questions=suggested_questions,
        summaries=summaries["summaries"],
        prompt_length=prompt_length,
        original_anwser=result["output_text"],
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

    return maybe_questions
        

