from typing import List
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from thoughts_gpt.core.prompts import STUFF_PROMPT
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from thoughts_gpt.core.embedding import FolderIndex
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel


class AnswerWithSources(BaseModel):
    answer: str
    sources: List[Document]
    match_sources: List[Document]
    suggested_questions: List[str]


def query_folder(
    query: str,
    folder_index: FolderIndex,
    llm: BaseChatModel,
    stuff_prompt: PromptTemplate=STUFF_PROMPT,
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
    )

    relevant_docs = folder_index.index.similarity_search(query, k=5)
    result = chain(
        {"input_documents": relevant_docs, "question": query}, return_only_outputs=True
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
    answer = answer.split("联想问题:")[0]

    return AnswerWithSources(
        answer=answer, 
        sources=sources, 
        match_sources=match_sources, 
        suggested_questions=suggested_questions
    )

def get_sources_key(line: str) -> List[str]:
    prefix_list = ["来源: ", "SOURCE: "]
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
    prefix_list = ["联想问题:"]

    maybe_questions = []
    for prefix in prefix_list:
        for line in answer.split(prefix)[1].split('\n'):
            if line.startswith('- '):
                maybe_questions.append(line.replace('- ', ''))

    return maybe_questions
        

